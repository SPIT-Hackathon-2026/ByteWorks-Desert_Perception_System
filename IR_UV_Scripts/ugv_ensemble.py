"""
UGV Weighted-Average Ensemble — PC Integration Layer
=====================================================
Combines three model sources per output:

  CAMOUFLAGE:
    Model A  → LogisticDot (from MicroPython CAMOUF_W / CAMOUF_B)  [weight W_A]
    Model B  → LogisticDot (from MicroPython OnlineRLS adapted)     [weight W_B]
    Model C  → RandomForest rf_camouf.pkl                           [weight W_C]
    ─────────────────────────────────────────────────────────
    ensemble_camouf  = softmax-weighted average of (A, B, C)

  TERRAIN HAZARD:
    Model A  → LogisticDot (TERRAIN_W / TERRAIN_B)                  [weight W_A]
    Model B  → LogisticDot (OnlineRLS adapted TERRAIN weights)       [weight W_B]
    Model C  → RandomForest rf_terrain.pkl                           [weight W_C]
    ─────────────────────────────────────────────────────────
    ensemble_terrain = softmax-weighted average of (A, B, C)

Weights can be set manually (direct floats) or auto-tuned from a
validation set via WeightOptimizer (grid search over simplex).

Usage:
    python ugv_ensemble.py                        # demo with random inputs
    python ugv_ensemble.py --tune val.csv         # tune weights from CSV
    python ugv_ensemble.py --live                 # stdin streaming mode

Serial streaming format (from UGV over UART/USB):
    Each line: "trans_rate,ir_ratio,variance,asymmetry,dist_cm"
    Example  : "0.425,0.490,0.250,0.010,8.3"
"""

import argparse
import json
import math
import sys
import os
import numpy as np
import joblib
import pandas as pd
from itertools import product as iproduct

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
_DIR   = os.path.dirname(os.path.abspath(__file__))
RF_CAMOUF_PATH  = os.path.join(_DIR, "models", "rf_camouf.pkl")
RF_TERRAIN_PATH = os.path.join(_DIR, "models", "rf_terrain.pkl")
WEIGHTS_PATH    = os.path.join(_DIR, "models", "ensemble_weights.json")

# ──────────────────────────────────────────────────────────────
# DEFAULT MANUAL WEIGHTS  (sum-to-1 not required — normalised internally)
# ──────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "camouf" : [0.25, 0.30, 0.45],   # [logistic_fixed, logistic_rls, rf]
    "terrain": [0.20, 0.30, 0.50],   # RF gets highest weight (best CV score)
}

# Calibrated ensemble thresholds (lower than per-model thresholds because
# the MicroPython logistic biases are intentionally conservative placeholders;
# the RF carries 45-50% weight and outputs crisp 0/1 probabilities, so the
# weighted average for a true positive sits around 0.45-0.57).
CAMOUF_ENS_THRESHOLD  = 0.40
TERRAIN_ENS_THRESHOLD = 0.40

# ──────────────────────────────────────────────────────────────
# REPLICA OF MICROPYTHON FIXED WEIGHTS  (copy from your device)
# Update these whenever you flash new weights to the UGV
# ──────────────────────────────────────────────────────────────
CAMOUF_W_FIXED   = [0.75, 0.85, 0.90, 0.65]
CAMOUF_B_FIXED   = -4.5

TERRAIN_W_FIXED  = [0.90, -0.80, 0.60, 1.10]
TERRAIN_B_FIXED  = -3.0

# Placeholder — replace with actual RLS-adapted weights exported from UGV
# (read CAMOUF_W / TERRAIN_W lists printed by the device after RLS updates)
CAMOUF_W_RLS     = [0.75, 0.85, 0.90, 0.65]   # update from device
CAMOUF_B_RLS     = -4.5
TERRAIN_W_RLS    = [0.90, -0.80, 0.60, 1.10]   # update from device
TERRAIN_B_RLS    = -3.0

MAX_RANGE_CM     = 400.0


# ══════════════════════════════════════════════════════════════
# MATH
# ══════════════════════════════════════════════════════════════
def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def _logistic(weights, bias, features):
    return _sigmoid(sum(w * f for w, f in zip(weights, features)) + bias)

def _normalise(weights):
    s = sum(weights)
    if s == 0:
        n = len(weights)
        return [1.0 / n] * n
    return [w / s for w in weights]

def _weighted_avg(probs, weights):
    w = _normalise(weights)
    return sum(p * wi for p, wi in zip(probs, w))


# ══════════════════════════════════════════════════════════════
# FEATURE DERIVATION  (mirrors MicroPython exactly)
# ══════════════════════════════════════════════════════════════
def derive_features(trans_rate, ir_ratio, variance, asymmetry, dist_cm):
    """
    Returns (camouf_feats, terrain_feats) from raw sensor values.

    Input:
        trans_rate  — IR transition rate  [0, 1]
        ir_ratio    — fraction of reflective IR readings [0, 1]
        variance    — Bernoulli variance = ir_ratio*(1-ir_ratio)
        asymmetry   — |ir_ratio - 0.5|
        dist_cm     — ultrasonic distance in cm (< 1 = unknown)
    """
    camouf_feats = [trans_rate, ir_ratio, variance, asymmetry]

    if dist_cm < 1.0:
        clearance_norm = 0.5
    else:
        clearance_norm = min(dist_cm / MAX_RANGE_CM, 1.0)

    combined_risk  = (1.0 - clearance_norm) * ir_ratio
    terrain_feats  = [ir_ratio, clearance_norm, variance, combined_risk]

    return camouf_feats, terrain_feats


# ══════════════════════════════════════════════════════════════
# ENSEMBLE
# ══════════════════════════════════════════════════════════════
class UGVEnsemble:
    """
    Weighted-average ensemble for camouflage and terrain hazard.

    Parameters
    ----------
    rf_camouf_path, rf_terrain_path : str
        Paths to joblib-serialised RandomForest models.
    weights : dict  {"camouf": [w_a, w_b, w_c], "terrain": [w_a, w_b, w_c]}
        Manual weights. Normalised internally — values don't need to sum to 1.
    camouf_threshold, terrain_threshold : float
        Decision boundary for binary classification.
    """

    def __init__(
        self,
        rf_camouf_path  = RF_CAMOUF_PATH,
        rf_terrain_path = RF_TERRAIN_PATH,
        weights         = None,
        camouf_threshold  = CAMOUF_ENS_THRESHOLD,
        terrain_threshold = TERRAIN_ENS_THRESHOLD,
    ):
        self.rf_c  = joblib.load(rf_camouf_path)
        self.rf_t  = joblib.load(rf_terrain_path)
        self.w     = weights if weights else DEFAULT_WEIGHTS
        self.thr_c = camouf_threshold
        self.thr_t = terrain_threshold

    # ── single-sample inference ───────────────────────────────
    def predict_proba(self, camouf_feats, terrain_feats):
        """
        Returns dict with per-model and ensemble probabilities.

        camouf_feats  : list[float] len 4
        terrain_feats : list[float] len 4
        """
        # ── Camouflage ────────────────────────────────────────
        p_c_a = _logistic(CAMOUF_W_FIXED, CAMOUF_B_FIXED, camouf_feats)
        p_c_b = _logistic(CAMOUF_W_RLS,   CAMOUF_B_RLS,   camouf_feats)
        p_c_rf= float(self.rf_c.predict_proba([camouf_feats])[0][1])
        ens_c = _weighted_avg([p_c_a, p_c_b, p_c_rf], self.w["camouf"])

        # ── Terrain hazard ────────────────────────────────────
        p_t_a = _logistic(TERRAIN_W_FIXED, TERRAIN_B_FIXED, terrain_feats)
        p_t_b = _logistic(TERRAIN_W_RLS,   TERRAIN_B_RLS,   terrain_feats)
        p_t_rf= float(self.rf_t.predict_proba([terrain_feats])[0][1])
        ens_t = _weighted_avg([p_t_a, p_t_b, p_t_rf], self.w["terrain"])

        return {
            # Per-model probabilities
            "camouf_logistic_fixed" : round(p_c_a,  4),
            "camouf_logistic_rls"   : round(p_c_b,  4),
            "camouf_rf"             : round(p_c_rf,  4),
            "camouf_ensemble"       : round(ens_c,   4),
            "camouf_decision"       : int(ens_c >= self.thr_c),

            "terrain_logistic_fixed": round(p_t_a,  4),
            "terrain_logistic_rls"  : round(p_t_b,  4),
            "terrain_rf"            : round(p_t_rf,  4),
            "terrain_ensemble"      : round(ens_t,   4),
            "terrain_decision"      : int(ens_t >= self.thr_t),
        }

    # ── batch inference from DataFrame ───────────────────────
    def predict_dataframe(self, df):
        """
        Expects columns: trans_rate, ir_ratio, variance, asymmetry, dist_cm
        Returns original df with result columns appended.
        """
        results = []
        for _, row in df.iterrows():
            cf, tf = derive_features(
                row["trans_rate"], row["ir_ratio"],
                row["variance"],   row["asymmetry"],
                row["dist_cm"]
            )
            results.append(self.predict_proba(cf, tf))
        return pd.concat([df.reset_index(drop=True),
                          pd.DataFrame(results)], axis=1)

    # ── update weights (manual) ───────────────────────────────
    def set_weights(self, camouf_weights=None, terrain_weights=None):
        if camouf_weights  is not None: self.w["camouf"]  = list(camouf_weights)
        if terrain_weights is not None: self.w["terrain"] = list(terrain_weights)

    def save_weights(self, path=WEIGHTS_PATH):
        with open(path, "w") as f:
            json.dump(self.w, f, indent=2)
        print(f"[WEIGHTS] Saved → {path}")

    def load_weights(self, path=WEIGHTS_PATH):
        with open(path) as f:
            self.w = json.load(f)
        print(f"[WEIGHTS] Loaded ← {path}")


# ══════════════════════════════════════════════════════════════
# WEIGHT OPTIMIZER  — grid search over simplex
# ══════════════════════════════════════════════════════════════
class WeightOptimizer:
    """
    Tunes ensemble weights by exhaustive grid search on a validation CSV.

    CSV must have columns:
        trans_rate, ir_ratio, variance, asymmetry, dist_cm,
        label_camouf, label_terrain

    Optimises F1 score for each output independently.
    """
    GRID_STEPS = 5    # 5 steps → (5+1)^2 = 36 combos per output (fast)

    def __init__(self, ensemble: UGVEnsemble):
        self.ens = ensemble

    @staticmethod
    def _f1(y_true, y_pred):
        tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
        fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
        fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        return 2 * p * r / (p + r + 1e-9)

    def tune(self, val_csv):
        df   = pd.read_csv(val_csv)
        rows = [
            derive_features(r.trans_rate, r.ir_ratio,
                            r.variance,   r.asymmetry, r.dist_cm)
            for _, r in df.iterrows()
        ]
        camouf_feats  = [x[0] for x in rows]
        terrain_feats = [x[1] for x in rows]
        y_c = df["label_camouf"].tolist()
        y_t = df["label_terrain"].tolist()

        steps = np.linspace(0, 1, self.GRID_STEPS + 1)
        best  = {"camouf": (0.0, None), "terrain": (0.0, None)}

        # Grid over (w_a, w_b) — w_c = 1 - w_a - w_b (normalised anyway)
        for w_a, w_b in iproduct(steps, steps):
            w_c = max(0.0, 1.0 - w_a - w_b)
            ww  = [w_a, w_b, w_c]

            preds_c, preds_t = [], []
            for cf, tf in zip(camouf_feats, terrain_feats):
                p_c_a  = _logistic(CAMOUF_W_FIXED, CAMOUF_B_FIXED, cf)
                p_c_b  = _logistic(CAMOUF_W_RLS,   CAMOUF_B_RLS,   cf)
                p_c_rf = float(self.ens.rf_c.predict_proba([cf])[0][1])
                ens_c  = _weighted_avg([p_c_a, p_c_b, p_c_rf], ww)
                preds_c.append(int(ens_c >= self.ens.thr_c))

                p_t_a  = _logistic(TERRAIN_W_FIXED, TERRAIN_B_FIXED, tf)
                p_t_b  = _logistic(TERRAIN_W_RLS,   TERRAIN_B_RLS,   tf)
                p_t_rf = float(self.ens.rf_t.predict_proba([tf])[0][1])
                ens_t  = _weighted_avg([p_t_a, p_t_b, p_t_rf], ww)
                preds_t.append(int(ens_t >= self.ens.thr_t))

            f1_c = self._f1(y_c, preds_c)
            f1_t = self._f1(y_t, preds_t)

            if f1_c > best["camouf"][0]:
                best["camouf"]  = (f1_c, ww[:])
            if f1_t > best["terrain"][0]:
                best["terrain"] = (f1_t, ww[:])

        print(f"[TUNE] Best camouf  weights={best['camouf'][1]}  F1={best['camouf'][0]:.4f}")
        print(f"[TUNE] Best terrain weights={best['terrain'][1]} F1={best['terrain'][0]:.4f}")
        self.ens.set_weights(best["camouf"][1], best["terrain"][1])
        self.ens.save_weights()
        return best


# ══════════════════════════════════════════════════════════════
# LIVE STDIN STREAMING MODE
# ══════════════════════════════════════════════════════════════
def live_mode(ens):
    """
    Read sensor lines from stdin and print ensemble decisions.
    Expected line format: "trans_rate,ir_ratio,variance,asymmetry,dist_cm"
    """
    print("[LIVE] Waiting for sensor data on stdin (Ctrl-C to quit)...")
    print(f"{'Input':<45} {'Cam':>6} {'Dec':>4}  {'Ter':>6} {'Dec':>4}")
    print("─" * 70)
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = [float(x) for x in line.split(",")]
            if len(parts) != 5:
                raise ValueError("expected 5 comma-separated values")
            tr, ir, va, asy, dist = parts
            cf, tf = derive_features(tr, ir, va, asy, dist)
            r      = ens.predict_proba(cf, tf)
            dec_c  = "CAMO!" if r["camouf_decision"]  else "clear"
            dec_t  = "HAZ!"  if r["terrain_decision"] else "safe "
            print(
                f"{line:<45} "
                f"{r['camouf_ensemble']:>6.3f} {dec_c:>5}  "
                f"{r['terrain_ensemble']:>6.3f} {dec_t:>5}"
            )
        except Exception as e:
            print(f"[WARN] Skipping line ({e}): {line!r}")


# ══════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════
def run_demo(ens):
    print("\n" + "═" * 70)
    print("  UGV Ensemble — Demo Inference")
    print("═" * 70)

    test_cases = [
        # (trans, ir_ratio, variance, asymmetry, dist_cm, description)
        (0.42, 0.49, 0.25, 0.01,   9.0,  "Classic camo + LOW CLEARANCE"),
        (0.40, 0.51, 0.25, 0.01, 150.0,  "Camo pattern only, safe distance"),
        (0.05, 0.92, 0.07, 0.42,  80.0,  "Clear reflective surface, mid dist"),
        (0.04, 0.10, 0.09, 0.40, 250.0,  "Clear absorptive surface, safe"),
        (0.38, 0.47, 0.25, 0.03,   5.5,  "Camo + very low clearance"),
        (0.06, 0.80, 0.16, 0.30,   7.0,  "Reflective + low clearance only"),
        (0.03, 0.50, 0.25, 0.00, 300.0,  "Borderline ir_ratio, high clearance"),
    ]

    hdr = (f"{'Description':<40} "
           f"{'CamA':>6} {'CamB':>6} {'CamRF':>6} {'CamEns':>7} {'Dec':>5}  "
           f"{'TerA':>6} {'TerB':>6} {'TerRF':>6} {'TerEns':>7} {'Dec':>5}")
    print(hdr)
    print("─" * len(hdr))

    for tr, ir, va, asy, dist, desc in test_cases:
        cf, tf = derive_features(tr, ir, va, asy, dist)
        r = ens.predict_proba(cf, tf)
        dc = "CAMO!" if r["camouf_decision"]  else "clear"
        dt = "HAZ!"  if r["terrain_decision"] else "safe"
        print(
            f"{desc:<40} "
            f"{r['camouf_logistic_fixed']:>6.3f} "
            f"{r['camouf_logistic_rls']:>6.3f} "
            f"{r['camouf_rf']:>6.3f} "
            f"{r['camouf_ensemble']:>7.3f} {dc:>5}  "
            f"{r['terrain_logistic_fixed']:>6.3f} "
            f"{r['terrain_logistic_rls']:>6.3f} "
            f"{r['terrain_rf']:>6.3f} "
            f"{r['terrain_ensemble']:>7.3f} {dt:>5}"
        )

    print("\n[WEIGHTS] Camouflage  (fixed / rls / rf):", _normalise(ens.w["camouf"]))
    print("[WEIGHTS] Terrain     (fixed / rls / rf):", _normalise(ens.w["terrain"]))


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UGV Weighted Ensemble")
    parser.add_argument("--tune",  metavar="CSV",  help="Tune weights from validation CSV")
    parser.add_argument("--live",  action="store_true", help="Live stdin streaming mode")
    parser.add_argument("--weights", metavar="JSON", help="Load saved weights JSON")
    args = parser.parse_args()

    ens = UGVEnsemble()

    if args.weights:
        ens.load_weights(args.weights)

    if args.tune:
        opt = WeightOptimizer(ens)
        opt.tune(args.tune)

    if args.live:
        live_mode(ens)
    else:
        run_demo(ens)
