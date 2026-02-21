"""
UGV Multi-Sensor ML Prediction Model
======================================
Self-contained — no sklearn, no external dependencies beyond math.
Works in CPython 3.x and MicroPython.

Three models available (all trained on the same 500-sample dataset):
  1. Logistic Regression  — 87% accuracy | fastest, pure math
  2. Decision Tree        — 86% accuracy | most interpretable, zero-cost inference
  3. Random Forest        — 83% accuracy | load via joblib (CPython only)

Label schema:
  0  SAFE_OPEN            Open ground, no hazards
  1  SAFE_REFLECTIVE      Hard/rock ground, adequate clearance
  2  LOW_CLEARANCE        Any surface but distance < 12 cm
  3  CAMOUFLAGED_OBJECT   Irregular IR flutter pattern detected
  4  TERRAIN_HAZARD       Soft/vegetated, close proximity risk

Features required (7 values):
  transition_rate   fraction of IR state changes in the window   [0..1]
  ir_ratio          fraction of "reflective" (LOW) IR readings   [0..1]
  variance          ir_ratio * (1 - ir_ratio)                    [0..0.25]
  asymmetry         abs(ir_ratio - 0.5)                          [0..0.5]
  distance_cm       ultrasonic ground clearance reading           [0..420]
  clearance_norm    distance_cm / 400.0  clamped to [0,1]        [0..1]
  combined_risk     (1 - clearance_norm) * ir_ratio               [0..1]

Usage:
  features = extract_features(ir_history, distance_cm)
  label, name, confidence = predict(features, model="lr")   # or "dt"
"""

import math

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
LABEL_NAMES = {
    0: "SAFE_OPEN",
    1: "SAFE_REFLECTIVE",
    2: "LOW_CLEARANCE",
    3: "CAMOUFLAGED_OBJECT",
    4: "TERRAIN_HAZARD",
}

FEATURES = [
    "transition_rate", "ir_ratio", "variance", "asymmetry",
    "distance_cm", "clearance_norm", "combined_risk",
]

MAX_RANGE_CM = 400.0

# ──────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (identical to ugv_perception.py v2.1)
# ──────────────────────────────────────────────────────────────

def extract_features(ir_history, distance_cm):
    """
    Compute the 7-element feature vector from raw sensor data.

    Args:
        ir_history   : list of int  — binary IR readings (0=reflective, 1=open)
                        should contain IR_WINDOW (40) samples
        distance_cm  : float — ultrasonic distance, -1.0 if invalid

    Returns:
        list[float] — [transition_rate, ir_ratio, variance, asymmetry,
                        distance_cm, clearance_norm, combined_risk]
    """
    n = len(ir_history)
    if n < 4:
        return [0.0, 0.0, 0.0, 0.0, max(distance_cm, 0.0), 0.0, 0.0]

    transitions  = sum(1 for i in range(1, n) if ir_history[i] != ir_history[i - 1])
    low_count    = ir_history.count(0)
    ir_ratio     = low_count / n
    variance     = ir_ratio * (1.0 - ir_ratio)
    asymmetry    = abs(ir_ratio - 0.5)
    trans_rate   = transitions / n

    d = distance_cm if distance_cm >= 1.0 else 0.0
    clearance_norm = min(d / MAX_RANGE_CM, 1.0)
    combined_risk  = (1.0 - clearance_norm) * ir_ratio

    return [
        round(trans_rate,    4),
        round(ir_ratio,      4),
        round(variance,      4),
        round(asymmetry,     4),
        round(d,             2),
        round(clearance_norm,4),
        round(combined_risk, 4),
    ]


# ──────────────────────────────────────────────────────────────
# MODEL 1 — LOGISTIC REGRESSION
# Trained on raw (unscaled) features.  87% test accuracy.
# ──────────────────────────────────────────────────────────────
# Weights shape: (5 classes, 7 features)
# Each row = weight vector for one class in a One-vs-Rest scheme.
# Prediction = argmax of softmax(X @ W.T + b)

_LR_COEF = [
    # transition  ir_ratio   variance   asymmetry  distance   cl_norm    comb_risk
    [-1.390228,   2.930367, -1.294209,  2.660738,  0.131583,  0.000134,  1.459472],   # 0 SAFE_OPEN
    [-2.443424,   0.547547,  0.727932, -0.966172,  0.130525,  0.000518,  1.128503],   # 1 SAFE_REFLECTIVE
    [-0.279624,   0.988048,  0.162495, -0.271103, -0.454254, -0.001176,  0.965585],   # 2 LOW_CLEARANCE
    [ 5.188144,  -1.000526,  0.781488, -2.315474,  0.119916,  0.000360, -0.394424],   # 3 CAMOUFLAGED_OBJECT
    [-1.074867,  -3.465437, -0.377706,  0.892012,  0.072231,  0.000164, -3.159136],   # 4 TERRAIN_HAZARD
]

_LR_INTERCEPT = [-6.141472, -3.291714, 6.873511, -1.535111, 4.094786]


def _softmax(logits):
    max_l = max(logits)
    exps  = [math.exp(v - max_l) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _lr_predict_proba(features):
    """Return softmax probability vector for all 5 classes."""
    logits = []
    for cls in range(5):
        score = _LR_INTERCEPT[cls]
        for j in range(7):
            score += _LR_COEF[cls][j] * features[j]
        logits.append(score)
    return _softmax(logits)


def predict_lr(features):
    """
    Logistic Regression inference.

    Returns:
        (label: int, label_name: str, confidence: float 0-1)
    """
    proba = _lr_predict_proba(features)
    label = proba.index(max(proba))
    return label, LABEL_NAMES[label], round(proba[label], 4)


# ──────────────────────────────────────────────────────────────
# MODEL 2 — DECISION TREE  (depth 6, raw thresholds)
# Trained on raw (unscaled) features.  86% test accuracy.
# No probabilities — hard classification only.
# Fully self-contained: copy this function to MicroPython as-is.
# ──────────────────────────────────────────────────────────────

def predict_dt(features):
    """
    Decision Tree inference — pure if/elif, zero dependencies.
    Compatible with MicroPython.

    Args:
        features : list[float] — 7-element vector from extract_features()

    Returns:
        (label: int, label_name: str, confidence: float)
        confidence is always 1.0 (hard decision — no proba in DT)
    """
    transition_rate = features[0]
    ir_ratio        = features[1]
    variance        = features[2]
    asymmetry       = features[3]
    distance_cm     = features[4]
    clearance_norm  = features[5]
    combined_risk   = features[6]

    # ── Tree depth 1 ──────────────────────────────────────────
    if transition_rate <= 0.22745:

        # ── Depth 2 ──────────────────────────────────────────
        if clearance_norm <= 0.0341:

            # ── Depth 3 ──────────────────────────────────────
            if ir_ratio <= 0.3134:
                if distance_cm <= 4.37:
                    label = 2  # LOW_CLEARANCE
                else:
                    label = 4  # TERRAIN_HAZARD

            else:  # ir_ratio > 0.3134
                if clearance_norm <= 0.0301:
                    if transition_rate <= 0.14195:
                        label = 2  # LOW_CLEARANCE
                    else:
                        if distance_cm <= 6.565:
                            label = 2  # LOW_CLEARANCE
                        else:
                            label = 4  # TERRAIN_HAZARD
                else:  # clearance_norm > 0.0301
                    if variance <= 0.2337:
                        label = 2  # LOW_CLEARANCE
                    else:
                        label = 4  # TERRAIN_HAZARD

        else:  # clearance_norm > 0.0341

            if ir_ratio <= 0.5048:
                label = 4  # TERRAIN_HAZARD

            else:  # ir_ratio > 0.5048
                if asymmetry <= 0.2066:
                    label = 1  # SAFE_REFLECTIVE
                else:
                    if asymmetry <= 0.36945:
                        if transition_rate <= 0.0778:
                            label = 0  # SAFE_OPEN
                        else:
                            label = 1  # SAFE_REFLECTIVE
                    else:
                        # combined_risk doesn't change outcome here — always SAFE_OPEN
                        label = 0  # SAFE_OPEN

    else:  # transition_rate > 0.22745
        label = 3  # CAMOUFLAGED_OBJECT

    return label, LABEL_NAMES[label], 1.0


# ──────────────────────────────────────────────────────────────
# UNIFIED PREDICT INTERFACE
# ──────────────────────────────────────────────────────────────

def predict(features, model="lr"):
    """
    Unified prediction interface.

    Args:
        features : list[float] from extract_features()
        model    : "lr" (Logistic Regression, default)
                   "dt" (Decision Tree — MicroPython compatible)

    Returns:
        (label: int, label_name: str, confidence: float)

    Example:
        feats  = extract_features(ir_history, distance_cm)
        label, name, conf = predict(feats, model="lr")
        print(f"{name}  ({conf*100:.1f}% confident)")
    """
    if model == "dt":
        return predict_dt(features)
    return predict_lr(features)


# ──────────────────────────────────────────────────────────────
# OPTIONAL: load pre-trained Random Forest (CPython only)
# ──────────────────────────────────────────────────────────────

def load_rf_model(path="ugv_rf_model.joblib"):
    """
    Load the saved Random Forest via joblib (requires sklearn).
    Returns a callable: rf_predict(features) -> (label, name, confidence)

    Only available on CPython. Not for MicroPython.
    """
    try:
        import joblib
        bundle = joblib.load(path)
        rf = bundle["model"]
    except Exception as e:
        raise ImportError(f"Could not load RF model from '{path}': {e}")

    def rf_predict(features):
        import numpy as np
        proba  = rf.predict_proba([features])[0]
        label  = int(np.argmax(proba))
        return label, LABEL_NAMES[label], round(float(proba[label]), 4)

    return rf_predict


# ──────────────────────────────────────────────────────────────
# DEMO / SELF-TEST
# ──────────────────────────────────────────────────────────────

def _run_demo():
    """
    Simulate five representative sensor scenarios and run all models.
    """
    print("=" * 60)
    print("  UGV ML Model — Demo / Self-Test")
    print("=" * 60)

    # IR convention: 0 = reflective (sensor triggered), 1 = open/absorptive
    # ir_ratio = count(0) / window_size  — high means mostly reflective surface
    # SAFE_OPEN      : high ir_ratio (0.70-0.98) — flat reflective ground, no obstruction
    # SAFE_REFLECTIVE: mid-high ir_ratio (0.55-0.85) — rock/packed surface, good clearance
    # TERRAIN_HAZARD : low ir_ratio (0.05-0.45)  — absorptive/dark soft ground
    test_cases = [
        {
            "name"       : "Open field, far obstacle",
            "ir_history" : [0]*34 + [1]*6,       # 85% reflective (ir_ratio=0.85) -> SAFE_OPEN
            "distance_cm": 180.0,
            "expected"   : "SAFE_OPEN",
        },
        {
            "name"       : "Rocky hard surface",
            "ir_history" : [0]*28 + [1]*12,      # 70% reflective -> SAFE_REFLECTIVE
            "distance_cm": 45.0,
            "expected"   : "SAFE_REFLECTIVE",
        },
        {
            "name"       : "Obstacle very close",
            "ir_history" : [0]*22 + [1]*18,      # 55% reflective, but distance < 12 cm
            "distance_cm": 6.5,
            "expected"   : "LOW_CLEARANCE",
        },
        {
            "name"       : "Camouflaged object (flutter)",
            "ir_history" : [0,1,0,1,0,0,1,0,1,0,1,1,0,1,0,1,0,0,1,0,
                            1,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,1],  # high transitions
            "distance_cm": 60.0,
            "expected"   : "CAMOUFLAGED_OBJECT",
        },
        {
            "name"       : "Soft vegetated ground close",
            "ir_history" : [1]*32 + [0]*8,       # 20% reflective (mostly absorptive)
            "distance_cm": 14.0,
            "expected"   : "TERRAIN_HAZARD",
        },
    ]

    header = f"{'Scenario':<34} {'Expected':<22} {'LR Result':<22} {'LR Conf':>8}  {'DT Result':<22}"
    print(header)
    print("-" * len(header))

    for tc in test_cases:
        feats = extract_features(tc["ir_history"], tc["distance_cm"])

        lr_lbl, lr_name, lr_conf = predict_lr(feats)
        dt_lbl, dt_name, _       = predict_dt(feats)

        lr_ok = "✔" if lr_name == tc["expected"] else "✘"
        dt_ok = "✔" if dt_name == tc["expected"] else "✘"

        print(
            f"{tc['name']:<34} "
            f"{tc['expected']:<22} "
            f"{lr_ok} {lr_name:<20} "
            f"{lr_conf*100:>6.1f}%  "
            f"{dt_ok} {dt_name}"
        )

    print()
    print("Feature vector for last test case:")
    feats = extract_features(test_cases[-1]["ir_history"], test_cases[-1]["distance_cm"])
    for fname, fval in zip(FEATURES, feats):
        print(f"  {fname:<18} = {fval}")

    print()
    print("Probability breakdown (Logistic Regression, last case):")
    proba = _lr_predict_proba(feats)
    for i, p in enumerate(proba):
        bar = "█" * int(p * 30)
        print(f"  {LABEL_NAMES[i]:<22}  {p*100:>5.1f}%  {bar}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────
# INTEGRATION SNIPPET  (for ugv_perception.py)
# ──────────────────────────────────────────────────────────────

INTEGRATION_GUIDE = """
─────────────────────────────────────────────────────────────
  HOW TO INTEGRATE INTO ugv_perception.py
─────────────────────────────────────────────────────────────

# 1. Import at top of ugv_perception.py:
from ugv_ml_model import extract_features, predict

# 2. Inside the main loop analysis block, replace manual model
#    calls with:

feats = extract_features(ir_history, distance_cm)
label, label_name, confidence = predict(feats, model="lr")
# or model="dt" for the MicroPython-compatible Decision Tree

# 3. Use the result:
if label == 2:
    emit_alert(f"LOW CLEARANCE — {distance_cm:.1f} cm")
elif label == 3:
    emit_alert(f"CAMOUFLAGED OBJECT  ({confidence*100:.0f}% conf)")
elif label == 4:
    emit_alert(f"TERRAIN HAZARD  [{label_name}]")

# 4. For MicroPython: copy ONLY predict_dt() and extract_features()
#    into your .py file. No imports needed beyond math.
─────────────────────────────────────────────────────────────
"""

# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _run_demo()
    print(INTEGRATION_GUIDE)
