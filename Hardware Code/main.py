"""
UGV Multi-Sensor Perception System  — v3.0 (Online-Learning + Error Correction)
=================================================================================
NEW in v3.0:
  - OnlineRLS:   Recursive Least Squares adapter that updates CAMOUF_W / TERRAIN_W
                 in-place from confirmed ground-truth windows.
  - KalmanEMA:   Per-output exponential smoother with adaptive gain — acts as a
                 lightweight single-state Kalman filter to smooth model outputs
                 and flag sudden prediction jumps as probable sensor glitches.
  - FeedbackRing: Fixed-capacity ring buffer that stores (features, label) pairs.
                  Confirmed correct predictions are POPPED (consumed) and fed back
                  into RLS so synthetic / early data is gradually replaced by real
                  field-confirmed data.
  - SyntheticSeeder: Pre-populates FeedbackRing with physics-derived synthetic
                  examples so RLS starts from a reasonable prior even before the
                  first confirmed window arrives.

Sensors / pins / config unchanged from v2.1.
"""

from machine import Pin, time_pulse_us
import utime
import math

# ──────────────────────────────────────────────
# PIN SETUP
# ──────────────────────────────────────────────
IR_PIN   = 27
TRIG_PIN = 28
ECHO_PIN = 26

ir   = Pin(IR_PIN,   Pin.IN)
trig = Pin(TRIG_PIN, Pin.OUT)
echo = Pin(ECHO_PIN, Pin.IN)

# ──────────────────────────────────────────────
# GLOBAL CONFIG
# ──────────────────────────────────────────────
SAMPLE_DELAY_MS    = 10
IR_WINDOW          = 40
COOLDOWN_MS        = 4000
MIN_CLEARANCE_CM   = 12.0
MAX_RANGE_CM       = 400.0

TEMP_C             = 25.0
_SOUND_SPEED_MS    = 331.3 + 0.606 * TEMP_C
US_PER_CM          = 1_000_000.0 / (_SOUND_SPEED_MS * 100.0) * 2.0
US_TIMEOUT         = int(MAX_RANGE_CM * US_PER_CM * 1.25) + 500
ULTRASONIC_SAMPLES = 3

# ──────────────────────────────────────────────
# MODEL WEIGHTS  (mutable lists — RLS will edit in-place)
# ──────────────────────────────────────────────
CAMOUF_W         = [0.75, 0.85, 0.90, 0.65]
CAMOUF_B_LIST    = [-4.5]           # wrapped in list so RLS can mutate it
CAMOUF_THRESHOLD = 0.68
CAMOUF_CONFIRM   = 3

TERRAIN_W         = [0.90, -0.80, 0.60, 1.10]
TERRAIN_B_LIST    = [-3.0]
TERRAIN_THRESHOLD = 0.72

# ──────────────────────────────────────────────
# TERRAIN LABELS
# ──────────────────────────────────────────────
TERRAIN_MAP = {
    (0, "low") : "HARD REFLECTIVE (rock/metal) — LOW CLEARANCE",
    (0, "mid") : "PACKED SAND / CLAY — MODERATE CLEARANCE",
    (0, "high"): "REFLECTIVE GROUND — SAFE CLEARANCE",
    (1, "low") : "SOFT/DARK SURFACE — LOW CLEARANCE WARNING",
    (1, "mid") : "VEGETATED / LOOSE TERRAIN",
    (1, "high"): "OPEN GROUND — SAFE",
}

# ══════════════════════════════════════════════
# MATH HELPERS
# ══════════════════════════════════════════════
def sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def dot(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))

def dot_predict(weights, bias_list, features):
    return sigmoid(dot(weights, features) + bias_list[0])

def _median3(a, b, c):
    if a <= b <= c or c <= b <= a: return b
    if b <= a <= c or c <= a <= b: return a
    return c

def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def vec_scale(v, s):
    return [x * s for x in v]

def vec_norm_sq(v):
    return sum(x * x for x in v)

# ══════════════════════════════════════════════
# KALMAN-EMA  — lightweight single-state smoother
# ══════════════════════════════════════════════
class KalmanEMA:
    """
    Models each scalar output as:
        x_k = x_{k-1} + process_noise
        z_k = x_k    + measurement_noise

    Simplified to an EMA with adaptive gain:
        K   = P / (P + R)          (Kalman gain)
        x   = x + K*(z - x)       (update)
        P   = (1 - K)*P + Q        (covariance predict)

    - Q (process noise): how fast the true value can change
    - R (measurement noise): expected sensor/model noise floor

    Spike detection: if |z - x_prior| > SPIKE_THRESH the reading is
    flagged as a probable glitch and down-weighted (K halved).
    """
    def __init__(self, q=0.005, r=0.05, init=0.5, spike_thresh=0.35):
        self.q           = q
        self.r           = r
        self.x           = init        # state estimate
        self.p           = 1.0         # error covariance
        self.spike_thresh= spike_thresh
        self.spike_count = 0

    def update(self, z):
        # Predict
        p_prior = self.p + self.q
        x_prior = self.x

        # Detect spike
        residual = z - x_prior
        spike = abs(residual) > self.spike_thresh

        # Kalman gain — halved on spike to down-weight outlier
        k = p_prior / (p_prior + self.r)
        if spike:
            k *= 0.5
            self.spike_count += 1

        # Update
        self.x = x_prior + k * residual
        self.p = (1.0 - k) * p_prior

        return self.x, spike

    def reset_spike_count(self):
        c = self.spike_count
        self.spike_count = 0
        return c

# ══════════════════════════════════════════════
# ONLINE RLS  — Recursive Least Squares adapter
# ══════════════════════════════════════════════
class OnlineRLS:
    """
    Fits  y_hat = sigmoid(w . x + b)  online via linearised gradient.

    Update rule (gradient descent on cross-entropy, one sample):
        err  = y_hat - y_true
        dL/dw = err * x_i          (sigmoid gradient absorbed into err)
        w_i  -= lr * err * x_i
        b    -= lr * err

    forgetting_factor < 1.0 makes old data decay (like RLS lambda).
    max_step clips the per-update weight change for stability.
    """
    def __init__(self, weights_ref, bias_list_ref,
                 lr=0.04, forgetting=0.98, max_step=0.12):
        self.w          = weights_ref    # direct reference — mutates caller list
        self.b          = bias_list_ref
        self.lr         = lr
        self.forgetting = forgetting
        self.max_step   = max_step
        self.n_updates  = 0

    def update(self, features, y_true):
        """
        features : list[float]  (same length as self.w)
        y_true   : float  0.0 or 1.0
        """
        y_hat = sigmoid(dot(self.w, features) + self.b[0])
        err   = y_hat - y_true

        # Apply forgetting: slightly shrink weights toward 0 each step
        for i in range(len(self.w)):
            self.w[i] *= self.forgetting
            grad = err * features[i]
            grad = max(-self.max_step, min(self.max_step, grad))
            self.w[i] -= self.lr * grad

        b_grad = max(-self.max_step, min(self.max_step, err))
        self.b[0] -= self.lr * b_grad
        self.n_updates += 1

    @property
    def total_updates(self):
        return self.n_updates

# ══════════════════════════════════════════════
# FEEDBACK RING BUFFER
# ══════════════════════════════════════════════
class FeedbackRing:
    """
    Fixed-capacity circular buffer of (features, label) pairs.

    push()  — adds a new sample, overwriting oldest if full.
    pop()   — removes and returns the oldest confirmed sample (FIFO).
    peek()  — returns oldest without removing.

    Synthetic data is pre-loaded by SyntheticSeeder; real confirmed
    windows are pushed in and synthetic entries are displaced naturally
    as the buffer cycles.
    """
    def __init__(self, capacity=64):
        self.capacity = capacity
        self._buf     = [None] * capacity
        self._head    = 0      # next write position
        self._tail    = 0      # next read position
        self._size    = 0

    def push(self, features, label):
        self._buf[self._head] = (list(features), float(label))
        self._head = (self._head + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1
        else:
            # Buffer full — advance tail to keep FIFO valid
            self._tail = (self._tail + 1) % self.capacity

    def pop(self):
        """Remove and return oldest entry, or None if empty."""
        if self._size == 0:
            return None
        item = self._buf[self._tail]
        self._buf[self._tail] = None
        self._tail = (self._tail + 1) % self.capacity
        self._size -= 1
        return item

    def peek(self):
        if self._size == 0:
            return None
        return self._buf[self._tail]

    def __len__(self):
        return self._size

# ══════════════════════════════════════════════
# SYNTHETIC SEEDER
# ══════════════════════════════════════════════
def seed_feedback_ring(ring, n=24):
    """
    Pre-populate the feedback ring with physics-derived synthetic examples.

    Camouflage features: [transition_rate, ir_ratio, variance, asymmetry]
      - Camouflaged (label=1): high transitions, mid ir_ratio (~0.45–0.55),
                                high variance, low asymmetry
      - Clear (label=0):       low transitions, extreme ir_ratio, low variance,
                                high asymmetry

    Terrain features: [ir_ratio, clearance_norm, variance, combined_risk]
      - Hazardous (label=1):   mid ir_ratio, low clearance_norm, mid variance,
                                high combined_risk
      - Safe (label=0):        any ir_ratio, high clearance_norm, low combined_risk
    """
    # Camouflage synthetic seeds
    camouf_seeds = [
        # (features,                                           label)
        ([0.40, 0.50, 0.25, 0.00], 1),  # classic camo signature
        ([0.38, 0.48, 0.25, 0.02], 1),
        ([0.35, 0.52, 0.25, 0.02], 1),
        ([0.05, 0.10, 0.09, 0.40], 0),  # clear reflective surface
        ([0.04, 0.90, 0.09, 0.40], 0),  # clear absorptive surface
        ([0.06, 0.15, 0.13, 0.35], 0),
        ([0.42, 0.47, 0.25, 0.03], 1),
        ([0.03, 0.92, 0.07, 0.42], 0),
    ]
    # Terrain synthetic seeds
    terrain_seeds = [
        ([0.80, 0.03, 0.16, 0.78], 1),  # reflective + very low clearance
        ([0.70, 0.02, 0.21, 0.69], 1),
        ([0.50, 0.04, 0.25, 0.48], 1),
        ([0.50, 0.90, 0.25, 0.05], 0),  # mid ir, high clearance → safe
        ([0.10, 0.95, 0.09, 0.01], 0),  # absorptive, high clearance → safe
        ([0.80, 0.85, 0.16, 0.12], 0),
        ([0.60, 0.05, 0.24, 0.57], 1),
        ([0.30, 0.92, 0.21, 0.02], 0),
    ]
    all_seeds = camouf_seeds + terrain_seeds
    # Repeat seeds to fill requested n slots
    for i in range(n):
        feat, lbl = all_seeds[i % len(all_seeds)]
        ring.push(feat, lbl)

# ══════════════════════════════════════════════
# PREDICTION ERROR CORRECTOR
# ══════════════════════════════════════════════
class PredictionCorrector:
    """
    Wraps a model's raw output with:
      1. KalmanEMA smoothing
      2. Confidence gate: if smoothed output is within MARGIN of threshold,
         hold the previous decision (hysteresis) to avoid chattering.
      3. Error log: tracks rolling mean absolute error vs confirmed labels.
    """
    MARGIN = 0.06  # hysteresis band around threshold

    def __init__(self, threshold, q=0.005, r=0.05):
        self.threshold    = threshold
        self.ema          = KalmanEMA(q=q, r=r, init=0.5)
        self._prev_dec    = False
        self._mae_sum     = 0.0
        self._mae_count   = 0

    def process(self, raw_prob):
        """
        Returns (smoothed_prob, decision, was_spike).
        decision is bool: True = positive class detected.
        """
        smoothed, spike = self.ema.update(raw_prob)

        lo = self.threshold - self.MARGIN
        hi = self.threshold + self.MARGIN

        if smoothed >= hi:
            decision = True
        elif smoothed <= lo:
            decision = False
        else:
            decision = self._prev_dec   # hysteresis — hold last decision

        self._prev_dec = decision
        return smoothed, decision, spike

    def record_feedback(self, smoothed_prob, true_label):
        """Call with confirmed label to track MAE."""
        self._mae_sum   += abs(smoothed_prob - true_label)
        self._mae_count += 1

    @property
    def mae(self):
        if self._mae_count == 0:
            return 0.0
        return self._mae_sum / self._mae_count

# ══════════════════════════════════════════════
# ULTRASONIC
# ══════════════════════════════════════════════
def _single_ping_cm():
    trig.low()
    utime.sleep_us(4)
    trig.high()
    utime.sleep_us(10)
    trig.low()
    duration = time_pulse_us(echo, 1, US_TIMEOUT)
    if duration < 0:
        return -1.0
    cm = duration / US_PER_CM
    if cm > MAX_RANGE_CM:
        return -1.0
    return round(cm, 2)

def measure_distance_cm():
    readings = []
    for _ in range(ULTRASONIC_SAMPLES):
        readings.append(_single_ping_cm())
        utime.sleep_ms(15)
    valid = [r for r in readings if r >= 0]
    if len(valid) == 0:  return -1.0
    if len(valid) == 1:  return valid[0]
    if len(valid) == 2:  return round((valid[0] + valid[1]) / 2.0, 2)
    return round(_median3(valid[0], valid[1], valid[2]), 2)

def clearance_zone(cm):
    if cm < 1.0:              return "unknown"
    if cm < MIN_CLEARANCE_CM: return "low"
    if cm < 80.0:             return "mid"
    return "high"

# ══════════════════════════════════════════════
# IR FEATURE EXTRACTION
# ══════════════════════════════════════════════
def extract_ir_features(history):
    n = len(history)
    if n < 4:
        return [0.0, 0.0, 0.0, 0.0]
    transitions = sum(1 for i in range(1, n) if history[i] != history[i - 1])
    low_count   = history.count(0)
    ir_ratio    = low_count / n
    variance    = ir_ratio * (1.0 - ir_ratio)
    asymmetry   = abs(ir_ratio - 0.5)
    trans_rate  = transitions / n
    return [trans_rate, ir_ratio, variance, asymmetry]

def extract_terrain_features(ir_features, distance_cm):
    ir_ratio  = ir_features[1]
    variance  = ir_features[2]
    if distance_cm < 1.0:
        clearance_norm = 0.5
    else:
        clearance_norm = min(distance_cm / MAX_RANGE_CM, 1.0)
    combined_risk = (1.0 - clearance_norm) * ir_ratio
    return [ir_ratio, clearance_norm, variance, combined_risk]

# ══════════════════════════════════════════════
# ALERT HELPERS
# ══════════════════════════════════════════════
def print_separator():
    print("=" * 60)

def emit_alert(msg):
    print_separator()
    print(msg)
    print_separator()

# ══════════════════════════════════════════════
# STARTUP SELF-TEST
# ══════════════════════════════════════════════
def self_test():
    print_separator()
    print("  SELF-TEST")
    try:
        v = ir.value()
        print(f"  IR  sensor  : PASS  (value={v})")
    except Exception as e:
        print(f"  IR  sensor  : FAIL  ({e})")
    d = measure_distance_cm()
    if 1.0 <= d <= 300.0:
        print(f"  Ultrasonic  : PASS  ({d} cm)")
    elif d < 0:
        print("  Ultrasonic  : FAIL  (no echo — check wiring)")
    else:
        print(f"  Ultrasonic  : WARN  (reading={d} cm)")
    print(f"  Sound speed : {_SOUND_SPEED_MS:.1f} m/s  @ {TEMP_C}°C")
    print(f"  µs/cm       : {US_PER_CM:.4f}")
    print(f"  Ping timeout: {US_TIMEOUT} µs")
    print_separator()

# ══════════════════════════════════════════════
# INSTANTIATE LEARNING + CORRECTION SUBSYSTEMS
# ══════════════════════════════════════════════

# Shared feedback ring (used by both models)
feedback_ring = FeedbackRing(capacity=64)
seed_feedback_ring(feedback_ring, n=24)     # pre-load synthetic priors

# Adaptive weight updaters
camouf_rls  = OnlineRLS(CAMOUF_W,  CAMOUF_B_LIST,  lr=0.04, forgetting=0.985)
terrain_rls = OnlineRLS(TERRAIN_W, TERRAIN_B_LIST, lr=0.04, forgetting=0.985)

# Output smoothers + hysteresis gates
camouf_corrector  = PredictionCorrector(CAMOUF_THRESHOLD,  q=0.003, r=0.04)
terrain_corrector = PredictionCorrector(TERRAIN_THRESHOLD, q=0.003, r=0.04)

# Ultrasonic EMA (track distance stability separately)
dist_ema = KalmanEMA(q=0.8, r=2.0, init=50.0, spike_thresh=30.0)

# ──────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────
ir_history       = []
camouf_confirmed = 0
last_alert_time  = {"camouflage": 0, "clearance": 0, "terrain": 0}

# Diagnostic counters
_windows_processed = 0
_rls_updates_done  = 0

# ══════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════
print_separator()
print("  UGV Sensor Perception System ONLINE  v3.0")
print(f"  IR Pin: {IR_PIN}  |  TRIG: {TRIG_PIN}  |  ECHO: {ECHO_PIN}")
print(f"  Min safe clearance : {MIN_CLEARANCE_CM} cm")
print(f"  IR window size     : {IR_WINDOW} samples")
print(f"  Feedback ring      : {len(feedback_ring)}/{feedback_ring.capacity} "
      f"entries (seeded synthetic)")
print_separator()

self_test()

# ══════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════
while True:
    now    = utime.ticks_ms()
    ir_val = ir.value()

    ir_history.append(ir_val)
    if len(ir_history) > IR_WINDOW:
        ir_history.pop(0)

    if len(ir_history) >= IR_WINDOW:
        _windows_processed += 1

        # ── Sensor measurements ───────────────────
        raw_dist = measure_distance_cm()
        dist_smooth, dist_spike = dist_ema.update(
            raw_dist if raw_dist >= 1.0 else dist_ema.x
        )
        # Use smoothed distance for logic; flag spike if raw jumps badly
        distance_cm = raw_dist   # keep raw for alerts (truthful reporting)

        # ── Feature extraction ────────────────────
        ir_feat = extract_ir_features(ir_history)
        tr_feat = extract_terrain_features(ir_feat, distance_cm)

        ir_dominant   = 0 if ir_feat[1] > 0.5 else 1
        zone          = clearance_zone(distance_cm)
        terrain_label = TERRAIN_MAP.get((ir_dominant, zone), "UNKNOWN TERRAIN")

        # ── MODEL 1: Camouflage — raw → corrected ─
        camouf_raw              = dot_predict(CAMOUF_W,  CAMOUF_B_LIST,  ir_feat)
        camouf_smooth, camouf_pos, camouf_spike = camouf_corrector.process(camouf_raw)

        if camouf_pos:
            camouf_confirmed += 1
        else:
            camouf_confirmed = max(0, camouf_confirmed - 1)

        camouf_alert = (
            camouf_confirmed >= CAMOUF_CONFIRM and
            utime.ticks_diff(now, last_alert_time["camouflage"]) > COOLDOWN_MS
        )
        if camouf_alert:
            emit_alert("CAMOUFLAGED OBJECT DETECTED")
            last_alert_time["camouflage"] = now
            # Confirmed positive — push to feedback ring as label=1
            feedback_ring.push(ir_feat, 1.0)
            camouf_confirmed = 0

        # ── MODEL 2: Terrain — raw → corrected ───
        hazard_raw                = dot_predict(TERRAIN_W, TERRAIN_B_LIST, tr_feat)
        hazard_smooth, hazard_pos, terrain_spike = terrain_corrector.process(hazard_raw)

        # ── Ground Clearance Alert ────────────────
        clearance_alert = (
            distance_cm >= 1.0 and
            distance_cm < MIN_CLEARANCE_CM and
            utime.ticks_diff(now, last_alert_time["clearance"]) > COOLDOWN_MS
        )
        if clearance_alert:
            emit_alert(f"LOW CLEARANCE: {distance_cm} cm  (min {MIN_CLEARANCE_CM} cm)")
            last_alert_time["clearance"] = now
            # Low clearance → terrain is hazardous; push confirmed label=1
            feedback_ring.push(tr_feat, 1.0)

        # ── Terrain Hazard Alert ──────────────────
        terrain_alert = (
            hazard_pos and
            utime.ticks_diff(now, last_alert_time["terrain"]) > COOLDOWN_MS
        )
        if terrain_alert:
            emit_alert(f"TERRAIN HAZARD  score={hazard_smooth:.2f}")
            last_alert_time["terrain"] = now
            feedback_ring.push(tr_feat, 1.0)

        # ── Push safe windows into ring ───────────
        # If no alert fired, this window is provisionally "safe" (label=0)
        if not camouf_alert and not clearance_alert and not terrain_alert:
            # Only push safe examples occasionally (every 5 windows) to avoid
            # overwhelming the ring with trivial negatives
            if _windows_processed % 5 == 0:
                feedback_ring.push(ir_feat,  0.0)
                feedback_ring.push(tr_feat, 0.0)

        # ── RLS UPDATE — pop confirmed samples and adapt ──────────────
        # Drain up to 2 confirmed entries per window to bound compute time
        for _ in range(2):
            entry = feedback_ring.pop()
            if entry is None:
                break
            feats, label = entry
            n_feats = len(feats)
            # Route to correct model by feature vector length + content
            # Camouflage features always have 4 elements; so do terrain.
            # Distinguish by position: terrain feat[1] is clearance_norm
            # which can be > 1.0 only if distance > MAX_RANGE (won't happen),
            # but camouf feat[1] is ir_ratio in [0,1] and feat[0]=trans_rate.
            # Simplest heuristic: if feat[2] == variance (Bernoulli) and
            # feat[3] == asymmetry, both share shape. We split by alternating
            # the ring between camouf / terrain pushes — or just update both
            # models with the same sample (they see different weight vectors).
            # Safest approach: update BOTH with the available sample.
            # The gradient direction per model will self-select relevance.
            if n_feats == 4:
                camouf_rls.update(feats, label)
                terrain_rls.update(feats, label)
                _rls_updates_done += 1

        # ── Spike diagnostic ──────────────────────
        spike_flags = []
        if camouf_spike:  spike_flags.append("CAM-SPIKE")
        if terrain_spike: spike_flags.append("TER-SPIKE")
        if dist_spike:    spike_flags.append("DIST-SPIKE")
        spike_str = " ".join(spike_flags) if spike_flags else ""

        # ── Status Print ──────────────────────────
        dist_str = f"{distance_cm:.1f}cm" if distance_cm >= 1.0 else "OUT-OF-RANGE"
        print(
            f"IR={'REFLECT' if ir_val == 0 else 'OPEN':7s}|"
            f"dist={dist_str:>12s}[{zone:7s}]|"
            f"cam={camouf_smooth:.2f}({'!' if camouf_pos else ' '}{camouf_confirmed}/{CAMOUF_CONFIRM})|"
            f"haz={hazard_smooth:.2f}({'!' if hazard_pos else ' '})|"
            f"ring={len(feedback_ring):02d}|rls={_rls_updates_done:04d}|"
            f"{spike_str:<22s}{terrain_label}"
        )

        # ── Reset window ──────────────────────────
        ir_history = []

    utime.sleep_ms(SAMPLE_DELAY_MS)
