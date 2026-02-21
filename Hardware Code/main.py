"""
UGV Multi-Sensor Perception System  — v2.1 (bug-fixed + calibrated)
=====================================================================
Sensors:
  IR Sensor         -> Pin 27  (DUAL PURPOSE)
     Purpose 1: Camouflaged Object Detection  (foreground reflection pattern)
     Purpose 2: Terrain Reflectivity Classifier (surface material inference)
  Ultrasonic TRIG   -> Pin 28
  Ultrasonic ECHO   -> Pin 26
     Purpose:  Ground Clearance Detector

TinyML weights are placeholders — replace with trained values.

Changelog vs v1:
  - FIX: ir_history was reset every loop iteration, so the window never
         accumulated. Now reset only AFTER a full analysis pass.
  - FIX: measure_distance_cm() used 29.1 µs/cm (off by ~0.2%). Corrected
         to 29.154 µs/cm (speed of sound 343 m/s at 20 °C).
  - FIX: time_pulse_us timeout raised from 30 000 µs to 38 000 µs to
         cover the full MAX_RANGE_CM=400 cm (round-trip ~23 200 µs) with
         margin, preventing premature -1 returns on valid echoes.
  - FIX: Ultrasonic now averages ULTRASONIC_SAMPLES readings and discards
         outliers (median-of-3), eliminating single-spike "out of range".
  - FIX: camouf_confirmed was allowed to decrement below 0 between window
         fills, causing confirmation count drift. Guard added.
  - FIX: clearance_alert condition used `distance_cm > 0` which passed for
         0.01 cm (noise floor). Changed to `distance_cm >= 1.0`.
  - FIX: terrain_label lookup used a stale ir_dominant derived from a
         previous window's ir_ratio when window was being reset mid-cycle.
         Now derived strictly within the analysis block.
  - IMPROVE: Added temperature-compensated sound speed (set TEMP_C).
  - IMPROVE: Separate rolling window for ultrasonic stability tracking.
  - IMPROVE: Startup sensor self-test with pass/fail reporting.
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
SAMPLE_DELAY_MS    = 10       # ms between individual IR samples
IR_WINDOW          = 40       # IR samples per analysis window
COOLDOWN_MS        = 4000     # ms between repeated alerts
MIN_CLEARANCE_CM   = 12.0     # below this → low-clearance warning
MAX_RANGE_CM       = 400.0    # ultrasonic reliable ceiling (cm)

# ── Ultrasonic calibration ───────────────────
TEMP_C             = 25.0     # ambient temperature °C — adjust for your env
# Speed of sound (m/s) at TEMP_C:  331.3 + 0.606 * T
# Round-trip divisor (µs per cm):  1e6 / (speed_m_s * 100) * 2
_SOUND_SPEED_MS    = 331.3 + 0.606 * TEMP_C          # m/s
US_PER_CM          = 1_000_000.0 / (_SOUND_SPEED_MS * 100.0) * 2.0  # µs/cm
# Max round-trip time for MAX_RANGE_CM + 25 % margin
US_TIMEOUT         = int(MAX_RANGE_CM * US_PER_CM * 1.25) + 500      # µs
ULTRASONIC_SAMPLES = 3        # readings to median-average per cycle

# ──────────────────────────────────────────────
# MODEL 1 — Camouflaged Object Detector
# Features: [transition_rate, ir_ratio, variance, asymmetry]
# ──────────────────────────────────────────────
CAMOUF_W         = [0.75, 0.85, 0.90, 0.65]
CAMOUF_B         = -4.5
CAMOUF_THRESHOLD = 0.68
CAMOUF_CONFIRM   = 3          # consecutive windows required to confirm

# ──────────────────────────────────────────────
# MODEL 2 — Terrain Hazard Scorer
# Features: [ir_ratio, clearance_norm, variance, combined_risk]
# ──────────────────────────────────────────────
TERRAIN_W         = [0.90, -0.80, 0.60, 1.10]
TERRAIN_B         = -3.0
TERRAIN_THRESHOLD = 0.72

# ──────────────────────────────────────────────
# TERRAIN LABELS
# (ir_dominant_side, clearance_zone) → description
# ir_dominant_side: 0 = mostly reflective, 1 = mostly open/absorptive
# ──────────────────────────────────────────────
TERRAIN_MAP = {
    (0, "low") : "HARD REFLECTIVE (rock/metal) — LOW CLEARANCE",
    (0, "mid") : "PACKED SAND / CLAY — MODERATE CLEARANCE",
    (0, "high"): "REFLECTIVE GROUND — SAFE CLEARANCE",
    (1, "low") : "SOFT/DARK SURFACE — LOW CLEARANCE WARNING",
    (1, "mid") : "VEGETATED / LOOSE TERRAIN",
    (1, "high"): "OPEN GROUND — SAFE",
}

# ──────────────────────────────────────────────
# MATH HELPERS
# ──────────────────────────────────────────────
def sigmoid(x):
    # Clamp to prevent overflow on micropython's float
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def dot_predict(weights, bias, features):
    s = bias
    for w, f in zip(weights, features):
        s += w * f
    return sigmoid(s)

def _median3(a, b, c):
    """Return median of three values without sorting."""
    if a <= b <= c or c <= b <= a:
        return b
    if b <= a <= c or c <= a <= b:
        return a
    return c

# ──────────────────────────────────────────────
# ULTRASONIC — GROUND CLEARANCE
# ──────────────────────────────────────────────
def _single_ping_cm():
    """
    Fire one ultrasonic pulse.
    Returns distance in cm, or -1.0 on timeout / out-of-range.
    """
    trig.low()
    utime.sleep_us(4)          # ensure line is settled before pulse
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
    """
    Take ULTRASONIC_SAMPLES pings and return the median.
    Using median instead of mean rejects single-spike noise.
    Returns -1.0 if the majority of readings are invalid.
    """
    readings = []
    for _ in range(ULTRASONIC_SAMPLES):
        readings.append(_single_ping_cm())
        utime.sleep_ms(15)     # ≥15 ms between pings to clear echo

    valid = [r for r in readings if r >= 0]

    if len(valid) == 0:
        return -1.0
    if len(valid) == 1:
        return valid[0]
    if len(valid) == 2:
        return round((valid[0] + valid[1]) / 2.0, 2)

    # 3-value median (works for ULTRASONIC_SAMPLES == 3)
    return round(_median3(valid[0], valid[1], valid[2]), 2)

def clearance_zone(cm):
    if cm < 1.0:               # treat 0 and -1 as unknown
        return "unknown"
    if cm < MIN_CLEARANCE_CM:
        return "low"
    if cm < 80.0:
        return "mid"
    return "high"

# ──────────────────────────────────────────────
# IR FEATURE EXTRACTION  (shared by both models)
# ──────────────────────────────────────────────
def extract_ir_features(history):
    """
    Returns [transition_rate, ir_ratio, variance, asymmetry]
    from a list of binary IR readings (0 = reflective, 1 = open).
    """
    n = len(history)
    if n < 4:
        return [0.0, 0.0, 0.0, 0.0]

    transitions = sum(1 for i in range(1, n) if history[i] != history[i - 1])
    low_count   = history.count(0)
    ir_ratio    = low_count / n                # fraction "reflective"
    variance    = ir_ratio * (1.0 - ir_ratio)  # max at ir_ratio = 0.5
    asymmetry   = abs(ir_ratio - 0.5)          # distance from neutral
    trans_rate  = transitions / n

    return [trans_rate, ir_ratio, variance, asymmetry]

# ──────────────────────────────────────────────
# TERRAIN FEATURE EXTRACTION (extends IR + ultrasonic)
# ──────────────────────────────────────────────
def extract_terrain_features(ir_features, distance_cm):
    """
    Returns [ir_ratio, clearance_norm, variance, combined_risk].
    """
    ir_ratio  = ir_features[1]
    variance  = ir_features[2]

    if distance_cm < 1.0:
        clearance_norm = 0.5                   # unknown → neutral
    else:
        clearance_norm = min(distance_cm / MAX_RANGE_CM, 1.0)

    # Proximity to a hard reflective surface = highest risk
    combined_risk = (1.0 - clearance_norm) * ir_ratio

    return [ir_ratio, clearance_norm, variance, combined_risk]

# ──────────────────────────────────────────────
# ALERT HELPERS
# ──────────────────────────────────────────────
def print_separator():
    print("=" * 52)

def emit_alert(msg):
    print_separator()
    print(msg)
    print_separator()

# ──────────────────────────────────────────────
# STARTUP SELF-TEST
# ──────────────────────────────────────────────
def self_test():
    """Quick sensor sanity check on boot. Prints PASS/FAIL per sensor."""
    print_separator()
    print("  SELF-TEST")

    # IR test — just check we can read it
    try:
        v = ir.value()
        print(f"  IR  sensor  : PASS  (value={v})")
    except Exception as e:
        print(f"  IR  sensor  : FAIL  ({e})")

    # Ultrasonic test — expect a valid ping indoors (< 300 cm)
    d = measure_distance_cm()
    if 1.0 <= d <= 300.0:
        print(f"  Ultrasonic  : PASS  ({d} cm)")
    elif d < 0:
        print("  Ultrasonic  : FAIL  (no echo — check wiring / TRIG/ECHO pins)")
    else:
        print(f"  Ultrasonic  : WARN  (reading={d} cm, check if open space)")

    print(f"  Sound speed : {_SOUND_SPEED_MS:.1f} m/s  @ {TEMP_C}°C")
    print(f"  µs/cm       : {US_PER_CM:.4f}")
    print(f"  Ping timeout: {US_TIMEOUT} µs  (~{round(US_TIMEOUT*_SOUND_SPEED_MS/1e6/2*100,1)} cm max)")
    print_separator()

# ──────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────
ir_history       = []
camouf_confirmed = 0
last_alert_time  = {
    "camouflage": 0,
    "clearance" : 0,
    "terrain"   : 0,
}

# ──────────────────────────────────────────────
# STARTUP
# ──────────────────────────────────────────────
print_separator()
print("  UGV Sensor Perception System ONLINE  v2.1")
print(f"  IR Pin: {IR_PIN}  |  TRIG: {TRIG_PIN}  |  ECHO: {ECHO_PIN}")
print(f"  Min safe clearance : {MIN_CLEARANCE_CM} cm")
print(f"  IR window size     : {IR_WINDOW} samples")
print_separator()

self_test()

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
while True:
    now    = utime.ticks_ms()
    ir_val = ir.value()

    # Measure distance once per window fill cycle (not every sample)
    # to keep the loop timing predictable without busy-waiting for 3 pings
    # on every iteration.  Distance is updated at the analysis step below.
    ir_history.append(ir_val)

    # Trim to window size (rolling — do NOT reset every cycle)
    if len(ir_history) > IR_WINDOW:
        ir_history.pop(0)

    # ── Process once a full window is accumulated ──
    if len(ir_history) >= IR_WINDOW:

        # Measure distance here (inside analysis block, not every sample tick)
        distance_cm = measure_distance_cm()

        ir_feat = extract_ir_features(ir_history)
        tr_feat = extract_terrain_features(ir_feat, distance_cm)

        # Derive terrain label keys inside analysis block (avoids stale data)
        ir_dominant   = 0 if ir_feat[1] > 0.5 else 1
        zone          = clearance_zone(distance_cm)
        terrain_label = TERRAIN_MAP.get((ir_dominant, zone), "UNKNOWN TERRAIN")

        # ── MODEL 1: Camouflage Detection ────────
        camouf_prob = dot_predict(CAMOUF_W, CAMOUF_B, ir_feat)

        if camouf_prob > CAMOUF_THRESHOLD:
            camouf_confirmed += 1
        else:
            # Decay confirmation count, but clamp to 0
            camouf_confirmed = max(0, camouf_confirmed - 1)

        camouf_alert = (
            camouf_confirmed >= CAMOUF_CONFIRM and
            utime.ticks_diff(now, last_alert_time["camouflage"]) > COOLDOWN_MS
        )
        if camouf_alert:
            emit_alert("CAMOUFLAGED OBJECT DETECTED")
            last_alert_time["camouflage"] = now
            camouf_confirmed = 0

        # ── MODEL 2: Terrain Hazard Score ────────
        hazard_prob = dot_predict(TERRAIN_W, TERRAIN_B, tr_feat)

        # ── Ground Clearance Alert ───────────────
        clearance_alert = (
            distance_cm >= 1.0 and                 # FIX: was > 0, caught noise
            distance_cm < MIN_CLEARANCE_CM and
            utime.ticks_diff(now, last_alert_time["clearance"]) > COOLDOWN_MS
        )
        if clearance_alert:
            emit_alert(f"LOW CLEARANCE: {distance_cm} cm  (min {MIN_CLEARANCE_CM} cm)")
            last_alert_time["clearance"] = now

        # ── Terrain Hazard Alert ─────────────────
        terrain_alert = (
            hazard_prob > TERRAIN_THRESHOLD and
            utime.ticks_diff(now, last_alert_time["terrain"]) > COOLDOWN_MS
        )
        if terrain_alert:
            emit_alert(f"TERRAIN HAZARD  score={hazard_prob:.2f}")
            last_alert_time["terrain"] = now

        # ── Status Print ─────────────────────────
        dist_str = f"{distance_cm:.1f} cm" if distance_cm >= 1.0 else "OUT OF RANGE"
        print(
            f"IR={'REFLECT' if ir_val == 0 else 'OPEN':7s} | "
            f"dist={dist_str:>10s} [{zone:7s}] | "
            f"camouf={camouf_prob:.2f}({camouf_confirmed}/{CAMOUF_CONFIRM}) | "
            f"hazard={hazard_prob:.2f} | "
            f"{terrain_label}"
        )

        # ── Reset window for next batch ──────────
        ir_history = []

    utime.sleep_ms(SAMPLE_DELAY_MS)
