"""
Posture assessment (Phase 7): GOOD / WARNING / POOR, judged against the user's OWN calibrated sitting position.

WHAT IS MEASURED (all from the webcam, no wearable)
    head tilt          sideways tilt of the head (eye line)                        -> face landmarks
    head nod           head dropped forward/down (pitch)                          -> face landmarks, 3D fit
    distance           face bigger/smaller than usual = leaning towards/away      -> eye-to-eye distance
    head position      head sunk lower / moved sideways in the picture            -> face box
    shoulder tilt      one shoulder higher than the other      (only if visible)  -> pose model
    head-shoulder gap  head sinking towards the shoulders      (only if visible)  -> pose model
    Shoulders are extra evidence: without them posture is judged from the head alone, and the result says so.

HOW IT DECIDES (same scheme as the fatigue rules, so it is easy to explain)
    Every indicator is a DEVIATION from the user's calibrated normal. Each has a WARNING and a POOR value
    (utils/config.py, POSTURE_RULE_THRESHOLDS). Its score is 0.5 at the warning value and 1.0 at the poor value.
    The posture score is the highest indicator score. Below 0.5 = GOOD, 0.5 up to 1.0 = WARNING, 1.0 or more = POOR.
    A worse posture must last POSTURE_ESCALATE_HOLD_S (5 s) before it is shown, so one bad frame never triggers a warning.

HONEST LIMITS
    * There is nothing about posture in the base paper. This whole phase comes from the synopsis, and every threshold is our own
      unvalidated starting value.
    * "Good" only means "close to how you sat when you calibrated". If you calibrate while slouching, slouching counts as normal.
      Calibrate sitting the way you want to sit.
    * A front-facing webcam cannot see slouching directly. Head nod, distance and shoulder cues are indirect.
    * Absolute head angles from a generic face model can be off by several degrees; only DEVIATIONS from your baseline are used.
    This is a monitoring indication, not medical advice.
"""

import math

import numpy as np

from computer_vision import head_pose
from ml.rules import indicator_score
from utils import config
from utils.smoothing import StateSmoother

GOOD, WARNING, POOR = "good", "warning", "poor"
LEVELS = [GOOD, WARNING, POOR]
DISCLAIMER = "This is a monitoring indication, not medical advice."
REQUIRED_BASELINE_KEYS = ("roll_deg", "pitch_deg", "eye_distance", "face_x", "face_y")


# ----------------------------------------------------------------------
# Measuring one frame
# ----------------------------------------------------------------------
def measure_posture(landmarks, box, pose, image_width, image_height):
    """
    landmarks: (478, 3) normalised face landmarks. box: the face box (x0, y0, x1, y1), normalised.
    pose: a PoseResult (shoulders) or None. Returns a metrics dict, or None if the head pose cannot be worked out.
    """
    roll = head_pose.eye_line_roll(landmarks, image_width, image_height)
    fit = head_pose.estimate_head_pose(landmarks, image_width, image_height)
    if roll is None or fit is None:
        return None
    yaw, pitch, _ = fit
    # Turning the head makes the eye-to-eye distance look smaller even at the same distance from the camera
    # (foreshortening), which could otherwise be misread as "leaning away". Divide by cos(yaw) to correct for
    # it. Capped at 60 degrees so a near-90-degree turn (cos close to 0) cannot blow up the correction.
    foreshortening = max(math.cos(math.radians(min(abs(yaw), 60.0))), 0.5)
    metrics = {
        "roll_deg": roll,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "eye_distance": head_pose.eye_distance(landmarks, image_width, image_height) / foreshortening,
        "face_x": (box[0] + box[2]) / 2.0,
        "face_y": (box[1] + box[3]) / 2.0,
        "shoulder_tilt_deg": None,
        "head_shoulder_ratio": None,
        "shoulder_width": None,
    }
    if pose is not None and pose.visible:
        left = np.array([pose.left_point[0] * image_width, pose.left_point[1] * image_height])
        right = np.array([pose.right_point[0] * image_width, pose.right_point[1] * image_height])
        width_px = float(np.linalg.norm(right - left))
        eyes_y = (landmarks[head_pose.EYE_CORNER_RIGHT, 1] + landmarks[head_pose.EYE_CORNER_LEFT, 1]) / 2.0
        gap_px = ((left[1] + right[1]) / 2.0 - float(eyes_y) * image_height)      # shoulders should be BELOW the eyes
        if width_px > 1e-6 and gap_px > 0:
            metrics["shoulder_tilt_deg"] = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
            metrics["head_shoulder_ratio"] = gap_px / width_px
            metrics["shoulder_width"] = width_px / image_width
    return metrics


# ----------------------------------------------------------------------
# The personal baseline
# ----------------------------------------------------------------------
def build_posture_baseline(samples):
    """
    Median of many steady measurements while the user sits normally. Returns (baseline, problem):
    baseline is None (and problem explains why) if there is not enough data.
    Shoulder values are included only if the shoulders were seen in most of the samples.
    """
    n = len(samples)
    if n < config.POSTURE_CAL_MIN_SAMPLES:
        return None, ("Your head position could not be measured often enough for the posture baseline. "
                      "Please keep your face inside the camera view and sit normally.")
    median = lambda key, rows=samples: float(np.median([row[key] for row in rows]))
    baseline = {key: median(key) for key in REQUIRED_BASELINE_KEYS}
    baseline["yaw_deg"] = median("yaw_deg")            # where the head points when facing the screen (for attention)
    baseline["samples"] = n
    if baseline["eye_distance"] <= 0:
        return None, "Your face size could not be measured. Please face the camera."
    with_shoulders = [row for row in samples if row["shoulder_tilt_deg"] is not None]
    baseline["shoulders"] = len(with_shoulders) >= config.POSTURE_CAL_MIN_SHOULDER_FRACTION * n
    if baseline["shoulders"]:
        for key in ("shoulder_tilt_deg", "head_shoulder_ratio", "shoulder_width"):
            baseline[key] = median(key, with_shoulders)
    return baseline, ""


def baseline_is_usable(baseline):
    """True for a dict with all required numbers (used when loading from the database)."""
    if not isinstance(baseline, dict):
        return False
    try:
        return all(math.isfinite(float(baseline[key])) for key in REQUIRED_BASELINE_KEYS) and float(baseline["eye_distance"]) > 0
    except (KeyError, TypeError, ValueError):
        return False


# ----------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------
def deviations(metrics, baseline):
    """Signed differences from the user's normal, for display: tilt/nod in degrees, distance in %, position in % of the picture."""
    out = {
        "head_tilt_deg": metrics["roll_deg"] - baseline["roll_deg"],
        "head_nod_deg": metrics["pitch_deg"] - baseline["pitch_deg"],           # positive = head lower than normal
        "distance_pct": (metrics["eye_distance"] / baseline["eye_distance"] - 1.0) * 100.0,
        "head_drop_pct": (metrics["face_y"] - baseline["face_y"]) * 100.0,
        "head_shift_pct": (metrics["face_x"] - baseline["face_x"]) * 100.0,
        "shoulder_tilt_deg": None,
    }
    if metrics.get("shoulder_tilt_deg") is not None and baseline.get("shoulders"):
        out["shoulder_tilt_deg"] = metrics["shoulder_tilt_deg"] - baseline["shoulder_tilt_deg"]
    return out


def indicator_values(metrics, baseline):
    """The indicator values (all >= 0). Shoulder indicators appear only when shoulders are usable in both places."""
    d = deviations(metrics, baseline)
    values = {
        "head_tilt_deg": abs(d["head_tilt_deg"]),
        "head_down_deg": max(0.0, d["head_nod_deg"]),
        "distance_change": abs(d["distance_pct"]) / 100.0,
        "head_drop": max(0.0, d["head_drop_pct"]) / 100.0,
        "head_shift": abs(d["head_shift_pct"]) / 100.0,
    }
    if d["shoulder_tilt_deg"] is not None and metrics.get("head_shoulder_ratio") is not None:
        values["shoulder_tilt_deg"] = abs(d["shoulder_tilt_deg"])
        values["head_shoulder_gap"] = max(0.0, 1.0 - metrics["head_shoulder_ratio"] / baseline["head_shoulder_ratio"])
    return values


def _reason(name, value, warning_level, poor_level, distance_pct):
    if name == "head_tilt_deg":
        return f"Head tilted {value:.0f} degrees sideways from your normal (warning {warning_level:.0f}, poor {poor_level:.0f})"
    if name == "head_down_deg":
        return f"Head dropped forward or down {value:.0f} degrees from your normal (warning {warning_level:.0f}, poor {poor_level:.0f})"
    if name == "distance_change":
        direction = "Leaning towards the screen: your face looks" if distance_pct > 0 else "Leaning away from the screen: your face looks"
        size = "larger" if distance_pct > 0 else "smaller"
        return f"{direction} {value * 100:.0f}% {size} than usual (warning {warning_level * 100:.0f}%, poor {poor_level * 100:.0f}%)"
    if name == "head_drop":
        return f"Head sunk lower than usual by {value * 100:.0f}% of the picture height (warning {warning_level * 100:.0f}%, poor {poor_level * 100:.0f}%)"
    if name == "head_shift":
        return f"Head moved sideways by {value * 100:.0f}% of the picture width (warning {warning_level * 100:.0f}%, poor {poor_level * 100:.0f}%)"
    if name == "shoulder_tilt_deg":
        return f"Shoulders uneven by {value:.0f} degrees compared with your normal (warning {warning_level:.0f}, poor {poor_level:.0f})"
    return f"Head sinking towards the shoulders by {value * 100:.0f}% (warning {warning_level * 100:.0f}%, poor {poor_level * 100:.0f}%)"


def classify(metrics, baseline):
    """Returns (score, state, reasons) for one frame."""
    values = indicator_values(metrics, baseline)
    distance_pct = deviations(metrics, baseline)["distance_pct"]
    best, reasons = 0.0, []
    for name, (warning_level, poor_level) in config.POSTURE_RULE_THRESHOLDS.items():
        if name not in values:
            continue
        score = indicator_score(values[name], warning_level, poor_level)
        if score >= 0.5:
            reasons.append((score, _reason(name, values[name], warning_level, poor_level, distance_pct)))
        best = max(best, score)
    reasons.sort(key=lambda item: -item[0])
    state = POOR if best >= 1.0 else WARNING if best >= 0.5 else GOOD
    return round(best, 3), state, [text for _, text in reasons]


# ----------------------------------------------------------------------
# The assessor
# ----------------------------------------------------------------------
class PostureAssessor:
    def __init__(self, baseline=None):
        """baseline: the user's posture baseline dict from calibration, or None (not calibrated for posture)."""
        self.baseline = baseline if baseline_is_usable(baseline) else None
        self._smoother = StateSmoother(LEVELS, config.POSTURE_ESCALATE_HOLD_S, config.POSTURE_RECOVERY_HOLD_S)
        self._events = {"warning": 0, "poor": 0}
        self._last_shown = GOOD

    def idle_snapshot(self):
        return self._snapshot(None, None, [], "Waiting for posture measurements.", "", None)

    def update(self, t, metrics, calibrating=False):
        """t: seconds. metrics: from measure_posture() or None when the face is not measured. Returns a snapshot dict."""
        if calibrating:
            return self._unavailable("Calibration is running.")
        if self.baseline is None:
            return self._unavailable("Posture is not calibrated yet. Please run the calibration (about 15 seconds) to enable it.")
        if metrics is None:
            return self._unavailable("Your face is not being measured right now, so posture cannot be assessed.")

        score, raw_state, reasons = classify(metrics, self.baseline)
        shown = self._smoother.update(t, raw_state)
        self._count(shown)
        if shown != GOOD and LEVELS.index(raw_state) < LEVELS.index(shown):
            reasons = ["Posture is improving; waiting for steady readings before lowering the state."]
        elif shown == GOOD:
            reasons = []
        shoulders_now = metrics.get("shoulder_tilt_deg") is not None and bool(self.baseline.get("shoulders"))
        note = "" if shoulders_now else "Judged from your head position only (shoulders not used)."
        return self._snapshot(shown, score, reasons, "", note, deviations(metrics, self.baseline), shoulders_now)

    # ------------------------------------------------------------------
    def _unavailable(self, message):
        self._smoother.reset()
        self._last_shown = GOOD
        return self._snapshot(None, None, [], message, "", None)

    def _count(self, shown):
        if shown != self._last_shown:
            if shown == WARNING and self._last_shown == GOOD:
                self._events["warning"] += 1
            elif shown == POOR:
                self._events["poor"] += 1
            self._last_shown = shown

    def _snapshot(self, state, score, reasons, message, note, dev, shoulders_used=False):
        return {
            "state": state or "unavailable",
            "score": score,
            "reasons": list(reasons),
            "message": message,
            "note": note,
            "shoulders_used": shoulders_used,
            "deviations": None if dev is None else {k: (None if v is None else round(v, 1)) for k, v in dev.items()},
            "events": dict(self._events),
            "disclaimer": DISCLAIMER,
        }
