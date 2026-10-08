"""
Fatigue assessment: turns the eye features into NORMAL / WARNING / FATIGUED.

TWO MODES (the dashboard always says which one is active)
    DEMO / RULE-BASED MODE   readable rules (ml/rules.py). Used when no trained model exists.
    TRAINED ML MODE          an SVM trained on real recordings (ml/train.py), as in the base paper.
                             Used only if models/fatigue_svm.joblib exists, loads, and is not a demo model.
Neither mode is validated on real users yet. This is a monitoring indication, not a medical diagnosis.

SMOOTHING (no flickering, no alarm on a single frame)
    The eye features already cover a 20-second window. On top of that the shown state changes only when
    the new state has lasted long enough:
        getting worse : FATIGUE_ESCALATE_HOLD_S seconds (3 s), except while the eyes are closed for
                        longer than the fatigue limit right now, which is shown at once
        getting better: FATIGUE_RECOVERY_HOLD_S seconds (10 s)
    The score is NOT smoothed (it is the raw current value); only the state is.

WHEN NO ASSESSMENT IS POSSIBLE the state is "unavailable" and a message says why (not calibrated,
calibration running, eyes not visible, or still collecting the first ~10 seconds of data).
"""

from utils import config
from utils.smoothing import StateSmoother as _SharedSmoother
from ml import features, rules
from ml.model import ModelLoadError, load_model

MODE_RULES = "rules"
MODE_ML = "ml"
MODE_LABELS = {MODE_RULES: "DEMO / RULE-BASED MODE", MODE_ML: "TRAINED ML MODE"}

DISCLAIMER = "This is a monitoring indication, not a medical diagnosis."
UNAVAILABLE = "unavailable"


class StateSmoother(_SharedSmoother):
    """Time-based hysteresis for the shown fatigue state (see utils/smoothing.py)."""

    def __init__(self, escalate_hold_s=None, recovery_hold_s=None):
        super().__init__(
            [rules.NORMAL, rules.WARNING, rules.FATIGUED],
            config.FATIGUE_ESCALATE_HOLD_S if escalate_hold_s is None else escalate_hold_s,
            config.FATIGUE_RECOVERY_HOLD_S if recovery_hold_s is None else recovery_hold_s,
        )


class FatigueAssessor:
    def __init__(self, model=None, model_message=""):
        """model: a LoadedModel (TRAINED ML MODE) or None (DEMO / RULE-BASED MODE)."""
        self.model = model
        self.model_message = model_message
        self._smoother = StateSmoother()
        self._events = {"warning": 0, "fatigued": 0}
        self._last_shown = rules.NORMAL

    @classmethod
    def from_files(cls, model_path=None):
        """Use the trained model if there is a usable one, otherwise fall back to the rules (and say why)."""
        try:
            model = load_model(model_path)
        except ModelLoadError as error:
            return cls(model=None, model_message=str(error))
        return cls(model=model, model_message=model.warning)

    @property
    def mode(self):
        return MODE_ML if self.model is not None else MODE_RULES

    def info(self):
        """Mode information for the dashboard, before any measurement."""
        return {
            "mode": self.mode,
            "mode_label": MODE_LABELS[self.mode],
            "mode_message": self.model_message,
            "model_summary": self.model.summary() if self.model else None,
        }

    def idle_snapshot(self):
        return self._snapshot(UNAVAILABLE, None, [], "Waiting for eye measurements.", None)

    # ------------------------------------------------------------------
    def update(self, t, eye):
        """t: time in seconds. eye: the feature dict from EyeStateTracker. Returns a snapshot dict."""
        reason = self._why_unavailable(eye)
        if reason:
            self._smoother.reset()
            self._last_shown = rules.NORMAL
            return self._snapshot(UNAVAILABLE, None, [], reason, None)

        rule = rules.classify(eye)
        if self.model is not None:
            probability = self.model.fatigue_probability(features.feature_dict(eye))
            raw_state = _state_from_probability(probability)
            score, immediate = probability, False
            reasons = [f"Trained model: {probability * 100:.0f}% likelihood of a fatigue pattern"] if raw_state != rules.NORMAL else []
        else:
            raw_state, score, immediate, reasons = rule.state, rule.score, rule.immediate, rule.reasons

        shown = self._smoother.update(t, raw_state, immediate)
        self._count_events(shown)
        if shown != rules.NORMAL and rules.STATE_ORDER[raw_state] < rules.STATE_ORDER[shown]:
            reasons = ["Signs are improving; waiting for steady readings before lowering the state."]
        elif shown == rules.NORMAL:
            reasons = []
        return self._snapshot(shown, score, reasons, "", rule.state)

    # ------------------------------------------------------------------
    def _why_unavailable(self, eye):
        if not eye:
            return "Waiting for eye measurements."
        if not eye.get("calibrated"):
            return "Please calibrate first (about 15 seconds) so fatigue can be assessed."
        if eye.get("paused"):
            return "Calibration is running."
        if eye.get("ear") is None:
            return "Your eyes are not visible, so fatigue cannot be assessed."
        if not features.window_features_ready(eye):
            remaining = max(0.0, config.FEATURE_MIN_COVERAGE_S - (eye.get("window_valid_s") or 0.0))
            return f"Collecting data ({remaining:.0f} s more) before fatigue can be assessed."
        return ""

    def _count_events(self, shown):
        if shown != self._last_shown:
            if shown == rules.WARNING and self._last_shown == rules.NORMAL:
                self._events["warning"] += 1
            elif shown == rules.FATIGUED:
                self._events["fatigued"] += 1
            self._last_shown = shown

    def _snapshot(self, state, score, reasons, message, rule_state):
        info = self.info()
        return {
            "mode": info["mode"],
            "mode_label": info["mode_label"],
            "state": state,
            "score": None if score is None else round(score, 3),
            "reasons": list(reasons),
            "message": message,
            "rule_state": rule_state,
            "events": dict(self._events),
            "disclaimer": DISCLAIMER,
        }


def _state_from_probability(probability):
    if probability >= config.ML_FATIGUED_PROBABILITY:
        return rules.FATIGUED
    if probability >= config.ML_WARNING_PROBABILITY:
        return rules.WARNING
    return rules.NORMAL
