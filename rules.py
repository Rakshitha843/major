"""
DEMO / RULE-BASED MODE: simple, readable fatigue rules.

This is what runs when no trained model exists. It is NOT validated on real users, and the
thresholds (utils/config.py, FATIGUE_RULE_THRESHOLDS) are our own starting values.

HOW IT WORKS (transparent on purpose)
    Four indicators are measured over the last 20 seconds:
        perclos            share of time the eyelids are at least 80% closed
        closure_s          longest single eye closure
        blink_rate         blinks per minute
        blink_duration_s   average blink duration
    Each has a WARNING level and a FATIGUED level. Its score is 0.5 at the warning level and 1.0 at
    the fatigued level (a straight line in between, 0 for no sign at all).
    The overall fatigue score is the HIGHEST indicator score.
        score below 0.5      -> normal
        score 0.5 to below 1 -> warning
        score 1.0 or more    -> fatigued
    The result also lists WHICH indicator caused it, so the user can see why.

This is a monitoring indication, not a medical diagnosis.
"""

from dataclasses import dataclass, field

from utils import config

NORMAL = "normal"
WARNING = "warning"
FATIGUED = "fatigued"
STATE_ORDER = {NORMAL: 0, WARNING: 1, FATIGUED: 2}


@dataclass
class RuleResult:
    score: float = 0.0
    state: str = NORMAL
    reasons: list = field(default_factory=list)
    immediate: bool = False        # a closure that is happening right now and is already too long


def indicator_values(eye, include_current=True):
    """The four indicator values from the eye features (None = not available)."""
    closure = eye.get("longest_closure_s")
    if include_current and eye.get("current_closure_s") is not None:
        closure = eye["current_closure_s"] if closure is None else max(closure, eye["current_closure_s"])
    return {
        "perclos": eye.get("perclos"),
        "closure_s": closure,
        "blink_rate": eye.get("blinks_per_min"),
        "blink_duration_s": eye.get("mean_blink_duration_s"),
    }


def indicator_score(value, warning_level, fatigued_level):
    """0 = no sign, 0.5 = at the warning level, 1.0 = at the fatigued level or beyond."""
    if value is None or value <= 0:
        return 0.0
    if value < warning_level:
        return 0.5 * value / warning_level
    if value < fatigued_level:
        return 0.5 + 0.5 * (value - warning_level) / (fatigued_level - warning_level)
    return 1.0


def state_from_score(score):
    if score >= 1.0:
        return FATIGUED
    if score >= 0.5:
        return WARNING
    return NORMAL


def _reason(name, value, warning_level, fatigued_level):
    if name == "perclos":
        text = f"Eyes mostly closed {value * 100:.0f}% of the last 20 s (warning {warning_level * 100:.0f}%, fatigue {fatigued_level * 100:.0f}%)"
    elif name == "closure_s":
        text = f"Longest eye closure {value:.1f} s (warning {warning_level:.1f} s, fatigue {fatigued_level:.1f} s)"
    elif name == "blink_rate":
        text = f"Frequent blinking, {value:.0f} per minute (warning {warning_level:.0f}, fatigue {fatigued_level:.0f})"
    else:
        text = f"Slow blinks, {value:.2f} s on average (warning {warning_level:.2f} s, fatigue {fatigued_level:.2f} s)"
    return text


def classify(eye, include_current=True):
    """Apply the rules to one set of eye features. Returns a RuleResult."""
    values = indicator_values(eye, include_current)
    best_score, reasons = 0.0, []
    for name, (warning_level, fatigued_level) in config.FATIGUE_RULE_THRESHOLDS.items():
        score = indicator_score(values.get(name), warning_level, fatigued_level)
        if score >= 0.5:
            reasons.append((score, _reason(name, values[name], warning_level, fatigued_level)))
        best_score = max(best_score, score)

    reasons.sort(key=lambda item: -item[0])
    current = eye.get("current_closure_s") if include_current else None
    immediate = bool(current is not None and current >= config.FATIGUE_RULE_THRESHOLDS["closure_s"][1])
    return RuleResult(
        score=round(best_score, 3),
        state=state_from_score(best_score),
        reasons=[text for _, text in reasons],
        immediate=immediate,
    )
