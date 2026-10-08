"""
DHRI -- Digital Health Risk Index (Phase 10): a single, transparent risk score from 0 (best) to 100 (worst),
combining fatigue, posture, attention and continuous exposure into ONE number, as the project brief asks for.

This is OUR OWN project-specific indicator, not a medically validated health score and not a diagnosis --
see docs/architecture.md for the full reasoning behind the weights and thresholds below, all of which are
unvalidated starting values.

HOW IT IS BUILT (transparent by design)
    Each of the four inputs already produces its own 0..1 "how bad is this right now" score, computed by
    earlier phases:
        fatigue    the fatigue assessor's score (rule-based score, or the trained model's probability)
        posture    the posture assessor's score
        attention  share of the last 60 s NOT attentive (Phase 8's "score")
        exposure   how long the person has been continuously active without a qualifying break (Phase 9's
                   session tracking), scored the same way as the fatigue/posture rules: 0.5 at the warning
                   length, 1.0 at the "high" length (utils/config.py: DHRI_EXPOSURE_WARNING_MIN/HIGH_MIN)

    DHRI = a WEIGHTED AVERAGE of whichever of these four are currently AVAILABLE, using DHRI_WEIGHTS
    (utils/config.py), re-normalised so the weights of the available inputs always sum to 1. An input is
    "available" only when its own card is not showing "unavailable" (not calibrated, calibrating, or not
    enough data yet). DHRI never guesses a value for a missing input -- it says which inputs are missing,
    and if NONE are available yet, DHRI itself is "unavailable".

WEIGHTS -- our own choice, not derived from the base paper or from data:
    fatigue 0.40, posture 0.25, attention 0.20, exposure 0.15
    Fatigue carries the most weight because it is the phase most directly built on the base paper's eye-
    closure concept (PERCLOS and related features). The paper's own idea is therefore already the single
    largest influence on DHRI, through the fatigue component -- it is not counted a second time as a
    separate "eye closure" input, to avoid double-weighting the same underlying signal.

SCORE AND RISK LEVEL
    DHRI is reported on a 0-100 scale (the weighted average times 100, one decimal) for readability.
        LOW RISK       DHRI < 35     (DHRI_LOW_MAX)
        MODERATE RISK  35 to 65      (DHRI_MODERATE_MAX)
        HIGH RISK      65 and above
    Both thresholds are configurable, as the brief asks. A worse level must be the CURRENT raw level,
    continuously, for DHRI_ESCALATE_HOLD_S (5 s) before it is shown; a better one for DHRI_RECOVERY_HOLD_S
    (5 s). This is DHRI's OWN smoothing, not the shared StateSmoother used by fatigue/posture/attention --
    see the note below for why.

WHY DHRI DOES NOT REUSE THE SHARED StateSmoother
    fatigue/posture/attention each smooth a single indicator that can genuinely DIP mid-escalation (for
    example a fatigue reading that touches WARNING, drops back, then reaches FATIGUED again), and their
    shared StateSmoother is deliberately conservative there: it only shows a level once EVERY reading in the
    whole waiting window was at least that bad, so a real dip correctly delays the escalation.
    DHRI's score is a weighted combination of already-computed indicators, and in practice it RAMPS
    smoothly upward as an underlying quantity (for example how long the eyes have been closed) accumulates
    -- passing briefly through MODERATE on the way to a settled HIGH is not a "dip", it is normal warm-up.
    Reusing the conservative rule here would make each level require its OWN full hold period in sequence
    (LOW to MODERATE, then MODERATE to HIGH), doubling or worse the intended delay for a value that is
    simply climbing and then staying high. DHRI instead tracks how long the CURRENT raw level has been the
    reading, restarting the clock whenever that raw level changes; a value that settles at HIGH reaches
    HIGH exactly DHRI_ESCALATE_HOLD_S after it stops changing, however it got there, while a genuine
    single-frame flicker (which changes the raw level again before the hold time is up) still cannot cause
    a promotion. This was found and fixed by scripting a full session through the real monitor (see
    docs/architecture.md) before this phase was shipped, not by a unit test of DHRI in isolation.

WHAT DHRI IS NOT
    Not a medical score, not validated against any ground truth, and not a diagnosis. It is a project-
    specific way to summarise four already-approximate indicators into one number for the dashboard.
"""

from ml import rules
from utils import config

LOW, MODERATE, HIGH = "low", "moderate", "high"
LEVELS = [LOW, MODERATE, HIGH]
UNAVAILABLE = "unavailable"

DISCLAIMER = ("DHRI is a project-specific digital wellbeing indicator, not a medically validated health "
              "score and not a diagnosis. It summarises fatigue, posture, attention and exposure into one number.")

INPUT_NAMES = ("fatigue", "posture", "attention", "exposure")


def exposure_score(continuous_active_s):
    """0..1 from how long someone has been continuously active without a qualifying break. None if unknown."""
    if continuous_active_s is None:
        return None
    warning_s = config.DHRI_EXPOSURE_WARNING_MIN * 60.0
    high_s = config.DHRI_EXPOSURE_HIGH_MIN * 60.0
    return rules.indicator_score(continuous_active_s, warning_s, high_s)


def _score_of(snapshot):
    """The 0..1 score from a fatigue/posture/attention snapshot, or None if that card is unavailable."""
    if not snapshot or snapshot.get("state") in (None, UNAVAILABLE):
        return None
    return snapshot.get("score")


def _level_from_score100(score100):
    if score100 >= config.DHRI_MODERATE_MAX:
        return HIGH
    if score100 >= config.DHRI_LOW_MAX:
        return MODERATE
    return LOW


class _SustainedLevelSmoother:
    """
    DHRI's own escalation logic (see the module docstring for why this differs from utils/smoothing.py):
    the shown level changes to the CURRENT raw level once that exact raw level has held continuously for
    the relevant hold time. Changing the raw level at all -- up, down, or to a different level entirely --
    restarts the clock, so a genuine flicker never causes a promotion, but a value that ramps through
    several levels and then SETTLES is shown at its settled level after just one hold period, not one per
    level crossed.
    """

    def __init__(self, levels, escalate_hold_s, recovery_hold_s):
        self.levels = list(levels)
        self.escalate_hold_s = escalate_hold_s
        self.recovery_hold_s = recovery_hold_s
        self.reset()

    def reset(self):
        self.shown = self.levels[0]
        self._candidate = None
        self._candidate_since = None

    def update(self, t, raw_level_name):
        if raw_level_name == self.shown:
            self._candidate = self._candidate_since = None
            return self.shown
        if raw_level_name != self._candidate:
            self._candidate, self._candidate_since = raw_level_name, t
        hold = (self.escalate_hold_s if self.levels.index(raw_level_name) > self.levels.index(self.shown)
                else self.recovery_hold_s)
        if t - self._candidate_since >= hold:
            self.shown = raw_level_name
            self._candidate = self._candidate_since = None
        return self.shown


class DHRIAssessor:
    def __init__(self):
        self._smoother = _SustainedLevelSmoother(LEVELS, config.DHRI_ESCALATE_HOLD_S, config.DHRI_RECOVERY_HOLD_S)
        self._events = {"moderate": 0, "high": 0}
        self._last_shown = LOW

    def idle_snapshot(self):
        return self._snapshot(None, None, UNAVAILABLE, list(INPUT_NAMES), {})

    def update(self, t, fatigue_snapshot, posture_snapshot, attention_snapshot, continuous_active_s):
        """
        t: seconds since the session started.
        fatigue_snapshot / posture_snapshot / attention_snapshot: the latest snapshot dicts from those
        assessors (as stored by the monitor). continuous_active_s: from the session tracker's live summary.
        Returns a snapshot dict.
        """
        inputs = {
            "fatigue": _score_of(fatigue_snapshot),
            "posture": _score_of(posture_snapshot),
            "attention": _score_of(attention_snapshot),
            "exposure": exposure_score(continuous_active_s),
        }
        available = {name: value for name, value in inputs.items() if value is not None}
        missing = [name for name in INPUT_NAMES if name not in available]

        if not available:
            self._smoother.reset()
            self._last_shown = LOW
            return self._snapshot(None, None, UNAVAILABLE, missing, {})

        total_weight = sum(config.DHRI_WEIGHTS[name] for name in available)
        raw_score = sum(config.DHRI_WEIGHTS[name] * value for name, value in available.items()) / total_weight
        score100 = round(raw_score * 100.0, 1)

        raw_level = _level_from_score100(score100)
        shown = self._smoother.update(t, raw_level)
        self._count(shown)

        # How much each available input actually contributed to the final 0-100 score, for the dashboard's
        # breakdown -- these add up to score100 (rounding aside).
        contributions = {
            name: round(config.DHRI_WEIGHTS[name] / total_weight * value * 100.0, 1)
            for name, value in available.items()
        }
        return self._snapshot(score100, contributions, shown, missing, dict(inputs))

    # ------------------------------------------------------------------
    def _count(self, shown):
        if shown != self._last_shown:
            if shown == MODERATE and self._last_shown == LOW:
                self._events["moderate"] += 1
            elif shown == HIGH:
                self._events["high"] += 1
            self._last_shown = shown

    def _snapshot(self, score100, contributions, level, missing, raw_inputs):
        return {
            "level": level,
            "score": score100,
            "contributions": contributions or {},
            "missing_inputs": missing,
            "raw_inputs": raw_inputs,
            "events": dict(self._events),
            "disclaimer": DISCLAIMER,
        }
