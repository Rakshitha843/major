"""
Attention (Phase 8): ATTENTIVE / DISTRACTED (LOOKING AWAY) / FACE NOT DETECTED.

WHAT THIS IS, AND IS NOT
    An OBSERVABLE indicator built from two things the camera can see:
      * is a face visible at all, and
      * does the head point roughly at the screen? "At the screen" means the head direction recorded during YOUR calibration,
        when you were looking at your screen (utils/config.py: ATTENTION_YAW_LIMIT_DEG, ATTENTION_PITCH_LIMIT_DEG).
    It cannot see where your EYES look, so reading a paper next to the screen with a slight head turn, or looking at a second
    monitor with only your eyes, is not detected. It cannot know what you are thinking. "Not attentive" only means "head turned away
    or face not visible".

THE THREE STATES
    attentive     the face is visible and the head points at the screen (within the limits)
    distracted    the face is visible but the head has been turned away (sideways, up or down) for ATTENTION_AWAY_HOLD_S (5 s)
    no_face       no face has been visible for ATTENTION_NO_FACE_HOLD_S (3 s). This covers two cases the camera cannot tell apart:
                  you are away from the camera, OR you turned so far that the face model lost your face.
    Looking away must LAST before it is shown, so a glance at the keyboard or the phone never triggers it, and a single missed
    detection never shows "face not detected". Coming back must last 2 s before ATTENTIVE returns.

OTHER OUTPUTS (used by the dashboard, and by the DHRI score and alerts in ml/)
    score / attentive_percent   share of the last 60 s (only counting moments that could be judged) spent NOT attentive
    continuous_away_s           how long the head has been turned away or the face missing right now
    extended_absence            the face has been missing for a long time (60 s)
    episodes                    how many times DISTRACTED / FACE NOT DETECTED was entered this session

WHEN IT CANNOT JUDGE the state is "unavailable" with a reason: not calibrated (or calibrated before Phase 8, which did not record head
direction), calibration running, or the picture has been unsteady for a while (moments right after a hand or a light change are skipped).

Nothing in the base paper covers attention; this phase comes from the synopsis, and every limit is our own unvalidated starting value.
"""

from collections import deque

from utils import config
from utils.smoothing import StateSmoother

ATTENTIVE, DISTRACTED, NO_FACE = "attentive", "distracted", "no_face"
LEVELS = [ATTENTIVE, DISTRACTED, NO_FACE]
UNAVAILABLE = "unavailable"

DISCLAIMER = ("Attention is estimated only from whether your face is visible and where your head points. "
              "It cannot know where your eyes look or what you are thinking.")

# Measurement problems (see computer_vision/eye_measurement.py): these mean "nobody can be seen"
NOBODY_VISIBLE = ("no_face", "too_dark")


def baseline_supports_attention(baseline):
    """True if the posture baseline also holds the head direction (baselines saved before Phase 8 do not)."""
    try:
        return baseline is not None and float(baseline["yaw_deg"]) == float(baseline["yaw_deg"]) \
            and float(baseline["pitch_deg"]) == float(baseline["pitch_deg"])
    except (KeyError, TypeError, ValueError):
        return False


class AttentionAssessor:
    def __init__(self, baseline=None):
        """baseline: the user's posture baseline dict (it holds yaw_deg and pitch_deg), or None."""
        self.baseline = baseline if baseline_supports_attention(baseline) else None
        self.has_old_baseline = baseline is not None and self.baseline is None      # calibrated before Phase 8
        self._smoother = StateSmoother(LEVELS, config.ATTENTION_AWAY_HOLD_S, config.ATTENTION_RECOVERY_HOLD_S)
        self._away_latched = False                       # hysteresis: once "away", stay away until clearly back
        self._history = deque()                          # (time, seconds since previous, not attentive)
        self._last_time = None
        self._away_since = None                          # since when the raw state has been "not attentive"
        self._no_face_since = None
        self._unknown_since = None
        self._events = {"distracted": 0, "no_face": 0}
        self._last_shown = ATTENTIVE
        self._last_snapshot = None

    def idle_snapshot(self):
        return self._snapshot(UNAVAILABLE, "Waiting for attention measurements.")

    # ------------------------------------------------------------------
    def update(self, t, reason, metrics, calibrating=False):
        """
        t: time in seconds.
        reason: why the eyes could not be measured ("" if fine), from EyeMeasurement.reason.
        metrics: posture metrics of this frame (with yaw_deg / pitch_deg) or None.
        Returns a snapshot dict.
        """
        if calibrating:
            return self._unavailable("Calibration is running.")
        if self.baseline is None:
            if self.has_old_baseline:
                return self._unavailable("Your calibration was made before attention monitoring existed. Please calibrate again "
                                         "(about 15 seconds) so the system knows where your head points when you look at your screen.")
            return self._unavailable("Please calibrate first (about 15 seconds) so the system knows where your head points "
                                     "when you look at your screen.")

        raw = self._raw_state(reason, metrics)
        if raw is None:                                   # unsteady picture: no information right now
            if self._unknown_since is None:
                self._unknown_since = t
            if t - self._unknown_since > config.ATTENTION_UNKNOWN_LIMIT_S:
                return self._unavailable("The picture is unsteady right now, so attention cannot be judged.")
            return self._last_snapshot or self.idle_snapshot()
        self._unknown_since = None

        self._track(t, raw)
        hold = config.ATTENTION_NO_FACE_HOLD_S if raw == NO_FACE else config.ATTENTION_AWAY_HOLD_S
        shown = self._smoother.update(t, raw, hold_s=hold)
        self._count(shown)
        return self._snapshot(shown, "", raw, metrics, t)

    # ------------------------------------------------------------------
    def _raw_state(self, reason, metrics):
        if reason in NOBODY_VISIBLE:
            self._away_latched = False
            return NO_FACE
        if metrics is None:
            return None
        factor = config.ATTENTION_EXIT_RATIO if self._away_latched else 1.0
        yaw_dev = abs(metrics["yaw_deg"] - self.baseline["yaw_deg"])
        pitch_dev = abs(metrics["pitch_deg"] - self.baseline["pitch_deg"])
        away = yaw_dev >= config.ATTENTION_YAW_LIMIT_DEG * factor or pitch_dev >= config.ATTENTION_PITCH_LIMIT_DEG * factor
        self._away_latched = away
        return DISTRACTED if away else ATTENTIVE

    def _track(self, t, raw):
        elapsed = 0.0 if self._last_time is None else min(1.0, max(0.0, t - self._last_time))
        self._last_time = t
        self._history.append((t, elapsed, raw != ATTENTIVE))
        while self._history and self._history[0][0] < t - config.ATTENTION_WINDOW_S:
            self._history.popleft()
        self._away_since = (self._away_since if self._away_since is not None else t) if raw != ATTENTIVE else None
        self._no_face_since = (self._no_face_since if self._no_face_since is not None else t) if raw == NO_FACE else None

    def _count(self, shown):
        if shown != self._last_shown:
            if shown == DISTRACTED:
                self._events["distracted"] += 1
            elif shown == NO_FACE:
                self._events["no_face"] += 1
            self._last_shown = shown

    def _unavailable(self, message):
        self._smoother.reset()
        self._last_shown = ATTENTIVE
        self._away_latched = False
        self._last_time = self._away_since = self._no_face_since = None
        self._history.clear()
        self._last_snapshot = self._snapshot(UNAVAILABLE, message)
        return self._last_snapshot

    def _snapshot(self, state, message, raw=None, metrics=None, t=None):
        observed = sum(dt for _, dt, _ in self._history)
        away_time = sum(dt for _, dt, away in self._history if away)
        score = None
        if state != UNAVAILABLE and observed >= config.ATTENTION_MIN_OBSERVED_S:
            score = round(away_time / observed, 3)
        deviations = None
        if metrics is not None and self.baseline is not None and state != UNAVAILABLE:
            deviations = {
                "head_turn_deg": round(metrics["yaw_deg"] - self.baseline["yaw_deg"], 1),
                "head_up_down_deg": round(metrics["pitch_deg"] - self.baseline["pitch_deg"], 1),   # positive = head lower
            }
        no_face_s = 0.0 if (t is None or self._no_face_since is None) else t - self._no_face_since
        snapshot = {
            "state": state,
            "score": score,
            "attentive_percent": None if score is None else round(100 * (1 - score)),
            "continuous_away_s": 0.0 if (t is None or self._away_since is None) else round(t - self._away_since, 1),
            "extended_absence": no_face_s >= config.ATTENTION_EXTENDED_ABSENCE_S,
            "reasons": self._reasons(state, raw, metrics),
            "message": message,
            "deviations": deviations,
            "events": dict(self._events),
            "disclaimer": DISCLAIMER,
        }
        self._last_snapshot = snapshot
        return snapshot

    def _reasons(self, state, raw, metrics):
        if state in (UNAVAILABLE, ATTENTIVE):
            return []
        if LEVELS.index(raw) < LEVELS.index(state):
            return ["Facing the screen again; waiting for steady readings before showing ATTENTIVE."]
        if state == NO_FACE:
            return ["Your face is not visible: you may be away from the camera, or turned too far for it to see your face."]
        reasons = []
        if metrics is not None:
            yaw_dev = metrics["yaw_deg"] - self.baseline["yaw_deg"]
            pitch_dev = metrics["pitch_deg"] - self.baseline["pitch_deg"]
            if abs(yaw_dev) >= config.ATTENTION_YAW_LIMIT_DEG * config.ATTENTION_EXIT_RATIO:
                reasons.append(f"Head turned {abs(yaw_dev):.0f} degrees sideways from facing your screen "
                               f"(limit {config.ATTENTION_YAW_LIMIT_DEG:.0f})")
            if abs(pitch_dev) >= config.ATTENTION_PITCH_LIMIT_DEG * config.ATTENTION_EXIT_RATIO:
                direction = "down" if pitch_dev > 0 else "up"
                reasons.append(f"Head tilted {direction} {abs(pitch_dev):.0f} degrees from facing your screen "
                               f"(limit {config.ATTENTION_PITCH_LIMIT_DEG:.0f})")
        return reasons or ["Head turned away from your screen."]
