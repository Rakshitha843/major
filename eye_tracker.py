"""
Turns a stream of eye measurements into the eye features used for fatigue analysis.

FROM THE BASE PAPER (Sun et al. 2020, section 3.3) - the three eye features:
    DEC  Duration of Eye Closure   -> here: blink duration, current closure time, longest closure
    FB   Frequency of Blinks       -> here: blinks per minute over a 20-second window
    PCT  eyelid closure over time  -> here: PERCLOS-style closure percentage over a 20-second window
    The paper also recomputes FB and PCT every 20 seconds instead of every minute; we use the
    same 20-second window length.

OUR OWN CHOICES (not from the paper)
    * Open/closed is decided on a PERSONAL scale. Calibration measures the user's own EAR with
      eyes open and with eyes closed, and "closure" runs from 0.0 (their open eyes) to 1.0
      (their closed eyes).
    * A hysteresis (enter 0.5, leave 0.3) stops a value hovering at the threshold from
      counting many false blinks.
    * PERCLOS-style = share of time the closure is at least 0.8 (eyelid "80% closed"). This mimics
      the usual PERCLOS definition, but our closure comes from EAR, not from measuring the
      eyelid over the pupil, so it is an approximation.
    * A closure up to BLINK_MAX_S is a blink; a longer one is a "long closure".
    All numbers are in utils/config.py and are starting values that need testing on real data.

Everything works on TIMESTAMPS (seconds), not frame counts, so the result does not depend on the
frame rate. Timing precision is about one frame (50 ms at 20 fps).

If no face is measured, an ongoing closure is discarded: when we cannot see the eyes we do not
guess what they were doing.

OPEN EYES FIRST: a blink is "open -> closed -> open". So after monitoring starts, and again after every
break in the data (face lost, gap), nothing is counted and no time is added to the 20-second window
until the eyes have been seen clearly open once. This stops the first unreliable measurements after
the face reappears (for example after covering and uncovering the camera) from becoming false blinks.

CONFIRMATION: a finished blink (or long closure) is only COUNTED after the eyes have stayed measurable for
BLINK_CONFIRM_S (1.0 s) more. If the eyes stop being measurable in that time (the picture went dark, the
face was lost), the event is DISCARDED, because a blink-like dip right before the data disappears is most
likely a distorted face, not a real blink. Counts therefore appear about 1 s late, and the number of
discarded events is reported (`discarded_events`) so problems can be diagnosed.
"""

from collections import deque

from utils import config


def closure_fraction(ear, baseline):
    """0.0 = the user's normal open eyes, 1.0 = the user's closed eyes (clipped to that range)."""
    span = baseline["ear_open"] - baseline["ear_closed"]
    if span <= 0:
        return 0.0
    return min(1.0, max(0.0, (baseline["ear_open"] - ear) / span))


class EyeStateTracker:
    def __init__(self, baseline=None):
        """baseline: {"ear_open": ..., "ear_closed": ...} from calibration, or None (not calibrated)."""
        self.baseline = baseline
        self.blink_count = 0
        self.long_closure_count = 0
        self._window_samples = deque()     # (time, seconds since previous sample, is_80_percent_closed)
        self._blinks = deque()             # (end time, duration)
        self._long_closures = deque()      # (end time, duration)
        self._closed = False
        self._closed_since = None
        self._last_time = None
        self._seen_open = False            # have the eyes been seen clearly open since the last break?
        self._pending = deque()            # finished closures waiting for confirmation: (end time, duration)
        self.discarded_events = 0          # closures thrown away because the data was lost right after them

    # ------------------------------------------------------------------
    def update(self, t, ear):
        """Add one measurement (time in seconds, EAR or None if no face) and return the features."""
        if ear is None:
            self._interrupt()
            return self._features(t, None, None)

        if self.baseline is None:                       # not calibrated: report EAR only
            return self._features(t, ear, None)

        elapsed = 0.0 if self._last_time is None else t - self._last_time
        if elapsed < 0 or elapsed > config.MAX_SAMPLE_GAP_S:
            self._interrupt()                           # a break in the data: start a new stretch
            elapsed = 0.0
        self._last_time = t

        closure = closure_fraction(ear, self.baseline)
        if closure <= config.EYE_CLOSE_EXIT:
            self._seen_open = True
        if not self._seen_open:                         # still waiting for a first clearly open eye
            return self._features(t, ear, closure)

        self._window_samples.append((t, elapsed, closure >= config.PERCLOS_CLOSURE))

        if not self._closed and closure >= config.EYE_CLOSE_ENTER:
            self._closed = True
            self._closed_since = t
        elif self._closed and closure <= config.EYE_CLOSE_EXIT:
            self._finish_closure(end_time=t)
        self._confirm_pending(t)

        return self._features(t, ear, closure)

    # ------------------------------------------------------------------
    def _interrupt(self):
        """The eyes cannot be seen (or the data has a gap): forget any closure in progress."""
        self._closed = False
        self._closed_since = None
        self._last_time = None
        self._seen_open = False
        self.discarded_events += len(self._pending)     # data was lost right after these: do not trust them
        self._pending.clear()

    def _finish_closure(self, end_time):
        duration = end_time - self._closed_since
        self._closed = False
        self._closed_since = None
        if duration >= config.BLINK_MIN_S:
            self._pending.append((end_time, duration))   # counted only once it is confirmed

    def _confirm_pending(self, now):
        while self._pending and now - self._pending[0][0] >= config.BLINK_CONFIRM_S:
            end_time, duration = self._pending.popleft()
            if duration <= config.BLINK_MAX_S:
                self.blink_count += 1
                self._blinks.append((end_time, duration))
            else:
                self.long_closure_count += 1
                self._long_closures.append((end_time, duration))

    def _drop_old_data(self, now):
        oldest = now - config.FEATURE_WINDOW_S
        while self._window_samples and self._window_samples[0][0] < oldest:
            self._window_samples.popleft()
        while self._blinks and self._blinks[0][0] < oldest:
            self._blinks.popleft()
        while self._long_closures and self._long_closures[0][0] < oldest:
            self._long_closures.popleft()

    def _features(self, t, ear, closure):
        features = {
            "ear": None if ear is None else round(ear, 4),
            "calibrated": self.baseline is not None,
            "closure": None, "eyes_closed": None,
            "current_closure_s": None, "long_closure": None,
            "blink_count": None, "long_closure_count": None, "discarded_events": None,
            "window_valid_s": None,
            "blinks_per_min": None, "mean_blink_duration_s": None,
            "perclos": None, "longest_closure_s": None,
        }
        if self.baseline is None:
            return features

        self._drop_old_data(t)
        valid_seconds = sum(sample[1] for sample in self._window_samples)
        current = (t - self._closed_since) if (self._closed and self._closed_since is not None) else 0.0

        features.update(
            closure=None if closure is None else round(closure, 3),
            eyes_closed=self._closed,
            current_closure_s=round(current, 2),
            long_closure=self._closed and current > config.BLINK_MAX_S,
            blink_count=self.blink_count,
            long_closure_count=self.long_closure_count,
            discarded_events=self.discarded_events,
            window_valid_s=round(valid_seconds, 1),
        )

        if valid_seconds >= config.FEATURE_MIN_COVERAGE_S:
            closed_seconds = sum(s[1] for s in self._window_samples if s[2])
            durations = ([d for _, d in self._blinks] + [d for _, d in self._long_closures]
                         + [d for _, d in self._pending] + [current])
            features.update(
                blinks_per_min=round(len(self._blinks) * 60.0 / valid_seconds, 1),
                mean_blink_duration_s=(
                    round(sum(d for _, d in self._blinks) / len(self._blinks), 2) if self._blinks else None
                ),
                perclos=round(closed_seconds / valid_seconds, 3),
                longest_closure_s=round(max(durations), 2),
            )
        return features
