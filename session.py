"""
Session and exposure tracking (Phase 9): how long the camera has run, how much of that was "active"
(a person actually there), and how many breaks were taken.

WHAT COUNTS AS A BREAK
    A "break" is a sustained absence: no face visible for at least SESSION_BREAK_MIN_S (30 s, our own starting
    value). A person glancing away, or the brief settle time after a hand blocks the lens, is NOT a break --
    those are already reflected in the attention indicator (Phase 8). A break only counts once it ends (the
    face returns) or the session ends, so its final duration is known.

    "Active duration" = session duration minus time spent on breaks. It is not the same as "time attentive"
    (Phase 8's score): a person can be present but distracted, which is active time, just not attentive time.

CONTINUOUS ACTIVE TIME (added in Phase 10, for the exposure input to the DHRI)
    How long the person has been active WITHOUT a qualifying break -- the clock resets to zero only when a
    break is CONFIRMED (see above), not on every brief glance away, so a run of short glances does not reset
    it repeatedly. While an absence is in progress but not yet confirmed as a break, the clock holds steady
    at its value from when the absence began, since it is not yet known whether this will turn out to be a
    real break or just a moment away.

WHAT THIS DOES NOT DO
    It does not decide whether a break was "deserved" or healthy -- that judgement, and any break reminders,
    are made by the recommendations module (ml/alerts.py), using the numbers this module produces. This
    module only counts and times what happened.
"""

from dataclasses import dataclass

from utils import config


@dataclass
class SessionSummary:
    duration_s: float
    active_duration_s: float
    break_count: int
    break_duration_s: float
    continuous_active_s: float


class SessionTracker:
    def __init__(self, break_min_s=None):
        self.break_min_s = config.SESSION_BREAK_MIN_S if break_min_s is None else break_min_s
        self.break_count = 0
        self.break_duration_s = 0.0
        self._away_since = None
        self._active_streak_start = 0.0     # resets to t whenever a break is CONFIRMED

    def update(self, t, present):
        """t: seconds since the session started. present: False when nobody is visible in this frame."""
        if present:
            self._close_break_if_long_enough(t)
        elif self._away_since is None:
            self._away_since = t

    def finalize(self, t):
        """Call once, when the session ends, so a break still in progress is still counted."""
        self._close_break_if_long_enough(t)

    def summary(self, t):
        """A snapshot as of time t (seconds since the session started). Safe to call mid-session or after finalize()."""
        duration = max(0.0, t)
        # An in-progress absence is not yet a counted break, but it should still reduce "active" time as it happens.
        open_break = 0.0 if self._away_since is None else max(0.0, t - self._away_since)
        active = max(0.0, duration - self.break_duration_s - open_break)
        reference_t = self._away_since if self._away_since is not None else t   # freeze the clock while away
        continuous_active = max(0.0, reference_t - self._active_streak_start)
        return SessionSummary(
            duration_s=round(duration, 1),
            active_duration_s=round(active, 1),
            break_count=self.break_count,
            break_duration_s=round(self.break_duration_s, 1),
            continuous_active_s=round(continuous_active, 1),
        )

    def _close_break_if_long_enough(self, t):
        if self._away_since is None:
            return
        duration = t - self._away_since
        if duration >= self.break_min_s:
            self.break_count += 1
            self.break_duration_s += duration
            self._active_streak_start = t         # a real break resets the continuous-active clock
        self._away_since = None
