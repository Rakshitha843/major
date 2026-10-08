"""
Time-based smoothing of a shown state, shared by fatigue (normal/warning/fatigued) and posture (good/warning/poor).

A state is described by an ordered list of levels, from best to worst. The SHOWN state changes only when the
new state has lasted long enough:
    getting worse : `escalate_hold_s` seconds (unless the caller says the change is `immediate`)
    getting better: `recovery_hold_s` seconds
This avoids alerts from a single bad frame and stops the display flickering around a threshold.
When the readings jump around, only what held for the WHOLE waiting time counts.
"""


class StateSmoother:
    def __init__(self, levels, escalate_hold_s, recovery_hold_s):
        self.levels = list(levels)                     # best first, e.g. ["good", "warning", "poor"]
        self.escalate_hold_s = escalate_hold_s
        self.recovery_hold_s = recovery_hold_s
        self.reset()

    def reset(self):
        self.shown = self.levels[0]
        self._up_since = self._up_lowest = None
        self._down_since = self._down_highest = None

    def update(self, t, raw_state, immediate=False, hold_s=None):
        """hold_s: use this waiting time (for getting worse) instead of the default, for this call only."""
        escalate_hold = self.escalate_hold_s if hold_s is None else hold_s
        shown_level = self.levels.index(self.shown)
        raw_level = self.levels.index(raw_state)

        if raw_level > shown_level:
            self._down_since = self._down_highest = None
            if immediate:
                self.shown, self._up_since, self._up_lowest = raw_state, None, None
                return self.shown
            if self._up_since is None:
                self._up_since, self._up_lowest = t, raw_level
            self._up_lowest = min(self._up_lowest, raw_level)      # only what held the WHOLE time counts
            if t - self._up_since >= escalate_hold:
                self.shown = self.levels[self._up_lowest]
                self._up_since = self._up_lowest = None
        elif raw_level < shown_level:
            self._up_since = self._up_lowest = None
            if self._down_since is None:
                self._down_since, self._down_highest = t, raw_level
            self._down_highest = max(self._down_highest, raw_level)
            if t - self._down_since >= self.recovery_hold_s:
                self.shown = self.levels[self._down_highest]
                self._down_since = self._down_highest = None
        else:
            self._up_since = self._up_lowest = None
            self._down_since = self._down_highest = None
        return self.shown
