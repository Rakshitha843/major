"""
Personal calibration (synopsis objective 2): measure THIS user's normal eye values.

WHY: people's eyes differ (size, glasses, camera angle). A fixed EAR threshold would suit some
users and fail for others. The base paper notes individual differences make fatigue detection
hard; measuring each user's own open and closed EAR is our (synopsis-driven) answer.

STEPS (about 15 seconds, all times in utils/config.py)
    1. ready   3 s  "Sit normally and look at the screen. Keep your face visible.
                     Maintain your normal sitting posture."   (nothing recorded)
    2. open    8 s  keep looking at the screen and blink normally  -> measures OPEN eyes
    3. close   4 s  close your eyes (the dashboard beeps to start and to finish) -> CLOSED eyes
    Result: {"ear_open": ..., "ear_closed": ...}

HOW THE NUMBERS ARE CHOSEN (our design)
    ear_open   = 60th percentile of the EAR values during step 2. Blinks are short and low, so
                 this ignores them and lands on the normal open-eye level.
    ear_closed = average of the lowest quarter of the EAR values during step 3, so the user
                 only has to keep their eyes closed for about one second of the four.
The calibration is REJECTED (with a reason) if the face was not visible often enough, the open-eye
value is unusual, or the eyes did not appear to close. We never store a doubtful baseline.

POSTURE (Phase 7): during the 8-second "open" step the head position is measured too, and its median becomes the
personal POSTURE baseline (head tilt, nod, distance and position; shoulders if they are clearly visible). This is why the
first instruction says "Maintain your normal sitting posture": whatever you do here counts as your normal. If the head
could not be measured often enough, the eye calibration is still kept and the result says the posture baseline is missing.
"""

import math

import numpy as np

from computer_vision.posture import build_posture_baseline
from utils import config

PHASE_READY = "ready"
PHASE_OPEN = "open"
PHASE_CLOSE = "close"
PHASE_DONE = "done"
PHASE_FAILED = "failed"

MESSAGES = {
    PHASE_READY: "Sit normally and look at the screen. Keep your face visible. "
                 "Maintain your normal sitting posture.",
    PHASE_OPEN: "Keep looking at the screen, blink normally and stay in your normal sitting position.",
    PHASE_CLOSE: "Now close your eyes gently. Open them when you hear the second beep.",
    PHASE_DONE: "Calibration complete. You can open your eyes.",
}


class Calibrator:
    def __init__(self, ready_s=None, open_s=None, close_s=None):
        self.ready_s = config.CAL_READY_S if ready_s is None else ready_s
        self.open_s = config.CAL_OPEN_S if open_s is None else open_s
        self.close_s = config.CAL_CLOSE_S if close_s is None else close_s
        self.total_s = self.ready_s + self.open_s + self.close_s
        self._start_time = None
        self._open_values, self._open_missed = [], 0
        self._close_values, self._close_missed = [], 0
        self._posture_samples = []          # head-position measurements taken during the "open" step
        self._phase = PHASE_READY
        self._elapsed = 0.0
        self.result = None      # set when finished: {"ok": True, "baseline": {...}} or {"ok": False, "reason": ...}

    @property
    def finished(self):
        return self._phase in (PHASE_DONE, PHASE_FAILED)

    def update(self, t, ear, posture=None):
        """Feed one measurement: time in seconds, EAR (None if the eyes were not measured) and, optionally, the
        posture metrics of that frame (from computer_vision.posture.measure_posture, None if not measured)."""
        if self.finished:
            return self.status()
        if self._start_time is None:
            self._start_time = t
        self._elapsed = max(0.0, t - self._start_time)

        if self._elapsed < self.ready_s:
            self._phase = PHASE_READY
        elif self._elapsed < self.ready_s + self.open_s:
            self._phase = PHASE_OPEN
            self._add(self._open_values, ear, is_open_step=True)
            if posture is not None:
                self._posture_samples.append(posture)
        elif self._elapsed < self.total_s:
            self._phase = PHASE_CLOSE
            self._add(self._close_values, ear, is_open_step=False)
        else:
            self._finish()
        return self.status()

    def cancel(self):
        if not self.finished:
            self._phase = PHASE_FAILED
            self.result = {"ok": False, "reason": "Calibration was cancelled."}

    def status(self):
        remaining_in_phase = {
            PHASE_READY: self.ready_s - self._elapsed,
            PHASE_OPEN: self.ready_s + self.open_s - self._elapsed,
            PHASE_CLOSE: self.total_s - self._elapsed,
        }.get(self._phase, 0.0)
        message = self.result["reason"] if self._phase == PHASE_FAILED else MESSAGES[self._phase]
        return {
            "active": not self.finished,
            "phase": self._phase,
            "message": message,
            "seconds_left": max(0, int(math.ceil(remaining_in_phase))),
            "progress": round(min(1.0, self._elapsed / self.total_s), 2),
        }

    # ------------------------------------------------------------------
    def _add(self, values, ear, is_open_step):
        if ear is None:
            if is_open_step:
                self._open_missed += 1
            else:
                self._close_missed += 1
        else:
            values.append(ear)

    def _fail(self, reason):
        self._phase = PHASE_FAILED
        self.result = {"ok": False, "reason": reason}

    def _finish(self):
        for values, missed in ((self._open_values, self._open_missed),
                               (self._close_values, self._close_missed)):
            seen = len(values)
            total = seen + missed
            if seen < config.CAL_MIN_SAMPLES or seen < config.CAL_MIN_VALID_FRACTION * total:
                return self._fail(
                    "Your eyes could not be measured often enough. Please keep your face inside "
                    "the camera view, make sure the room is well lit, and try again."
                )

        ear_open = float(np.percentile(self._open_values, 60))
        ordered = np.sort(self._close_values)
        ear_closed = float(ordered[: max(1, len(ordered) // 4)].mean())

        low, high = config.CAL_OPEN_EAR_RANGE
        if not low <= ear_open <= high:
            return self._fail(
                f"Your open-eye measurement ({ear_open:.2f}) looks unusual. Please face the camera "
                "directly, reduce glare on your glasses, and try again."
            )
        if ear_open - ear_closed < config.CAL_MIN_SEPARATION * ear_open:
            return self._fail(
                "We could not see your eyes close. Please close your eyes fully during step 2 "
                "(after the first beep) and check that the lighting is good, then try again."
            )

        self._phase = PHASE_DONE
        seen = len(self._open_values) + len(self._close_values)
        posture_baseline, posture_problem = build_posture_baseline(self._posture_samples)
        self.result = {
            "ok": True,
            "posture_baseline": posture_baseline,          # None if the head could not be measured often enough
            "posture_problem": posture_problem,
            "baseline": {
                "ear_open": round(ear_open, 4),
                "ear_closed": round(ear_closed, 4),
                "open_samples": len(self._open_values),
                "close_samples": len(self._close_values),
                "sample_rate_hz": round(seen / (self.open_s + self.close_s), 1),
            },
        }
