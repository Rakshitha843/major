"""
Decides, frame by frame, whether the eyes can be measured reliably, and if so measures them.

WHY THIS EXISTS
    A real test showed that covering the lens with a hand made the blink counter go up. While a hand
    approaches or leaves, the picture changes quickly and the face model can return a distorted face whose eye
    values flap open and closed. Those flaps look like blinks. So eye values are only trusted when the picture is
    steady. This used to be decided in three separate places (dashboard monitor, eye test, data recorder); it
    now lives here, so all of them behave the same way.

A FRAME IS NOT USED (eyes "not measured") when:
    no_face          no face was found
    too_dark         the picture is almost black (for example a palm on the lens)
    unstable_image   the average brightness changed by more than 50% within the last half second
    face_jump        the face box suddenly grew, shrank or moved far between two frames
    settling         the picture has been steady for less than EYE_SETTLE_S (1.0 s) since the last problem,
                     so the first frames after a hand leaves are ignored too
All limits are in utils/config.py. They are our own starting values, chosen from one observed failure, and
should be revisited with more recordings.

The tracker adds a second protection: a blink is only counted if the eyes are still measurable BLINK_CONFIRM_S (1.0 s)
after it (see eye_tracker.py). A blink-like flap just before the picture goes dark is therefore discarded.
"""

from collections import deque
from dataclasses import dataclass

from computer_vision import lighting
from computer_vision.eye_features import eye_aspect_ratios
from utils import config

REASON_TEXT = {
    "": "",
    "no_face": "no face",
    "too_dark": "image too dark",
    "unstable_image": "image changing too fast",
    "face_jump": "face position jumped",
    "settling": "settling after a break",
    "eyes_unreadable": "eyes could not be read",
}


@dataclass
class EyeMeasurement:
    values: "tuple | None"      # (left EAR, right EAR, mean EAR) or None when the eyes are not measured
    reason: str                 # "" when measured, otherwise one of the reasons above
    brightness: float           # average brightness of this frame, 0-255


class EyeMeasurer:
    def __init__(self):
        self._brightness_history = deque()        # (time, brightness) for the last EYE_UNSTABLE_WINDOW_S
        self._last_box = None                     # face box in the previous frame
        self._steady_since = None                 # since when every frame has been usable

    def measure(self, t, frame, face_result):
        """t: time in seconds. frame: the camera image. face_result: from FaceDetector.detect()."""
        height, width = frame.shape[:2]
        brightness = lighting.mean_brightness(frame)
        self._remember_brightness(t, brightness)

        reason = self._problem(face_result, brightness)
        if reason:
            self._steady_since = None
            return EyeMeasurement(None, reason, brightness)

        if self._steady_since is None:
            self._steady_since = t
        if t - self._steady_since < config.EYE_SETTLE_S:
            return EyeMeasurement(None, "settling", brightness)

        values = eye_aspect_ratios(face_result.landmarks, width, height)
        if values is None:
            return EyeMeasurement(None, "eyes_unreadable", brightness)
        return EyeMeasurement(values, "", brightness)

    # ------------------------------------------------------------------
    def _remember_brightness(self, t, brightness):
        self._brightness_history.append((t, brightness))
        while self._brightness_history and self._brightness_history[0][0] < t - config.EYE_UNSTABLE_WINDOW_S:
            self._brightness_history.popleft()

    def _problem(self, face_result, brightness):
        if face_result.landmarks is None:
            self._last_box = None
            return "no_face"
        jumped = self._face_jumped(face_result.bbox)
        self._last_box = face_result.bbox
        if brightness < config.EYE_ANALYSIS_MIN_BRIGHTNESS:
            return "too_dark"
        values = [b for _, b in self._brightness_history]
        if max(values) / max(min(values), 1.0) > config.EYE_BRIGHTNESS_CHANGE_RATIO:
            return "unstable_image"
        if jumped:
            return "face_jump"
        return ""

    def _face_jumped(self, box):
        if self._last_box is None:
            return False                          # first face after a break: the settle time covers it
        old_area = max(1e-6, (self._last_box[2] - self._last_box[0]) * (self._last_box[3] - self._last_box[1]))
        new_area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
        ratio = new_area / old_area
        shift = max(abs((box[0] + box[2]) / 2 - (self._last_box[0] + self._last_box[2]) / 2),
                    abs((box[1] + box[3]) / 2 - (self._last_box[1] + self._last_box[3]) / 2))
        return (ratio > config.EYE_FACE_JUMP_AREA_RATIO or ratio < 1 / config.EYE_FACE_JUMP_AREA_RATIO
                or shift > config.EYE_FACE_JUMP_SHIFT)
