"""
Draws the detection result on the preview image so the user can SEE what the system found.

Only lines are drawn, never text: the browser shows the preview mirrored, and any text
drawn here would read backwards. Messages are shown by the dashboard instead.

Colours are OpenCV's blue-green-red order.
"""

import cv2
import numpy as np

from computer_vision.face_landmarks import (
    FACE_OK,
    LEFT_EYE_OUTLINE,
    RIGHT_EYE_OUTLINE,
)

GREEN = (80, 200, 80)       # main face, everything fine
ORANGE = (0, 165, 255)      # main face has a warning, or an additional face
YELLOW = (0, 255, 255)      # eye outlines
RED = (0, 0, 255)           # eye outlines while the eyes are closed


def _to_pixel_box(box, width, height):
    x0, y0, x1, y1 = box
    return (int(round(x0 * width)), int(round(y0 * height)),
            int(round(x1 * width)), int(round(y1 * height)))


def draw_face_overlay(frame, result, eyes_closed=False):
    """Draw on `frame` in place. Safe to call when no face was found. Eye outlines turn red while closed."""
    if result is None or result.landmarks is None:
        return frame

    height, width = frame.shape[:2]

    for box in result.other_bboxes:
        x0, y0, x1, y1 = _to_pixel_box(box, width, height)
        cv2.rectangle(frame, (x0, y0), (x1, y1), ORANGE, 1)

    x0, y0, x1, y1 = _to_pixel_box(result.bbox, width, height)
    cv2.rectangle(frame, (x0, y0), (x1, y1), GREEN if result.status == FACE_OK else ORANGE, 2)

    scale = np.array([width, height], dtype=np.float32)
    for outline in (RIGHT_EYE_OUTLINE, LEFT_EYE_OUTLINE):
        points = np.round(result.landmarks[outline, :2] * scale).astype(np.int32)
        cv2.polylines(frame, [points], isClosed=True, color=RED if eyes_closed else YELLOW, thickness=1)

    return frame
