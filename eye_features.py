"""
Eye Aspect Ratio (EAR) from facial landmarks.

FROM THE BASE PAPER (Sun et al. 2020, equation 2)
    EAR = ( |P38 - P42| + |P39 - P41| ) / ( 2 * |P37 - P40| )
    i.e. two vertical eyelid distances divided by twice the eye width. The ratio is large
    when the eye is open and drops towards zero when it closes. The paper then decides
    open/closed with a threshold on this ratio.

WHAT WE CHANGED
    The paper numbers dlib's 68 landmarks (P37..P42 are one eye). MediaPipe numbers them
    differently, so we use the equivalent six points below. Naming (as in the paper):
        p1 = outer corner   p2, p3 = upper eyelid   p4 = inner corner   p5, p6 = lower eyelid
    and the vertical pairs are (p2, p6) and (p3, p5).
    We also average the left and right eye. This mapping is our own; tests check that each
    upper point sits opposite its lower partner, but nothing here is from the paper's code.

IMPORTANT DETAIL
    MediaPipe coordinates are fractions of the image width and height. A camera image is wider
    than it is tall, so distances must be measured in PIXELS. Measuring in fractions would
    stretch the vertical direction and give a wrong EAR.

KNOWN LIMITS
    EAR also falls when the user looks down (for example at the keyboard) and it is affected by
    head turning and by thick glasses frames or glare. Calibration (Phase 5) measures the user's
    own open and closed values, which reduces but does not remove these effects.
"""

import numpy as np

# MediaPipe face-mesh numbers, in the order p1, p2, p3, p4, p5, p6 (see above).
# "Right" and "left" are the SUBJECT's right and left eye.
RIGHT_EYE_EAR_POINTS = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR_POINTS = (362, 385, 387, 263, 373, 380)


def _ear_of_one_eye(points):
    """`points` is a (6, 2) array of pixel positions p1..p6. Returns the EAR or None."""
    p1, p2, p3, p4, p5, p6 = points
    eye_width = np.linalg.norm(p1 - p4)
    if eye_width < 1e-6:
        return None
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    return float(vertical / (2.0 * eye_width))


def eye_aspect_ratios(landmarks, image_width, image_height):
    """
    landmarks: (478, 3) array of normalised coordinates for the monitored face.
    Returns (left_ear, right_ear, mean_ear), or None if an eye could not be measured.
    """
    scale = np.array([image_width, image_height], dtype=np.float64)
    right = _ear_of_one_eye(landmarks[list(RIGHT_EYE_EAR_POINTS), :2].astype(np.float64) * scale)
    left = _ear_of_one_eye(landmarks[list(LEFT_EYE_EAR_POINTS), :2].astype(np.float64) * scale)
    if right is None or left is None:
        return None
    return left, right, (left + right) / 2.0
