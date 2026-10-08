"""
Head pose from the face landmarks (Phase 7).

WHAT WE MEASURE
    roll    sideways head tilt, measured DIRECTLY from the line between the two outer eye corners.
            Positive = the eye on the right of the camera picture is lower (clockwise in the picture).
    pitch   head nodding up/down, from a 3D pose fit of six face points. Positive = head tilted DOWN (chin towards the chest).
    yaw     head turning left/right (also from the fit). Positive = face turned towards the left of the camera picture.
            Not used for posture; attention monitoring (Phase 8) will use it.

HOW pitch AND yaw ARE FOUND
    OpenCV's solvePnP finds the rotation of a generic 3D face model (six points: nose tip, chin, both outer eye corners,
    both mouth corners) that best explains where those six landmarks appear in the picture. The model is a rough average
    face, not YOUR face, and the camera's lens is unknown (we assume a focal length equal to the picture width). So the
    absolute angles can be off by several degrees. That is fine here: posture is always compared with the user's OWN
    calibrated baseline, so a constant error cancels out. The tests check the maths on synthetic faces built from the same
    model (exact recovery); they cannot say how accurate the angles are on real faces.
"""

import math

import cv2
import numpy as np

# MediaPipe face-mesh landmark numbers: nose tip, chin, outer corner of the SUBJECT's right eye (left of the picture),
# outer corner of the subject's left eye (right of the picture), mouth corners (subject's right, then left).
POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
EYE_CORNER_RIGHT, EYE_CORNER_LEFT = 33, 263            # left and right of the PICTURE respectively

# Generic face model in arbitrary units (x right, y down, z away from the camera; the nose tip is the origin).
FACE_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, 330.0, 65.0),       # chin
    (-225.0, -170.0, 135.0),  # outer corner, left of the picture
    (225.0, -170.0, 135.0),   # outer corner, right of the picture
    (-150.0, 150.0, 125.0),   # mouth corner, left of the picture
    (150.0, 150.0, 125.0),    # mouth corner, right of the picture
], dtype=np.float64)


def eye_line_roll(landmarks, image_width, image_height):
    """Sideways head tilt in degrees from the eye line (0 = level). None if the eye corners coincide."""
    scale = np.array([image_width, image_height], dtype=np.float64)
    left_of_picture = landmarks[EYE_CORNER_RIGHT, :2].astype(np.float64) * scale
    right_of_picture = landmarks[EYE_CORNER_LEFT, :2].astype(np.float64) * scale
    dx, dy = right_of_picture - left_of_picture
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx))


def eye_distance(landmarks, image_width, image_height):
    """Distance between the outer eye corners as a fraction of the picture width (bigger = closer to the camera)."""
    scale = np.array([image_width, image_height], dtype=np.float64)
    a = landmarks[EYE_CORNER_RIGHT, :2].astype(np.float64) * scale
    b = landmarks[EYE_CORNER_LEFT, :2].astype(np.float64) * scale
    return float(np.linalg.norm(a - b) / image_width)


def estimate_head_pose(landmarks, image_width, image_height):
    """
    Returns (yaw, pitch, roll) in degrees from the 3D fit, or None if the fit fails.
    Use eye_line_roll() for the sideways tilt used by posture; roll here is for completeness.
    """
    scale = np.array([image_width, image_height], dtype=np.float64)
    image_points = landmarks[list(POSE_LANDMARKS), :2].astype(np.float64) * scale
    focal = float(image_width)                                   # assumed lens: focal length = picture width
    camera = np.array([[focal, 0, image_width / 2.0], [0, focal, image_height / 2.0], [0, 0, 1.0]])
    try:
        ok, rotation_vector, translation = cv2.solvePnP(
            FACE_MODEL_POINTS, image_points, camera, None, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None
    if not ok or translation[2][0] <= 0:                          # the face must be in FRONT of the camera
        return None
    rotation, _ = cv2.Rodrigues(rotation_vector)
    horizontal = math.hypot(rotation[0, 0], rotation[1, 0])
    yaw = math.degrees(math.atan2(-rotation[2, 0], horizontal))
    pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
    roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return yaw, pitch, roll
