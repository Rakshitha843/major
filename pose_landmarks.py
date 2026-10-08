"""
Shoulder detection (Phase 7) with MediaPipe's Pose Landmarker.

WHY OPTIONAL
    Sitting at a webcam, the shoulders are often only partly in the picture. Posture therefore works from the HEAD alone
    (head tilt, nodding, distance, position) and uses the shoulders as extra evidence only when they are clearly visible.
    If the pose model is missing or cannot be loaded, posture still works, judged from the head only.

WHAT IS RETURNED
    Just the two shoulder points (normalised picture coordinates), ordered left-of-picture then right-of-picture, so that
    nothing depends on which shoulder is "left" or "right". Shoulders are only reported when both points are confident,
    inside the picture and reasonably far apart (utils/config.py: SHOULDER_MIN_VISIBILITY, SHOULDER_MIN_WIDTH).

The pose model is heavier than the face model, so the monitor runs it only on every 4th frame (POSE_EVERY_N_FRAMES).
The pose model file was downloaded in Phase 1; whether it finds YOUR shoulders is checked with scripts/posture_test.py.
"""

from dataclasses import dataclass

from computer_vision.face_landmarks import ModelError
from utils import config

LEFT_SHOULDER_INDEX = 11      # MediaPipe pose landmark numbers
RIGHT_SHOULDER_INDEX = 12


@dataclass
class PoseResult:
    visible: bool = False                 # were both shoulders found and trustworthy?
    left_point: "tuple | None" = None     # (x, y) of the shoulder on the LEFT of the picture
    right_point: "tuple | None" = None    # (x, y) of the shoulder on the RIGHT of the picture


def extract_shoulders(pose_landmarks):
    """
    pose_landmarks: the 33 landmarks of one person (objects with x, y and visibility).
    Returns a PoseResult; visible=False when the shoulders cannot be trusted.
    """
    try:
        a, b = pose_landmarks[LEFT_SHOULDER_INDEX], pose_landmarks[RIGHT_SHOULDER_INDEX]
    except (IndexError, TypeError):
        return PoseResult()
    for point in (a, b):
        visibility = getattr(point, "visibility", None)
        if visibility is None or visibility < config.SHOULDER_MIN_VISIBILITY:
            return PoseResult()
        if not (0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0):
            return PoseResult()                                   # a guessed point outside the picture
    left, right = sorted(((a.x, a.y), (b.x, b.y)), key=lambda p: p[0])
    if right[0] - left[0] < config.SHOULDER_MIN_WIDTH:
        return PoseResult()
    return PoseResult(visible=True, left_point=left, right_point=right)


class PoseDetector:
    def __init__(self, model_path=None):
        self.model_path = config.POSE_LANDMARKER_FILE if model_path is None else model_path
        self._landmarker = None
        self._mp = None
        self._last_timestamp_ms = -1

    def load(self):
        """Load the model. Raises ModelError with a fix-it message if that is not possible."""
        if not self.model_path.is_file():
            raise ModelError("Shoulder model not found. Run this once (needs internet): python scripts/download_models.py")
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError:
            raise ModelError("MediaPipe is not installed. Run: pip install mediapipe")
        try:
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=self.model_path.read_bytes()),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self._mp = mp
        except Exception as error:
            raise ModelError(
                f"The shoulder model could not be loaded ({error}). Delete models/pose_landmarker_lite.task "
                "and run: python scripts/download_models.py"
            )

    def detect(self, frame_bgr, timestamp_ms):
        """Find the shoulders in one camera frame. Returns a PoseResult (visible=False if not trustworthy)."""
        if self._landmarker is None:
            raise ModelError("The shoulder model has not been loaded.")
        import cv2

        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)      # must strictly increase
        self._last_timestamp_ms = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        raw = self._landmarker.detect_for_video(image, timestamp_ms)
        if not raw.pose_landmarks:
            return PoseResult()
        return extract_shoulders(raw.pose_landmarks[0])

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
