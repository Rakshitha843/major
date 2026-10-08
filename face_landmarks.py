"""
Face detection and facial landmarks (Phase 4).

WHAT IT DOES
    Finds the face in each camera frame and returns 478 landmark points (x, y, z) for it.
    Later phases read the eye points (Phase 5), head/shoulder points (Phases 7-8) from this.

    Coordinates are NORMALISED: x and y run from 0 to 1 across the image (x=0 left edge,
    y=0 top edge). Multiply by the image width/height to get pixels.

WHY MEDIAPIPE (and how it relates to the base paper)
    The base paper (Sun et al. 2020) finds facial landmarks with dlib. We use MediaPipe's
    Face Landmarker instead because it installs with pip and gives more points. The paper's
    eye-region idea is kept: we still locate the eye landmarks and measure them in Phase 5.
    This change is our own design choice, not something taken from the paper.

ONE PRIMARY USER
    The system monitors ONE person: the largest face in view. If a second person of similar
    size is in view we show a warning and keep monitoring the largest face only.

The logic that decides "no face / one face / several faces" is in plain functions
(analyse_faces, StatusDebouncer) so it can be tested without a camera or a model file.
"""

from dataclasses import dataclass, field

import numpy as np

from utils import config

# ---------------------------------------------------------------------------
# Face status values and the messages shown to the user
# ---------------------------------------------------------------------------
FACE_OK = "ok"
FACE_NONE = "no_face"
FACE_MULTIPLE = "multiple_faces"
FACE_PARTIAL = "partial_face"

FACE_MESSAGES = {
    FACE_OK: "",
    FACE_NONE: "Please position your face inside the camera view.",
    FACE_MULTIPLE: "More than one face detected. Only the largest face is monitored.",
    FACE_PARTIAL: "Part of your face is outside the camera view. Please move back a little.",
}

# ---------------------------------------------------------------------------
# Landmark numbers of the two eyes (MediaPipe face mesh numbering).
# In MediaPipe, "right eye" means the SUBJECT's right eye (it appears on the LEFT of the image).
# Each list walks once around the eye outline; a test checks them against MediaPipe's own tables.
# ---------------------------------------------------------------------------
RIGHT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_OUTLINE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]


class ModelError(Exception):
    """The face model is missing or cannot be loaded. The message can be shown to the user."""


@dataclass
class FaceResult:
    status: str = FACE_NONE
    faces_found: int = 0                       # faces that count (small background faces are ignored)
    landmarks: "np.ndarray | None" = None      # main face, shape (478, 3), normalised
    bbox: "tuple | None" = None                # main face box (x0, y0, x1, y1), normalised
    other_bboxes: list = field(default_factory=list)   # boxes of the additional counted faces


# ---------------------------------------------------------------------------
# Plain logic (no camera and no model needed)
# ---------------------------------------------------------------------------
def bounding_box(landmarks):
    """Smallest box around all landmarks: (x0, y0, x1, y1)."""
    x0, y0 = landmarks[:, 0].min(), landmarks[:, 1].min()
    x1, y1 = landmarks[:, 0].max(), landmarks[:, 1].max()
    return float(x0), float(y0), float(x1), float(y1)


def _box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _is_partly_outside(box):
    margin = config.FACE_EDGE_MARGIN
    return box[0] < margin or box[1] < margin or box[2] > 1 - margin or box[3] > 1 - margin


def analyse_faces(landmark_arrays):
    """
    Turn the raw list of detected faces into one FaceResult.

    Rules:
      * no faces                                   -> "no_face"
      * the LARGEST face is the person we monitor
      * other faces count only if at least SECONDARY_FACE_MIN_AREA_RATIO of its size
      * more than one counted face                 -> "multiple_faces"
      * else, main face touching the image edge    -> "partial_face"
      * else                                       -> "ok"
    """
    if not landmark_arrays:
        return FaceResult()

    boxes = [bounding_box(face) for face in landmark_arrays]
    main_index = max(range(len(boxes)), key=lambda i: _box_area(boxes[i]))
    main_box = boxes[main_index]
    main_area = _box_area(main_box)

    other_boxes = [
        box for i, box in enumerate(boxes)
        if i != main_index and _box_area(box) >= config.SECONDARY_FACE_MIN_AREA_RATIO * main_area
    ]

    if other_boxes:
        status = FACE_MULTIPLE
    elif _is_partly_outside(main_box):
        status = FACE_PARTIAL
    else:
        status = FACE_OK

    return FaceResult(
        status=status,
        faces_found=1 + len(other_boxes),
        landmarks=landmark_arrays[main_index],
        bbox=main_box,
        other_bboxes=other_boxes,
    )


class StatusDebouncer:
    """
    Stops the shown status flickering (consecutive-frame logic).

    A new status is only adopted after it has been seen `frames_needed` frames in a row.
    Before any face has been confirmed the shown status is "no_face".
    """

    def __init__(self, frames_needed=None):
        self.frames_needed = config.STATUS_CHANGE_FRAMES if frames_needed is None else frames_needed
        self.shown = FACE_NONE
        self._candidate = FACE_NONE
        self._count = 0

    def update(self, raw_status):
        if raw_status == self.shown:
            self._candidate, self._count = raw_status, 0
            return self.shown
        if raw_status == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = raw_status, 1
        if self._count >= self.frames_needed:
            self.shown = raw_status
            self._count = 0
        return self.shown


# ---------------------------------------------------------------------------
# The MediaPipe detector
# ---------------------------------------------------------------------------
class FaceDetector:
    def __init__(self, model_path=None, max_faces=None):
        self.model_path = config.FACE_LANDMARKER_FILE if model_path is None else model_path
        self.max_faces = config.MAX_FACES if max_faces is None else max_faces
        self._landmarker = None
        self._mp = None
        self._last_timestamp_ms = -1

    def load(self):
        """Load the model. Raises ModelError with a fix-it message if that is not possible."""
        if not self.model_path.is_file():
            raise ModelError("Face model not found. Run this once (needs internet): python scripts/download_models.py")
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError:
            raise ModelError("MediaPipe is not installed. Run: pip install mediapipe")

        try:
            # We hand MediaPipe the file's bytes instead of its path, so unusual folder
            # names (accents, non-English letters) can never stop the model from loading.
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=self.model_path.read_bytes()),
                running_mode=vision.RunningMode.VIDEO,     # tracks the face from frame to frame
                num_faces=self.max_faces,
                min_face_detection_confidence=config.FACE_MIN_DETECTION_CONFIDENCE,
                min_face_presence_confidence=config.FACE_MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=config.FACE_MIN_TRACKING_CONFIDENCE,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
        except Exception as error:      # MediaPipe reports a damaged model in several ways
            raise ModelError(
                f"The face model could not be loaded ({error}). Delete models/face_landmarker.task "
                "and run: python scripts/download_models.py"
            )

    def detect(self, frame_bgr, timestamp_ms):
        """Find faces in one camera frame (OpenCV BGR image). Returns a FaceResult."""
        if self._landmarker is None:
            raise ModelError("The face model has not been loaded.")
        import cv2

        # MediaPipe needs strictly increasing timestamps in video mode
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)          # MediaPipe expects RGB
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        raw = self._landmarker.detect_for_video(image, timestamp_ms)

        faces = [
            np.array([(point.x, point.y, point.z) for point in face], dtype=np.float32)
            for face in raw.face_landmarks
        ]
        return analyse_faces(faces)

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
