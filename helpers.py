"""Shared test tools: a temporary database and a fake camera."""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import math

from computer_vision.camera import CameraError
from computer_vision.head_pose import FACE_MODEL_POINTS, POSE_LANDMARKS
from computer_vision.pose_landmarks import PoseResult
from computer_vision.face_landmarks import ModelError, analyse_faces
from computer_vision.monitor import CameraMonitor
from ml.fatigue import FatigueAssessor
from database import connection as db
from utils import config


class TempDatabaseTestCase(unittest.TestCase):
    """Base class: every test gets a fresh, empty database in a temporary folder."""

    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self._original_path = config.DATABASE_PATH
        config.DATABASE_PATH = Path(self._folder.name) / "test.db"
        db.init_db()
        self.connection = db.connect()

    def tearDown(self):
        self.connection.close()
        config.DATABASE_PATH = self._original_path
        self._folder.cleanup()


class FakeCamera:
    """Pretends to be a webcam so tests need no hardware."""

    def __init__(self, fail_open=False, fail_after_reads=None, raise_on_read=False, gray_level=120):
        self.fail_open = fail_open
        self.fail_after_reads = fail_after_reads
        self.raise_on_read = raise_on_read
        self.gray_level = gray_level
        self.reads = 0
        self.opened = False
        self.released = False

    def open(self):
        if self.fail_open:
            raise CameraError("Camera not detected.")
        self.opened = True

    def read(self):
        self.reads += 1
        if self.raise_on_read:
            raise RuntimeError("driver crashed")
        if self.fail_after_reads is not None and self.reads > self.fail_after_reads:
            return None
        return np.full((480, 640, 3), self.gray_level, dtype=np.uint8)

    def release(self):
        self.released = True


def wait_for(condition, timeout=4.0):
    """Wait until condition() is true (checks every 20 ms). Returns whether it became true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


EYE_WIDTH_PX = 40.0


def set_eye_openness(points, ear, image_width=640, image_height=480, anchored=False):
    """Move the six EAR points of both eyes so the eye aspect ratio is exactly `ear`.
    anchored=True keeps the two outer eye corners (landmarks 33 and 263) where the head pose put them."""
    from computer_vision.eye_features import LEFT_EYE_EAR_POINTS, RIGHT_EYE_EAR_POINTS
    vertical = ear * EYE_WIDTH_PX                       # EAR = vertical / width for this shape
    if anchored:
        anchors = [(points[33, 0] * image_width, points[33, 1] * image_height),
                   (points[263, 0] * image_width - EYE_WIDTH_PX, points[263, 1] * image_height)]
    else:
        centre_y = points[0, 1] * image_height + 60     # somewhere inside the face box
        anchors = [(points[0, 0] * image_width + 40, centre_y), (points[0, 0] * image_width + 120, centre_y)]
    for indices, (corner_x, centre_y) in zip((RIGHT_EYE_EAR_POINTS, LEFT_EYE_EAR_POINTS), anchors):
        p1, p2, p3, p4, p5, p6 = indices
        pixel_positions = {
            p1: (corner_x, centre_y),
            p4: (corner_x + EYE_WIDTH_PX, centre_y),
            p2: (corner_x + 13, centre_y - vertical / 2),
            p6: (corner_x + 13, centre_y + vertical / 2),
            p3: (corner_x + 27, centre_y - vertical / 2),
            p5: (corner_x + 27, centre_y + vertical / 2),
        }
        for index, (x_px, y_px) in pixel_positions.items():
            points[index, 0] = x_px / image_width
            points[index, 1] = y_px / image_height
    return points


def rotation_matrix(yaw, pitch, roll):
    """Rotation used by computer_vision/head_pose.py: R = Rz(roll) @ Ry(yaw) @ Rx(pitch), angles in degrees."""
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def set_head_pose(points, cx=0.5, cy=0.5, yaw=0.0, pitch=0.0, roll=0.0, distance=2500.0,
                  image_width=640, image_height=480):
    """Place the six head-pose landmarks as a camera with focal length = picture width would see the generic
    face model turned by (yaw, pitch, roll) degrees, `distance` model units away, centred on (cx, cy)."""
    focal = float(image_width)
    shift = np.array([(cx * image_width - image_width / 2) * distance / focal,
                      (cy * image_height - image_height / 2) * distance / focal, distance])
    camera_points = (rotation_matrix(yaw, pitch, roll) @ FACE_MODEL_POINTS.T).T + shift
    pixels = focal * camera_points[:, :2] / camera_points[:, 2:3] + np.array([image_width / 2, image_height / 2])
    for index, (x_px, y_px) in zip(POSE_LANDMARKS, pixels):
        points[index, 0], points[index, 1] = x_px / image_width, y_px / image_height
    return points


def make_face_landmarks(cx=0.5, cy=0.5, size=0.4, ear=None, head=None):
    """A fake face: 478 points scattered inside a square of the given size (normalised coordinates).
    Two corner points are pinned so the bounding box is exactly cx +/- size/2, cy +/- size/2.
    If `ear` is given, the eyes are shaped to have exactly that eye aspect ratio (640x480 image)."""
    rng = np.random.default_rng(0)
    points = np.zeros((478, 3), dtype=np.float32)
    points[:, :2] = rng.uniform(-0.5, 0.5, (478, 2)) * size + np.array([cx, cy])
    points[0, :2] = (cx - size / 2, cy - size / 2)
    points[1, :2] = (cx + size / 2, cy + size / 2)
    if head is not None:                                # head: {"roll": .., "pitch": .., "yaw": .., "distance": ..}
        set_head_pose(points, cx=cx, cy=cy, **head)
    if ear is not None:
        set_eye_openness(points, ear, anchored=head is not None)
    return points


class FakeDetector:
    """Pretends to be the face detector. `faces` is a list of (cx, cy, size) tuples, one per face.
    `ear` is the eye aspect ratio of every fake face; a test can change it while the monitor runs
    (a number, or a function that returns one) to make the eyes open and close."""

    def __init__(self, faces=((0.5, 0.5, 0.4),), fail_load=None, ear=0.30, head=None):
        self.faces = list(faces)
        self.ear = ear
        self.head = head                # {"roll":..,"pitch":..,"distance":..} or a function returning it; None = random layout
        self.fail_load = fail_load
        self.loaded = False
        self.closed = False
        self.detect_calls = 0

    def load(self):
        if self.fail_load:
            raise ModelError(self.fail_load)
        self.loaded = True

    def detect(self, frame, timestamp_ms):
        self.detect_calls += 1
        ear = self.ear() if callable(self.ear) else self.ear
        head = self.head() if callable(self.head) else self.head
        return analyse_faces([make_face_landmarks(*face, ear=ear, head=head) for face in self.faces])

    def close(self):
        self.closed = True


def shoulders_result(cx=0.5, y=0.78, width=0.55, tilt_deg=0.0, image_width=640, image_height=480):
    """A PoseResult with both shoulders visible, `tilt_deg` degrees of tilt (positive = right of the picture lower)."""
    dy = (width * image_width * math.tan(math.radians(tilt_deg))) / 2 / image_height
    return PoseResult(visible=True, left_point=(cx - width / 2, y - dy), right_point=(cx + width / 2, y + dy))


class FakePoseDetector:
    """Pretends to be the shoulder detector. `result` is a PoseResult, a function returning one, or None (no shoulders)."""

    def __init__(self, result=None, fail_load=None):
        self.result = result
        self.fail_load = fail_load
        self.loaded = False
        self.closed = False
        self.detect_calls = 0

    def load(self):
        if self.fail_load:
            raise ModelError(self.fail_load)
        self.loaded = True

    def detect(self, frame, timestamp_ms):
        self.detect_calls += 1
        result = self.result() if callable(self.result) else self.result
        return result if result is not None else PoseResult()

    def close(self):
        self.closed = True


def new_monitor(camera_factory, detector_factory=None, **options):   # options: e.g. calibrator_factory
    """A CameraMonitor for tests: uses the fake face detector unless another one is given."""
    options.setdefault("assessor_factory", lambda: FatigueAssessor(model=None))     # rules, never a local model file
    options.setdefault("pose_detector_factory", lambda: FakePoseDetector())         # never the real shoulder model
    return CameraMonitor(
        camera_factory=camera_factory,
        detector_factory=detector_factory or (lambda: FakeDetector()),
        **options,
    )
