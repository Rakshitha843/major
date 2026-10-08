"""
Phase 4 tests: face detection logic, overlay drawing, model loading and the monitor integration.

The face LOGIC is tested with synthetic landmarks and a fake detector, so these tests need
no webcam and no model file. Whether the real MediaPipe model finds YOUR face is checked
by scripts/face_test.py and by you in the browser (not by these tests).

Run from the project root:
    python -m unittest discover -s tests -t . -v
"""

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from computer_vision import drawing
from computer_vision import face_landmarks as fl
from computer_vision import monitor as monitor_module
from tests.helpers import (
    FakeCamera,
    FakeDetector,
    TempDatabaseTestCase,
    make_face_landmarks,
    new_monitor,
    wait_for,
)
from utils import config, env_check

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)


def edges_of_loop(indices):
    """The set of (unordered) point pairs joined by a closed loop through `indices`."""
    return {frozenset((indices[i], indices[(i + 1) % len(indices)])) for i in range(len(indices))}


class EyeLandmarkTableTests(unittest.TestCase):
    def test_outlines_are_16_distinct_valid_points(self):
        for outline in (fl.RIGHT_EYE_OUTLINE, fl.LEFT_EYE_OUTLINE):
            self.assertEqual(len(outline), 16)
            self.assertEqual(len(set(outline)), 16)
            self.assertTrue(all(0 <= index < 478 for index in outline))

    def test_outlines_match_mediapipes_own_eye_tables(self):
        try:
            from mediapipe.tasks.python import vision
            tables = vision.FaceLandmarksConnections
            right = {frozenset((c.start, c.end)) for c in tables.FACE_LANDMARKS_RIGHT_EYE}
            left = {frozenset((c.start, c.end)) for c in tables.FACE_LANDMARKS_LEFT_EYE}
        except (ImportError, AttributeError):
            self.skipTest("this MediaPipe version has no FaceLandmarksConnections")
        self.assertEqual(edges_of_loop(fl.RIGHT_EYE_OUTLINE), right)
        self.assertEqual(edges_of_loop(fl.LEFT_EYE_OUTLINE), left)


class AnalyseFacesTests(unittest.TestCase):
    def analyse(self, *faces):
        return fl.analyse_faces([make_face_landmarks(*face) for face in faces])

    def test_no_faces(self):
        result = fl.analyse_faces([])
        self.assertEqual(result.status, fl.FACE_NONE)
        self.assertEqual(result.faces_found, 0)
        self.assertIsNone(result.landmarks)

    def test_one_centred_face_is_ok(self):
        result = self.analyse((0.5, 0.5, 0.4))
        self.assertEqual(result.status, fl.FACE_OK)
        self.assertEqual(result.faces_found, 1)
        for got, expected in zip(result.bbox, (0.3, 0.3, 0.7, 0.7)):
            self.assertAlmostEqual(got, expected, places=5)

    def test_two_similar_faces_warn_and_the_larger_is_monitored(self):
        result = self.analyse((0.3, 0.5, 0.30), (0.7, 0.5, 0.36))
        self.assertEqual(result.status, fl.FACE_MULTIPLE)
        self.assertEqual(result.faces_found, 2)
        self.assertAlmostEqual(result.bbox[0], 0.7 - 0.18, places=5)        # the 0.36 face
        np.testing.assert_allclose(result.landmarks, make_face_landmarks(0.7, 0.5, 0.36))
        self.assertEqual(len(result.other_bboxes), 1)

    def test_tiny_background_face_is_ignored(self):
        result = self.analyse((0.5, 0.5, 0.4), (0.9, 0.2, 0.08))
        self.assertEqual(result.status, fl.FACE_OK)
        self.assertEqual(result.faces_found, 1)
        self.assertEqual(result.other_bboxes, [])

    def test_face_touching_the_edge_is_partial(self):
        for centre in ((0.9, 0.5), (0.1, 0.5), (0.5, 0.1), (0.5, 0.95)):
            self.assertEqual(self.analyse((*centre, 0.4)).status, fl.FACE_PARTIAL, centre)

    def test_multiple_faces_take_priority_over_partial(self):
        result = self.analyse((0.9, 0.5, 0.4), (0.4, 0.5, 0.35))
        self.assertEqual(result.status, fl.FACE_MULTIPLE)

    def test_every_status_has_a_message(self):
        for status in (fl.FACE_OK, fl.FACE_NONE, fl.FACE_MULTIPLE, fl.FACE_PARTIAL):
            self.assertIn(status, fl.FACE_MESSAGES)
        self.assertEqual(fl.FACE_MESSAGES[fl.FACE_NONE], "Please position your face inside the camera view.")
        self.assertEqual(fl.FACE_MESSAGES[fl.FACE_OK], "")


class StatusDebouncerTests(unittest.TestCase):
    def feed(self, debouncer, status, times):
        shown = None
        for _ in range(times):
            shown = debouncer.update(status)
        return shown

    def test_starts_as_no_face(self):
        self.assertEqual(fl.StatusDebouncer().shown, fl.FACE_NONE)

    def test_needs_consecutive_frames_to_change(self):
        debouncer = fl.StatusDebouncer(frames_needed=6)
        self.assertEqual(self.feed(debouncer, fl.FACE_OK, 5), fl.FACE_NONE)   # 5 is not enough
        self.assertEqual(debouncer.update(fl.FACE_OK), fl.FACE_OK)            # the 6th does it

    def test_a_single_missed_frame_does_not_change_the_status(self):
        debouncer = fl.StatusDebouncer(frames_needed=6)
        self.feed(debouncer, fl.FACE_OK, 6)
        self.assertEqual(debouncer.update(fl.FACE_NONE), fl.FACE_OK)
        self.assertEqual(debouncer.update(fl.FACE_OK), fl.FACE_OK)

    def test_flickering_never_changes_the_status(self):
        debouncer = fl.StatusDebouncer(frames_needed=3)
        self.feed(debouncer, fl.FACE_OK, 3)
        for _ in range(20):
            self.assertEqual(debouncer.update(fl.FACE_NONE), fl.FACE_OK)
            self.assertEqual(debouncer.update(fl.FACE_OK), fl.FACE_OK)

    def test_lost_face_is_reported_after_enough_frames(self):
        debouncer = fl.StatusDebouncer(frames_needed=6)
        self.feed(debouncer, fl.FACE_OK, 6)
        self.assertEqual(self.feed(debouncer, fl.FACE_NONE, 5), fl.FACE_OK)
        self.assertEqual(debouncer.update(fl.FACE_NONE), fl.FACE_NONE)

    def test_changing_between_two_new_statuses_restarts_the_count(self):
        debouncer = fl.StatusDebouncer(frames_needed=4)
        self.feed(debouncer, fl.FACE_OK, 4)
        self.feed(debouncer, fl.FACE_NONE, 3)
        self.feed(debouncer, fl.FACE_MULTIPLE, 3)          # switching candidate resets the count
        self.assertEqual(debouncer.shown, fl.FACE_OK)
        self.assertEqual(debouncer.update(fl.FACE_MULTIPLE), fl.FACE_MULTIPLE)


class DrawingTests(unittest.TestCase):
    def black_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_face_box_and_eyes_are_drawn(self):
        frame = self.black_frame()
        drawing.draw_face_overlay(frame, fl.analyse_faces([make_face_landmarks()]))
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertTrue(np.any(np.all(frame == drawing.GREEN, axis=2)), "green face box missing")
        self.assertTrue(np.any(np.all(frame == drawing.YELLOW, axis=2)), "eye outlines missing")

    def test_extra_faces_and_warnings_are_orange(self):
        frame = self.black_frame()
        result = fl.analyse_faces([make_face_landmarks(0.3, 0.5, 0.3), make_face_landmarks(0.7, 0.5, 0.36)])
        drawing.draw_face_overlay(frame, result)
        self.assertTrue(np.any(np.all(frame == drawing.ORANGE, axis=2)))
        self.assertFalse(np.any(np.all(frame == drawing.GREEN, axis=2)))    # status is "multiple"

    def test_nothing_drawn_when_there_is_no_face(self):
        frame = self.black_frame()
        drawing.draw_face_overlay(frame, fl.analyse_faces([]))
        drawing.draw_face_overlay(frame, None)
        self.assertEqual(int(frame.sum()), 0)

    def test_face_partly_outside_the_image_does_not_crash(self):
        frame = self.black_frame()
        drawing.draw_face_overlay(frame, fl.analyse_faces([make_face_landmarks(1.0, 1.0, 0.6)]))
        self.assertGreater(int(frame.sum()), 0)

    def test_no_text_is_drawn_because_the_preview_is_mirrored(self):
        source = Path(drawing.__file__).read_text(encoding="utf-8")
        self.assertNotIn("putText", source)


class FaceDetectorLoadingTests(unittest.TestCase):
    def test_missing_model_gives_a_fix_it_message(self):
        with tempfile.TemporaryDirectory() as folder:
            detector = fl.FaceDetector(model_path=Path(folder) / "nothing.task")
            with self.assertRaises(fl.ModelError) as caught:
                detector.load()
        self.assertIn("download_models.py", str(caught.exception))

    def test_damaged_model_gives_a_clear_message_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            damaged = Path(folder) / "damaged.task"
            damaged.write_bytes(b"this is not a real model" * 100)
            detector = fl.FaceDetector(model_path=damaged)
            with self.assertRaises(fl.ModelError) as caught:
                detector.load()
        self.assertIn("could not be loaded", str(caught.exception))
        self.assertIn("download_models.py", str(caught.exception))

    def test_detect_before_load_is_refused_and_close_is_safe(self):
        detector = fl.FaceDetector(model_path=Path("does_not_matter.task"))
        with self.assertRaises(fl.ModelError):
            detector.detect(np.zeros((10, 10, 3), dtype=np.uint8), 0)
        detector.close()
        detector.close()

    def test_environment_check_reports_a_damaged_model(self):
        with tempfile.TemporaryDirectory() as folder:
            damaged = Path(folder) / "damaged.task"
            damaged.write_bytes(b"junk" * 500)
            result = env_check.check_face_model_loads(damaged)
        self.assertFalse(result["ok"])
        self.assertIn("could not be loaded", result["detail"])


class MonitorFaceTests(unittest.TestCase):
    def setUp(self):
        self._monitors = []

    def tearDown(self):
        for monitor in self._monitors:
            monitor.stop()

    def run_monitor(self, faces, camera=None):
        camera = camera or FakeCamera()
        detector = FakeDetector(faces=faces)
        monitor = new_monitor(lambda: camera, detector_factory=lambda: detector)
        self._monitors.append(monitor)
        monitor.start(1)
        return monitor, camera, detector

    def status_reaches(self, monitor, face_status):
        return wait_for(lambda: monitor.get_status(1).get("face_status") == face_status)

    def test_one_face_is_reported_ok(self):
        monitor, _, _ = self.run_monitor([(0.5, 0.5, 0.4)])
        self.assertTrue(self.status_reaches(monitor, "ok"))
        status = monitor.get_status(1)
        self.assertEqual(status["face_message"], "")
        self.assertEqual(status["faces_found"], 1)
        self.assertGreaterEqual(status["processing_ms"], 0)

    def test_no_face_shows_the_required_message(self):
        monitor, _, _ = self.run_monitor([])
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("processing_ms") is not None))
        status = monitor.get_status(1)
        self.assertEqual(status["face_status"], "no_face")
        self.assertEqual(status["face_message"], "Please position your face inside the camera view.")
        self.assertEqual(status["faces_found"], 0)

    def test_multiple_faces_show_a_warning(self):
        monitor, _, _ = self.run_monitor([(0.3, 0.5, 0.4), (0.75, 0.5, 0.36)])
        self.assertTrue(self.status_reaches(monitor, "multiple_faces"))
        self.assertIn("More than one face", monitor.get_status(1)["face_message"])
        self.assertEqual(monitor.get_status(1)["faces_found"], 2)

    def test_partly_visible_face_shows_a_warning(self):
        monitor, _, _ = self.run_monitor([(0.92, 0.5, 0.4)])
        self.assertTrue(self.status_reaches(monitor, "partial_face"))
        self.assertIn("outside the camera view", monitor.get_status(1)["face_message"])

    def test_preview_image_contains_the_drawn_overlay(self):
        monitor, _, _ = self.run_monitor([(0.5, 0.5, 0.4)])
        self.assertTrue(self.status_reaches(monitor, "ok"))
        _, jpeg = monitor.get_frame(1)
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        changed = np.abs(image.astype(int) - 120).max(axis=2) > 40      # camera image is plain gray 120
        self.assertGreater(int(changed.sum()), 50)

    def test_missing_model_gives_a_message_and_the_camera_is_never_switched_on(self):
        camera = FakeCamera()
        detector = FakeDetector(fail_load="Face model not found. Run this once: python scripts/download_models.py")
        monitor = new_monitor(lambda: camera, detector_factory=lambda: detector)
        self._monitors.append(monitor)
        status = monitor.start(1)
        self.assertEqual(status["state"], "error")
        self.assertIn("download_models.py", status["message"])
        self.assertFalse(camera.opened)                  # privacy: no camera light for a setup problem

    def test_detector_and_camera_are_released_on_stop(self):
        monitor, camera, detector = self.run_monitor([(0.5, 0.5, 0.4)])
        self.assertTrue(wait_for(lambda: detector.detect_calls > 2))
        monitor.stop()
        self.assertTrue(detector.closed)
        self.assertTrue(camera.released)

    def test_detector_is_released_when_the_camera_cannot_open(self):
        detector = FakeDetector()
        monitor = new_monitor(lambda: FakeCamera(fail_open=True), detector_factory=lambda: detector)
        self._monitors.append(monitor)
        self.assertEqual(monitor.start(1)["message"], "Camera not detected.")
        self.assertTrue(detector.closed)

    def test_nothing_is_written_to_disk_by_the_face_code(self):
        for module in (fl, drawing, monitor_module):
            source = Path(module.__file__).read_text(encoding="utf-8")
            for forbidden in ("imwrite", "VideoWriter", "write_bytes", "write_text", ".write("):
                self.assertNotIn(forbidden, source, module.__name__)
            self.assertIsNone(re.search(r"(?<![\w.])open\(", source), module.__name__)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class FaceApiTests(TempDatabaseTestCase):
    def setUp(self):
        super().setUp()
        self.camera = FakeCamera()
        self._original_monitor = monitor_module.monitor

    def tearDown(self):
        monitor_module.monitor.stop()
        monitor_module.monitor = self._original_monitor
        super().tearDown()

    def client_with(self, detector):
        from fastapi.testclient import TestClient
        from backend.main import app
        monitor_module.monitor = new_monitor(lambda: self.camera, detector_factory=lambda: detector)
        client = TestClient(app)
        client.post("/api/auth/register", json={"username": "asha", "password": "password123"})
        return client

    def test_status_includes_face_information(self):
        client = self.client_with(FakeDetector())
        client.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json().get("face_status") == "ok"))
        body = client.get("/api/monitor/status").json()
        for key in ("face_status", "face_message", "faces_found", "processing_ms"):
            self.assertIn(key, body)

    def test_missing_model_returns_503_and_camera_stays_off(self):
        client = self.client_with(FakeDetector(fail_load="Face model not found. Run: python scripts/download_models.py"))
        response = client.post("/api/monitor/start")
        self.assertEqual(response.status_code, 503)
        self.assertIn("download_models.py", response.json()["detail"])
        self.assertFalse(self.camera.opened)


if __name__ == "__main__":
    unittest.main()
