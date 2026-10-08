"""
Phase 3 tests: webcam wrapper, lighting check, camera monitor, and the camera API.

All camera behaviour is tested with a FAKE camera, so these tests need no webcam.
(The real webcam is checked by scripts/check_environment.py and by you, in the browser.)

Run from the project root:
    python -m unittest discover -s tests -t . -v
"""

import importlib.util
import threading
import time
import unittest

import numpy as np

from computer_vision import lighting
from computer_vision import monitor as monitor_module
from computer_vision.camera import Camera, CameraError
from computer_vision.monitor import CameraBusyError, CameraMonitor
from tests.helpers import FakeCamera, TempDatabaseTestCase, new_monitor, wait_for
from utils import config

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)

JPEG_MAGIC = b"\xff\xd8"   # every JPEG file starts with these two bytes


class LightingTests(unittest.TestCase):
    def test_brightness_of_black_white_and_gray_frames(self):
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        white = np.full((480, 640, 3), 255, dtype=np.uint8)
        gray = np.full((480, 640, 3), 100, dtype=np.uint8)
        self.assertAlmostEqual(lighting.mean_brightness(black), 0.0, places=1)
        self.assertAlmostEqual(lighting.mean_brightness(white), 255.0, delta=0.5)
        self.assertAlmostEqual(lighting.mean_brightness(gray), 100.0, delta=0.5)

    def test_classification_uses_the_configured_thresholds(self):
        self.assertEqual(lighting.classify_lighting(config.LIGHT_TOO_DARK_BELOW - 1)[0], lighting.LIGHTING_TOO_DARK)
        self.assertEqual(lighting.classify_lighting(120)[0], lighting.LIGHTING_OK)
        self.assertEqual(lighting.classify_lighting(config.LIGHT_TOO_BRIGHT_ABOVE + 1)[0], lighting.LIGHTING_TOO_BRIGHT)
        self.assertEqual(lighting.classify_lighting(None), (lighting.LIGHTING_OK, ""))

    def test_dark_message_tells_the_user_what_to_do(self):
        self.assertIn("more light", lighting.classify_lighting(10)[1])

    def test_smoothing(self):
        self.assertEqual(lighting.smooth(None, 80), 80)                 # first value is used as is
        self.assertAlmostEqual(lighting.smooth(100, 200, alpha=0.1), 110)


class CameraWrapperTests(unittest.TestCase):
    def test_missing_camera_gives_the_required_message(self):
        # No real camera exists at index 99 on any computer
        with self.assertRaises(CameraError) as caught:
            Camera(index=99).open()
        self.assertEqual(str(caught.exception), "Camera not detected.")

    def test_read_before_open_and_double_release_are_safe(self):
        camera = Camera(index=99)
        self.assertIsNone(camera.read())
        camera.release()
        camera.release()


class MonitorTestCase(unittest.TestCase):
    """Base class: builds monitors and always switches them off afterwards."""

    def setUp(self):
        self._monitors = []

    def tearDown(self):
        for monitor in self._monitors:
            monitor.stop()

    def make_monitor(self, camera=None, **options):
        camera = camera or FakeCamera()
        monitor = new_monitor(lambda: camera, **options)
        self._monitors.append(monitor)
        return monitor, camera


class MonitorLifecycleTests(MonitorTestCase):
    def test_camera_is_off_until_started(self):
        monitor, camera = self.make_monitor()
        status = monitor.get_status(1)
        self.assertEqual(status["state"], "stopped")
        self.assertFalse(status["camera_active"])
        self.assertFalse(camera.opened)

    def test_start_delivers_jpeg_frames_and_status(self):
        monitor, camera = self.make_monitor()
        status = monitor.start(user_id=1)
        self.assertEqual(status["state"], "running")
        self.assertTrue(status["owned_by_me"])
        self.assertTrue(wait_for(lambda: monitor.get_frame(1) is not None))

        frame_id, jpeg = monitor.get_frame(1)
        self.assertTrue(jpeg.startswith(JPEG_MAGIC))
        self.assertGreater(frame_id, 0)

        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("fps", 0) > 0))
        status = monitor.get_status(1)
        self.assertEqual((status["frame_width"], status["frame_height"]), (640, 480))
        self.assertEqual(status["lighting"], "ok")

    def test_frames_keep_updating(self):
        monitor, _ = self.make_monitor()
        monitor.start(1)
        self.assertTrue(wait_for(lambda: monitor.get_frame(1) is not None))
        first_id = monitor.get_frame(1)[0]
        self.assertTrue(wait_for(lambda: monitor.get_frame(1)[0] > first_id + 2))

    def test_stop_releases_camera_and_deletes_the_last_frame(self):
        monitor, camera = self.make_monitor()
        monitor.start(1)
        self.assertTrue(wait_for(lambda: monitor.get_frame(1) is not None))
        monitor.stop()
        self.assertTrue(camera.released)                 # camera light is off
        self.assertIsNone(monitor.get_frame(1))          # privacy: no image kept after Stop
        self.assertEqual(monitor.get_status(1)["state"], "stopped")

    def test_stop_when_nothing_is_running_is_safe(self):
        monitor, _ = self.make_monitor()
        monitor.stop()
        monitor.stop()

    def test_can_start_again_after_stopping(self):
        cameras = []

        def factory():
            cameras.append(FakeCamera())
            return cameras[-1]

        monitor = new_monitor(factory)
        self._monitors.append(monitor)
        monitor.start(1)
        monitor.stop()
        self.assertEqual(monitor.start(1)["state"], "running")
        self.assertEqual(len(cameras), 2)
        self.assertTrue(cameras[0].released)

    def test_starting_twice_for_the_same_user_opens_only_one_camera(self):
        opened = []

        def factory():
            opened.append(1)
            return FakeCamera()

        monitor = new_monitor(factory)
        self._monitors.append(monitor)
        monitor.start(1)
        monitor.start(1)
        self.assertEqual(len(opened), 1)

    def test_dark_room_is_reported(self):
        monitor, _ = self.make_monitor(camera=FakeCamera(gray_level=10))
        monitor.start(1)
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("lighting") == "too_dark"))
        self.assertIn("more light", monitor.get_status(1)["lighting_message"])


class MonitorErrorTests(MonitorTestCase):
    def test_no_camera_gives_clear_error_and_leaves_camera_off(self):
        monitor, _ = self.make_monitor(camera=FakeCamera(fail_open=True))
        status = monitor.start(1)
        self.assertEqual(status["state"], "error")
        self.assertEqual(status["message"], "Camera not detected.")
        self.assertFalse(status["camera_active"])
        monitor.stop()                                   # Stop clears the error
        self.assertEqual(monitor.get_status(1)["message"], "")

    def test_camera_lost_while_running(self):
        monitor, camera = self.make_monitor(
            camera=FakeCamera(fail_after_reads=3), max_read_failures=3
        )
        monitor.start(1)
        self.assertTrue(wait_for(lambda: monitor.get_status(1)["state"] == "error"))
        self.assertIn("stopped sending images", monitor.get_status(1)["message"])
        self.assertTrue(wait_for(lambda: camera.released))
        self.assertIsNone(monitor.get_frame(1))

    def test_unexpected_crash_becomes_an_error_message_not_a_dead_thread(self):
        monitor, camera = self.make_monitor(camera=FakeCamera(raise_on_read=True))
        monitor.start(1)
        self.assertTrue(wait_for(lambda: monitor.get_status(1)["state"] == "error"))
        self.assertIn("Unexpected camera problem", monitor.get_status(1)["message"])
        self.assertTrue(camera.released)

    def test_can_retry_after_an_error(self):
        attempts = [FakeCamera(fail_open=True), FakeCamera()]
        monitor = new_monitor(lambda: attempts.pop(0))
        self._monitors.append(monitor)
        self.assertEqual(monitor.start(1)["state"], "error")
        self.assertEqual(monitor.start(1)["state"], "running")


class MonitorPrivacyTests(MonitorTestCase):
    def test_second_user_cannot_take_or_see_the_camera(self):
        monitor, _ = self.make_monitor()
        monitor.start(1)
        self.assertTrue(wait_for(lambda: monitor.get_frame(1) is not None))

        with self.assertRaises(CameraBusyError):
            monitor.start(2)

        other_view = monitor.get_status(2)
        self.assertTrue(other_view["busy_other_user"])
        self.assertFalse(other_view["owned_by_me"])
        self.assertNotIn("fps", other_view)              # no measurements leak to user 2
        self.assertIsNone(monitor.get_frame(2))          # no picture leaks to user 2

    def test_only_the_owner_can_stop_the_camera(self):
        monitor, _ = self.make_monitor()
        monitor.start(1)
        self.assertFalse(monitor.stop_if_owner(2))
        self.assertTrue(monitor.get_status(1)["camera_active"])
        self.assertTrue(monitor.stop_if_owner(1))
        self.assertFalse(monitor.get_status(1)["camera_active"])

    def test_camera_switches_off_when_the_dashboard_goes_away(self):
        monitor, camera = self.make_monitor(idle_timeout_s=0.4)
        monitor.start(1)
        self.assertTrue(wait_for(lambda: monitor.get_status(1)["state"] == "stopped", timeout=4))
        self.assertIn("automatically", monitor.get_status(1)["message"])
        self.assertTrue(camera.released)
        self.assertIsNone(monitor.get_frame(1))

    def test_heartbeat_keeps_the_camera_on(self):
        monitor, _ = self.make_monitor(idle_timeout_s=0.6)
        monitor.start(1)
        end = time.monotonic() + 1.4                     # longer than the timeout
        while time.monotonic() < end:
            monitor.touch(1)
            time.sleep(0.1)
        self.assertEqual(monitor.get_status(1)["state"], "running")

    def test_heartbeat_from_another_user_does_not_keep_the_camera_on(self):
        monitor, _ = self.make_monitor(idle_timeout_s=0.5)
        monitor.start(1)
        end = time.monotonic() + 1.5
        while time.monotonic() < end and monitor.get_status(1)["state"] == "running":
            monitor.touch(2)
            time.sleep(0.1)
        self.assertEqual(monitor.get_status(1)["state"], "stopped")

    def test_no_background_threads_are_left_after_stop(self):
        monitor, _ = self.make_monitor()
        monitor.start(1)
        monitor.stop()
        self.assertFalse(any(t.name == "camera-monitor" and t.is_alive() for t in threading.enumerate()))

    def test_nothing_is_written_to_disk(self):
        # The monitor must never save images: no image/video writers and no file opening.
        # (Simple text check of the source file; `camera.open()` is allowed, it opens the camera.)
        import re
        from pathlib import Path
        source = Path(monitor_module.__file__).read_text(encoding="utf-8")
        for forbidden in ("imwrite", "VideoWriter", "write_bytes", "write_text", ".write("):
            self.assertNotIn(forbidden, source)
        self.assertIsNone(re.search(r"(?<![\w.])open\(", source))   # no builtin open(...)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class MonitorApiTests(TempDatabaseTestCase):
    def setUp(self):
        super().setUp()
        self.fake_camera = FakeCamera()
        self.fake_monitor = new_monitor(lambda: self.fake_camera)
        self._original_monitor = monitor_module.monitor
        monitor_module.monitor = self.fake_monitor        # the routes now use the fake camera

    def tearDown(self):
        self.fake_monitor.stop()
        monitor_module.monitor = self._original_monitor
        super().tearDown()

    def new_client(self, username=None):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        if username:
            client.post("/api/auth/register", json={"username": username, "password": "password123"})
        return client

    def test_login_is_required(self):
        client = self.new_client()
        self.assertEqual(client.post("/api/monitor/start").status_code, 401)
        self.assertEqual(client.get("/api/monitor/status").status_code, 401)
        self.assertEqual(client.get("/api/monitor/frame").status_code, 401)

    def test_start_preview_stop_cycle(self):
        client = self.new_client("asha")
        self.assertFalse(client.get("/api/monitor/status").json()["camera_active"])

        started = client.post("/api/monitor/start")
        self.assertEqual(started.status_code, 200)
        self.assertTrue(started.json()["owned_by_me"])

        self.assertTrue(wait_for(lambda: client.get("/api/monitor/frame").status_code == 200))
        frame = client.get("/api/monitor/frame")
        self.assertEqual(frame.headers["content-type"], "image/jpeg")
        self.assertTrue(frame.content.startswith(JPEG_MAGIC))
        self.assertIn("no-store", frame.headers["cache-control"])
        self.assertEqual(client.get("/api/health").json()["camera"], "on")

        client.post("/api/monitor/stop")
        self.assertFalse(client.get("/api/monitor/status").json()["camera_active"])
        self.assertEqual(client.get("/api/monitor/frame").status_code, 204)
        self.assertEqual(client.get("/api/health").json()["camera"], "off")

    def test_no_camera_returns_503_with_message(self):
        self.fake_camera.fail_open = True
        client = self.new_client("asha")
        response = client.post("/api/monitor/start")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Camera not detected.")

    def test_second_account_is_blocked_and_sees_no_picture(self):
        first, second = self.new_client("asha"), self.new_client("bala")
        first.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: first.get("/api/monitor/frame").status_code == 200))

        self.assertEqual(second.post("/api/monitor/start").status_code, 409)
        self.assertTrue(second.get("/api/monitor/status").json()["busy_other_user"])
        self.assertEqual(second.get("/api/monitor/frame").status_code, 204)

        second.post("/api/monitor/stop")                  # not theirs: must not switch it off
        self.assertTrue(first.get("/api/monitor/status").json()["camera_active"])

    def test_logout_switches_the_camera_off(self):
        client = self.new_client("asha")
        client.post("/api/monitor/start")
        client.post("/api/auth/logout")
        self.assertTrue(wait_for(lambda: self.fake_camera.released))
        self.assertFalse(self.fake_monitor.get_status(1)["camera_active"])


if __name__ == "__main__":
    unittest.main()
