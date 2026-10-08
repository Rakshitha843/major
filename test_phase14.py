"""
Phase 14 tests: whole-pipeline interaction scenarios not yet exercised together by earlier phases, plus a
capstone end-to-end test walking through the project brief's own 20-item testing list (section 25).

WHY THIS PHASE EXISTS, GIVEN HOW MUCH IS ALREADY TESTED
    Phases 2-13 each unit-test their own subsystem thoroughly, and several ALSO include one whole-pipeline
    scripted test (a simulated camera driving the real monitor end to end). That specific technique found
    two real bugs no unit test caught: the Phase 8 posture/attention interaction bug (turning the head looked
    like leaning away) and the Phase 10 DHRI escalation "stairstep" bug. Phase 14's job is to run MORE such
    whole-pipeline scenarios, specifically ones earlier phases had no reason to try, looking for the same
    class of bug: something that only shows up when several already-correct pieces interact.

A REAL BUG FOUND WHILE WRITING THIS PHASE
    The server's shutdown handler (backend/main.py's `lifespan`) stopped the camera on shutdown but never
    saved the session that was running -- meaning every session running when the server was stopped (Ctrl+C,
    or `--reload` auto-restarting after a code change, which happens on every file save during development)
    was silently discarded. Confirmed by reproducing it directly (a session ran, the equivalent of the old
    shutdown code ran, and 0 rows existed in the database afterwards) before fixing it. See
    ServerShutdownTests below and docs/architecture.md for the full story.
"""

import importlib.util
import time
import unittest
from unittest import mock

from computer_vision import monitor as monitor_module
from computer_vision.calibrator import Calibrator
from database import sessions as sessions_db
from tests.helpers import FakeCamera, FakeDetector, FakePoseDetector, TempDatabaseTestCase, new_monitor, shoulders_result, wait_for
from utils import config

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)

HEAD = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "distance": 2500.0}


def calibrate(monitor, detector, user_id=1):
    monitor.begin_calibration(user_id)
    end = time.monotonic() + 10
    while time.monotonic() < end:
        status = monitor.get_calibration_status(user_id)
        if status is not None:
            detector.ear = 0.10 if status["phase"] == "close" else 0.30
            if not status["active"]:
                break
        time.sleep(0.02)
    return monitor.take_calibration_result(user_id)


def build_monitor(**options):
    camera = FakeCamera()
    detector = FakeDetector(ear=0.30, head=dict(HEAD))
    options.setdefault("calibrator_factory", lambda: Calibrator(ready_s=1.0, open_s=2.5, close_s=2.0))
    monitor = new_monitor(lambda: camera, detector_factory=lambda: detector,
                          pose_detector_factory=lambda: FakePoseDetector(result=shoulders_result()), **options)
    return monitor, camera, detector


# ======================================================================
# The server-shutdown data-loss bug: reproduced, then confirmed fixed
# ======================================================================
class ServerShutdownTests(TempDatabaseTestCase):
    def setUp(self):
        super().setUp()
        from backend import auth_service
        self.user, _ = auth_service.register(self.connection, "asha", "password123")
        self._monitors = []

    def tearDown(self):
        for monitor in self._monitors:
            monitor.stop()
        super().tearDown()

    def build(self):
        monitor, camera, detector = build_monitor()
        self._monitors.append(monitor)
        return monitor, camera, detector

    def shutdown_like_the_server_does(self, monitor):
        """Exactly what backend/main.py's lifespan shutdown handler does, after the Phase 14 fix."""
        owner_id = monitor.get_active_owner()
        monitor.stop()
        if owner_id is not None:
            summary = monitor.take_session_result(owner_id)
            if sessions_db.should_save_session(summary):
                sessions_db.save_session(self.connection, owner_id, summary)
        return owner_id

    def test_a_session_running_at_shutdown_is_saved(self):
        monitor, _, _ = self.build()
        monitor.start(self.user["id"])
        self.assertTrue(wait_for(lambda: monitor.get_status(self.user["id"]).get("session", {}).get("duration_s", 0) > 0.1))
        self.shutdown_like_the_server_does(monitor)
        saved = sessions_db.get_recent_sessions(self.connection, self.user["id"])
        self.assertEqual(len(saved), 1)
        self.assertGreater(saved[0]["duration_s"], 0)

    def test_reproduces_as_data_loss_without_the_fix(self):
        # Documents the exact bug: stopping the camera WITHOUT also collecting and saving the session
        # (what the code did before this phase) silently discards it.
        monitor, _, _ = self.build()
        monitor.start(self.user["id"])
        self.assertTrue(wait_for(lambda: monitor.get_status(self.user["id"]).get("session", {}).get("duration_s", 0) > 0.1))
        monitor.stop()      # the old, incomplete shutdown behavior: stop only, nothing saved
        saved = sessions_db.get_recent_sessions(self.connection, self.user["id"])
        self.assertEqual(len(saved), 0, "this documents the bug: stop() alone saves nothing")

    def test_shutdown_with_no_active_session_saves_nothing_and_does_not_error(self):
        monitor, _, _ = self.build()      # never started
        self.shutdown_like_the_server_does(monitor)      # must not raise
        self.assertEqual(len(sessions_db.get_recent_sessions(self.connection, self.user["id"])), 0)

    def test_shutdown_right_after_stop_was_already_called_saves_nothing_extra(self):
        monitor, _, _ = self.build()
        monitor.start(self.user["id"])
        self.assertTrue(wait_for(lambda: monitor.get_status(self.user["id"]).get("session", {}).get("duration_s", 0) > 0.1))
        monitor.stop()
        summary = monitor.take_session_result(self.user["id"])          # the route already collected it
        sessions_db.save_session(self.connection, self.user["id"], summary)
        self.shutdown_like_the_server_does(monitor)                     # nothing left to collect
        self.assertEqual(len(sessions_db.get_recent_sessions(self.connection, self.user["id"])), 1)

    def test_get_active_owner_reflects_who_is_using_the_camera(self):
        monitor, _, _ = self.build()
        self.assertIsNone(monitor.get_active_owner())
        monitor.start(self.user["id"])
        self.assertEqual(monitor.get_active_owner(), self.user["id"])
        monitor.stop()
        self.assertIsNone(monitor.get_active_owner())


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class ServerShutdownViaRealAppTests(TempDatabaseTestCase):
    """Same scenario, but running the ACTUAL FastAPI lifespan shutdown code, for maximum fidelity."""

    def setUp(self):
        super().setUp()
        self._original_monitor = monitor_module.monitor
        self.camera = FakeCamera()
        self.detector = FakeDetector(ear=0.30)
        monitor_module.monitor = new_monitor(lambda: self.camera, detector_factory=lambda: self.detector)

    def tearDown(self):
        monitor_module.monitor.stop()
        monitor_module.monitor = self._original_monitor
        super().tearDown()

    def test_the_real_lifespan_shutdown_saves_an_active_session(self):
        import asyncio
        from backend.main import app, lifespan
        from fastapi.testclient import TestClient

        client = TestClient(app)
        client.post("/api/auth/register", json={"username": "asha", "password": "password123"})
        client.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json().get("session", {}).get("duration_s", 0) > 0.1))

        # Drive the real lifespan context manager's shutdown half directly (this is exactly what uvicorn
        # runs when the process stops or --reload restarts it).
        async def run_shutdown():
            async with lifespan(app):
                pass      # the "yield" point: startup already happened via TestClient's own app lifecycle
        asyncio.run(run_shutdown())

        from database import connection as db
        connection = db.connect()
        try:
            row = connection.execute("SELECT id FROM users WHERE username = 'asha'").fetchone()
            saved = connection.execute(
                "SELECT COUNT(*) AS n FROM monitoring_sessions WHERE user_id = ? AND end_time IS NOT NULL",
                (row["id"] if row else -1,),
            ).fetchone()
            self.assertEqual(saved["n"], 1)
        finally:
            connection.close()


# ======================================================================
# Recalibrating mid-session with accumulated fatigue/posture/attention/DHRI/alert state
# ======================================================================
class RecalibrationMidSessionTests(unittest.TestCase):
    def setUp(self):
        self._monitors = []

    def tearDown(self):
        for monitor in self._monitors:
            monitor.stop()

    def test_recalibrating_after_triggering_several_conditions_does_not_crash_or_corrupt_state(self):
        with mock.patch.object(config, "POSTURE_CAL_MIN_SAMPLES", 10):
            monitor, camera, detector = build_monitor()
            self._monitors.append(monitor)
            monitor.start(1)
            calibrate(monitor, detector)

            # Rack up real fatigue, posture, attention and DHRI/alert activity before recalibrating.
            detector.head = {**HEAD, "roll": 25.0}
            detector.ear = 0.10
            self.assertTrue(wait_for(lambda: monitor.get_status(1)["alerts"]["total_count"] > 0, timeout=8))
            events_before = monitor.get_status(1)["fatigue"]["events"]

            # Recalibrate with a DIFFERENT baseline while all of that is active.
            detector.head = {**HEAD, "yaw": 8.0}
            detector.ear = 0.30
            result = calibrate(monitor, detector)
            self.assertTrue(result["ok"], result)

            # Everything must still be alive and responsive afterwards, using the NEW baseline.
            self.assertTrue(wait_for(lambda: monitor.get_status(1)["fatigue"]["state"] == "normal", timeout=15))
            status = monitor.get_status(1)
            self.assertEqual(status["state"], "running")
            self.assertTrue(status["calibrated"])
            self.assertTrue(status["posture_calibrated"])
            self.assertTrue(status["attention_calibrated"])
            # Event counts from before are preserved, not reset by recalibrating mid-session.
            self.assertGreaterEqual(monitor.get_status(1)["fatigue"]["events"]["fatigued"], events_before["fatigued"])

    def test_recalibrating_does_not_lose_the_camera_or_produce_two_sessions(self):
        with mock.patch.object(config, "POSTURE_CAL_MIN_SAMPLES", 10):
            monitor, camera, detector = build_monitor()
            self._monitors.append(monitor)
            monitor.start(1)
            calibrate(monitor, detector)
            duration_before = monitor.get_status(1)["session"]["duration_s"]
            calibrate(monitor, detector)      # recalibrate again
            # The SAME session keeps running (its clock does not reset just because calibration reran).
            self.assertGreaterEqual(monitor.get_status(1)["session"]["duration_s"], duration_before)
            self.assertEqual(monitor.get_status(1)["state"], "running")


# ======================================================================
# Alert cooldowns and a fresh camera start (a design choice worth locking in with a test + documenting)
# ======================================================================
class AlertCooldownAcrossRestartTests(unittest.TestCase):
    def setUp(self):
        self._monitors = []

    def tearDown(self):
        for monitor in self._monitors:
            monitor.stop()

    def test_stopping_and_starting_again_resets_alert_cooldowns(self):
        # Documented design choice (see docs/architecture.md): cooldowns are per-SESSION, not per-account
        # forever, because a fresh AlertManager is created each time the camera starts. This means an alert
        # that just fired can fire again immediately in a NEW session, even though its cooldown has not
        # really elapsed in wall-clock time. This test exists so that behavior is deliberate and known, not
        # an accident someone discovers later.
        with mock.patch.object(config, "ALERT_COOLDOWN_S", {**config.ALERT_COOLDOWN_S, "posture": 10_000.0}):
            monitor, camera, detector = build_monitor()
            self._monitors.append(monitor)
            monitor.start(1)
            calibrate(monitor, detector)
            detector.head = {**HEAD, "roll": 25.0}
            self.assertTrue(wait_for(lambda: monitor.get_status(1)["alerts"]["total_count"] > 0, timeout=8))
            first_session_count = monitor.get_status(1)["alerts"]["total_count"]

            monitor.stop()
            monitor.take_session_result(1)
            monitor.start(1, baseline=None)     # a fresh session; posture stays bad from the start
            # Recalibrate with the head NEUTRAL (otherwise the new baseline would treat the 25-degree tilt
            # as normal, and it would never look "poor" again -- a mistake in an earlier version of this test).
            detector.head = dict(HEAD)
            with mock.patch.object(config, "POSTURE_CAL_MIN_SAMPLES", 10):
                calibrate(monitor, detector)
            detector.head = {**HEAD, "roll": 25.0}
            self.assertTrue(wait_for(lambda: monitor.get_status(1)["alerts"]["total_count"] > 0, timeout=8))
            # A fresh AlertManager means this fired again despite the 10,000 s cooldown from the old session.
            self.assertGreaterEqual(monitor.get_status(1)["alerts"]["total_count"], 1)
            self.assertNotEqual(first_session_count, -1)   # (sanity: the earlier count was real)


# ======================================================================
# Logging out / stopping while calibration is actively running
# ======================================================================
class InterruptedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self._monitors = []

    def tearDown(self):
        for monitor in self._monitors:
            monitor.stop()

    def test_stopping_the_camera_while_calibration_is_running_does_not_crash(self):
        monitor, camera, detector = build_monitor(calibrator_factory=lambda: Calibrator(ready_s=1.0, open_s=3.0, close_s=3.0))
        self._monitors.append(monitor)
        monitor.start(1)
        monitor.begin_calibration(1)
        self.assertTrue(wait_for(lambda: monitor.get_calibration_status(1) is not None and monitor.get_calibration_status(1)["active"]))
        monitor.stop()      # equivalent to a logout while "Sit normally..." is still on screen
        self.assertEqual(monitor.get_status(1)["state"], "stopped")
        self.assertTrue(camera.released)
        # No stale calibration progress should be reported for a fresh status check on this (now-stopped) camera.
        self.assertNotIn("calibrated", monitor.get_status(1))     # that field only exists while the camera runs

    def test_a_new_session_can_start_cleanly_after_an_interrupted_calibration(self):
        monitor, camera, detector = build_monitor(calibrator_factory=lambda: Calibrator(ready_s=1.0, open_s=3.0, close_s=3.0))
        self._monitors.append(monitor)
        monitor.start(1)
        monitor.begin_calibration(1)
        self.assertTrue(wait_for(lambda: monitor.get_calibration_status(1) is not None and monitor.get_calibration_status(1)["active"]))
        monitor.stop()
        monitor.start(1)
        final = calibrate(monitor, detector)
        self.assertTrue(final["ok"], final)


# ======================================================================
# Capstone: the project brief's own 20-item testing list (section 25), end to end
# ======================================================================
@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class BriefTestingChecklistTests(TempDatabaseTestCase):
    """
    One test per numbered item in the brief's section 25 list, run as a single continuous session against
    the real FastAPI app (items 4-8, 20 use a simulated camera/face; item 4's REAL camera and item 5's REAL
    face model cannot be verified here -- see docs/testing_checklist.md for the manual steps that cover those).
    """

    def setUp(self):
        super().setUp()
        self._original_monitor = monitor_module.monitor
        self.camera = FakeCamera()
        self.detector = FakeDetector(ear=0.30, head=dict(HEAD))
        monitor_module.monitor = new_monitor(
            lambda: self.camera, detector_factory=lambda: self.detector,
            pose_detector_factory=lambda: FakePoseDetector(result=shoulders_result()),
            calibrator_factory=lambda: Calibrator(ready_s=1.0, open_s=2.5, close_s=2.0))

    def tearDown(self):
        monitor_module.monitor.stop()
        monitor_module.monitor = self._original_monitor
        super().tearDown()

    def new_client(self, username):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app), username

    def calibrate_via_api(self, client):
        client.post("/api/calibration/start")
        end = time.monotonic() + 10
        while time.monotonic() < end:
            body = client.get("/api/calibration/status").json()
            self.detector.ear = 0.10 if body["phase"] == "close" else 0.30
            if not body["active"]:
                break
            time.sleep(0.02)

    def test_the_full_20_item_checklist_end_to_end(self):
        # Fatigue's "fatigued" state needs FEATURE_MIN_COVERAGE_S (10s) of accumulated window data
        # PLUS FATIGUE_ESCALATE_HOLD_S (3s) before it can appear at all -- and a gap in frame
        # processing does not count towards that coverage (deliberately: see
        # tests/test_phase5.py's "a gap in the data is not counted as valid time" -- using stale
        # data would over-report PERCLOS). That makes an end-to-end test that waits on real
        # wall-clock time for the real thresholds fundamentally fragile whenever the machine
        # running it is briefly busy (this project's own full test suite makes it so, running many
        # camera-simulation threads back to back) -- through no fault of the fatigue logic itself.
        # Every OTHER config-dependent timing in this project's tests already shrinks the window
        # for exactly this reason (see tests/test_phase6.py's
        # test_normal_then_fatigued_then_recovered_through_the_whole_pipeline); this test does the
        # same, since what it actually needs to verify -- the real FastAPI/lifespan wiring -- does
        # not depend on the window being 20s long.
        with mock.patch.object(config, "FEATURE_WINDOW_S", 3), \
             mock.patch.object(config, "FEATURE_MIN_COVERAGE_S", 1.0):
            client, _ = self.new_client("asha")

            # 1. Registration
            response = client.post("/api/auth/register", json={"username": "asha", "password": "password123"})
            self.assertEqual(response.status_code, 201)

            # 2. Login (logout then log back in)
            client.post("/api/auth/logout")
            self.assertEqual(client.get("/api/auth/me").status_code, 401)
            login = client.post("/api/auth/login", json={"username": "asha", "password": "password123"})
            self.assertEqual(login.status_code, 200)

            # 3. Calibration
            self.calibrate_via_api(client)
            self.assertTrue(client.get("/api/auth/me").json()["calibrated"])
            # calibrate_via_api's last poll leaves the simulated eyes CLOSED (calibration's own last
            # step is "close your eyes" -- see computer_vision/calibrator.py's phase order). A real
            # user naturally opens their eyes again once calibration finishes; simulate that here.
            # Without it, the eye tracker's "must see the eyes open before trusting a closure" guard
            # (computer_vision/eye_tracker.py's _seen_open -- there precisely so a closure already in
            # progress before tracking resumes is never miscounted) would never see an open reading
            # to arm itself, and window_valid_s would stay 0 forever, so fatigue could never assess.
            self.detector.ear = 0.30

            # 4-8. Camera access, face detection, eye detection, blink detection, eye closure
            # (simulated here; the REAL camera/face model are verified manually -- see the testing checklist doc)
            self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json().get("eye", {}).get("ear") is not None))
            self.assertEqual(client.get("/api/monitor/status").json()["face_status"], "ok")
            self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json()["eye"].get("eyes_closed") is False))

            # 9. Fatigue detection
            self.detector.ear = 0.10
            self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json()["fatigue"]["state"] == "fatigued", timeout=15))
            self.detector.ear = 0.30

        # 10. Posture detection (real POSTURE_ESCALATE_HOLD_S = 5s; margin for the same reason as above)
        self.detector.head = {**HEAD, "roll": 25.0}
        self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json()["posture"]["state"] == "poor", timeout=12))
        self.detector.head = dict(HEAD)

        # 11. Attention detection (real ATTENTION_AWAY_HOLD_S = 5s; same margin)
        self.detector.head = {**HEAD, "yaw": 40.0}
        self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json()["attention"]["state"] == "distracted", timeout=12))
        self.detector.head = dict(HEAD)

        # 12. Session timer
        self.assertGreater(client.get("/api/monitor/status").json()["session"]["duration_s"], 0)

        # 13. DHRI calculation
        self.assertIn(client.get("/api/monitor/status").json()["dhri"]["level"], ("low", "moderate", "high"))

        # 14. Alerts
        self.assertGreater(client.get("/api/monitor/status").json()["alerts"]["total_count"], 0)

        # 15. Recommendations
        # (may be empty if conditions have since cleared; check the field exists and is a list)
        self.assertIsInstance(client.get("/api/monitor/status").json()["alerts"]["recommendations"], list)

        # 16. Database storage (stop the session and confirm a row was written)
        client.post("/api/monitor/stop")
        history = client.get("/api/monitor/history").json()
        self.assertEqual(len(history["recent"]), 1)

        # 17. History
        self.assertGreater(history["today"]["session_count"], 0)
        full = client.get("/api/monitor/sessions").json()
        self.assertEqual(full["total_count"], 1)

        # 18. Dashboard graphs -- the chart-drawing JavaScript is tested separately (Node-based dashboard
        # tests); this confirms the DATA those charts read is present and correctly shaped.
        row = history["recent"][0]
        for field in ("fatigue_warning_count", "fatigue_event_count", "posture_poor_count",
                      "attention_distracted_count", "dhri_high_count", "alert_count"):
            self.assertIn(field, row)

        # 19. Logout
        self.assertEqual(client.post("/api/auth/logout").status_code, 200)
        self.assertEqual(client.get("/api/auth/me").status_code, 401)

        # 20. Multiple users (a second, completely independent account)
        second, _ = self.new_client("bala")
        second.post("/api/auth/register", json={"username": "bala", "password": "password456"})
        self.assertFalse(second.get("/api/auth/me").json()["calibrated"])
        self.assertEqual(len(second.get("/api/monitor/history").json()["recent"]), 0)


if __name__ == "__main__":
    unittest.main()
