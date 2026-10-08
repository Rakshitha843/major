"""
Phase 13 tests: the full-history pagination (database layer and the /api/monitor/sessions route), and an
END-TO-END concurrent multi-user integration test that runs two accounts through calibration, monitoring,
alerts and history together, interleaved in real time -- going beyond the many per-feature "two users stay
separate" tests already written in Phases 2-12, which each check ONE subsystem's isolation in the small.
This one exercises the whole stack together the way two people actually taking turns on one computer would.

Run from the project root:
    python -m unittest discover -s tests -t . -v
"""

import importlib.util
import time
import unittest
from datetime import datetime, timedelta, timezone

from computer_vision import monitor as monitor_module
from computer_vision.calibrator import Calibrator
from database import sessions as sessions_db
from tests.helpers import FakeCamera, FakeDetector, FakePoseDetector, TempDatabaseTestCase, new_monitor, shoulders_result, wait_for

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)


def a_summary(start_time, **overrides):
    base = {
        "start_time": start_time, "duration_s": 600.0, "active_duration_s": 600.0,
        "break_count": 0, "break_duration_s": 0.0,
        "fatigue_events": {"warning": 0, "fatigued": 0}, "posture_events": {"warning": 0, "poor": 0},
        "attention_events": {"distracted": 0, "no_face": 0}, "dhri_events": {"moderate": 0, "high": 0},
        "alert_count": 0,
    }
    base.update(overrides)
    return base


# ======================================================================
# Pagination (database layer)
# ======================================================================
class PaginationTests(TempDatabaseTestCase):
    def make_user(self, name="asha"):
        from backend import auth_service
        user, _ = auth_service.register(self.connection, name, "password123")
        return user["id"]

    def save_n_sessions(self, user_id, n, start=None):
        start = start or datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
        for i in range(n):
            sessions_db.save_session(self.connection, user_id, a_summary(start + timedelta(hours=i)))

    def test_offset_skips_the_right_number_of_rows(self):
        user_id = self.make_user()
        self.save_n_sessions(user_id, 25)
        first_page = sessions_db.get_recent_sessions(self.connection, user_id, limit=20, offset=0)
        second_page = sessions_db.get_recent_sessions(self.connection, user_id, limit=20, offset=20)
        self.assertEqual(len(first_page), 20)
        self.assertEqual(len(second_page), 5)
        # newest-first ordering: the first page's last row should be immediately newer than the second page's first
        first_times = [row["start_time"] for row in first_page]
        second_times = [row["start_time"] for row in second_page]
        self.assertEqual(set(first_times) & set(second_times), set())          # no overlap between pages
        self.assertGreater(min(first_times), max(second_times))                # page 1 is strictly newer than page 2

    def test_count_finished_sessions_matches_what_was_saved(self):
        user_id = self.make_user()
        self.save_n_sessions(user_id, 7)
        self.assertEqual(sessions_db.count_finished_sessions(self.connection, user_id), 7)

    def test_count_is_zero_for_a_fresh_user(self):
        user_id = self.make_user()
        self.assertEqual(sessions_db.count_finished_sessions(self.connection, user_id), 0)

    def test_offset_past_the_end_returns_nothing(self):
        user_id = self.make_user()
        self.save_n_sessions(user_id, 3)
        self.assertEqual(sessions_db.get_recent_sessions(self.connection, user_id, limit=20, offset=100), [])

    def test_pagination_is_per_user(self):
        first, second = self.make_user("asha"), self.make_user("bala")
        self.save_n_sessions(first, 15)
        self.save_n_sessions(second, 3)
        self.assertEqual(sessions_db.count_finished_sessions(self.connection, first), 15)
        self.assertEqual(sessions_db.count_finished_sessions(self.connection, second), 3)
        self.assertEqual(len(sessions_db.get_recent_sessions(self.connection, second, limit=20, offset=0)), 3)

    def test_default_offset_is_zero_for_backward_compatibility(self):
        # get_recent_sessions is also called (with no offset argument) by the Phase 12 trend-chart route;
        # this must keep working exactly as before.
        user_id = self.make_user()
        self.save_n_sessions(user_id, 5)
        self.assertEqual(len(sessions_db.get_recent_sessions(self.connection, user_id)),
                         len(sessions_db.get_recent_sessions(self.connection, user_id, offset=0)))


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class SessionsApiTests(TempDatabaseTestCase):
    def setUp(self):
        super().setUp()
        self._original_monitor = monitor_module.monitor
        self.camera = FakeCamera()
        self.detector = FakeDetector(ear=0.30)
        monitor_module.monitor = new_monitor(
            lambda: self.camera, detector_factory=lambda: self.detector,
            pose_detector_factory=lambda: FakePoseDetector(result=shoulders_result()))

    def tearDown(self):
        monitor_module.monitor.stop()
        monitor_module.monitor = self._original_monitor
        super().tearDown()

    def client(self, username="asha"):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        client.post("/api/auth/register", json={"username": username, "password": "password123"})
        return client

    def make_sessions(self, client, n):
        for _ in range(n):
            client.post("/api/monitor/start")
            self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json().get("session", {}).get("duration_s", 0) > 0.05))
            client.post("/api/monitor/stop")

    def test_default_page_and_total_count(self):
        client = self.client()
        self.make_sessions(client, 3)
        body = client.get("/api/monitor/sessions").json()
        self.assertEqual(len(body["sessions"]), 3)
        self.assertEqual(body["total_count"], 3)
        self.assertEqual(body["offset"], 0)

    def test_pagination_via_query_params(self):
        client = self.client()
        self.make_sessions(client, 5)
        page1 = client.get("/api/monitor/sessions?offset=0&limit=2").json()
        page2 = client.get("/api/monitor/sessions?offset=2&limit=2").json()
        self.assertEqual(len(page1["sessions"]), 2)
        self.assertEqual(len(page2["sessions"]), 2)
        self.assertEqual(page1["total_count"], 5)
        self.assertNotEqual(page1["sessions"][0]["start_time"], page2["sessions"][0]["start_time"])

    def test_limit_is_capped_to_a_sensible_maximum(self):
        client = self.client()
        self.make_sessions(client, 3)
        body = client.get("/api/monitor/sessions?limit=99999").json()
        self.assertEqual(body["limit"], 100)                    # capped, not 99999

    def test_negative_offset_is_treated_as_zero(self):
        client = self.client()
        self.make_sessions(client, 2)
        body = client.get("/api/monitor/sessions?offset=-5").json()
        self.assertEqual(body["offset"], 0)
        self.assertEqual(len(body["sessions"]), 2)

    def test_login_is_required(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        self.assertEqual(TestClient(app).get("/api/monitor/sessions").status_code, 401)

    def test_sessions_route_and_history_route_use_the_same_field_set(self):
        # Both /history and /sessions must return exactly the fields the dashboard's charts and full-history
        # table need -- they now share one projection function specifically to prevent them drifting apart
        # (see the Phase 12 alert_count bug this was written to guard against).
        client = self.client()
        self.make_sessions(client, 1)
        history_row = client.get("/api/monitor/history").json()["recent"][0]
        sessions_row = client.get("/api/monitor/sessions").json()["sessions"][0]
        self.assertEqual(set(history_row.keys()), set(sessions_row.keys()))

    def test_each_users_full_history_is_separate(self):
        first, second = self.client("asha"), self.client("bala")
        self.make_sessions(first, 4)
        self.make_sessions(second, 1)
        self.assertEqual(first.get("/api/monitor/sessions").json()["total_count"], 4)
        self.assertEqual(second.get("/api/monitor/sessions").json()["total_count"], 1)


# ======================================================================
# End-to-end concurrent multi-user integration
# ======================================================================
@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class ConcurrentMultiUserTests(TempDatabaseTestCase):
    """
    Simulates two people sharing one computer, taking turns, with realistic interleaving: both register,
    one calibrates and starts monitoring while the other repeatedly tries (and is refused, since there is
    only one webcam), the first finishes and logs out, the second then calibrates and uses the camera, and
    finally both check their OWN history and see only their own session -- exercising auth, calibration,
    camera ownership, alerts and history together in one realistic sequence, not as separate assertions.
    """

    def setUp(self):
        super().setUp()
        self._original_monitor = monitor_module.monitor
        self.camera = FakeCamera()
        self.detector = FakeDetector(ear=0.30, head={"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "distance": 2500.0})
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
        client = TestClient(app)
        client.post("/api/auth/register", json={"username": username, "password": "password123"})
        return client

    def calibrate(self, client, end_yaw=0.0):
        client.post("/api/calibration/start")
        end = time.monotonic() + 10
        while time.monotonic() < end:
            body = client.get("/api/calibration/status").json()
            self.detector.ear = 0.10 if body["phase"] == "close" else 0.30
            if not body["active"]:
                break
            time.sleep(0.02)

    def test_two_people_sharing_one_computer_end_to_end(self):
        asha, bala = self.new_client("asha"), self.new_client("bala")

        # Asha calibrates and starts monitoring first.
        self.calibrate(asha)
        self.assertTrue(asha.get("/api/auth/me").json()["calibrated"])
        self.assertTrue(asha.post("/api/monitor/start").json()["state"] in ("starting", "running"))

        # While Asha is using the camera, Bala tries several things -- all must be refused or empty, never
        # touching Asha's camera, calibration, or data.
        self.assertEqual(bala.post("/api/monitor/start").status_code, 409)
        self.assertEqual(bala.post("/api/calibration/start").status_code, 409)
        self.assertTrue(bala.get("/api/monitor/status").json()["busy_other_user"])
        self.assertNotIn("fps", bala.get("/api/monitor/status").json())            # no data leaks to Bala
        self.assertFalse(bala.get("/api/auth/me").json()["calibrated"])            # Bala's own status, unaffected

        # Asha triggers a real fatigue alert while using the camera.
        # FEATURE_MIN_COVERAGE_S is 10s by itself, so a 10s timeout here is exactly on the edge of
        # possible even on a fast machine -- give it real margin, as the equivalent wait elsewhere
        # in the suite (tests/test_phase14.py) already does.
        self.assertTrue(wait_for(lambda: asha.get("/api/monitor/status").json()["fatigue"]["state"] == "normal", timeout=15))
        self.detector.ear = 0.10
        self.assertTrue(wait_for(lambda: asha.get("/api/monitor/status").json()["alerts"]["total_count"] > 0, timeout=6))
        self.detector.ear = 0.30

        # Asha finishes; her camera and session are cleanly released. She stays logged in for now --
        # the "no cross-contamination" check below needs her session to still be authenticated, so
        # logging her out happens only at the very end of this test, once that check is done.
        asha.post("/api/monitor/stop")
        asha_history = asha.get("/api/monitor/history").json()
        self.assertEqual(len(asha_history["recent"]), 1)
        self.assertGreater(asha_history["recent"][0]["alert_count"], 0)
        self.assertFalse(bala.get("/api/monitor/status").json()["camera_active"])   # confirmed free for Bala too

        # Now Bala calibrates and uses the camera; her calibration/session must not see anything of Asha's.
        self.calibrate(bala)
        self.assertTrue(bala.get("/api/auth/me").json()["calibrated"])
        bala.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: bala.get("/api/monitor/status").json().get("session", {}).get("duration_s", 0) > 0.1))
        bala.post("/api/monitor/stop")

        # Final check: each account's history contains exactly its own session, with no cross-contamination.
        asha_final = asha.get("/api/monitor/history").json()
        bala_final = bala.get("/api/monitor/history").json()
        self.assertEqual(len(asha_final["recent"]), 1)          # still just the one session from earlier
        self.assertEqual(len(bala_final["recent"]), 1)          # exactly the one she just finished
        self.assertEqual(bala_final["recent"][0]["alert_count"], 0)   # Bala's session had no fatigue trigger
        self.assertNotEqual(asha.get("/api/auth/me").json(), bala.get("/api/auth/me").json())

        # Only now, with every authenticated check done, confirm logout itself works.
        asha.post("/api/auth/logout")
        self.assertEqual(asha.get("/api/monitor/history").status_code, 401)

    def test_logging_out_the_camera_owner_lets_the_other_account_start_immediately(self):
        asha, bala = self.new_client("asha"), self.new_client("bala")
        self.calibrate(asha)
        asha.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: asha.get("/api/monitor/status").json()["camera_active"]))
        asha.post("/api/auth/logout")
        self.calibrate(bala)
        started = bala.post("/api/monitor/start")
        self.assertIn(started.json()["state"], ("starting", "running"))


if __name__ == "__main__":
    unittest.main()
