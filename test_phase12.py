"""
Phase 12 tests: the /api/monitor/history contract that the dashboard's trend charts depend on.

Phase 12 itself (the charts) is client-side JavaScript (frontend/index.html), which is NOT run by these
Python tests -- it is tested separately with a Node.js "fake browser" harness (see the project's dashboard
test tooling) that exercises the real page script against scripted API responses, checking chronological
ordering, bar counts, and per-metric values. These Python tests check the BACKEND CONTRACT the charts rely
on: every field the six trend charts read must actually be present in the API response, and sessions must
come back newest-first (the chart code reverses this client-side to read oldest-to-newest, left-to-right).

This file also closes a real gap found while building Phase 12: `alert_count` was added to the database and
to monitor.take_session_result() in Phase 11, and a Phase 11 test asserted it appears in the history
response -- but the route's per-row projection never actually included it, so that assertion would have
failed the moment it ran with FastAPI installed. It went undetected here because FastAPI is not available in
this sandbox. `test_every_field_the_trend_charts_need_is_present` below checks the full set in one place, so
a future omission like this is caught by a single, obviously-named test rather than relying on remembering
to check each field individually.

Run from the project root:
    python -m unittest discover -s tests -t . -v
"""

import importlib.util
import time
import unittest
from datetime import datetime, timedelta, timezone

from computer_vision import monitor as monitor_module
from database import sessions as sessions_db
from tests.helpers import FakeCamera, FakeDetector, FakePoseDetector, TempDatabaseTestCase, new_monitor, shoulders_result, wait_for

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)

# Every field one of the six trend charts (session duration, fatigue, posture, attention, DHRI, alerts) reads
# from a history row, per frontend/index.html's renderTrendCharts(). If a field is renamed or removed from the
# route without updating this list (or vice versa), this test's failure is the signal to update the other side.
FIELDS_THE_CHARTS_NEED = {
    "start_time",                                              # x-axis label
    "duration_s",                                              # session duration chart
    "fatigue_warning_count", "fatigue_event_count",             # fatigue chart
    "posture_warning_count", "posture_poor_count",              # posture chart
    "attention_distracted_count", "attention_no_face_count",    # attention chart
    "dhri_moderate_count", "dhri_high_count",                   # DHRI chart
    "alert_count",                                              # alerts chart
}


def a_summary(start_time, **overrides):
    base = {
        "start_time": start_time, "duration_s": 600.0, "active_duration_s": 600.0,
        "break_count": 0, "break_duration_s": 0.0,
        "fatigue_events": {"warning": 1, "fatigued": 0}, "posture_events": {"warning": 0, "poor": 1},
        "attention_events": {"distracted": 1, "no_face": 0}, "dhri_events": {"moderate": 1, "high": 0},
        "alert_count": 3,
    }
    base.update(overrides)
    return base


class HistoryFieldCompletenessTests(TempDatabaseTestCase):
    """Checks the raw database layer returns everything the charts need (no FastAPI required)."""

    def make_user(self):
        from backend import auth_service
        user, _ = auth_service.register(self.connection, "asha", "password123")
        return user["id"]

    def test_every_field_the_trend_charts_need_is_present_in_the_database_row(self):
        user_id = self.make_user()
        sessions_db.save_session(self.connection, user_id, a_summary(datetime(2026, 9, 22, 9, 0, tzinfo=timezone.utc)))
        row = sessions_db.get_recent_sessions(self.connection, user_id)[0]
        missing = FIELDS_THE_CHARTS_NEED - set(row.keys())
        self.assertEqual(missing, set(), f"the database row is missing fields the charts need: {missing}")

    def test_sessions_come_back_newest_first(self):
        # The chart code relies on this: it reverses the list client-side to draw oldest-to-newest, left-to-right.
        user_id = self.make_user()
        for hour in (8, 12, 10):
            sessions_db.save_session(self.connection, user_id, a_summary(datetime(2026, 9, 22, hour, 0, tzinfo=timezone.utc)))
        rows = sessions_db.get_recent_sessions(self.connection, user_id)
        hours = [int(row["start_time"][11:13]) for row in rows]
        self.assertEqual(hours, [12, 10, 8])

    def test_more_than_ten_sessions_can_be_fetched_for_charting(self):
        # Phase 12 raised the dashboard's fetch limit from 10 to 20 sessions so the charts have more history
        # to show; this checks the underlying query itself supports fetching that many.
        user_id = self.make_user()
        for i in range(15):
            sessions_db.save_session(self.connection, user_id,
                                     a_summary(datetime(2026, 9, 22, 0, 0, tzinfo=timezone.utc) + timedelta(hours=i)))
        self.assertEqual(len(sessions_db.get_recent_sessions(self.connection, user_id, limit=20)), 15)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class HistoryApiCompletenessTests(TempDatabaseTestCase):
    """The same completeness/ordering checks, but through the real HTTP route (needs FastAPI)."""

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

    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        client.post("/api/auth/register", json={"username": "asha", "password": "password123"})
        return client

    def test_every_field_the_trend_charts_need_is_present_in_the_api_response(self):
        client = self.client()
        client.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json().get("session", {}).get("duration_s", 0) > 0.1))
        client.post("/api/monitor/stop")
        row = client.get("/api/monitor/history").json()["recent"][0]
        missing = FIELDS_THE_CHARTS_NEED - set(row.keys())
        self.assertEqual(missing, set(), f"the API response is missing fields the charts need: {missing}")

    def test_the_route_can_return_more_than_ten_recent_sessions(self):
        client = self.client()
        for _ in range(12):
            client.post("/api/monitor/start")
            self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json().get("session", {}).get("duration_s", 0) > 0.05))
            client.post("/api/monitor/stop")
        recent = client.get("/api/monitor/history").json()["recent"]
        self.assertEqual(len(recent), 12)


if __name__ == "__main__":
    unittest.main()
