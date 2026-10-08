"""
Phase 9 tests: session/exposure tracking (SessionTracker), the database migration and storage for sessions,
and the monitor + API integration.

Run from the project root:
    python -m unittest discover -s tests -t . -v
"""

import importlib.util
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from computer_vision import monitor as monitor_module
from computer_vision.session import SessionSummary, SessionTracker
from database import connection as db
from database import sessions as sessions_db
from tests.helpers import FakeCamera, FakeDetector, FakePoseDetector, TempDatabaseTestCase, new_monitor, shoulders_result, wait_for
from utils import config

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)


# ======================================================================
# SessionTracker
# ======================================================================
class SessionTrackerTests(unittest.TestCase):
    def test_fully_present_session_has_no_breaks(self):
        t = SessionTracker(break_min_s=10.0)
        for i in range(20):
            t.update(i * 1.0, True)
        t.finalize(20.0)
        s = t.summary(20.0)
        self.assertEqual(s.break_count, 0)
        self.assertEqual(s.break_duration_s, 0.0)
        self.assertEqual(s.active_duration_s, 20.0)
        self.assertEqual(s.duration_s, 20.0)

    def test_a_short_absence_is_not_a_break(self):
        t = SessionTracker(break_min_s=10.0)
        t.update(0.0, True)
        t.update(5.0, False)     # away
        t.update(8.0, True)      # back after only 3 s -- not long enough
        t.finalize(10.0)
        s = t.summary(10.0)
        self.assertEqual(s.break_count, 0)
        self.assertEqual(s.active_duration_s, 10.0)   # nothing was subtracted

    def test_a_long_absence_is_one_break(self):
        t = SessionTracker(break_min_s=10.0)
        t.update(0.0, True)
        t.update(5.0, False)
        t.update(20.0, True)     # away for 15 s -- a break
        t.finalize(25.0)
        s = t.summary(25.0)
        self.assertEqual(s.break_count, 1)
        self.assertAlmostEqual(s.break_duration_s, 15.0)
        self.assertAlmostEqual(s.active_duration_s, 10.0)

    def test_multiple_breaks_are_all_counted(self):
        t = SessionTracker(break_min_s=5.0)
        for start, end, present in ((0, 10, True), (10, 20, False), (20, 30, True), (30, 40, False), (40, 45, True)):
            tt = start
            while tt < end:
                t.update(tt, present)
                tt += 1.0
        t.finalize(45.0)
        s = t.summary(45.0)
        self.assertEqual(s.break_count, 2)
        self.assertAlmostEqual(s.break_duration_s, 20.0, delta=1.5)

    def test_an_absence_still_in_progress_when_the_session_ends_still_counts_if_long_enough(self):
        t = SessionTracker(break_min_s=5.0)
        t.update(0.0, True)
        t.update(3.0, False)      # away, never comes back before the session ends
        t.finalize(20.0)          # 17 s away when finalized
        s = t.summary(20.0)
        self.assertEqual(s.break_count, 1)
        self.assertAlmostEqual(s.break_duration_s, 17.0)

    def test_a_short_final_absence_is_not_counted_even_at_finalize(self):
        t = SessionTracker(break_min_s=10.0)
        t.update(0.0, True)
        t.update(18.0, False)
        t.finalize(20.0)          # only 2 s away
        s = t.summary(20.0)
        self.assertEqual(s.break_count, 0)
        self.assertAlmostEqual(s.active_duration_s, 20.0)

    def test_live_summary_reflects_an_in_progress_absence_before_it_is_confirmed_as_a_break(self):
        t = SessionTracker(break_min_s=10.0)
        t.update(0.0, True)
        t.update(5.0, False)
        live = t.summary(8.0)                            # 3 s into an absence, not yet a "break"
        self.assertEqual(live.break_count, 0)             # not confirmed yet
        self.assertAlmostEqual(live.active_duration_s, 5.0)   # but it is not counted as active either

    def test_default_threshold_comes_from_config(self):
        t = SessionTracker()
        self.assertEqual(t.break_min_s, config.SESSION_BREAK_MIN_S)

    def test_summary_is_a_plain_dataclass(self):
        s = SessionTracker().summary(5.0)
        self.assertIsInstance(s, SessionSummary)
        self.assertEqual(s.duration_s, 5.0)


# ======================================================================
# Database: migration
# ======================================================================
class MigrationTests(unittest.TestCase):
    def make_old_style_db(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute("""CREATE TABLE monitoring_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            start_time TEXT NOT NULL, end_time TEXT, duration_s REAL)""")
        connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)")
        connection.execute("INSERT INTO users VALUES (1, 'asha')")
        connection.execute("INSERT INTO monitoring_sessions (user_id, start_time) VALUES (1, 'old-row')")
        connection.commit()
        return connection

    def test_migration_adds_every_new_column(self):
        connection = self.make_old_style_db()
        db.migrate_schema(connection)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(monitoring_sessions)")}
        for column in ("active_duration_s", "break_count", "break_duration_s", "fatigue_warning_count",
                      "fatigue_event_count", "posture_warning_count", "posture_poor_count",
                      "attention_distracted_count", "attention_no_face_count", "alert_count"):
            self.assertIn(column, columns)

    def test_migration_preserves_existing_rows(self):
        connection = self.make_old_style_db()
        db.migrate_schema(connection)
        row = connection.execute("SELECT * FROM monitoring_sessions").fetchone()
        self.assertEqual(row["user_id"], 1)
        self.assertEqual(row["start_time"], "old-row")

    def test_migration_can_run_more_than_once_without_error(self):
        connection = self.make_old_style_db()
        db.migrate_schema(connection)
        db.migrate_schema(connection)          # a second server start-up: must not raise or duplicate data
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM monitoring_sessions").fetchone()[0], 1)

    def test_a_fresh_database_already_has_the_columns_and_migration_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(config, "DATABASE_PATH", Path(folder) / "fresh.db"):
                db.init_db()             # runs the real schema.sql AND migrate_schema together
                connection = db.connect()
                try:
                    columns = {row["name"] for row in connection.execute("PRAGMA table_info(monitoring_sessions)")}
                    self.assertIn("active_duration_s", columns)
                finally:
                    connection.close()


# ======================================================================
# Database: saving and reading sessions
# ======================================================================
def a_summary(**overrides):
    base = {
        "start_time": datetime(2026, 9, 20, 9, 0, 0, tzinfo=timezone.utc),
        "duration_s": 1800.0, "active_duration_s": 1650.0, "break_count": 2, "break_duration_s": 150.0,
        "fatigue_events": {"warning": 1, "fatigued": 0},
        "posture_events": {"warning": 2, "poor": 1},
        "attention_events": {"distracted": 3, "no_face": 1},
    }
    base.update(overrides)
    return base


class SaveDecisionTests(unittest.TestCase):
    def test_a_zero_length_session_should_not_be_saved(self):
        self.assertFalse(sessions_db.should_save_session(a_summary(duration_s=0.0)))

    def test_none_should_not_be_saved(self):
        self.assertFalse(sessions_db.should_save_session(None))

    def test_a_real_session_should_be_saved(self):
        self.assertTrue(sessions_db.should_save_session(a_summary(duration_s=0.1)))


class SessionStorageTests(TempDatabaseTestCase):
    def make_user(self, name="asha"):
        from backend import auth_service
        user, _ = auth_service.register(self.connection, name, "password123")
        return user["id"]

    def test_save_and_read_back_a_session(self):
        user_id = self.make_user()
        sessions_db.save_session(self.connection, user_id, a_summary())
        rows = sessions_db.get_recent_sessions(self.connection, user_id)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["duration_s"], 1800.0)
        self.assertEqual(row["active_duration_s"], 1650.0)
        self.assertEqual(row["break_count"], 2)
        self.assertEqual(row["break_duration_s"], 150.0)
        self.assertEqual(row["fatigue_warning_count"], 1)
        self.assertEqual(row["fatigue_event_count"], 0)
        self.assertEqual(row["posture_warning_count"], 2)
        self.assertEqual(row["posture_poor_count"], 1)
        self.assertEqual(row["attention_distracted_count"], 3)
        self.assertEqual(row["attention_no_face_count"], 1)
        self.assertEqual(row["alert_count"], 0)

    def test_end_time_is_start_time_plus_duration(self):
        user_id = self.make_user()
        start = datetime(2026, 9, 20, 9, 0, 0, tzinfo=timezone.utc)
        sessions_db.save_session(self.connection, user_id, a_summary(start_time=start, duration_s=600.0))
        row = sessions_db.get_recent_sessions(self.connection, user_id)[0]
        self.assertEqual(row["end_time"], (start + timedelta(seconds=600)).isoformat())

    def test_recent_sessions_come_back_newest_first(self):
        user_id = self.make_user()
        for hour in (8, 10, 9):
            sessions_db.save_session(self.connection, user_id,
                                     a_summary(start_time=datetime(2026, 9, 20, hour, 0, tzinfo=timezone.utc)))
        rows = sessions_db.get_recent_sessions(self.connection, user_id)
        hours = [int(row["start_time"][11:13]) for row in rows]
        self.assertEqual(hours, [10, 9, 8])

    def test_recent_sessions_respects_the_limit(self):
        user_id = self.make_user()
        for i in range(5):
            sessions_db.save_session(self.connection, user_id,
                                     a_summary(start_time=datetime(2026, 9, 20, 8, i, tzinfo=timezone.utc)))
        self.assertEqual(len(sessions_db.get_recent_sessions(self.connection, user_id, limit=3)), 3)

    def test_each_user_only_sees_their_own_sessions(self):
        first, second = self.make_user("asha"), self.make_user("bala")
        sessions_db.save_session(self.connection, first, a_summary())
        self.assertEqual(len(sessions_db.get_recent_sessions(self.connection, first)), 1)
        self.assertEqual(len(sessions_db.get_recent_sessions(self.connection, second)), 0)

    def test_today_summary_totals_only_todays_sessions(self):
        user_id = self.make_user()
        now = datetime(2026, 9, 20, 15, 0, 0, tzinfo=timezone.utc)
        sessions_db.save_session(self.connection, user_id, a_summary(
            start_time=datetime(2026, 9, 20, 9, 0, tzinfo=timezone.utc), duration_s=1000.0,
            active_duration_s=900.0, break_count=1))
        sessions_db.save_session(self.connection, user_id, a_summary(
            start_time=datetime(2026, 9, 20, 13, 0, tzinfo=timezone.utc), duration_s=500.0,
            active_duration_s=450.0, break_count=2))
        sessions_db.save_session(self.connection, user_id, a_summary(              # yesterday: must be excluded
            start_time=datetime(2026, 9, 19, 23, 0, tzinfo=timezone.utc), duration_s=99999.0, break_count=99))
        today = sessions_db.get_today_summary(self.connection, user_id, now=now)
        self.assertEqual(today["session_count"], 2)
        self.assertAlmostEqual(today["duration_s"], 1500.0)
        self.assertAlmostEqual(today["active_duration_s"], 1350.0)
        self.assertEqual(today["break_count"], 3)

    def test_today_summary_is_all_zero_with_no_sessions(self):
        user_id = self.make_user()
        today = sessions_db.get_today_summary(self.connection, user_id)
        self.assertEqual(today["session_count"], 0)
        self.assertEqual(today["duration_s"], 0)
        self.assertEqual(today["break_count"], 0)


# ======================================================================
# Monitor integration
# ======================================================================
class MonitorSessionTests(unittest.TestCase):
    def setUp(self):
        self._monitors = []

    def tearDown(self):
        for monitor in self._monitors:
            monitor.stop()

    def build(self, break_min_s=None, **options):
        camera = FakeCamera()
        detector = FakeDetector(ear=0.30, head={"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "distance": 2500.0})
        with mock.patch.object(config, "SESSION_BREAK_MIN_S", break_min_s if break_min_s is not None else config.SESSION_BREAK_MIN_S):
            monitor = new_monitor(lambda: camera, detector_factory=lambda: detector,
                                  pose_detector_factory=lambda: FakePoseDetector(result=shoulders_result()), **options)
            self._monitors.append(monitor)
            monitor.start(1)
        return monitor, camera, detector

    def test_live_session_status_appears_and_grows(self):
        monitor, _, _ = self.build()
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("session", {}).get("duration_s", 0) > 0))
        first = monitor.get_status(1)["session"]["duration_s"]
        time.sleep(0.5)
        self.assertGreater(monitor.get_status(1)["session"]["duration_s"], first)

    def test_stopping_produces_a_one_shot_result_with_a_start_time(self):
        monitor, _, _ = self.build()
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("session", {}).get("duration_s", 0) > 0))
        monitor.stop()
        result = monitor.take_session_result(1)
        self.assertIsNotNone(result)
        self.assertIn("start_time", result)
        self.assertGreater(result["duration_s"], 0)
        self.assertIsNone(monitor.take_session_result(1))     # one-shot: gone after the first collection

    def test_a_break_is_detected_end_to_end(self):
        monitor, camera, detector = self.build(break_min_s=1.0)
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("session", {}).get("duration_s", 0) > 0.3))
        detector.faces = []
        time.sleep(1.6)
        detector.faces = [(0.5, 0.5, 0.4)]
        self.assertTrue(wait_for(lambda: monitor.get_status(1)["session"]["break_count"] == 1, timeout=4))
        monitor.stop()
        result = monitor.take_session_result(1)
        self.assertEqual(result["break_count"], 1)
        self.assertGreater(result["break_duration_s"], 1.0)

    def test_covering_the_camera_also_counts_as_absence_for_breaks(self):
        monitor, camera, _ = self.build(break_min_s=1.0)
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("session", {}).get("duration_s", 0) > 0.3))
        camera.gray_level = 3
        time.sleep(1.6)
        camera.gray_level = 120
        self.assertTrue(wait_for(lambda: monitor.get_status(1)["session"]["break_count"] == 1, timeout=4))

    def test_event_counts_are_pulled_from_the_last_assessor_snapshots(self):
        monitor, _, detector = self.build()
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("eye", {}).get("ear") is not None))
        detector.head = {"roll": 25.0, "pitch": 0.0, "yaw": 0.0, "distance": 2500.0}   # (no posture baseline yet: no effect, just exercising the path)
        time.sleep(0.3)
        monitor.stop()
        result = monitor.take_session_result(1)
        for key in ("fatigue_events", "posture_events", "attention_events"):
            self.assertIn(key, result)
            self.assertIsInstance(result[key], dict)

    def test_stopping_immediately_still_returns_a_usable_zero_length_result(self):
        monitor, _, _ = self.build()
        monitor.stop()      # stop almost immediately, possibly before any frame was processed
        result = monitor.take_session_result(1)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["duration_s"], 0.0)
        self.assertEqual(result["break_count"], 0)

    def test_the_idle_timeout_still_produces_a_session_result(self):
        monitor, _, _ = self.build()
        with mock.patch.object(monitor, "_idle_timeout_s", 0.3):
            self.assertTrue(wait_for(lambda: monitor.get_status(1)["state"] == "stopped", timeout=4))
        result = monitor.take_session_result(1)
        self.assertIsNotNone(result)
        self.assertGreater(result["duration_s"], 0)

    def test_a_camera_lost_error_still_produces_a_session_result(self):
        monitor, camera, _ = self.build()
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("session", {}).get("duration_s", 0) > 0.1))
        camera.fail_after_reads = 0             # every subsequent read returns None
        self.assertTrue(wait_for(lambda: monitor.get_status(1)["state"] == "error", timeout=4))
        result = monitor.take_session_result(1)
        self.assertIsNotNone(result)

    def test_a_second_account_cannot_collect_someone_elses_session_result(self):
        monitor, _, _ = self.build()
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("session", {}).get("duration_s", 0) > 0.1))
        monitor.stop()
        self.assertIsNone(monitor.take_session_result(2))
        self.assertIsNotNone(monitor.take_session_result(1))    # still there for the real owner

    def test_a_new_session_starts_its_own_tracking_from_zero(self):
        monitor, camera, detector = self.build(break_min_s=1.0)
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("session", {}).get("duration_s", 0) > 0.3))
        monitor.stop()
        monitor.take_session_result(1)
        monitor.start(1)
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("session", {}).get("duration_s", 1) < 0.5))
        self.assertEqual(monitor.get_status(1)["session"]["break_count"], 0)

    def test_a_camera_setup_failure_produces_no_session_result_at_all(self):
        monitor = new_monitor(lambda: FakeCamera(fail_open=True))
        self._monitors.append(monitor)
        monitor.start(1)
        self.assertIsNone(monitor.take_session_result(1))    # the camera never actually ran


# ======================================================================
# API
# ======================================================================
@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class SessionApiTests(TempDatabaseTestCase):
    def setUp(self):
        super().setUp()
        self._original_monitor = monitor_module.monitor
        self.camera = FakeCamera()
        self.detector = FakeDetector(ear=0.30)
        monitor_module.monitor = new_monitor(lambda: self.camera, detector_factory=lambda: self.detector,
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

    def test_stopping_saves_a_session_row(self):
        client = self.client()
        client.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json().get("session", {}).get("duration_s", 0) > 0.1))
        client.post("/api/monitor/stop")
        history = client.get("/api/monitor/history").json()
        self.assertEqual(len(history["recent"]), 1)
        self.assertGreater(history["recent"][0]["duration_s"], 0)
        self.assertEqual(history["today"]["session_count"], 1)

    def test_stopping_without_ever_starting_saves_nothing(self):
        client = self.client()
        client.post("/api/monitor/stop")     # never started
        history = client.get("/api/monitor/history").json()
        self.assertEqual(len(history["recent"]), 0)
        self.assertEqual(history["today"]["session_count"], 0)

    def test_two_stop_calls_in_a_row_save_only_one_session(self):
        client = self.client()
        client.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json().get("session", {}).get("duration_s", 0) > 0.1))
        client.post("/api/monitor/stop")
        client.post("/api/monitor/stop")     # pressing Stop twice, or a retried request
        history = client.get("/api/monitor/history").json()
        self.assertEqual(len(history["recent"]), 1)

    def test_login_is_required_for_history(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        self.assertEqual(TestClient(app).get("/api/monitor/history").status_code, 401)

    def test_each_users_history_is_separate(self):
        first, second = self.client("asha"), self.client("bala")
        first.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: first.get("/api/monitor/status").json().get("session", {}).get("duration_s", 0) > 0.1))
        first.post("/api/monitor/stop")
        self.assertEqual(len(first.get("/api/monitor/history").json()["recent"]), 1)
        self.assertEqual(len(second.get("/api/monitor/history").json()["recent"]), 0)

    def test_another_account_stopping_does_not_save_the_owners_session(self):
        first, second = self.client("asha"), self.client("bala")
        first.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: first.get("/api/monitor/status").json().get("session", {}).get("duration_s", 0) > 0.1))
        second.post("/api/monitor/stop")     # does not own the camera: must do nothing
        self.assertTrue(first.get("/api/monitor/status").json()["camera_active"])
        self.assertEqual(len(second.get("/api/monitor/history").json()["recent"]), 0)
        first.post("/api/monitor/stop")
        self.assertEqual(len(first.get("/api/monitor/history").json()["recent"]), 1)


if __name__ == "__main__":
    unittest.main()
