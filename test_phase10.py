"""
Phase 10 tests: the Digital Health Risk Index (DHRI) -- weighted combination, its own escalation smoothing,
database persistence, and the monitor integration.

IMPORTANT: DHRI combines already-computed fatigue/posture/attention/exposure scores. These tests use
hand-built snapshot dicts and a simulated monitor, so they check the WEIGHTING, AVAILABILITY and SMOOTHING
logic works as designed. They say nothing about whether DHRI's weights or thresholds are meaningful risk
indicators for real people -- that is explicitly out of scope, as documented in ml/dhri.py.

Run from the project root:
    python -m unittest discover -s tests -t . -v
"""

import importlib.util
import sqlite3
import time
import unittest
from datetime import datetime, timezone
from unittest import mock

from computer_vision import monitor as monitor_module
from computer_vision.calibrator import Calibrator
from database import connection as db
from database import sessions as sessions_db
from ml.dhri import LOW, MODERATE, HIGH, UNAVAILABLE, DHRIAssessor, _SustainedLevelSmoother, exposure_score
from tests.helpers import FakeCamera, FakeDetector, FakePoseDetector, TempDatabaseTestCase, new_monitor, shoulders_result, wait_for
from utils import config

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)


def snap(state, score):
    """A minimal fatigue/posture/attention-shaped snapshot."""
    return {"state": state, "score": score}


UNAVAIL = {"state": "unavailable", "score": None}


# ======================================================================
# Exposure scoring
# ======================================================================
class ExposureScoreTests(unittest.TestCase):
    def test_none_when_unknown(self):
        self.assertIsNone(exposure_score(None))

    def test_zero_is_zero_risk(self):
        self.assertEqual(exposure_score(0.0), 0.0)

    def test_half_at_the_warning_length_and_one_at_the_high_length(self):
        warning_s = config.DHRI_EXPOSURE_WARNING_MIN * 60.0
        high_s = config.DHRI_EXPOSURE_HIGH_MIN * 60.0
        self.assertAlmostEqual(exposure_score(warning_s), 0.5)
        self.assertAlmostEqual(exposure_score(high_s), 1.0)
        self.assertEqual(exposure_score(high_s * 10), 1.0)          # capped, never above 1.0


# ======================================================================
# Weighting and availability
# ======================================================================
class WeightingTests(unittest.TestCase):
    def test_all_good_gives_a_low_score(self):
        a = DHRIAssessor()
        s = a.update(0.0, snap("normal", 0.0), snap("good", 0.0), snap("attentive", 0.0), continuous_active_s=0.0)
        self.assertEqual(s["score"], 0.0)
        self.assertEqual(s["missing_inputs"], [])

    def test_weights_are_used_directly_when_all_four_are_available(self):
        a = DHRIAssessor()
        s = a.update(0.0, snap("fatigued", 1.0), snap("poor", 1.0), snap("attentive", 1.0), continuous_active_s=0.0)
        # exposure=0 here, so score = 100*(0.40+0.25+0.20) = 85
        self.assertAlmostEqual(s["score"], 85.0, delta=0.1)
        self.assertAlmostEqual(s["contributions"]["fatigue"], 40.0, delta=0.1)
        self.assertAlmostEqual(s["contributions"]["posture"], 25.0, delta=0.1)
        self.assertAlmostEqual(s["contributions"]["attention"], 20.0, delta=0.1)

    def test_missing_inputs_are_reweighted_not_treated_as_zero(self):
        a = DHRIAssessor()
        # only fatigue available, maxed out -- if treated as zero for the rest, score would be 40; instead
        # it should be re-normalised to 100 (fatigue is the only available weight, so it becomes the whole score)
        s = a.update(0.0, snap("fatigued", 1.0), UNAVAIL, UNAVAIL, continuous_active_s=None)
        self.assertAlmostEqual(s["score"], 100.0, delta=0.1)
        self.assertEqual(set(s["missing_inputs"]), {"posture", "attention", "exposure"})
        self.assertEqual(set(s["contributions"]), {"fatigue"})

    def test_two_available_inputs_are_reweighted_between_themselves(self):
        a = DHRIAssessor()
        # fatigue (0.40) and posture (0.25) available, both maxed: renormalised weight is 0.40/0.65 and 0.25/0.65
        s = a.update(0.0, snap("fatigued", 1.0), snap("poor", 1.0), UNAVAIL, continuous_active_s=None)
        self.assertAlmostEqual(s["score"], 100.0, delta=0.1)                 # both maxed -> the whole thing is bad
        self.assertAlmostEqual(s["contributions"]["fatigue"] + s["contributions"]["posture"], 100.0, delta=0.2)
        self.assertAlmostEqual(s["contributions"]["fatigue"] / s["contributions"]["posture"], 0.40 / 0.25, delta=0.05)

    def test_nothing_available_is_unavailable(self):
        a = DHRIAssessor()
        s = a.update(0.0, UNAVAIL, UNAVAIL, UNAVAIL, continuous_active_s=None)
        self.assertEqual(s["level"], UNAVAILABLE)
        self.assertIsNone(s["score"])
        self.assertEqual(set(s["missing_inputs"]), {"fatigue", "posture", "attention", "exposure"})

    def test_idle_snapshot_is_unavailable(self):
        self.assertEqual(DHRIAssessor().idle_snapshot()["level"], UNAVAILABLE)

    def test_a_calibrating_input_counts_as_unavailable(self):
        a = DHRIAssessor()
        calibrating_posture = {"state": "unavailable", "score": None, "message": "Calibration is running."}
        s = a.update(0.0, snap("normal", 0.1), calibrating_posture, snap("attentive", 0.0), continuous_active_s=0.0)
        self.assertIn("posture", s["missing_inputs"])

    def test_none_snapshot_counts_as_unavailable(self):
        a = DHRIAssessor()
        s = a.update(0.0, None, snap("good", 0.0), snap("attentive", 0.0), continuous_active_s=0.0)
        self.assertIn("fatigue", s["missing_inputs"])

    def test_weights_configured_sum_to_one(self):
        self.assertAlmostEqual(sum(config.DHRI_WEIGHTS.values()), 1.0)

    def test_risk_thresholds_are_ordered(self):
        self.assertLess(config.DHRI_LOW_MAX, config.DHRI_MODERATE_MAX)

    def test_score_boundaries_map_to_the_correct_risk_level(self):
        # Direct test of the LOW/MODERATE/HIGH boundaries themselves (crafted fatigue-only scores land exactly
        # on each side of the two configured thresholds, since with only fatigue available score100 == fatigue*100).
        low_max, moderate_max = config.DHRI_LOW_MAX, config.DHRI_MODERATE_MAX
        cases = [
            (low_max / 100 - 0.005, LOW), (low_max / 100 + 0.005, MODERATE),
            (moderate_max / 100 - 0.005, MODERATE), (moderate_max / 100 + 0.005, HIGH),
        ]
        for fatigue_value, expected in cases:
            a = DHRIAssessor()
            # settle past both hold times so the smoothed level reflects the raw classification directly
            snapshot = None
            for i in range(200):
                snapshot = a.update(i * 0.1, snap("fatigued", fatigue_value), UNAVAIL, UNAVAIL, continuous_active_s=None)
            self.assertEqual(snapshot["level"], expected, (fatigue_value, snapshot["score"]))


# ======================================================================
# DHRI's own sustained-level smoothing (see ml/dhri.py for why it differs from utils/smoothing.py)
# ======================================================================
class SustainedLevelSmootherTests(unittest.TestCase):
    def new(self, escalate=5.0, recovery=5.0):
        return _SustainedLevelSmoother(["low", "moderate", "high"], escalate, recovery)

    def test_starts_at_the_first_level(self):
        self.assertEqual(self.new().shown, "low")

    def test_a_sustained_worse_level_is_shown_after_its_hold_time(self):
        s = self.new(escalate=3.0)
        s.update(0.0, "high")
        self.assertEqual(s.update(2.9, "high"), "low")
        self.assertEqual(s.update(3.0, "high"), "high")

    def test_a_momentary_flicker_never_promotes(self):
        s = self.new(escalate=3.0)
        s.update(0.0, "high")
        s.update(1.0, "low")          # flickers back before the hold time
        self.assertEqual(s.update(5.0, "low"), "low")

    def test_ramping_through_an_intermediate_level_then_settling_takes_one_hold_period_not_two(self):
        # Regression test: found via a scripted end-to-end run before Phase 10 shipped. A value that ramps
        # low -> moderate -> high and then STAYS at high must reach "high" ~escalate_hold_s after it settles,
        # not ~2x escalate_hold_s (one hold period per level crossed on the way up).
        s = self.new(escalate=5.0)
        s.update(0.0, "low")
        s.update(1.0, "moderate")      # ramping
        s.update(1.2, "high")          # now settled
        for t in (2.0, 4.0, 6.0, 6.15, 6.25):
            result = s.update(t, "high")
        self.assertEqual(result, "high")
        # and it must NOT have already reached "high" well before one hold period after settling at t=1.2
        s2 = self.new(escalate=5.0)
        s2.update(0.0, "low")
        s2.update(1.0, "moderate")
        s2.update(1.2, "high")
        self.assertEqual(s2.update(5.5, "high"), "low")          # only ~4.3 s since settling: not yet

    def test_recovery_uses_its_own_hold_time(self):
        s = self.new(escalate=1.0, recovery=8.0)
        s.update(0.0, "high")
        s.update(1.0, "high")              # escalated to "high"
        self.assertEqual(s.shown, "high")
        s.update(2.0, "low")
        self.assertEqual(s.update(9.9, "low"), "high")           # 7.9 s of "low": not yet (needs 8.0)
        self.assertEqual(s.update(10.0, "low"), "low")

    def test_a_new_candidate_restarts_the_clock_even_between_two_worse_levels(self):
        s = self.new(escalate=3.0)
        s.update(0.0, "moderate")
        s.update(2.0, "high")              # a DIFFERENT raw level: must restart, not continue moderate's clock
        self.assertEqual(s.update(4.5, "high"), "low")            # only 2.5 s at "high"
        self.assertEqual(s.update(5.0, "high"), "high")           # 3.0 s at "high": now it is shown

    def test_reset(self):
        s = self.new()
        s.update(0.0, "high")
        s.reset()
        self.assertEqual(s.shown, "low")


# ======================================================================
# Monitor integration
# ======================================================================
class MonitorDHRITests(unittest.TestCase):
    def setUp(self):
        self._monitors = []

    def tearDown(self):
        for monitor in self._monitors:
            monitor.stop()

    def build(self, **options):
        camera = FakeCamera()
        detector = FakeDetector(ear=0.30, head={"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "distance": 2500.0})
        options.setdefault("calibrator_factory", lambda: Calibrator(ready_s=1.0, open_s=2.5, close_s=2.0))
        monitor = new_monitor(lambda: camera, detector_factory=lambda: detector,
                              pose_detector_factory=lambda: FakePoseDetector(result=shoulders_result()), **options)
        self._monitors.append(monitor)
        monitor.start(1)
        monitor.begin_calibration(1)
        end = time.monotonic() + 10
        while time.monotonic() < end:
            status = monitor.get_calibration_status(1)
            if status is not None:
                detector.ear = 0.10 if status["phase"] == "close" else 0.30
                if not status["active"]:
                    break
            time.sleep(0.02)
        monitor.take_calibration_result(1)
        return monitor, camera, detector

    def test_dhri_appears_after_calibration_and_warm_up(self):
        monitor, _, _ = self.build()
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("dhri", {}).get("level") not in (None, "unavailable"), timeout=15))
        d = monitor.get_status(1)["dhri"]
        for key in ("level", "score", "contributions", "missing_inputs", "events", "disclaimer"):
            self.assertIn(key, d)

    def test_dhri_rises_when_fatigue_and_posture_are_bad_and_recovers(self):
        with mock.patch.object(config, "DHRI_ESCALATE_HOLD_S", 1.0), mock.patch.object(config, "DHRI_RECOVERY_HOLD_S", 1.0):
            monitor, _, detector = self.build()
            self.assertTrue(wait_for(lambda: monitor.get_status(1)["dhri"]["level"] == "low", timeout=15))
            detector.head = {"roll": 25.0, "pitch": 0.0, "yaw": 0.0, "distance": 2500.0}
            detector.ear = 0.10
            self.assertTrue(wait_for(lambda: monitor.get_status(1)["dhri"]["level"] in ("moderate", "high"), timeout=10))
            self.assertGreater(monitor.get_status(1)["dhri"]["score"], config.DHRI_LOW_MAX)
            detector.head = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "distance": 2500.0}
            detector.ear = 0.30
            self.assertTrue(wait_for(lambda: monitor.get_status(1)["dhri"]["level"] == "low", timeout=15))

    def test_the_dhri_events_counter_is_reflected_in_the_session_result(self):
        with mock.patch.object(config, "DHRI_ESCALATE_HOLD_S", 0.5):
            monitor, _, detector = self.build()
            self.assertTrue(wait_for(lambda: monitor.get_status(1)["dhri"]["level"] == "low", timeout=15))
            detector.head = {"roll": 25.0, "pitch": 0.0, "yaw": 0.0, "distance": 2500.0}
            detector.ear = 0.10
            self.assertTrue(wait_for(lambda: monitor.get_status(1)["dhri"]["level"] in ("moderate", "high"), timeout=10))
            monitor.stop()
            result = monitor.take_session_result(1)
            self.assertIn("dhri_events", result)
            self.assertGreater(result["dhri_events"]["moderate"] + result["dhri_events"]["high"], 0)

    def test_not_yet_calibrated_dhri_is_unavailable(self):
        camera = FakeCamera()
        detector = FakeDetector(ear=0.30)
        monitor = new_monitor(lambda: camera, detector_factory=lambda: detector)
        self._monitors.append(monitor)
        monitor.start(1)
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("dhri", {}).get("level") == "unavailable", timeout=15))
        self.assertIn("fatigue", monitor.get_status(1)["dhri"]["missing_inputs"])

    def test_another_account_never_sees_dhri(self):
        monitor, _, _ = self.build()
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("dhri", {}).get("level")))
        self.assertNotIn("dhri", monitor.get_status(2))


# ======================================================================
# Database
# ======================================================================
class DHRIMigrationTests(unittest.TestCase):
    def test_migration_adds_the_dhri_columns(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute("""CREATE TABLE monitoring_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            start_time TEXT NOT NULL, end_time TEXT, duration_s REAL)""")
        db.migrate_schema(connection)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(monitoring_sessions)")}
        self.assertIn("dhri_moderate_count", columns)
        self.assertIn("dhri_high_count", columns)

    def test_migration_from_a_phase_9_style_database_also_gets_the_dhri_columns(self):
        # Simulates a database already upgraded to Phase 9 (has the Phase 9 columns) but not Phase 10.
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        columns_sql = ", ".join(f"{c} {t}" for c, t in db._SESSION_COLUMNS_ADDED_IN_PHASE_9.items())
        connection.execute(f"""CREATE TABLE monitoring_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            start_time TEXT NOT NULL, end_time TEXT, duration_s REAL, {columns_sql})""")
        db.migrate_schema(connection)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(monitoring_sessions)")}
        self.assertIn("dhri_moderate_count", columns)
        self.assertIn("active_duration_s", columns)             # the Phase 9 columns are still there


def a_summary(**overrides):
    base = {
        "start_time": datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc),
        "duration_s": 600.0, "active_duration_s": 600.0, "break_count": 0, "break_duration_s": 0.0,
        "fatigue_events": {"warning": 0, "fatigued": 0}, "posture_events": {"warning": 0, "poor": 0},
        "attention_events": {"distracted": 0, "no_face": 0}, "dhri_events": {"moderate": 1, "high": 2},
    }
    base.update(overrides)
    return base


class DHRIStorageTests(TempDatabaseTestCase):
    def make_user(self):
        from backend import auth_service
        user, _ = auth_service.register(self.connection, "asha", "password123")
        return user["id"]

    def test_dhri_event_counts_are_saved(self):
        user_id = self.make_user()
        sessions_db.save_session(self.connection, user_id, a_summary())
        row = sessions_db.get_recent_sessions(self.connection, user_id)[0]
        self.assertEqual(row["dhri_moderate_count"], 1)
        self.assertEqual(row["dhri_high_count"], 2)

    def test_missing_dhri_events_defaults_to_zero(self):
        user_id = self.make_user()
        summary = a_summary()
        del summary["dhri_events"]
        sessions_db.save_session(self.connection, user_id, summary)          # must not raise
        row = sessions_db.get_recent_sessions(self.connection, user_id)[0]
        self.assertEqual(row["dhri_moderate_count"], 0)
        self.assertEqual(row["dhri_high_count"], 0)


# ======================================================================
# API
# ======================================================================
@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class DHRIApiTests(TempDatabaseTestCase):
    def setUp(self):
        super().setUp()
        self._original_monitor = monitor_module.monitor
        self.camera = FakeCamera()
        self.detector = FakeDetector(ear=0.30)
        monitor_module.monitor = new_monitor(
            lambda: self.camera, detector_factory=lambda: self.detector,
            pose_detector_factory=lambda: FakePoseDetector(result=shoulders_result()),
            calibrator_factory=lambda: Calibrator(ready_s=1.0, open_s=2.5, close_s=2.0))

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

    def test_dhri_is_in_the_status_response(self):
        client = self.client()
        client.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json().get("dhri", {}).get("level"), timeout=6))
        body = client.get("/api/monitor/status").json()
        self.assertIn("dhri", body)
        for key in ("level", "score", "missing_inputs", "disclaimer"):
            self.assertIn(key, body["dhri"])

    def test_dhri_counts_appear_in_saved_history(self):
        client = self.client()
        client.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json().get("session", {}).get("duration_s", 0) > 0.1))
        client.post("/api/monitor/stop")
        row = client.get("/api/monitor/history").json()["recent"][0]
        self.assertIn("dhri_moderate_count", row)
        self.assertIn("dhri_high_count", row)


if __name__ == "__main__":
    unittest.main()
