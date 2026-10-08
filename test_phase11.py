"""
Phase 11 tests: alerts (cooldown-gated notifications) and recommendations (always-current advice), the
monitor integration, and session persistence of alert_count.

IMPORTANT: alert/recommendation CONDITIONS are read from already-smoothed fatigue/posture/attention/DHRI
states, which Phases 6-10 already test extensively. These tests check the ALERT/RECOMMENDATION LOGIC itself
(cooldowns fire and expire correctly, the right message for the right condition, recommendations track
current state with no cooldown) using hand-built snapshot dicts, not real camera data.

Run from the project root:
    python -m unittest discover -s tests -t . -v
"""

import importlib.util
import time
import unittest
from datetime import datetime, timezone
from unittest import mock

from computer_vision import monitor as monitor_module
from computer_vision.calibrator import Calibrator
from database import sessions as sessions_db
from ml.alerts import ALERT_MESSAGES, AlertManager, _alert_conditions, _recommendations
from tests.helpers import FakeCamera, FakeDetector, FakePoseDetector, TempDatabaseTestCase, new_monitor, shoulders_result, wait_for
from utils import config

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)

GOOD_FATIGUE = {"state": "normal", "score": 0.1, "reasons": []}
GOOD_POSTURE = {"state": "good", "score": 0.0}
GOOD_ATTENTION = {"state": "attentive", "score": 0.0}
GOOD_DHRI = {"level": "low", "score": 5.0}

CLOSURE_FATIGUE = {"state": "fatigued", "score": 1.0,
                   "reasons": ["Longest eye closure 1.4 s (warning 0.5 s, fatigue 1.0 s)"]}
BLINK_RATE_FATIGUE = {"state": "fatigued", "score": 1.0,
                      "reasons": ["Frequent blinking, 50 per minute (warning 30, fatigue 45)"]}
WARNING_FATIGUE = {"state": "warning", "score": 0.6, "reasons": ["Frequent blinking, 35 per minute (warning 30, fatigue 45)"]}
POOR_POSTURE = {"state": "poor", "score": 1.0}
WARNING_POSTURE = {"state": "warning", "score": 0.6}
DISTRACTED = {"state": "distracted", "score": 0.6}
HIGH_DHRI = {"level": "high", "score": 80.0}


# ======================================================================
# Alert conditions (pure function)
# ======================================================================
class AlertConditionTests(unittest.TestCase):
    def all_good(self, **overrides):
        base = dict(fatigue=GOOD_FATIGUE, posture=GOOD_POSTURE, attention=GOOD_ATTENTION, dhri=GOOD_DHRI,
                   continuous_active_s=0.0)
        base.update(overrides)
        return _alert_conditions(**base)

    def test_all_good_triggers_nothing(self):
        self.assertFalse(any(self.all_good().values()))

    def test_take_a_break_only_from_dhri_high(self):
        self.assertFalse(self.all_good()["take_a_break"])
        self.assertTrue(self.all_good(dhri=HIGH_DHRI)["take_a_break"])
        self.assertFalse(self.all_good(dhri={"level": "moderate", "score": 50.0})["take_a_break"])

    def test_eye_closure_needs_fatigued_state_AND_a_closure_reason(self):
        self.assertTrue(self.all_good(fatigue=CLOSURE_FATIGUE)["eye_closure"])
        self.assertFalse(self.all_good(fatigue=BLINK_RATE_FATIGUE)["eye_closure"])   # fatigued, but NOT from closure
        self.assertFalse(self.all_good(fatigue=WARNING_FATIGUE)["eye_closure"])      # closure-worded but only warning

    def test_posture_only_from_poor_not_warning(self):
        self.assertTrue(self.all_good(posture=POOR_POSTURE)["posture"])
        self.assertFalse(self.all_good(posture=WARNING_POSTURE)["posture"])

    def test_exposure_uses_the_high_length_not_the_warning_length(self):
        warning_s = config.DHRI_EXPOSURE_WARNING_MIN * 60.0
        high_s = config.DHRI_EXPOSURE_HIGH_MIN * 60.0
        self.assertFalse(self.all_good(continuous_active_s=warning_s)["exposure"])
        self.assertTrue(self.all_good(continuous_active_s=high_s)["exposure"])
        self.assertFalse(self.all_good(continuous_active_s=None)["exposure"])

    def test_attention_only_from_distracted(self):
        self.assertTrue(self.all_good(attention=DISTRACTED)["attention"])
        self.assertFalse(self.all_good(attention={"state": "no_face", "score": 1.0})["attention"])

    def test_missing_snapshots_do_not_crash(self):
        conditions = _alert_conditions(None, None, None, None, None)
        self.assertFalse(any(conditions.values()))

    def test_every_condition_has_a_message(self):
        conditions = self.all_good()
        for alert_id in conditions:
            self.assertIn(alert_id, ALERT_MESSAGES)
            self.assertIn(alert_id, config.ALERT_COOLDOWN_S)


# ======================================================================
# Recommendations (pure function, no cooldown)
# ======================================================================
class RecommendationTests(unittest.TestCase):
    def recs(self, **overrides):
        base = dict(fatigue=GOOD_FATIGUE, posture=GOOD_POSTURE, attention=GOOD_ATTENTION, dhri=GOOD_DHRI,
                   continuous_active_s=0.0)
        base.update(overrides)
        return _recommendations(**base)

    def test_all_good_gives_no_recommendations(self):
        self.assertEqual(self.recs(), [])

    def test_general_fatigue_warning_recommends_a_short_break(self):
        self.assertEqual(self.recs(fatigue=WARNING_FATIGUE), ["Consider taking a short break."])

    def test_closure_based_fatigue_recommends_resting_the_eyes_specifically(self):
        result = self.recs(fatigue=CLOSURE_FATIGUE)
        self.assertIn("Rest your eyes for a short period.", result)
        self.assertNotIn("Consider taking a short break.", result)     # the more specific tip replaces the general one

    def test_non_closure_fatigued_state_still_gets_the_general_recommendation(self):
        self.assertEqual(self.recs(fatigue=BLINK_RATE_FATIGUE), ["Consider taking a short break."])

    def test_poor_posture_recommends_adjusting_it(self):
        self.assertIn("Adjust your sitting position and keep your back and neck aligned.", self.recs(posture=POOR_POSTURE))

    def test_warning_posture_gives_no_posture_recommendation(self):
        self.assertEqual(self.recs(posture=WARNING_POSTURE), [])

    def test_exposure_recommendation_appears_at_the_warning_length_earlier_than_the_alert(self):
        warning_s = config.DHRI_EXPOSURE_WARNING_MIN * 60.0
        self.assertIn("Consider taking a screen break.", self.recs(continuous_active_s=warning_s))
        self.assertEqual(self.recs(continuous_active_s=warning_s - 1), [])

    def test_distracted_attention_recommends_refocusing(self):
        self.assertIn("Try to reduce distractions and refocus on your screen.", self.recs(attention=DISTRACTED))

    def test_high_dhri_recommends_a_longer_break(self):
        self.assertIn("Your combined indicators suggest elevated risk; consider a longer break.", self.recs(dhri=HIGH_DHRI))

    def test_multiple_conditions_give_multiple_recommendations(self):
        result = self.recs(fatigue=CLOSURE_FATIGUE, posture=POOR_POSTURE, attention=DISTRACTED)
        self.assertEqual(len(result), 3)

    def test_missing_snapshots_do_not_crash(self):
        self.assertEqual(_recommendations(None, None, None, None, None), [])

    def test_none_of_the_recommendations_are_medical_treatment(self):
        # Sanity check on wording: none should sound like a diagnosis or a treatment instruction.
        all_text = " ".join(self.recs(fatigue=CLOSURE_FATIGUE, posture=POOR_POSTURE, attention=DISTRACTED,
                                       dhri=HIGH_DHRI, continuous_active_s=config.DHRI_EXPOSURE_WARNING_MIN * 60.0))
        for forbidden in ("diagnos", "treat", "medic", "prescri"):
            self.assertNotIn(forbidden, all_text.lower())


# ======================================================================
# AlertManager: cooldowns and history
# ======================================================================
class AlertManagerTests(unittest.TestCase):
    def test_a_new_condition_fires_immediately(self):
        m = AlertManager()
        s = m.update(0.0, GOOD_FATIGUE, POOR_POSTURE, GOOD_ATTENTION, GOOD_DHRI, 0.0)
        self.assertEqual([a["id"] for a in s["new_alerts"]], ["posture"])
        self.assertEqual(s["total_count"], 1)

    def test_the_same_condition_does_not_refire_before_its_cooldown(self):
        with mock.patch.object(config, "ALERT_COOLDOWN_S", {**config.ALERT_COOLDOWN_S, "posture": 100.0}):
            m = AlertManager()
            m.update(0.0, GOOD_FATIGUE, POOR_POSTURE, GOOD_ATTENTION, GOOD_DHRI, 0.0)
            s = m.update(50.0, GOOD_FATIGUE, POOR_POSTURE, GOOD_ATTENTION, GOOD_DHRI, 0.0)   # 50 s later, still < 100 s
            self.assertEqual(s["new_alerts"], [])
            self.assertEqual(s["total_count"], 1)

    def test_it_refires_once_the_cooldown_has_passed_even_if_the_condition_never_cleared(self):
        with mock.patch.object(config, "ALERT_COOLDOWN_S", {**config.ALERT_COOLDOWN_S, "posture": 10.0}):
            m = AlertManager()
            m.update(0.0, GOOD_FATIGUE, POOR_POSTURE, GOOD_ATTENTION, GOOD_DHRI, 0.0)
            s = m.update(10.0, GOOD_FATIGUE, POOR_POSTURE, GOOD_ATTENTION, GOOD_DHRI, 0.0)   # condition still true
            self.assertEqual([a["id"] for a in s["new_alerts"]], ["posture"])
            self.assertEqual(s["total_count"], 2)

    def test_the_condition_clearing_and_returning_does_not_reset_the_cooldown(self):
        with mock.patch.object(config, "ALERT_COOLDOWN_S", {**config.ALERT_COOLDOWN_S, "posture": 100.0}):
            m = AlertManager()
            m.update(0.0, GOOD_FATIGUE, POOR_POSTURE, GOOD_ATTENTION, GOOD_DHRI, 0.0)      # fires
            m.update(10.0, GOOD_FATIGUE, GOOD_POSTURE, GOOD_ATTENTION, GOOD_DHRI, 0.0)      # clears
            s = m.update(20.0, GOOD_FATIGUE, POOR_POSTURE, GOOD_ATTENTION, GOOD_DHRI, 0.0)  # true again, 20 s since first fire
            self.assertEqual(s["new_alerts"], [])       # cooldown is time-based from the original firing, not episode-based
            self.assertEqual(s["total_count"], 1)

    def test_different_alert_types_have_independent_cooldowns(self):
        m = AlertManager()
        s = m.update(0.0, CLOSURE_FATIGUE, POOR_POSTURE, GOOD_ATTENTION, GOOD_DHRI, 0.0)
        self.assertEqual(set(a["id"] for a in s["new_alerts"]), {"eye_closure", "posture"})
        self.assertEqual(s["total_count"], 2)

    def test_recent_alerts_are_newest_first_and_capped(self):
        with mock.patch.object(config, "ALERT_COOLDOWN_S", {**config.ALERT_COOLDOWN_S, "posture": 1.0}):
            m = AlertManager()
            for i in range(15):
                s = m.update(i * 1.0, GOOD_FATIGUE, POOR_POSTURE, GOOD_ATTENTION, GOOD_DHRI, 0.0)
            self.assertLessEqual(len(s["recent_alerts"]), 10)
            self.assertGreater(s["recent_alerts"][0]["at_s"], s["recent_alerts"][-1]["at_s"])
            self.assertEqual(s["total_count"], 15)          # the total keeps counting even once history is capped

    def test_idle_snapshot_has_the_documented_fields_and_no_alerts(self):
        snapshot = AlertManager().idle_snapshot()
        for key in ("new_alerts", "recent_alerts", "recommendations", "total_count"):
            self.assertIn(key, snapshot)
        self.assertEqual(snapshot["total_count"], 0)

    def test_recommendations_are_included_in_every_update_regardless_of_cooldowns(self):
        with mock.patch.object(config, "ALERT_COOLDOWN_S", {**config.ALERT_COOLDOWN_S, "posture": 1000.0}):
            m = AlertManager()
            m.update(0.0, GOOD_FATIGUE, POOR_POSTURE, GOOD_ATTENTION, GOOD_DHRI, 0.0)
            s = m.update(1.0, GOOD_FATIGUE, POOR_POSTURE, GOOD_ATTENTION, GOOD_DHRI, 0.0)   # alert on cooldown
            self.assertIn("Adjust your sitting position and keep your back and neck aligned.", s["recommendations"])


# ======================================================================
# Monitor integration
# ======================================================================
class MonitorAlertsTests(unittest.TestCase):
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

    def test_alerts_block_appears_after_calibration(self):
        monitor, _, _ = self.build()
        self.assertTrue(wait_for(lambda: "alerts" in monitor.get_status(1) and monitor.get_status(1)["alerts"] is not None, timeout=15))
        a = monitor.get_status(1)["alerts"]
        for key in ("new_alerts", "recent_alerts", "recommendations", "total_count"):
            self.assertIn(key, a)

    def test_an_eye_closure_alert_fires_end_to_end(self):
        monitor, _, detector = self.build()
        self.assertTrue(wait_for(lambda: monitor.get_status(1)["fatigue"]["state"] == "normal", timeout=15))
        detector.ear = 0.10
        self.assertTrue(wait_for(lambda: monitor.get_status(1)["alerts"]["total_count"] > 0, timeout=6))
        messages = [a["message"] for a in monitor.get_status(1)["alerts"]["recent_alerts"]]
        self.assertIn("Your eyes appear closed for longer than usual.", messages)

    def test_the_alert_count_is_reflected_in_the_session_result(self):
        monitor, _, detector = self.build()
        self.assertTrue(wait_for(lambda: monitor.get_status(1)["fatigue"]["state"] == "normal", timeout=15))
        detector.ear = 0.10
        self.assertTrue(wait_for(lambda: monitor.get_status(1)["alerts"]["total_count"] > 0, timeout=6))
        monitor.stop()
        result = monitor.take_session_result(1)
        self.assertIn("alert_count", result)
        self.assertGreater(result["alert_count"], 0)

    def test_another_account_never_sees_alerts(self):
        monitor, _, _ = self.build()
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("alerts", {}).get("total_count") is not None))
        self.assertNotIn("alerts", monitor.get_status(2))


# ======================================================================
# Database persistence
# ======================================================================
def a_summary(**overrides):
    base = {
        "start_time": datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc),
        "duration_s": 600.0, "active_duration_s": 600.0, "break_count": 0, "break_duration_s": 0.0,
        "fatigue_events": {"warning": 0, "fatigued": 0}, "posture_events": {"warning": 0, "poor": 0},
        "attention_events": {"distracted": 0, "no_face": 0}, "dhri_events": {"moderate": 0, "high": 0},
        "alert_count": 7,
    }
    base.update(overrides)
    return base


class AlertStorageTests(TempDatabaseTestCase):
    def make_user(self):
        from backend import auth_service
        user, _ = auth_service.register(self.connection, "asha", "password123")
        return user["id"]

    def test_alert_count_is_saved(self):
        user_id = self.make_user()
        sessions_db.save_session(self.connection, user_id, a_summary())
        row = sessions_db.get_recent_sessions(self.connection, user_id)[0]
        self.assertEqual(row["alert_count"], 7)

    def test_missing_alert_count_defaults_to_zero(self):
        user_id = self.make_user()
        summary = a_summary()
        del summary["alert_count"]
        sessions_db.save_session(self.connection, user_id, summary)
        row = sessions_db.get_recent_sessions(self.connection, user_id)[0]
        self.assertEqual(row["alert_count"], 0)


# ======================================================================
# API
# ======================================================================
@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class AlertsApiTests(TempDatabaseTestCase):
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

    def test_alerts_are_in_the_status_response(self):
        client = self.client()
        client.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: "alerts" in client.get("/api/monitor/status").json(), timeout=6))
        body = client.get("/api/monitor/status").json()
        for key in ("new_alerts", "recent_alerts", "recommendations", "total_count"):
            self.assertIn(key, body["alerts"])

    def test_alert_count_appears_in_saved_history(self):
        client = self.client()
        client.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json().get("session", {}).get("duration_s", 0) > 0.1))
        client.post("/api/monitor/stop")
        row = client.get("/api/monitor/history").json()["recent"][0]
        self.assertIn("alert_count", row)


if __name__ == "__main__":
    unittest.main()
