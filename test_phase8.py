"""
Phase 8 tests: attention (ATTENTIVE / DISTRACTED / FACE NOT DETECTED).

IMPORTANT: all head directions here are SIMULATED. The tests check that the logic does what it is designed to do
("looking away for 5 s is DISTRACTED; a 3 s glance is not"). They say nothing about how accurate the measured head
direction is on real faces, or whether the limits are right for real people. scripts/posture_test.py shows the real
numbers on your camera.

Run from the project root:
    python -m unittest discover -s tests -t . -v
"""

import importlib.util
import re
import time
import unittest
from pathlib import Path
from unittest import mock

from computer_vision import attention as attention_module
from computer_vision import monitor as monitor_module
from computer_vision.attention import (ATTENTIVE, DISTRACTED, NO_FACE, AttentionAssessor, baseline_supports_attention)
from computer_vision.calibrator import Calibrator
from computer_vision.posture import build_posture_baseline
from tests.helpers import FakeCamera, FakeDetector, FakePoseDetector, TempDatabaseTestCase, new_monitor, wait_for
from tests.test_phase7 import BASE as HEAD_ONLY_BASELINE, head, metrics_for
from utils import config
from utils.smoothing import StateSmoother

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)

EYE_BASELINE = {"ear_open": 0.30, "ear_closed": 0.10}
BASE = {"yaw_deg": 2.0, "pitch_deg": 5.0}               # where this (simulated) user's head points when facing the screen
FPS = 20.0


def m(yaw=None, pitch=None):
    """Head measurements: `yaw`/`pitch` are the DEVIATIONS from BASE in degrees (0 = facing the screen)."""
    return {"yaw_deg": BASE["yaw_deg"] + (yaw or 0.0), "pitch_deg": BASE["pitch_deg"] + (pitch or 0.0)}


def feed(assessor, seconds, metrics=None, reason="", start=0.0):
    """Feed one measurement every 1/FPS seconds. Returns (last snapshot, time after the last measurement)."""
    t, snapshot = start, None
    for _ in range(int(round(seconds * FPS))):
        snapshot = assessor.update(t, reason, metrics)
        t += 1.0 / FPS
    return snapshot, t


def facing():
    return m()


# ======================================================================
# Shared smoother: a per-call hold time
# ======================================================================
class SmootherHoldOverrideTests(unittest.TestCase):
    def test_a_per_call_hold_replaces_the_default_for_getting_worse(self):
        default, quick = StateSmoother(["a", "b"], 5.0, 2.0), StateSmoother(["a", "b"], 5.0, 2.0)
        self.assertEqual(default.update(0.0, "b"), "a")
        self.assertEqual(default.update(2.0, "b"), "a")                       # default hold: 5 s
        self.assertEqual(quick.update(0.0, "b", hold_s=1.0), "a")
        self.assertEqual(quick.update(1.0, "b", hold_s=1.0), "b")             # this call: 1 s

    def test_without_the_override_nothing_changes(self):
        s = StateSmoother(["a", "b"], 3.0, 2.0)
        s.update(0.0, "b")
        self.assertEqual(s.update(3.0, "b"), "b")


# ======================================================================
# The baseline now records where the head points
# ======================================================================
class BaselineYawTests(unittest.TestCase):
    def test_the_posture_baseline_holds_the_head_direction_as_a_median(self):
        samples = [metrics_for(head(yaw=y)) for y in (8.0, 9.0, 10.0, 11.0, 12.0) * 6]
        baseline, _ = build_posture_baseline(samples)
        self.assertAlmostEqual(baseline["yaw_deg"], 10.0, delta=0.6)
        self.assertTrue(baseline_supports_attention(baseline))

    def test_baselines_without_head_direction_are_recognised_as_old(self):
        old = {k: v for k, v in HEAD_ONLY_BASELINE.items() if k != "yaw_deg"}
        self.assertFalse(baseline_supports_attention(old))
        for bad in (None, {}, {"yaw_deg": "x", "pitch_deg": 1}, {"yaw_deg": float("nan"), "pitch_deg": 1}):
            self.assertFalse(baseline_supports_attention(bad), bad)
        self.assertTrue(baseline_supports_attention({"yaw_deg": 1, "pitch_deg": 2}))


# ======================================================================
# When no judgement is possible
# ======================================================================
class UnavailableTests(unittest.TestCase):
    def test_reasons(self):
        cases = [
            (AttentionAssessor(BASE), facing(), True, "Calibration is running"),
            (AttentionAssessor(None), facing(), False, "calibrate first"),
            (AttentionAssessor({"roll_deg": 0, "pitch_deg": 1}), facing(), False, "before attention monitoring existed"),
        ]
        for assessor, metrics, calibrating, expected in cases:
            snapshot = assessor.update(0.0, "", metrics, calibrating=calibrating)
            self.assertEqual(snapshot["state"], "unavailable")
            self.assertIn(expected, snapshot["message"])
            self.assertIsNone(snapshot["score"])

    def test_the_idle_snapshot(self):
        self.assertEqual(AttentionAssessor(BASE).idle_snapshot()["state"], "unavailable")


# ======================================================================
# The three states
# ======================================================================
class AttentionStateTests(unittest.TestCase):
    def test_facing_the_screen_is_attentive_and_carries_the_honesty_note(self):
        snapshot, _ = feed(AttentionAssessor(BASE), 10, facing())
        self.assertEqual(snapshot["state"], ATTENTIVE)
        self.assertEqual(snapshot["reasons"], [])
        self.assertEqual(snapshot["events"], {"distracted": 0, "no_face": 0})
        self.assertIn("cannot know where your eyes look or what you are thinking", snapshot["disclaimer"])

    def test_small_head_movements_are_still_attentive(self):
        for yaw, pitch in ((10, 0), (-15, 0), (0, 12), (0, -18), (20, 20)):
            snapshot, _ = feed(AttentionAssessor(BASE), 8, m(yaw, pitch))
            self.assertEqual(snapshot["state"], ATTENTIVE, (yaw, pitch))

    def test_a_short_glance_away_is_not_distraction(self):
        a = AttentionAssessor(BASE)
        _, t = feed(a, 5, facing())
        snapshot, t = feed(a, 3.5, m(yaw=40), start=t)                       # a 3.5 s glance
        self.assertEqual(snapshot["state"], ATTENTIVE)
        snapshot, _ = feed(a, 5, facing(), start=t)
        self.assertEqual(snapshot["state"], ATTENTIVE)
        self.assertEqual(snapshot["events"]["distracted"], 0)

    def test_looking_away_for_the_hold_time_is_distraction(self):
        a = AttentionAssessor(BASE)
        _, t = feed(a, 5, facing())
        early, t = feed(a, config.ATTENTION_AWAY_HOLD_S - 0.5, m(yaw=40), start=t)
        self.assertEqual(early["state"], ATTENTIVE)                            # not yet
        snapshot, t = feed(a, 1.5, m(yaw=40), start=t)
        self.assertEqual(snapshot["state"], DISTRACTED)
        self.assertEqual(snapshot["events"]["distracted"], 1)
        self.assertIn("40 degrees sideways", snapshot["reasons"][0])
        self.assertAlmostEqual(snapshot["deviations"]["head_turn_deg"], 40, delta=0.1)
        self.assertGreater(snapshot["continuous_away_s"], 5)

    def test_turning_either_way_sideways_counts(self):
        for yaw in (-40, 40):
            snapshot, _ = feed(AttentionAssessor(BASE), 7, m(yaw=yaw))
            self.assertEqual(snapshot["state"], DISTRACTED, yaw)
            self.assertAlmostEqual(abs(snapshot["deviations"]["head_turn_deg"]), 40, delta=0.1)

    def test_looking_down_or_up_counts_too_and_says_which(self):
        for pitch, word in ((32, "down"), (-32, "up")):
            snapshot, _ = feed(AttentionAssessor(BASE), 7, m(pitch=pitch))
            self.assertEqual(snapshot["state"], DISTRACTED, pitch)
            self.assertIn(f"tilted {word}", snapshot["reasons"][0])

    def test_the_limits_come_from_the_configuration(self):
        just_inside = m(yaw=config.ATTENTION_YAW_LIMIT_DEG - 1)
        just_outside = m(yaw=config.ATTENTION_YAW_LIMIT_DEG + 1)
        self.assertEqual(feed(AttentionAssessor(BASE), 8, just_inside)[0]["state"], ATTENTIVE)
        self.assertEqual(feed(AttentionAssessor(BASE), 8, just_outside)[0]["state"], DISTRACTED)

    def test_hovering_near_the_limit_does_not_flicker(self):
        a = AttentionAssessor(BASE)
        _, t = feed(a, 3, facing())
        snapshot, t = feed(a, 6, m(yaw=27), start=t)                           # past the limit: distracted
        self.assertEqual(snapshot["state"], DISTRACTED)
        inside_but_above_exit = config.ATTENTION_YAW_LIMIT_DEG * config.ATTENTION_EXIT_RATIO + 2      # e.g. 22 of 25
        snapshot, t = feed(a, 8, m(yaw=inside_but_above_exit), start=t)        # back inside the limit, not clearly back
        self.assertEqual(snapshot["state"], DISTRACTED)                        # hysteresis keeps it
        snapshot, _ = feed(a, config.ATTENTION_RECOVERY_HOLD_S + 0.5, m(yaw=10), start=t)
        self.assertEqual(snapshot["state"], ATTENTIVE)

    def test_coming_back_needs_a_moment_before_attentive_returns(self):
        a = AttentionAssessor(BASE)
        _, t = feed(a, 8, m(yaw=40))
        snapshot, t = feed(a, 1.0, facing(), start=t)
        self.assertEqual(snapshot["state"], DISTRACTED)
        self.assertIn("Facing the screen again", snapshot["reasons"][0])
        snapshot, _ = feed(a, config.ATTENTION_RECOVERY_HOLD_S, facing(), start=t)
        self.assertEqual(snapshot["state"], ATTENTIVE)
        self.assertEqual(snapshot["reasons"], [])

    def test_no_face_is_shown_only_after_the_hold_time(self):
        a = AttentionAssessor(BASE)
        _, t = feed(a, 5, facing())
        snapshot, t = feed(a, config.ATTENTION_NO_FACE_HOLD_S - 0.5, None, reason="no_face", start=t)
        self.assertEqual(snapshot["state"], ATTENTIVE)                         # a brief detection glitch changes nothing
        snapshot, t = feed(a, 1.5, None, reason="no_face", start=t)
        self.assertEqual(snapshot["state"], NO_FACE)
        self.assertEqual(snapshot["events"]["no_face"], 1)
        self.assertIn("away from the camera, or turned too far", snapshot["reasons"][0])
        snapshot, _ = feed(a, config.ATTENTION_RECOVERY_HOLD_S + 0.5, facing(), start=t)
        self.assertEqual(snapshot["state"], ATTENTIVE)

    def test_a_black_picture_counts_as_nobody_visible(self):
        snapshot, _ = feed(AttentionAssessor(BASE), 6, None, reason="too_dark")
        self.assertEqual(snapshot["state"], NO_FACE)

    def test_extended_absence_is_flagged_after_a_long_time(self):
        a = AttentionAssessor(BASE)
        snapshot, t = feed(a, config.ATTENTION_EXTENDED_ABSENCE_S - 5, None, reason="no_face")
        self.assertFalse(snapshot["extended_absence"])
        snapshot, t = feed(a, 10, None, reason="no_face", start=t)
        self.assertTrue(snapshot["extended_absence"])
        snapshot, _ = feed(a, 3, facing(), start=t)
        self.assertFalse(snapshot["extended_absence"])

    def test_turning_away_until_the_face_is_lost_goes_from_distracted_to_no_face(self):
        a = AttentionAssessor(BASE)
        _, t = feed(a, 3, facing())
        _, t = feed(a, 3, m(yaw=45), start=t)                                  # turning
        snapshot, t = feed(a, 1.0, None, reason="no_face", start=t)            # the face model loses the face
        self.assertEqual(snapshot["state"], DISTRACTED)                        # the 3 s of turning already counted
        snapshot, t = feed(a, 4.0, None, reason="no_face", start=t)            # ... then its own 3 s hold passes
        self.assertEqual(snapshot["state"], NO_FACE)
        self.assertEqual(snapshot["events"], {"distracted": 1, "no_face": 1})


class UnsteadyPictureTests(unittest.TestCase):
    def test_moments_without_information_change_nothing(self):
        a = AttentionAssessor(BASE)
        _, t = feed(a, 6, facing())
        for reason in ("unstable_image", "face_jump", "settling", "eyes_unreadable"):
            snapshot, t = feed(a, 1.0, None, reason=reason, start=t)
            self.assertEqual(snapshot["state"], ATTENTIVE, reason)

    def test_unknown_moments_are_not_counted_as_looking_away(self):
        a = AttentionAssessor(BASE)
        _, t = feed(a, 30, facing())
        _, t = feed(a, 3, None, reason="settling", start=t)
        snapshot, _ = feed(a, 30, facing(), start=t)
        self.assertEqual(snapshot["score"], 0.0)

    def test_a_long_unsteady_spell_makes_it_unavailable_with_a_reason(self):
        a = AttentionAssessor(BASE)
        _, t = feed(a, 6, facing())
        snapshot, _ = feed(a, config.ATTENTION_UNKNOWN_LIMIT_S + 2, None, reason="unstable_image", start=t)
        self.assertEqual(snapshot["state"], "unavailable")
        self.assertIn("unsteady", snapshot["message"])


# ======================================================================
# The "last minute" figure
# ======================================================================
class AttentionScoreTests(unittest.TestCase):
    def test_no_figure_until_enough_has_been_observed(self):
        snapshot, _ = feed(AttentionAssessor(BASE), config.ATTENTION_MIN_OBSERVED_S - 2, facing())
        self.assertIsNone(snapshot["score"])
        self.assertIsNone(snapshot["attentive_percent"])

    def test_half_the_time_away_is_about_fifty_percent(self):
        a = AttentionAssessor(BASE)
        _, t = feed(a, 30, facing())
        snapshot, _ = feed(a, 30, None, reason="no_face", start=t)
        self.assertAlmostEqual(snapshot["score"], 0.5, delta=0.04)
        self.assertAlmostEqual(snapshot["attentive_percent"], 50, delta=4)

    def test_fully_attentive_is_100_percent(self):
        snapshot, _ = feed(AttentionAssessor(BASE), 20, facing())
        self.assertEqual(snapshot["attentive_percent"], 100)

    def test_the_window_forgets_old_behaviour(self):
        a = AttentionAssessor(BASE)
        _, t = feed(a, 40, m(yaw=45))
        snapshot, _ = feed(a, config.ATTENTION_WINDOW_S + 10, facing(), start=t)
        self.assertEqual(snapshot["score"], 0.0)

    def test_score_is_reported_without_smoothing_so_it_follows_reality(self):
        a = AttentionAssessor(BASE)
        _, t = feed(a, 20, facing())
        snapshot, _ = feed(a, 2, m(yaw=45), start=t)                           # only 2 s away: state still attentive
        self.assertEqual(snapshot["state"], ATTENTIVE)
        self.assertGreater(snapshot["score"], 0.05)


class ConfigTests(unittest.TestCase):
    def test_limits_are_sensible(self):
        self.assertGreater(config.ATTENTION_YAW_LIMIT_DEG, 0)
        self.assertGreater(config.ATTENTION_PITCH_LIMIT_DEG, 0)
        self.assertTrue(0 < config.ATTENTION_EXIT_RATIO < 1)
        self.assertGreater(config.ATTENTION_AWAY_HOLD_S, 1)                      # far more than a single frame
        self.assertGreater(config.ATTENTION_NO_FACE_HOLD_S, 1)
        self.assertGreater(config.ATTENTION_EXTENDED_ABSENCE_S, config.ATTENTION_NO_FACE_HOLD_S)
        self.assertGreater(config.ATTENTION_WINDOW_S, config.ATTENTION_MIN_OBSERVED_S)


class PrivacyTests(unittest.TestCase):
    def test_attention_saves_no_images_and_opens_no_files(self):
        source = Path(attention_module.__file__).read_text(encoding="utf-8")
        for forbidden in ("imwrite", "VideoWriter", "write_bytes", "write_text", ".write(", "import cv2"):
            self.assertNotIn(forbidden, source)
        self.assertIsNone(re.search(r"(?<![\w.])open\(", source))


# ======================================================================
# Through the whole monitor
# ======================================================================
def posture_baseline_from(head_pose_=None):
    baseline, problem = build_posture_baseline([metrics_for(head_pose_) for _ in range(30)])
    assert baseline is not None, problem
    return baseline


class MonitorAttentionTests(unittest.TestCase):
    def setUp(self):
        self._monitors = []

    def tearDown(self):
        for monitor in self._monitors:
            monitor.stop()

    def build(self, posture_baseline, camera=None, detector=None, **options):
        camera = camera or FakeCamera()
        detector = detector or FakeDetector(ear=0.30, head=head())
        monitor = new_monitor(lambda: camera, detector_factory=lambda: detector, **options)
        self._monitors.append(monitor)
        monitor.start(1, baseline=EYE_BASELINE, posture_baseline=posture_baseline)
        return monitor, camera, detector

    def attention(self, monitor):
        return monitor.get_status(1).get("attention", {})

    def test_without_calibration_attention_says_to_calibrate(self):
        monitor, _, _ = self.build(None)
        self.assertTrue(wait_for(lambda: "calibrate first" in self.attention(monitor).get("message", "")))
        self.assertFalse(monitor.get_status(1)["attention_calibrated"])

    def test_a_calibration_from_before_this_phase_asks_for_a_new_one(self):
        old = {k: v for k, v in posture_baseline_from().items() if k != "yaw_deg"}
        monitor, _, _ = self.build(old)
        self.assertTrue(wait_for(lambda: "before attention monitoring existed" in self.attention(monitor).get("message", "")))
        self.assertFalse(monitor.get_status(1)["attention_calibrated"])

    def test_facing_away_and_leaving_the_view_through_the_whole_pipeline(self):
        with mock.patch.object(config, "ATTENTION_AWAY_HOLD_S", 0.6), mock.patch.object(config, "ATTENTION_NO_FACE_HOLD_S", 0.6), \
             mock.patch.object(config, "ATTENTION_RECOVERY_HOLD_S", 0.5):
            monitor, _, detector = self.build(posture_baseline_from())
            self.assertTrue(wait_for(lambda: self.attention(monitor).get("state") == ATTENTIVE, timeout=6))
            self.assertTrue(monitor.get_status(1)["attention_calibrated"])

            detector.head = head(yaw=40)                                      # head turned away
            self.assertTrue(wait_for(lambda: self.attention(monitor).get("state") == DISTRACTED, timeout=6))
            self.assertTrue(any("sideways" in r for r in self.attention(monitor)["reasons"]))
            self.assertAlmostEqual(self.attention(monitor)["deviations"]["head_turn_deg"], 40, delta=1.0)

            detector.head = head()                                            # facing the screen again
            self.assertTrue(wait_for(lambda: self.attention(monitor).get("state") == ATTENTIVE, timeout=6))

            detector.faces = []                                               # left the camera view
            self.assertTrue(wait_for(lambda: self.attention(monitor).get("state") == NO_FACE, timeout=6))
            detector.faces = [(0.5, 0.5, 0.4)]
            self.assertTrue(wait_for(lambda: self.attention(monitor).get("state") == ATTENTIVE, timeout=8))
            self.assertEqual(self.attention(monitor)["events"], {"distracted": 1, "no_face": 1})

    def test_covering_the_lens_shows_face_not_detected_not_distracted(self):
        with mock.patch.object(config, "ATTENTION_NO_FACE_HOLD_S", 0.6), mock.patch.object(config, "ATTENTION_AWAY_HOLD_S", 0.6):
            monitor, camera, _ = self.build(posture_baseline_from())
            self.assertTrue(wait_for(lambda: self.attention(monitor).get("state") == ATTENTIVE, timeout=6))
            camera.gray_level = 3
            self.assertTrue(wait_for(lambda: self.attention(monitor).get("state") == NO_FACE, timeout=6))
            self.assertEqual(self.attention(monitor)["events"]["distracted"], 0)

    def test_calibration_through_the_monitor_records_the_head_direction_and_activates_attention(self):
        with mock.patch.object(config, "POSTURE_CAL_MIN_SAMPLES", 10):
            monitor, _, detector = self.build(None, calibrator_factory=lambda: Calibrator(ready_s=1.2, open_s=2.0, close_s=2.0))
            detector.head = head(yaw=6.0)                                     # this user faces the screen at yaw +6
            self.assertTrue(monitor.begin_calibration(1))
            end = time.monotonic() + 10
            while time.monotonic() < end:
                status = monitor.get_calibration_status(1)
                if status is not None:
                    detector.ear = 0.10 if status["phase"] == "close" else 0.30
                    if not status["active"]:
                        break
                time.sleep(0.02)
            result = monitor.take_calibration_result(1)
            self.assertTrue(result["ok"], result)
            self.assertAlmostEqual(result["posture_baseline"]["yaw_deg"], 6.0, delta=1.0)
            self.assertTrue(monitor.get_status(1)["attention_calibrated"])
            self.assertTrue(wait_for(lambda: self.attention(monitor).get("state") == ATTENTIVE, timeout=6))

    def test_another_account_never_sees_attention(self):
        monitor, _, _ = self.build(posture_baseline_from())
        self.assertTrue(wait_for(lambda: self.attention(monitor).get("state")))
        self.assertNotIn("attention", monitor.get_status(2))


class AttentionProfileTests(TempDatabaseTestCase):
    def test_the_profile_says_whether_attention_is_calibrated(self):
        from backend import auth_service
        from database import calibration as calibration_db
        user, _ = auth_service.register(self.connection, "asha", "password123")
        row = auth_service.get_user_from_token(self.connection, auth_service.login(self.connection, "asha", "password123")[1])
        old = {k: v for k, v in posture_baseline_from().items() if k != "yaw_deg"}
        calibration_db.save_calibration(self.connection, user["id"], dict(EYE_BASELINE), old)
        profile = auth_service.describe_user(self.connection, row)
        self.assertTrue(profile["posture_calibrated"])
        self.assertFalse(profile["attention_calibrated"])                     # calibrated before Phase 8
        calibration_db.save_calibration(self.connection, user["id"], dict(EYE_BASELINE), posture_baseline_from())
        self.assertTrue(auth_service.describe_user(self.connection, row)["attention_calibrated"])
        calibration_db.save_calibration(self.connection, user["id"], dict(EYE_BASELINE), None)
        self.assertFalse(auth_service.describe_user(self.connection, row)["attention_calibrated"])


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class AttentionApiTests(TempDatabaseTestCase):
    def setUp(self):
        super().setUp()
        self.detector = FakeDetector(ear=0.30, head=head(yaw=4.0))
        self._original_monitor = monitor_module.monitor
        monitor_module.monitor = new_monitor(
            lambda: FakeCamera(), detector_factory=lambda: self.detector,
            calibrator_factory=lambda: Calibrator(ready_s=1.2, open_s=2.0, close_s=2.0))
        patch = mock.patch.object(config, "POSTURE_CAL_MIN_SAMPLES", 10)
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        monitor_module.monitor.stop()
        monitor_module.monitor = self._original_monitor
        super().tearDown()

    def test_the_saved_calibration_carries_the_head_direction_to_attention(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        client.post("/api/auth/register", json={"username": "asha", "password": "password123"})
        client.post("/api/calibration/start")
        end = time.monotonic() + 10
        while time.monotonic() < end:
            body = client.get("/api/calibration/status").json()
            self.detector.ear = 0.10 if body["phase"] == "close" else 0.30
            if not body["active"]:
                break
            time.sleep(0.05)
        client.post("/api/monitor/stop")
        started = client.post("/api/monitor/start").json()
        self.assertTrue(started["attention_calibrated"])
        self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json().get("attention", {}).get("state")))
        for key in ("state", "score", "attentive_percent", "reasons", "message", "events", "disclaimer", "extended_absence"):
            self.assertIn(key, client.get("/api/monitor/status").json()["attention"])


if __name__ == "__main__":
    unittest.main()
