"""
Phase 5 tests: eye aspect ratio, blink / closure / PERCLOS tracking, personal calibration,
saving the baseline, and the monitor + API integration.

The eye LOGIC is tested with synthetic eye measurements whose correct answers are known
(e.g. "one 0.15-second closure = one blink"). Whether the real face model gives sensible EAR values
on YOUR face is checked by scripts/eye_test.py and the dashboard, not by these tests.

Run from the project root:
    python -m unittest discover -s tests -t . -v
"""

import importlib.util
import json
import re
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from backend import auth_service
from computer_vision import calibrator as calibrator_module
from computer_vision import drawing, eye_features, eye_tracker
from computer_vision import face_landmarks as fl
from computer_vision import monitor as monitor_module
from computer_vision.calibrator import Calibrator
from computer_vision.eye_tracker import EyeStateTracker, closure_fraction
from database import calibration as calibration_db
from tests.helpers import (
    FakeCamera,
    FakeDetector,
    TempDatabaseTestCase,
    make_face_landmarks,
    new_monitor,
    wait_for,
)
from utils import config

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)

BASELINE = {"ear_open": 0.30, "ear_closed": 0.10}
OPEN, CLOSED = 0.30, 0.10          # EAR values at exactly 0% and 100% closure for BASELINE


def run(tracker, pattern, fps=20.0, start=0.0):
    """
    Feed a tracker a list of (seconds, ear) steps, one measurement every 1/fps seconds.
    ear may be None (no face). Returns (last features, time after the last step).
    """
    t, features = start, None
    for seconds, ear in pattern:
        for _ in range(int(round(seconds * fps))):
            features = tracker.update(t, ear)
            t += 1.0 / fps
    return features, t


# ======================================================================
# Eye aspect ratio
# ======================================================================
class EyeAspectRatioTests(unittest.TestCase):
    def test_known_shape_gives_the_formula_result(self):
        # eye width 30, both vertical distances 10  ->  (10 + 10) / (2 * 30) = 0.3333
        points = np.array([(0, 0), (10, -5), (20, -5), (30, 0), (20, 5), (10, 5)], dtype=float)
        self.assertAlmostEqual(eye_features._ear_of_one_eye(points), 1 / 3, places=6)

    def test_closed_eye_gives_zero(self):
        points = np.array([(0, 0), (10, 0), (20, 0), (30, 0), (20, 0), (10, 0)], dtype=float)
        self.assertAlmostEqual(eye_features._ear_of_one_eye(points), 0.0, places=6)

    def test_eye_with_no_width_returns_none(self):
        points = np.zeros((6, 2))
        self.assertIsNone(eye_features._ear_of_one_eye(points))

    def test_fake_face_reproduces_the_requested_ear(self):
        for ear in (0.05, 0.10, 0.25, 0.30, 0.40):
            left, right, mean = eye_features.eye_aspect_ratios(make_face_landmarks(ear=ear), 640, 480)
            for value in (left, right, mean):
                self.assertAlmostEqual(value, ear, places=3)

    def test_distances_are_measured_in_pixels_not_image_fractions(self):
        landmarks = make_face_landmarks(ear=0.30)
        correct = eye_features.eye_aspect_ratios(landmarks, 640, 480)[2]
        naive = eye_features.eye_aspect_ratios(landmarks, 1, 1)[2]      # treating fractions as if square
        self.assertAlmostEqual(correct, 0.30, places=3)
        self.assertGreater(abs(naive - correct), 0.05)                  # about 33% too large

    def test_ear_points_follow_the_outline_layout(self):
        # Corners are the two ends of the outline; each upper point sits at the same position along the
        # upper lid as its lower partner sits along the lower lid (so the pairs are vertical).
        for points, outline in ((eye_features.RIGHT_EYE_EAR_POINTS, fl.RIGHT_EYE_OUTLINE),
                                (eye_features.LEFT_EYE_EAR_POINTS, fl.LEFT_EYE_OUTLINE)):
            p1, p2, p3, p4, p5, p6 = points
            self.assertEqual({p1, p4}, {outline[0], outline[8]})
            lower = outline[1:8]                        # outer corner -> inner corner
            upper = list(reversed(outline[9:]))         # outer corner -> inner corner
            self.assertEqual(upper.index(p2), lower.index(p6))
            self.assertEqual(upper.index(p3), lower.index(p5))
            self.assertNotEqual(upper.index(p2), upper.index(p3))


# ======================================================================
# Closure fraction and configuration sanity
# ======================================================================
class ClosureFractionTests(unittest.TestCase):
    def test_personal_scale(self):
        self.assertAlmostEqual(closure_fraction(0.30, BASELINE), 0.0)
        self.assertAlmostEqual(closure_fraction(0.10, BASELINE), 1.0)
        self.assertAlmostEqual(closure_fraction(0.20, BASELINE), 0.5)

    def test_values_outside_the_range_are_clipped(self):
        self.assertEqual(closure_fraction(0.45, BASELINE), 0.0)
        self.assertEqual(closure_fraction(0.02, BASELINE), 1.0)

    def test_broken_baseline_does_not_divide_by_zero(self):
        self.assertEqual(closure_fraction(0.2, {"ear_open": 0.2, "ear_closed": 0.2}), 0.0)


class ConfigSanityTests(unittest.TestCase):
    def test_thresholds_are_consistent(self):
        self.assertGreater(config.EYE_CLOSE_ENTER, config.EYE_CLOSE_EXIT)      # hysteresis gap
        self.assertGreaterEqual(config.PERCLOS_CLOSURE, config.EYE_CLOSE_ENTER)
        self.assertLess(config.BLINK_MIN_S, config.BLINK_MAX_S)
        self.assertLessEqual(config.FEATURE_MIN_COVERAGE_S, config.FEATURE_WINDOW_S)
        self.assertGreaterEqual(config.TARGET_FPS, config.MIN_RELIABLE_FPS)
        self.assertEqual(config.FEATURE_WINDOW_S, 20)          # the base paper's window length


# ======================================================================
# Blink / closure / PERCLOS tracking
# ======================================================================
class EyeStateTrackerTests(unittest.TestCase):
    def test_uncalibrated_tracker_reports_only_the_ear(self):
        features = EyeStateTracker(None).update(0.0, 0.28)
        self.assertEqual(features["ear"], 0.28)
        self.assertFalse(features["calibrated"])
        for key in ("blink_count", "eyes_closed", "perclos", "blinks_per_min", "closure"):
            self.assertIsNone(features[key])

    def test_one_short_closure_is_one_blink(self):
        tracker = EyeStateTracker(BASELINE)
        features, _ = run(tracker, [(6, OPEN), (0.15, CLOSED), (6, OPEN)])
        self.assertEqual(features["blink_count"], 1)
        self.assertEqual(features["long_closure_count"], 0)
        self.assertFalse(features["eyes_closed"])

    def test_blink_duration_is_measured_within_one_frame(self):
        tracker = EyeStateTracker(BASELINE)
        features, _ = run(tracker, [(6, OPEN), (0.20, CLOSED), (6, OPEN)])
        self.assertAlmostEqual(features["mean_blink_duration_s"], 0.20, delta=0.06)

    def test_blink_rate_over_the_20_second_window(self):
        tracker = EyeStateTracker(BASELINE)
        pattern = [(0.1, OPEN)]
        for _ in range(6):                                  # 6 blinks in about 20 s -> about 18 per minute
            pattern += [(0.15, CLOSED), (3.2, OPEN)]
        features, _ = run(tracker, pattern)
        self.assertEqual(features["blink_count"], 6)
        self.assertAlmostEqual(features["blinks_per_min"], 18, delta=2.0)

    def test_long_closure_is_reported_while_it_happens_and_counted_afterwards(self):
        tracker = EyeStateTracker(BASELINE)
        features, t = run(tracker, [(11, OPEN), (0.8, CLOSED)])
        self.assertTrue(features["eyes_closed"])
        self.assertTrue(features["long_closure"])
        self.assertAlmostEqual(features["current_closure_s"], 0.75, delta=0.1)
        features, _ = run(tracker, [(1.5, CLOSED), (1.6, OPEN)], start=t)      # (a count is confirmed 1 s after it ends)
        self.assertEqual(features["long_closure_count"], 1)
        self.assertEqual(features["blink_count"], 0)         # a long closure is not a blink
        self.assertAlmostEqual(features["longest_closure_s"], 2.3, delta=0.15)

    def test_perclos_is_the_share_of_time_with_closed_eyes(self):
        tracker = EyeStateTracker(BASELINE)
        features, _ = run(tracker, [(9, OPEN), (2, CLOSED), (9, OPEN)])          # 2 s of 20 s
        self.assertAlmostEqual(features["perclos"], 0.10, delta=0.02)

    def test_perclos_is_one_when_the_eyes_stay_closed(self):
        tracker = EyeStateTracker(BASELINE)
        features, _ = run(tracker, [(1, OPEN), (14, CLOSED)])
        self.assertAlmostEqual(features["perclos"], 1.0, delta=0.08)

    def test_half_closed_eyes_count_as_closed_for_perclos_only_from_80_percent(self):
        tracker = EyeStateTracker(BASELINE)
        # EAR 0.16 = 70% closure: above the blink threshold, below the 80% PERCLOS level
        features, _ = run(tracker, [(1, OPEN), (14, 0.16)])
        self.assertEqual(features["perclos"], 0.0)
        self.assertTrue(features["eyes_closed"])

    def test_values_hovering_at_the_threshold_do_not_count_extra_blinks(self):
        tracker = EyeStateTracker(BASELINE)
        _, t = run(tracker, [(2, OPEN), (0.2, CLOSED)])
        for _ in range(20):                       # 2 s alternating between 55% and 40% closure at 20 fps
            for ear in (0.19, 0.22):              # 0.22 is below the "closed" level but above the "reopened" level
                features = tracker.update(t, ear)
                t += 0.05
            self.assertTrue(features["eyes_closed"])
        features, _ = run(tracker, [(1.6, OPEN)], start=t)
        self.assertEqual(features["blink_count"], 0)
        self.assertEqual(features["long_closure_count"], 1)   # one long closure, not many small ones

    def test_eyes_closed_from_the_very_start_are_not_a_blink(self):
        tracker = EyeStateTracker(BASELINE)
        features, _ = run(tracker, [(1.0, CLOSED), (3, OPEN)])          # no open eye was ever seen before it
        self.assertEqual(features["blink_count"], 0)
        self.assertEqual(features["long_closure_count"], 0)

    def test_unreliable_first_frames_after_the_face_returns_are_not_blinks(self):
        tracker = EyeStateTracker(BASELINE)
        _, t = run(tracker, [(3, OPEN), (1, None)])                     # face lost (for example a hand over the lens)
        features, t = run(tracker, [(0.3, CLOSED), (3, OPEN)], start=t)  # bad first frames, then normal open eyes
        self.assertEqual(features["blink_count"], 0)
        self.assertEqual(features["long_closure_count"], 0)

    def test_real_blinks_still_count_after_the_face_returned(self):
        tracker = EyeStateTracker(BASELINE)
        _, t = run(tracker, [(3, OPEN), (1, None), (0.3, CLOSED), (3, OPEN)])
        features, _ = run(tracker, [(0.15, CLOSED), (3, OPEN)], start=t)
        self.assertEqual(features["blink_count"], 1)

    def test_unreliable_frames_do_not_inflate_the_closed_share(self):
        tracker = EyeStateTracker(BASELINE)
        features, _ = run(tracker, [(12, OPEN), (1, None), (0.5, CLOSED), (6, OPEN)])
        self.assertEqual(features["perclos"], 0.0)

    def test_a_gap_in_the_data_also_requires_open_eyes_first(self):
        tracker = EyeStateTracker(BASELINE)
        _, t = run(tracker, [(2, OPEN)])
        features = tracker.update(t + 3.0, CLOSED)                      # data resumes with a "closed" value
        features, _ = run(tracker, [(1, OPEN)], start=t + 3.05)
        self.assertEqual(features["blink_count"] + features["long_closure_count"], 0)

    def test_a_dip_that_never_reaches_the_threshold_is_not_a_blink(self):
        tracker = EyeStateTracker(BASELINE)
        features, _ = run(tracker, [(3, OPEN), (0.3, 0.22), (3, OPEN)])     # only 40% closure
        self.assertEqual(features["blink_count"], 0)

    def test_closures_shorter_than_the_minimum_are_ignored(self):
        tracker = EyeStateTracker(BASELINE)
        tracker.update(0.00, OPEN)
        tracker.update(0.01, CLOSED)
        features = tracker.update(0.02, OPEN)                 # closed for only 0.01 s
        self.assertEqual(features["blink_count"], 0)

    def test_losing_the_face_during_a_closure_cancels_it(self):
        tracker = EyeStateTracker(BASELINE)
        _, t = run(tracker, [(2, OPEN), (0.2, CLOSED)])
        features = tracker.update(t, None)                    # face lost
        self.assertFalse(features["eyes_closed"])
        self.assertIsNone(features["ear"])
        features, _ = run(tracker, [(1, OPEN)], start=t + 0.05)
        self.assertEqual(features["blink_count"], 0)          # nothing was guessed

    def test_a_gap_in_the_data_is_not_counted_as_valid_time_or_as_a_closure(self):
        tracker = EyeStateTracker(BASELINE)
        _, t = run(tracker, [(2, OPEN), (0.2, CLOSED)])
        features = tracker.update(t + 4.0, OPEN)              # next measurement 4 s later
        self.assertEqual(features["blink_count"], 0)
        self.assertLess(features["window_valid_s"], 3.0)      # the 4 s gap was not added

    def test_old_blinks_leave_the_window_but_stay_in_the_total(self):
        tracker = EyeStateTracker(BASELINE)
        features, _ = run(tracker, [(1, OPEN), (0.15, CLOSED), (25, OPEN)])
        self.assertEqual(features["blink_count"], 1)
        self.assertEqual(features["blinks_per_min"], 0.0)

    def test_window_measures_wait_for_enough_data(self):
        tracker = EyeStateTracker(BASELINE)
        features, _ = run(tracker, [(5, OPEN), (0.15, CLOSED), (2, OPEN)])       # only about 7 s
        self.assertIsNone(features["blinks_per_min"])
        self.assertIsNone(features["perclos"])
        self.assertEqual(features["blink_count"], 1)          # counts are available immediately

    def test_results_do_not_depend_on_the_frame_rate(self):
        for fps in (10, 20, 30):
            tracker = EyeStateTracker(BASELINE)
            features, _ = run(tracker, [(6, OPEN), (0.2, CLOSED), (6, OPEN)], fps=fps)
            self.assertEqual(features["blink_count"], 1, f"fps={fps}")
            self.assertAlmostEqual(features["mean_blink_duration_s"], 0.2, delta=1.2 / fps, msg=f"fps={fps}")

    def test_time_going_backwards_is_treated_as_a_break(self):
        tracker = EyeStateTracker(BASELINE)
        _, t = run(tracker, [(1, OPEN), (0.2, CLOSED)])
        features = tracker.update(t - 10, OPEN)
        self.assertFalse(features["eyes_closed"])
        self.assertEqual(features["blink_count"], 0)


# ======================================================================
# Personal calibration
# ======================================================================
def run_calibration(calibrator, ear_at, fps=20.0):
    """Feed the calibrator ear_at(t) until it finishes (synthetic time, so this is instant)."""
    t = 0.0
    while not calibrator.finished and t < 60:
        calibrator.update(t, ear_at(t))
        t += 1.0 / fps
    return calibrator


def typical_user(t, open_ear=0.30, closed_ear=0.10, noise=0.0, seed=1):
    """Ready 0-3 s, eyes open with a blink every 3 s from 3-11 s, then eyes closed after a 0.5 s reaction."""
    rng = np.random.default_rng(int(t * 1000) + seed)
    jitter = rng.normal(0, noise) if noise else 0.0
    if t < 11:
        blinking = t >= 3 and (t - 3) % 3 < 0.15
        return (closed_ear if blinking else open_ear) + jitter
    return (open_ear if t < 11.5 else closed_ear) + jitter


class CalibratorTests(unittest.TestCase):
    def test_a_normal_calibration_finds_the_users_open_and_closed_values(self):
        calibrator = run_calibration(Calibrator(), typical_user)
        self.assertTrue(calibrator.result["ok"])
        baseline = calibrator.result["baseline"]
        self.assertAlmostEqual(baseline["ear_open"], 0.30, delta=0.02)
        self.assertAlmostEqual(baseline["ear_closed"], 0.10, delta=0.03)
        self.assertAlmostEqual(baseline["sample_rate_hz"], 20, delta=1.5)

    def test_it_still_works_with_measurement_noise(self):
        calibrator = run_calibration(Calibrator(), lambda t: typical_user(t, noise=0.01))
        self.assertTrue(calibrator.result["ok"])
        self.assertAlmostEqual(calibrator.result["baseline"]["ear_open"], 0.30, delta=0.03)
        self.assertAlmostEqual(calibrator.result["baseline"]["ear_closed"], 0.10, delta=0.04)

    def test_a_different_user_gets_different_baselines(self):
        calibrator = run_calibration(Calibrator(), lambda t: typical_user(t, open_ear=0.24, closed_ear=0.12))
        self.assertAlmostEqual(calibrator.result["baseline"]["ear_open"], 0.24, delta=0.02)
        self.assertAlmostEqual(calibrator.result["baseline"]["ear_closed"], 0.12, delta=0.03)

    def test_the_steps_run_in_order_with_the_required_instructions(self):
        calibrator = Calibrator()
        seen = []
        t = 0.0
        while not calibrator.finished:
            status = calibrator.update(t, typical_user(t))
            if not seen or seen[-1]["phase"] != status["phase"]:
                seen.append(status)
            t += 0.05
        self.assertEqual([s["phase"] for s in seen], ["ready", "open", "close", "done"])
        ready_text = seen[0]["message"]
        for sentence in ("Sit normally and look at the screen.", "Keep your face visible.",
                         "Maintain your normal sitting posture."):
            self.assertIn(sentence, ready_text)
        self.assertEqual(seen[0]["seconds_left"], 3)
        self.assertEqual(seen[-1]["progress"], 1.0)
        self.assertFalse(seen[-1]["active"])

    def test_countdown_goes_down(self):
        calibrator = Calibrator()
        self.assertEqual(calibrator.update(0.0, 0.3)["seconds_left"], 3)
        self.assertEqual(calibrator.update(2.2, 0.3)["seconds_left"], 1)
        self.assertEqual(calibrator.update(3.5, 0.3)["phase"], "open")

    def test_no_face_fails_with_a_clear_reason(self):
        calibrator = run_calibration(Calibrator(), lambda t: None)
        self.assertFalse(calibrator.result["ok"])
        self.assertIn("could not be measured often enough", calibrator.result["reason"])

    def test_a_face_that_is_often_missing_fails(self):
        calibrator = run_calibration(
            Calibrator(), lambda t: None if int(t * 20) % 2 == 0 else typical_user(t))   # half the frames lost
        self.assertFalse(calibrator.result["ok"])

    def test_too_few_measurements_per_step_fails(self):
        calibrator = run_calibration(Calibrator(), typical_user, fps=1.0)               # about 8 open samples
        self.assertFalse(calibrator.result["ok"])

    def test_eyes_that_never_close_are_rejected(self):
        calibrator = run_calibration(Calibrator(), lambda t: 0.30)
        self.assertFalse(calibrator.result["ok"])
        self.assertIn("could not see your eyes close", calibrator.result["reason"])

    def test_unusual_open_eye_values_are_rejected(self):
        for open_ear, closed_ear in ((0.80, 0.10), (0.06, 0.02)):
            calibrator = run_calibration(Calibrator(), lambda t, o=open_ear, c=closed_ear: typical_user(t, o, c))
            self.assertFalse(calibrator.result["ok"], open_ear)
            self.assertIn("looks unusual", calibrator.result["reason"])

    def test_cancel(self):
        calibrator = Calibrator()
        calibrator.update(0.0, 0.3)
        calibrator.cancel()
        self.assertTrue(calibrator.finished)
        self.assertEqual(calibrator.status()["phase"], "failed")
        self.assertIn("cancelled", calibrator.result["reason"])

    def test_a_finished_calibration_does_not_change_any_more(self):
        calibrator = run_calibration(Calibrator(), typical_user)
        before = dict(calibrator.result)
        calibrator.update(100.0, None)
        calibrator.cancel()
        self.assertEqual(calibrator.result, before)


# ======================================================================
# Saving the baseline
# ======================================================================
class BaselineDatabaseTests(TempDatabaseTestCase):
    def make_user(self, name="asha"):
        user, _ = auth_service.register(self.connection, name, "password123")
        return user["id"]

    def test_save_and_load(self):
        user_id = self.make_user()
        self.assertIsNone(calibration_db.get_eye_baseline(self.connection, user_id))
        calibration_db.save_eye_baseline(self.connection, user_id, {"ear_open": 0.31, "ear_closed": 0.11, "open_samples": 150})
        loaded = calibration_db.get_eye_baseline(self.connection, user_id)
        self.assertEqual((loaded["ear_open"], loaded["ear_closed"]), (0.31, 0.11))
        self.assertEqual(loaded["open_samples"], 150)
        self.assertIn("calibrated_at", loaded)

    def test_recalibrating_replaces_the_old_baseline_and_keeps_posture_data(self):
        user_id = self.make_user()
        calibration_db.save_eye_baseline(self.connection, user_id, {"ear_open": 0.30, "ear_closed": 0.10})
        self.connection.execute("UPDATE calibrations SET posture_baseline_json = '{\"x\": 1}' WHERE user_id = ?", (user_id,))
        self.connection.commit()
        calibration_db.save_eye_baseline(self.connection, user_id, {"ear_open": 0.28, "ear_closed": 0.09})
        self.assertEqual(calibration_db.get_eye_baseline(self.connection, user_id)["ear_open"], 0.28)
        row = calibration_db.get_calibration(self.connection, user_id)
        self.assertEqual(json.loads(row["posture_baseline_json"]), {"x": 1})
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM calibrations").fetchone()[0], 1)

    def test_damaged_or_unusable_data_counts_as_not_calibrated(self):
        user_id = self.make_user()
        for bad in ("not json", "{}", '{"ear_open": "x", "ear_closed": 0.1}', '{"ear_open": 0.1, "ear_closed": 0.3}',
                    '{"ear_open": 0.3, "ear_closed": 0}', "null"):
            self.connection.execute("DELETE FROM calibrations")
            self.connection.execute(
                "INSERT INTO calibrations VALUES (?, ?, '{}', 'now')", (user_id, bad))
            self.connection.commit()
            self.assertIsNone(calibration_db.get_eye_baseline(self.connection, user_id), bad)

    def test_each_user_has_their_own_baseline(self):
        first, second = self.make_user("asha"), self.make_user("bala")
        calibration_db.save_eye_baseline(self.connection, first, {"ear_open": 0.30, "ear_closed": 0.10})
        self.assertIsNone(calibration_db.get_eye_baseline(self.connection, second))
        calibration_db.save_eye_baseline(self.connection, second, {"ear_open": 0.25, "ear_closed": 0.12})
        self.assertEqual(calibration_db.get_eye_baseline(self.connection, first)["ear_open"], 0.30)
        self.assertEqual(calibration_db.get_eye_baseline(self.connection, second)["ear_open"], 0.25)

    def test_profile_reports_the_calibration(self):
        user_id = self.make_user()
        user = auth_service.get_user_from_token(self.connection, auth_service.login(self.connection, "asha", "password123")[1])
        self.assertFalse(auth_service.describe_user(self.connection, user)["calibrated"])
        calibration_db.save_eye_baseline(self.connection, user_id, {"ear_open": 0.3012, "ear_closed": 0.1}) 
        profile = auth_service.describe_user(self.connection, user)
        self.assertTrue(profile["calibrated"])
        self.assertEqual(profile["ear_open"], 0.301)


# ======================================================================
# Drawing
# ======================================================================
class EyeDrawingTests(unittest.TestCase):
    def test_eye_outline_is_red_while_the_eyes_are_closed(self):
        result = fl.analyse_faces([make_face_landmarks()])
        open_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        closed_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        drawing.draw_face_overlay(open_frame, result, eyes_closed=False)
        drawing.draw_face_overlay(closed_frame, result, eyes_closed=True)
        self.assertTrue(np.any(np.all(open_frame == drawing.YELLOW, axis=2)))
        self.assertFalse(np.any(np.all(open_frame == drawing.RED, axis=2)))
        self.assertTrue(np.any(np.all(closed_frame == drawing.RED, axis=2)))
        self.assertFalse(np.any(np.all(closed_frame == drawing.YELLOW, axis=2)))


# ======================================================================
# Monitor integration
# ======================================================================
def short_calibrator():
    return Calibrator(ready_s=1.2, open_s=2.0, close_s=2.0)     # ready step longer than the 0.8 s settle time; needs >= 10 fps


class MonitorEyeTestCase(unittest.TestCase):
    def setUp(self):
        self._monitors = []

    def tearDown(self):
        for monitor in self._monitors:
            monitor.stop()

    def build(self, ear=0.30, faces=((0.5, 0.5, 0.4),), baseline=None, calibrator_factory=short_calibrator):
        camera = FakeCamera()
        detector = FakeDetector(faces=faces, ear=ear)
        monitor = new_monitor(lambda: camera, detector_factory=lambda: detector, calibrator_factory=calibrator_factory)
        self._monitors.append(monitor)
        monitor.start(1, baseline=baseline)
        return monitor, detector

    def eye(self, monitor):
        return monitor.get_status(1).get("eye", {})

    def open_eyes_measured(self, monitor):
        """True once the eyes have really been MEASURED as open (not merely 'not closed because not measured')."""
        eye = self.eye(monitor)
        return eye.get("ear") is not None and eye.get("eyes_closed") is False


class MonitorEyeTests(MonitorEyeTestCase):
    def test_not_calibrated_shows_ear_but_no_blink_analysis(self):
        monitor, _ = self.build()
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("ear") is not None))
        eye = self.eye(monitor)
        self.assertAlmostEqual(eye["ear"], 0.30, places=2)
        self.assertFalse(eye["calibrated"])
        self.assertIsNone(eye["blink_count"])
        self.assertFalse(monitor.get_status(1)["calibrated"])

    def test_left_and_right_ear_are_reported(self):
        monitor, _ = self.build(ear=0.27)
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("ear_left") is not None))
        self.assertAlmostEqual(self.eye(monitor)["ear_left"], 0.27, places=2)
        self.assertAlmostEqual(self.eye(monitor)["ear_right"], 0.27, places=2)

    def test_closed_eyes_are_detected_and_a_long_closure_is_flagged(self):
        monitor, detector = self.build(baseline=BASELINE)
        self.assertTrue(wait_for(lambda: self.open_eyes_measured(monitor)))
        detector.ear = 0.10
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("eyes_closed") is True))
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("long_closure") is True, timeout=5))
        self.assertGreater(self.eye(monitor)["current_closure_s"], 0.4)

    def test_preview_outline_turns_red_when_the_eyes_close(self):
        monitor, detector = self.build(baseline=BASELINE)
        self.assertTrue(wait_for(lambda: self.open_eyes_measured(monitor)))   # open eyes seen first
        detector.ear = 0.10
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("eyes_closed") is True))
        self.assertTrue(wait_for(lambda: monitor.get_frame(1) is not None))

        def red_pixels():
            _, jpeg = monitor.get_frame(1)
            image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            blue, green, red = image[..., 0].astype(int), image[..., 1].astype(int), image[..., 2].astype(int)
            return int(((red > 180) & (green < 90) & (blue < 90)).sum())

        self.assertTrue(wait_for(lambda: red_pixels() > 10))

    def test_a_blink_is_counted_once(self):
        monitor, detector = self.build(baseline=BASELINE)
        self.assertTrue(wait_for(lambda: self.open_eyes_measured(monitor)))
        detector.ear = 0.10
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("eyes_closed") is True))
        detector.ear = 0.30
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("blink_count") == 1 or
                                 self.eye(monitor).get("long_closure_count") == 1))
        self.assertEqual(self.eye(monitor)["blink_count"] + self.eye(monitor)["long_closure_count"], 1)

    def test_no_face_means_no_eye_values(self):
        monitor, _ = self.build(faces=[], baseline=BASELINE)
        self.assertTrue(wait_for(lambda: "ear" in self.eye(monitor)))          # first frame has been analysed
        eye = self.eye(monitor)
        self.assertIsNone(eye["ear"])
        self.assertFalse(eye["eyes_closed"])

    def test_a_black_image_is_not_used_to_measure_the_eyes(self):
        # A hand over the lens: the (fake) model still returns a face, but the image is almost black.
        camera = FakeCamera(gray_level=5)
        detector = FakeDetector(ear=0.10)                               # would look like closed eyes
        monitor = new_monitor(lambda: camera, detector_factory=lambda: detector)
        self._monitors.append(monitor)
        monitor.start(1, baseline=BASELINE)
        self.assertTrue(wait_for(lambda: "ear" in self.eye(monitor)))
        self.assertTrue(wait_for(lambda: detector.detect_calls > 10))
        eye = self.eye(monitor)
        self.assertIsNone(eye["ear"])
        self.assertFalse(eye["eyes_closed"])
        self.assertEqual(eye["blink_count"], 0)
        self.assertEqual(monitor.get_status(1)["lighting"], "too_dark")   # the user is told why

    def test_brightness_limit_is_below_the_dark_lighting_warning(self):
        self.assertLess(config.EYE_ANALYSIS_MIN_BRIGHTNESS, config.LIGHT_TOO_DARK_BELOW)

    def test_low_frame_rate_gives_a_warning(self):
        with mock.patch.object(config, "MIN_RELIABLE_FPS", 10_000), \
             mock.patch.object(config, "FPS_WARNING_WARMUP_S", 0):
            monitor, _ = self.build()
            self.assertTrue(wait_for(lambda: "frame rate is low" in monitor.get_status(1).get("fps_warning", "")))

    def test_no_warning_when_the_frame_rate_is_fine(self):
        monitor, _ = self.build()
        self.assertTrue(wait_for(lambda: monitor.get_status(1).get("fps", 0) > 0))
        self.assertEqual(monitor.get_status(1)["fps_warning"], "")


class MonitorCalibrationTests(MonitorEyeTestCase):
    def follow_the_instructions(self, monitor, detector, timeout=8):
        """Play the part of the user: eyes open, and closed during the 'close' step. Returns the result."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            status = monitor.get_calibration_status(1)
            if status is not None:
                detector.ear = 0.10 if status["phase"] == "close" else 0.30
                if not status["active"]:
                    break
            time.sleep(0.02)
        return monitor.take_calibration_result(1)

    def test_calibration_produces_a_baseline_and_applies_it_immediately(self):
        monitor, detector = self.build()
        self.assertFalse(monitor.get_status(1)["calibrated"])
        self.assertTrue(monitor.begin_calibration(1))
        self.assertTrue(monitor.get_status(1)["calibrating"])

        result = self.follow_the_instructions(monitor, detector)
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["baseline"]["ear_open"], 0.30, delta=0.03)
        self.assertAlmostEqual(result["baseline"]["ear_closed"], 0.10, delta=0.03)

        status = monitor.get_status(1)
        self.assertTrue(status["calibrated"])
        self.assertFalse(status["calibrating"])
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("calibrated") is True))
        self.assertIsNone(monitor.take_calibration_result(1))          # the result can be collected only once

    def test_blink_counting_is_paused_during_calibration(self):
        monitor, detector = self.build(ear=0.30, baseline=BASELINE)
        self.assertTrue(wait_for(lambda: self.open_eyes_measured(monitor)))   # open eyes seen first
        detector.ear = 0.10                                                                # then eyes closed ...
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("eyes_closed") is True))
        monitor.begin_calibration(1)                                                       # ... and calibration starts
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("paused") is True))
        self.assertIsNone(self.eye(monitor)["eyes_closed"])                # no blink analysis meanwhile
        self.assertTrue(wait_for(lambda: not monitor.get_calibration_status(1)["active"], timeout=8))
        self.assertFalse(monitor.take_calibration_result(1)["ok"])         # "open" EAR 0.10 is unusable -> fails
        detector.ear = 0.30                                                # eyes open again
        self.assertTrue(wait_for(lambda: self.open_eyes_measured(monitor)))
        eye = self.eye(monitor)
        self.assertEqual(eye["long_closure_count"], 0)                     # the calibration period was not counted
        self.assertEqual(eye["blink_count"], 0)

    def test_failed_calibration_reports_why_and_keeps_the_camera_running(self):
        monitor, detector = self.build(faces=[])
        monitor.begin_calibration(1)
        result = self.follow_the_instructions(monitor, detector)
        self.assertFalse(result["ok"])
        self.assertIn("could not be measured often enough", result["reason"])
        self.assertFalse(monitor.get_status(1)["calibrated"])
        self.assertEqual(monitor.get_status(1)["state"], "running")

    def test_failed_calibration_keeps_the_old_baseline(self):
        monitor, detector = self.build(faces=[], baseline=BASELINE)
        monitor.begin_calibration(1)
        self.follow_the_instructions(monitor, detector)
        self.assertTrue(monitor.get_status(1)["calibrated"])

    def test_cancelling(self):
        monitor, _ = self.build()
        monitor.begin_calibration(1)
        monitor.cancel_calibration(1)
        self.assertTrue(wait_for(lambda: not monitor.get_calibration_status(1)["active"]))
        result = monitor.take_calibration_result(1)
        self.assertFalse(result["ok"])
        self.assertIn("cancelled", result["reason"])

    def test_other_accounts_cannot_use_or_see_the_calibration(self):
        monitor, _ = self.build()
        self.assertFalse(monitor.begin_calibration(2))
        monitor.begin_calibration(1)
        self.assertIsNone(monitor.get_calibration_status(2))
        self.assertIsNone(monitor.take_calibration_result(2))
        monitor.cancel_calibration(2)
        self.assertTrue(monitor.get_calibration_status(1)["active"])

    def test_calibration_needs_a_running_camera(self):
        monitor = new_monitor(lambda: FakeCamera())
        self._monitors.append(monitor)
        self.assertFalse(monitor.begin_calibration(1))


# ======================================================================
# The new code never saves images or video
# ======================================================================
class PrivacyTests(unittest.TestCase):
    def test_new_modules_write_no_files(self):
        for module in (eye_features, eye_tracker, calibrator_module):
            source = Path(module.__file__).read_text(encoding="utf-8")
            for forbidden in ("imwrite", "VideoWriter", "write_bytes", "write_text", ".write("):
                self.assertNotIn(forbidden, source, module.__name__)
            self.assertIsNone(re.search(r"(?<![\w.])open\(", source), module.__name__)


# ======================================================================
# API (needs fastapi; skipped otherwise)
# ======================================================================
@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class CalibrationApiTests(TempDatabaseTestCase):
    def setUp(self):
        super().setUp()
        self.detector = FakeDetector(ear=0.30)
        self._original_monitor = monitor_module.monitor
        monitor_module.monitor = new_monitor(
            lambda: FakeCamera(), detector_factory=lambda: self.detector, calibrator_factory=short_calibrator)

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

    def calibrate(self, client):
        """Start a calibration and behave like a user until it finishes. Returns every status seen."""
        self.assertEqual(client.post("/api/calibration/start").status_code, 200)
        seen, end = [], time.monotonic() + 10
        while time.monotonic() < end:
            body = client.get("/api/calibration/status").json()
            seen.append(body)
            self.detector.ear = 0.10 if body["phase"] == "close" else 0.30
            if not body["active"]:
                break
            time.sleep(0.05)
        return seen

    def test_login_is_required(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        self.assertEqual(client.post("/api/calibration/start").status_code, 401)
        self.assertEqual(client.get("/api/calibration/status").status_code, 401)

    def test_full_calibration_is_saved_to_the_account(self):
        client = self.new_client("asha")
        self.assertFalse(client.get("/api/auth/me").json()["calibrated"])
        seen = self.calibrate(client)
        self.assertEqual(sum(1 for body in seen if body["saved"]), 1)          # saved exactly once
        final = [body for body in seen if body["result"]][0]
        self.assertTrue(final["result"]["ok"])
        me = client.get("/api/auth/me").json()
        self.assertTrue(me["calibrated"])
        self.assertAlmostEqual(me["ear_open"], 0.30, delta=0.03)
        self.assertAlmostEqual(me["ear_closed"], 0.10, delta=0.03)

    def test_the_saved_baseline_is_used_the_next_time_monitoring_starts(self):
        client = self.new_client("asha")
        self.calibrate(client)
        client.post("/api/monitor/stop")
        self.assertFalse(client.get("/api/monitor/status").json()["camera_active"])
        started = client.post("/api/monitor/start").json()
        self.assertTrue(started["calibrated"])

    def test_failed_calibration_saves_nothing(self):
        self.detector.faces = []
        client = self.new_client("asha")
        seen = self.calibrate(client)
        result = [body for body in seen if body["result"]][0]["result"]
        self.assertFalse(result["ok"])
        self.assertFalse(any(body["saved"] for body in seen))
        self.assertFalse(client.get("/api/auth/me").json()["calibrated"])

    def test_calibrations_of_two_users_are_separate(self):
        first, second = self.new_client("asha"), self.new_client("bala")
        self.calibrate(first)
        first.post("/api/monitor/stop")
        self.assertTrue(first.get("/api/auth/me").json()["calibrated"])
        self.assertFalse(second.get("/api/auth/me").json()["calibrated"])

    def test_second_account_cannot_calibrate_while_the_camera_is_in_use(self):
        first, second = self.new_client("asha"), self.new_client("bala")
        first.post("/api/monitor/start")
        self.assertEqual(second.post("/api/calibration/start").status_code, 409)

    def test_missing_camera_gives_503_and_the_message(self):
        monitor_module.monitor.stop()
        monitor_module.monitor = new_monitor(lambda: FakeCamera(fail_open=True), detector_factory=lambda: self.detector)
        client = self.new_client("asha")
        response = client.post("/api/calibration/start")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Camera not detected.")


if __name__ == "__main__":
    unittest.main()
