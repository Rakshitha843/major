"""
Phase 7 tests: head pose, shoulders, posture metrics, personal baseline, rules, smoothing, calibration, storage,
and the monitor/API integration.

IMPORTANT: everything here uses SIMULATED faces and shoulders built from the same generic face model the head-pose maths
uses. So these tests prove the maths and the logic are self-consistent (a face turned by exactly 15 degrees is measured as
15 degrees; a 12 degree tilt is a WARNING; one bad frame does not trigger anything). They say NOTHING about how accurate the
angles are on real faces, and nothing about whether the thresholds are right for real people. scripts/posture_test.py checks
that on your own camera.

Run from the project root:
    python -m unittest discover -s tests -t . -v
"""

import importlib.util
import re
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend import auth_service
from computer_vision import face_landmarks as fl
from computer_vision import head_pose, pose_landmarks, posture
from computer_vision import monitor as monitor_module
from computer_vision.calibrator import Calibrator
from computer_vision.face_landmarks import ModelError
from computer_vision.pose_landmarks import PoseDetector, PoseResult, extract_shoulders
from computer_vision.posture import (GOOD, POOR, WARNING, PostureAssessor, build_posture_baseline, classify,
                                     deviations, measure_posture)
from database import calibration as calibration_db
from tests.helpers import (FakeCamera, FakeDetector, FakePoseDetector, TempDatabaseTestCase, make_face_landmarks,
                           new_monitor, shoulders_result, wait_for)
from utils import config, env_check

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)

W, H = 640, 480
HEAD0 = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "distance": 2500.0}
EYE_BASELINE = {"ear_open": 0.30, "ear_closed": 0.10}


def head(**changes):
    return {**HEAD0, **changes}


def distance_for_size_ratio(ratio):
    """Model distance that makes the face `ratio` times as wide as at HEAD0. The eye corners sit 135 units behind the
    nose tip, so the eye-to-eye width in the picture scales with 1 / (distance + 135)."""
    return (HEAD0["distance"] + 135.0) / ratio - 135.0


def metrics_for(head_pose_=None, cx=0.5, cy=0.5, pose=None):
    """Posture metrics of a simulated face (size 0.7 keeps every face point inside the face box)."""
    result = fl.analyse_faces([make_face_landmarks(cx, cy, 0.7, ear=0.30, head=head_pose_ or HEAD0)])
    return measure_posture(result.landmarks, result.bbox, pose, W, H)


def baseline_with(pose=None, n=30):
    baseline, problem = build_posture_baseline([metrics_for(pose=pose) for _ in range(n)])
    assert baseline is not None, problem
    return baseline


BASE = baseline_with()                                 # head only
BASE_WITH_SHOULDERS = baseline_with(pose=shoulders_result())


# ======================================================================
# Head pose maths
# ======================================================================
class HeadPoseTests(unittest.TestCase):
    def landmarks(self, **pose):
        return make_face_landmarks(0.5, 0.5, 0.7, head={**HEAD0, **pose})

    def test_sideways_tilt_is_read_directly_from_the_eye_line(self):
        for roll in (-25, -10, -3, 0, 4, 12, 30):
            self.assertAlmostEqual(head_pose.eye_line_roll(self.landmarks(roll=roll), W, H), roll, places=1)

    def test_pitch_and_yaw_are_recovered_from_the_fit(self):
        for yaw, pitch, roll in [(0, 0, 0), (20, 0, 0), (-25, 10, 5), (0, -15, 0), (0, 20, -12), (30, -10, 15), (10, 25, 0)]:
            estimate = head_pose.estimate_head_pose(self.landmarks(yaw=yaw, pitch=pitch, roll=roll), W, H)
            self.assertIsNotNone(estimate)
            for got, expected in zip(estimate, (yaw, pitch, roll)):
                self.assertAlmostEqual(got, expected, delta=0.3)

    def test_the_sign_of_pitch_matches_the_documented_convention(self):
        # Convention (see head_pose.py): pitch grows when the head is tilted DOWN. Looking down turns the top of the head
        # towards the camera, so the LOWER face (nose to chin) is foreshortened compared with the upper face (eyes to nose).
        def lower_to_upper_ratio(landmarks):
            nose_y, chin_y = landmarks[1, 1], landmarks[152, 1]
            eyes_y = (landmarks[33, 1] + landmarks[263, 1]) / 2
            return (chin_y - nose_y) / (nose_y - eyes_y)
        neutral, down, up = self.landmarks(pitch=0), self.landmarks(pitch=15), self.landmarks(pitch=-15)
        self.assertLess(lower_to_upper_ratio(down), lower_to_upper_ratio(neutral))       # head down: chin foreshortened
        self.assertGreater(lower_to_upper_ratio(up), lower_to_upper_ratio(neutral))      # head up: chin stretched
        self.assertGreater(head_pose.estimate_head_pose(down, W, H)[1], 10)
        self.assertLess(head_pose.estimate_head_pose(up, W, H)[1], -10)

    def test_a_face_further_away_is_smaller(self):
        near = head_pose.eye_distance(self.landmarks(distance=2000), W, H)
        far = head_pose.eye_distance(self.landmarks(distance=3000), W, H)
        self.assertGreater(near, far)
        self.assertAlmostEqual(near / far, (3000 + 135) / (2000 + 135), places=3)

    def test_moving_the_face_does_not_change_the_measured_angles(self):
        for cx, cy in ((0.35, 0.4), (0.65, 0.6)):
            landmarks = make_face_landmarks(cx, cy, 0.7, head=head(pitch=12, roll=7))
            yaw, pitch, _ = head_pose.estimate_head_pose(landmarks, W, H)
            self.assertAlmostEqual(pitch, 12, delta=0.5)
            self.assertAlmostEqual(head_pose.eye_line_roll(landmarks, W, H), 7, delta=0.5)

    def test_broken_input_gives_none_not_a_crash(self):
        flat = np.zeros((478, 3), dtype=np.float32)
        self.assertIsNone(head_pose.eye_line_roll(flat, W, H))                      # both eye corners at the same point
        estimate = head_pose.estimate_head_pose(flat, W, H)                        # all six points identical
        self.assertTrue(estimate is None or all(np.isfinite(estimate)))

    def test_roll_uses_pixels_not_picture_fractions(self):
        landmarks = self.landmarks(roll=20)
        naive = np.degrees(np.arctan2(landmarks[263, 1] - landmarks[33, 1], landmarks[263, 0] - landmarks[33, 0]))
        self.assertGreater(abs(naive - 20), 3)                                     # measuring in fractions would be wrong
        self.assertAlmostEqual(head_pose.eye_line_roll(landmarks, W, H), 20, places=1)


# ======================================================================
# Shoulders
# ======================================================================
class Pt:
    def __init__(self, x, y, visibility=0.95):
        self.x, self.y, self.visibility = x, y, visibility


def pose_points(left=(0.25, 0.8, 0.95), right=(0.75, 0.8, 0.95)):
    points = [Pt(0.5, 0.3, 0.9) for _ in range(33)]
    points[11], points[12] = Pt(*left), Pt(*right)
    return points


class ShoulderTests(unittest.TestCase):
    def test_two_confident_wide_shoulders_are_reported_left_to_right(self):
        result = extract_shoulders(pose_points(left=(0.75, 0.82), right=(0.25, 0.8)))     # given in "wrong" order
        self.assertTrue(result.visible)
        self.assertEqual(result.left_point, (0.25, 0.8))
        self.assertEqual(result.right_point, (0.75, 0.82))

    def test_unreliable_shoulders_are_not_used(self):
        cases = {
            "low confidence": pose_points(left=(0.25, 0.8, 0.3)),
            "no confidence value": pose_points(left=(0.25, 0.8, None)),
            "guessed outside the picture": pose_points(right=(0.75, 1.2)),
            "too narrow": pose_points(left=(0.45, 0.8), right=(0.55, 0.8)),
        }
        for name, points in cases.items():
            self.assertFalse(extract_shoulders(points).visible, name)

    def test_missing_or_short_lists_do_not_crash(self):
        for bad in (None, [], [Pt(0, 0)] * 5):
            self.assertFalse(extract_shoulders(bad).visible)

    def test_limits_come_from_the_configuration(self):
        borderline = pose_points(left=(0.5 - config.SHOULDER_MIN_WIDTH / 2 + 0.01, 0.8),
                                 right=(0.5 + config.SHOULDER_MIN_WIDTH / 2 - 0.01, 0.8))
        self.assertFalse(extract_shoulders(borderline).visible)


class PoseDetectorLoadingTests(unittest.TestCase):
    def test_missing_model_gives_a_fix_it_message(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ModelError) as caught:
                PoseDetector(model_path=Path(folder) / "none.task").load()
        self.assertIn("download_models.py", str(caught.exception))

    def test_damaged_model_gives_a_clear_message_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            damaged = Path(folder) / "damaged.task"
            damaged.write_bytes(b"not a real model" * 200)
            with self.assertRaises(ModelError) as caught:
                PoseDetector(model_path=damaged).load()
        self.assertIn("could not be loaded", str(caught.exception))

    def test_detect_before_load_is_refused_and_close_is_safe(self):
        detector = PoseDetector(model_path=Path("nothing.task"))
        with self.assertRaises(ModelError):
            detector.detect(np.zeros((10, 10, 3), dtype=np.uint8), 0)
        detector.close()
        detector.close()

    def test_environment_check_treats_the_pose_model_as_optional(self):
        with tempfile.TemporaryDirectory() as folder:
            result = env_check.check_pose_model_loads(Path(folder) / "none.task")
        self.assertFalse(result["ok"])
        self.assertIn("optional", result["name"].lower())


# ======================================================================
# Posture metrics
# ======================================================================
class MeasurePostureTests(unittest.TestCase):
    def test_metrics_follow_the_simulated_head(self):
        m = metrics_for(head(roll=9, pitch=14), cx=0.42, cy=0.55)
        self.assertAlmostEqual(m["roll_deg"], 9, delta=0.5)
        self.assertAlmostEqual(m["pitch_deg"], 14, delta=0.5)
        self.assertAlmostEqual(m["face_x"], 0.42, places=2)
        self.assertAlmostEqual(m["face_y"], 0.55, places=2)
        self.assertIsNone(m["shoulder_tilt_deg"])

    def test_turning_the_head_does_not_look_like_leaning_away(self):
        # Regression test: a real scripted run showed that turning the head 40 degrees away (attention: DISTRACTED)
        # ALSO shrank the measured eye-to-eye distance enough to trigger posture's "leaning away" warning (15%).
        # Foreshortening correction (measure_posture) must keep the deviation well under that limit.
        for yaw in (20, 30, 40, 50):
            ratio = metrics_for(head(yaw=yaw))["eye_distance"] / metrics_for()["eye_distance"]
            self.assertLess(abs(ratio - 1.0), 0.5 * config.POSTURE_RULE_THRESHOLDS["distance_change"][0], yaw)

    def test_shoulder_metrics_appear_only_with_visible_shoulders(self):
        m = metrics_for(pose=shoulders_result(tilt_deg=6))
        self.assertAlmostEqual(m["shoulder_tilt_deg"], 6, delta=0.3)
        self.assertGreater(m["head_shoulder_ratio"], 0)
        self.assertAlmostEqual(m["shoulder_width"], 0.55, places=2)
        self.assertIsNone(metrics_for(pose=PoseResult())["shoulder_tilt_deg"])

    def test_shoulders_above_the_eyes_are_rejected_as_nonsense(self):
        m = metrics_for(pose=shoulders_result(y=0.2))               # "shoulders" higher than the eye line
        self.assertIsNone(m["shoulder_tilt_deg"])
        self.assertIsNotNone(m["roll_deg"])                          # the head measurements are still fine

    def test_degenerate_faces_give_none(self):
        flat = np.zeros((478, 3), dtype=np.float32)
        self.assertIsNone(measure_posture(flat, (0.4, 0.4, 0.6, 0.6), None, W, H))


# ======================================================================
# The personal baseline
# ======================================================================
class BaselineTests(unittest.TestCase):
    def test_baseline_is_the_median_of_the_samples(self):
        samples = [metrics_for(head(roll=r)) for r in (1.0, 2.0, 3.0, 40.0, 2.5) * 6]         # one wild value per group
        baseline, _ = build_posture_baseline(samples)
        self.assertAlmostEqual(baseline["roll_deg"], 2.5, delta=0.3)                            # the median ignores the outlier
        self.assertEqual(baseline["samples"], 30)

    def test_too_few_samples_give_no_baseline_and_a_reason(self):
        baseline, problem = build_posture_baseline([metrics_for()] * (config.POSTURE_CAL_MIN_SAMPLES - 1))
        self.assertIsNone(baseline)
        self.assertIn("could not be measured often enough", problem)
        self.assertIsNone(build_posture_baseline([])[0])

    def test_shoulders_enter_the_baseline_only_if_seen_in_most_samples(self):
        with_pose, without = metrics_for(pose=shoulders_result()), metrics_for()
        mostly, _ = build_posture_baseline([with_pose] * 20 + [without] * 10)                 # 67% >= 60%
        rarely, _ = build_posture_baseline([with_pose] * 10 + [without] * 20)                 # 33% < 60%
        self.assertTrue(mostly["shoulders"])
        self.assertIn("head_shoulder_ratio", mostly)
        self.assertFalse(rarely["shoulders"])
        self.assertNotIn("shoulder_tilt_deg", rarely)

    def test_usability_check_for_stored_baselines(self):
        self.assertTrue(posture.baseline_is_usable(BASE))
        for bad in (None, {}, "text", {**BASE, "eye_distance": 0}, {**BASE, "roll_deg": float("nan")},
                    {k: v for k, v in BASE.items() if k != "face_x"}):
            self.assertFalse(posture.baseline_is_usable(bad), bad)


# ======================================================================
# Rules
# ======================================================================
class PostureRuleTests(unittest.TestCase):
    def state_of(self, m, baseline=BASE):
        return classify(m, baseline)[1]

    def test_the_baseline_position_itself_is_good(self):
        score, state, reasons = classify(metrics_for(), BASE)
        self.assertEqual(state, GOOD)
        self.assertLess(score, 0.05)
        self.assertEqual(reasons, [])

    def test_each_head_indicator_gives_warning_then_poor(self):
        cases = {
            "tilt": (lambda v: metrics_for(head(roll=v)), 8, 15),
            "head down": (lambda v: metrics_for(head(pitch=v)), 12, 20),
            "leaning closer": (lambda v: metrics_for(head(distance=distance_for_size_ratio(1 + v))), 0.15, 0.30),
            "leaning away": (lambda v: metrics_for(head(distance=distance_for_size_ratio(1 - v))), 0.15, 0.30),
            "sunk lower": (lambda v: metrics_for(cy=0.5 + v), 0.06, 0.12),
            "shifted sideways": (lambda v: metrics_for(cx=0.5 + v), 0.10, 0.20),
        }
        for name, (make, warning_level, poor_level) in cases.items():
            self.assertEqual(self.state_of(make(warning_level * 0.6)), GOOD, name)
            self.assertEqual(self.state_of(make(warning_level * 1.05)), WARNING, name)
            self.assertEqual(self.state_of(make(poor_level * 1.05)), POOR, name)

    def test_looking_up_or_tilting_the_other_way_is_handled_sensibly(self):
        self.assertEqual(self.state_of(metrics_for(head(pitch=-30))), GOOD)               # head UP is not "dropped"
        self.assertEqual(self.state_of(metrics_for(head(roll=-16))), POOR)                # tilt counts in both directions
        self.assertEqual(self.state_of(metrics_for(cy=0.35)), GOOD)                        # sitting HIGHER is not a slouch

    def test_shoulder_indicators_need_shoulders_in_the_frame_and_in_the_baseline(self):
        tilted = metrics_for(pose=shoulders_result(tilt_deg=12))
        self.assertEqual(self.state_of(tilted, BASE_WITH_SHOULDERS), POOR)
        self.assertEqual(self.state_of(tilted, BASE), GOOD)                                # baseline has no shoulders
        self.assertEqual(self.state_of(metrics_for(), BASE_WITH_SHOULDERS), GOOD)          # shoulders not visible now
        sunk = metrics_for(pose=shoulders_result(y=0.62))
        self.assertEqual(self.state_of(sunk, BASE_WITH_SHOULDERS), POOR)                   # head sinking to the shoulders

    def test_the_highest_indicator_decides_and_reasons_are_worst_first(self):
        m = metrics_for(head(roll=9, pitch=25))
        score, state, reasons = classify(m, BASE)
        self.assertEqual(state, POOR)
        self.assertEqual(len(reasons), 2)
        self.assertIn("dropped forward or down", reasons[0])
        self.assertIn("tilted", reasons[1])

    def test_the_distance_reason_says_which_way(self):
        self.assertIn("towards the screen", classify(metrics_for(head(distance=1900)), BASE)[2][0])
        self.assertIn("larger", classify(metrics_for(head(distance=1900)), BASE)[2][0])
        self.assertIn("away from the screen", classify(metrics_for(head(distance=3400)), BASE)[2][0])
        self.assertIn("smaller", classify(metrics_for(head(distance=3400)), BASE)[2][0])

    def test_deviations_are_signed_and_relative_to_the_baseline(self):
        d = deviations(metrics_for(head(roll=-6, pitch=10), cy=0.56), BASE)
        self.assertAlmostEqual(d["head_tilt_deg"], -6, delta=0.5)
        self.assertAlmostEqual(d["head_nod_deg"], 10, delta=0.5)
        self.assertAlmostEqual(d["head_drop_pct"], 6, delta=0.5)
        self.assertIsNone(d["shoulder_tilt_deg"])

    def test_a_different_baseline_changes_what_counts_as_good(self):
        slouch_baseline = baseline_with()
        slouch_baseline = {**slouch_baseline, "pitch_deg": 30.0}                            # user calibrated with head down
        self.assertEqual(self.state_of(metrics_for(head(pitch=30)), slouch_baseline), GOOD)
        self.assertEqual(self.state_of(metrics_for(head(pitch=30)), BASE), POOR)            # same pose, other baseline

    def test_thresholds_are_consistent(self):
        for name, (warning_level, poor_level) in config.POSTURE_RULE_THRESHOLDS.items():
            self.assertLess(0, warning_level, name)
            self.assertLess(warning_level, poor_level, name)
        self.assertGreater(config.POSTURE_ESCALATE_HOLD_S, 1)         # far more than one frame


# ======================================================================
# The assessor and smoothing
# ======================================================================
def run_assessor(assessor, seconds, metrics, start=0.0, fps=20.0, calibrating=False):
    t, snapshot = start, None
    for _ in range(int(round(seconds * fps))):
        snapshot = assessor.update(t, metrics, calibrating=calibrating)
        t += 1.0 / fps
    return snapshot, t


class PostureAssessorTests(unittest.TestCase):
    def test_reasons_when_no_assessment_is_possible(self):
        cases = [
            (PostureAssessor(BASE), None, False, "not being measured"),
            (PostureAssessor(None), metrics_for(), False, "not calibrated"),
            (PostureAssessor(BASE), metrics_for(), True, "Calibration is running"),
        ]
        for assessor, m, calibrating, expected in cases:
            snapshot = assessor.update(0.0, m, calibrating=calibrating)
            self.assertEqual(snapshot["state"], "unavailable")
            self.assertIn(expected, snapshot["message"])
            self.assertIsNone(snapshot["score"])

    def test_a_damaged_baseline_counts_as_not_calibrated(self):
        self.assertIn("not calibrated", PostureAssessor({"roll_deg": 1}).update(0.0, metrics_for())["message"])

    def test_good_posture_stays_good(self):
        snapshot, _ = run_assessor(PostureAssessor(BASE), 20, metrics_for())
        self.assertEqual(snapshot["state"], "good")
        self.assertEqual(snapshot["events"], {"warning": 0, "poor": 0})
        self.assertIn("head position only", snapshot["note"])          # no shoulders in this baseline

    def test_one_bad_frame_or_a_short_slouch_never_triggers_a_warning(self):
        a = PostureAssessor(BASE)
        _, t = run_assessor(a, 5, metrics_for())
        _, t = run_assessor(a, 0.05, metrics_for(head(roll=25)), start=t)                   # a single bad frame
        snapshot, t = run_assessor(a, 3, metrics_for(), start=t)
        self.assertEqual(snapshot["state"], "good")
        snapshot, t = run_assessor(a, config.POSTURE_ESCALATE_HOLD_S - 1.0, metrics_for(head(roll=25)), start=t)
        self.assertEqual(snapshot["state"], "good")                                          # not yet: hold time not reached
        self.assertEqual(snapshot["events"]["poor"], 0)

    def test_sustained_bad_posture_is_shown_after_the_hold_time_and_recovers(self):
        a = PostureAssessor(BASE)
        _, t = run_assessor(a, 5, metrics_for())
        snapshot, t = run_assessor(a, config.POSTURE_ESCALATE_HOLD_S + 1.0, metrics_for(head(roll=12)), start=t)
        self.assertEqual(snapshot["state"], "warning")
        self.assertEqual(snapshot["events"]["warning"], 1)
        self.assertTrue(any("tilted" in reason for reason in snapshot["reasons"]))
        snapshot, t = run_assessor(a, config.POSTURE_ESCALATE_HOLD_S + 1.0, metrics_for(head(roll=22)), start=t)
        self.assertEqual(snapshot["state"], "poor")
        self.assertEqual(snapshot["events"]["poor"], 1)
        snapshot, t = run_assessor(a, 1.0, metrics_for(), start=t)                           # improving, not yet accepted
        self.assertEqual(snapshot["state"], "poor")
        self.assertIn("improving", snapshot["reasons"][0])
        snapshot, _ = run_assessor(a, config.POSTURE_RECOVERY_HOLD_S + 0.5, metrics_for(), start=t)
        self.assertEqual(snapshot["state"], "good")
        self.assertEqual(snapshot["reasons"], [])

    def test_losing_the_face_resets_the_state_so_nothing_stale_is_shown(self):
        a = PostureAssessor(BASE)
        run_assessor(a, config.POSTURE_ESCALATE_HOLD_S + 1.0, metrics_for(head(roll=25)))
        gone = a.update(20.0, None)
        self.assertEqual(gone["state"], "unavailable")
        back = a.update(20.05, metrics_for(head(roll=25)))
        self.assertEqual(back["state"], "good")                                              # must be confirmed again

    def test_shoulders_are_reported_as_used_only_when_both_baseline_and_frame_have_them(self):
        a = PostureAssessor(BASE_WITH_SHOULDERS)
        used = a.update(0.0, metrics_for(pose=shoulders_result()))
        self.assertTrue(used["shoulders_used"])
        self.assertEqual(used["note"], "")
        self.assertIsNotNone(used["deviations"]["shoulder_tilt_deg"])
        unused = a.update(0.05, metrics_for())
        self.assertFalse(unused["shoulders_used"])
        self.assertIn("head position only", unused["note"])

    def test_every_snapshot_carries_the_disclaimer_and_the_documented_fields(self):
        snapshot = PostureAssessor(BASE).update(0.0, metrics_for())
        for key in ("state", "score", "reasons", "message", "note", "shoulders_used", "deviations", "events", "disclaimer"):
            self.assertIn(key, snapshot)
        self.assertIn("not medical advice", snapshot["disclaimer"])


# ======================================================================
# Calibration collects the posture baseline
# ======================================================================
def run_calibration(calibrator, posture_by_time, fps=20.0):
    t = 0.0
    while not calibrator.finished and t < 60:
        calibrator.update(t, 0.10 if 11 <= t < 15 else 0.30 if t >= 3 else 0.30, posture_by_time(t))
        t += 1.0 / fps
    return calibrator


class CalibratorPostureTests(unittest.TestCase):
    def eyes_ok_result(self, calibrator):
        # eye calibration needs the usual closed-eye step; reuse a simple version
        return calibrator.result

    def calibrate(self, posture_by_time):
        calibrator = Calibrator()
        t = 0.0
        while not calibrator.finished and t < 60:
            ear = 0.10 if 11.5 <= t < 15 else 0.30
            calibrator.update(t, ear, posture_by_time(t))
            t += 0.05
        return calibrator

    def test_the_baseline_comes_from_the_open_eyes_step_only(self):
        good, other = metrics_for(head(roll=2)), metrics_for(head(roll=30))
        calibrator = self.calibrate(lambda t: good if 3 <= t < 11 else other)                 # slouching in other steps
        self.assertTrue(calibrator.result["ok"])
        self.assertAlmostEqual(calibrator.result["posture_baseline"]["roll_deg"], 2, delta=0.5)
        self.assertAlmostEqual(calibrator.result["posture_baseline"]["samples"], 160, delta=2)      # 8 s at 20 samples/s

    def test_missing_head_data_keeps_the_eye_calibration_but_reports_the_posture_problem(self):
        calibrator = self.calibrate(lambda t: None)
        self.assertTrue(calibrator.result["ok"])                                                # eyes are fine
        self.assertIsNone(calibrator.result["posture_baseline"])
        self.assertIn("could not be measured often enough", calibrator.result["posture_problem"])

    def test_older_callers_without_posture_data_still_work(self):
        calibrator = Calibrator()
        t = 0.0
        while not calibrator.finished and t < 60:
            calibrator.update(t, 0.10 if 11.5 <= t < 15 else 0.30)                              # no posture argument at all
            t += 0.05
        self.assertTrue(calibrator.result["ok"])
        self.assertIsNone(calibrator.result["posture_baseline"])

    def test_the_sitting_instruction_is_part_of_the_steps(self):
        from computer_vision.calibrator import MESSAGES
        self.assertIn("Maintain your normal sitting posture.", MESSAGES["ready"])
        self.assertIn("normal sitting position", MESSAGES["open"])

    def test_shoulders_are_included_when_seen_most_of_the_time(self):
        with_pose = metrics_for(pose=shoulders_result())
        calibrator = self.calibrate(lambda t: with_pose)
        self.assertTrue(calibrator.result["posture_baseline"]["shoulders"])


# ======================================================================
# Storage
# ======================================================================
class PostureStorageTests(TempDatabaseTestCase):
    def make_user(self, name="asha"):
        user, _ = auth_service.register(self.connection, name, "password123")
        return user["id"]

    def test_both_baselines_are_saved_and_loaded(self):
        uid = self.make_user()
        calibration_db.save_calibration(self.connection, uid, dict(EYE_BASELINE), dict(BASE))
        self.assertAlmostEqual(calibration_db.get_posture_baseline(self.connection, uid)["eye_distance"], BASE["eye_distance"])
        self.assertEqual(calibration_db.get_eye_baseline(self.connection, uid)["ear_open"], 0.30)

    def test_a_new_calibration_without_posture_clears_the_old_posture_baseline(self):
        uid = self.make_user()
        calibration_db.save_calibration(self.connection, uid, dict(EYE_BASELINE), dict(BASE))
        calibration_db.save_calibration(self.connection, uid, dict(EYE_BASELINE), None)
        self.assertIsNone(calibration_db.get_posture_baseline(self.connection, uid))
        self.assertIsNotNone(calibration_db.get_eye_baseline(self.connection, uid))

    def test_saving_only_the_eye_baseline_keeps_the_posture_baseline(self):
        uid = self.make_user()
        calibration_db.save_calibration(self.connection, uid, dict(EYE_BASELINE), dict(BASE))
        calibration_db.save_eye_baseline(self.connection, uid, {"ear_open": 0.31, "ear_closed": 0.09})
        self.assertIsNotNone(calibration_db.get_posture_baseline(self.connection, uid))

    def test_damaged_or_missing_posture_data_counts_as_not_calibrated(self):
        uid = self.make_user()
        self.assertIsNone(calibration_db.get_posture_baseline(self.connection, uid))          # no row at all
        calibration_db.save_calibration(self.connection, uid, dict(EYE_BASELINE), None)
        self.assertIsNone(calibration_db.get_posture_baseline(self.connection, uid))          # "{}"
        for bad in ("not json", '{"roll_deg": 1}', '{"roll_deg": "x", "pitch_deg": 0, "eye_distance": 0.2, "face_x": 0.5, "face_y": 0.5}',
                    '{"roll_deg": 1, "pitch_deg": 0, "eye_distance": 0, "face_x": 0.5, "face_y": 0.5}', "null"):
            self.connection.execute("UPDATE calibrations SET posture_baseline_json = ? WHERE user_id = ?", (bad, uid))
            self.connection.commit()
            self.assertIsNone(calibration_db.get_posture_baseline(self.connection, uid), bad)

    def test_each_user_has_their_own_posture_baseline(self):
        first, second = self.make_user("asha"), self.make_user("bala")
        calibration_db.save_calibration(self.connection, first, dict(EYE_BASELINE), dict(BASE))
        self.assertIsNone(calibration_db.get_posture_baseline(self.connection, second))

    def test_the_profile_reports_posture_calibration_separately(self):
        uid = self.make_user()
        token = auth_service.login(self.connection, "asha", "password123")[1]
        user = auth_service.get_user_from_token(self.connection, token)
        calibration_db.save_calibration(self.connection, uid, dict(EYE_BASELINE), None)
        profile = auth_service.describe_user(self.connection, user)
        self.assertTrue(profile["calibrated"])
        self.assertFalse(profile["posture_calibrated"])
        calibration_db.save_calibration(self.connection, uid, dict(EYE_BASELINE), dict(BASE))
        self.assertTrue(auth_service.describe_user(self.connection, user)["posture_calibrated"])


# ======================================================================
# Monitor integration
# ======================================================================
class MonitorPostureTests(unittest.TestCase):
    def setUp(self):
        self._monitors = []

    def tearDown(self):
        for monitor in self._monitors:
            monitor.stop()

    def build(self, head_pose_=None, baseline=None, pose=None, pose_factory=None, eye_baseline=EYE_BASELINE, calibrator_factory=None):
        camera = FakeCamera()
        detector = FakeDetector(ear=0.30, head=head_pose_ if head_pose_ is not None else dict(HEAD0))
        pose = pose if pose is not None else FakePoseDetector()
        options = {"pose_detector_factory": pose_factory or (lambda: pose)}
        if calibrator_factory:
            options["calibrator_factory"] = calibrator_factory
        monitor = new_monitor(lambda: camera, detector_factory=lambda: detector, **options)
        self._monitors.append(monitor)
        monitor.start(1, baseline=eye_baseline, posture_baseline=baseline)
        return monitor, detector, pose

    def posture(self, monitor):
        return monitor.get_status(1).get("posture", {})

    def test_not_calibrated_for_posture_says_so(self):
        monitor, _, _ = self.build(baseline=None)
        self.assertTrue(wait_for(lambda: "not calibrated" in self.posture(monitor).get("message", "")))
        self.assertFalse(monitor.get_status(1)["posture_calibrated"])

    def test_good_then_poor_then_good_through_the_whole_pipeline(self):
        with mock.patch.object(config, "POSTURE_ESCALATE_HOLD_S", 0.6), mock.patch.object(config, "POSTURE_RECOVERY_HOLD_S", 0.6):
            monitor, detector, _ = self.build(baseline=BASE)
            self.assertTrue(wait_for(lambda: self.posture(monitor).get("state") == "good", timeout=6))

            detector.head = head(roll=22)                                                     # head tilted sideways
            self.assertTrue(wait_for(lambda: self.posture(monitor).get("state") == "poor", timeout=6))
            snapshot = self.posture(monitor)
            self.assertTrue(any("tilted" in reason for reason in snapshot["reasons"]))
            self.assertEqual(snapshot["events"]["poor"], 1)
            self.assertAlmostEqual(snapshot["deviations"]["head_tilt_deg"], 22, delta=1.0)

            detector.head = dict(HEAD0)                                                       # sitting normally again
            self.assertTrue(wait_for(lambda: self.posture(monitor).get("state") == "good", timeout=6))

    def test_shoulders_from_the_pose_model_are_used_when_visible(self):
        with mock.patch.object(config, "POSTURE_ESCALATE_HOLD_S", 0.5):
            baseline = BASE_WITH_SHOULDERS
            monitor, _, pose = self.build(baseline=baseline, pose=FakePoseDetector(result=shoulders_result(tilt_deg=0)))
            self.assertTrue(wait_for(lambda: self.posture(monitor).get("shoulders_used") is True, timeout=6))
            pose.result = shoulders_result(tilt_deg=13)
            self.assertTrue(wait_for(lambda: self.posture(monitor).get("state") == "poor", timeout=6))
            self.assertTrue(any("Shoulders uneven" in r for r in self.posture(monitor)["reasons"]))

    def test_the_shoulder_model_runs_only_on_some_frames(self):
        monitor, detector, pose = self.build(baseline=BASE, pose=FakePoseDetector(result=shoulders_result()))
        self.assertTrue(wait_for(lambda: self.posture(monitor).get("state") == "good", timeout=6))
        time.sleep(2.0)
        frames = monitor.get_status(1)["fps"] * monitor.get_status(1)["uptime_s"]
        self.assertGreater(pose.detect_calls, 3)
        self.assertLess(pose.detect_calls, detector.detect_calls * 0.5)                          # roughly every 4th frame

    def test_a_missing_shoulder_model_does_not_stop_the_camera_and_is_explained(self):
        pose = FakePoseDetector(fail_load="Shoulder model not found. Run this once (needs internet): python scripts/download_models.py")
        monitor, _, _ = self.build(baseline=BASE, pose=pose)
        self.assertEqual(monitor.get_status(1)["state"], "running")
        self.assertTrue(wait_for(lambda: "head position only" in self.posture(monitor).get("shoulder_model_message", "")))
        self.assertTrue(wait_for(lambda: self.posture(monitor).get("state") == "good", timeout=6))      # head-only posture works
        self.assertTrue(pose.closed or not pose.loaded)

    def test_the_shoulder_model_is_released_when_monitoring_stops(self):
        monitor, _, pose = self.build(baseline=BASE)
        self.assertTrue(wait_for(lambda: pose.loaded))
        monitor.stop()
        self.assertTrue(pose.closed)

    def test_calibration_through_the_monitor_produces_a_posture_baseline_and_activates_posture(self):
        with mock.patch.object(config, "POSTURE_CAL_MIN_SAMPLES", 10):
            monitor, detector, _ = self.build(baseline=None, eye_baseline=None,
                                              calibrator_factory=lambda: Calibrator(ready_s=1.2, open_s=2.0, close_s=2.0))
            self.assertTrue(monitor.begin_calibration(1))
            result, end = None, time.monotonic() + 10
            while time.monotonic() < end:
                status = monitor.get_calibration_status(1)
                if status is not None:
                    detector.ear = 0.10 if status["phase"] == "close" else 0.30
                    if not status["active"]:
                        break
                time.sleep(0.02)
            result = monitor.take_calibration_result(1)
            self.assertTrue(result["ok"], result)
            self.assertIsNotNone(result["posture_baseline"])
            self.assertAlmostEqual(result["posture_baseline"]["roll_deg"], 0, delta=1.0)
            self.assertTrue(monitor.get_status(1)["posture_calibrated"])
            self.assertTrue(wait_for(lambda: self.posture(monitor).get("state") == "good", timeout=6))

    def test_another_account_never_sees_the_posture(self):
        monitor, _, _ = self.build(baseline=BASE)
        self.assertTrue(wait_for(lambda: self.posture(monitor).get("state")))
        self.assertNotIn("posture", monitor.get_status(2))


class PrivacyTests(unittest.TestCase):
    def test_posture_code_saves_no_images_and_opens_no_files(self):
        for module in (head_pose, pose_landmarks, posture):
            source = Path(module.__file__).read_text(encoding="utf-8")
            for forbidden in ("imwrite", "VideoWriter", "write_bytes", "write_text", ".write("):
                self.assertNotIn(forbidden, source, module.__name__)
            self.assertIsNone(re.search(r"(?<![\w.])open\(", source), module.__name__)


# ======================================================================
# API (needs fastapi)
# ======================================================================
@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class PostureApiTests(TempDatabaseTestCase):
    def setUp(self):
        super().setUp()
        self.detector = FakeDetector(ear=0.30, head=dict(HEAD0))
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

    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        client.post("/api/auth/register", json={"username": "asha", "password": "password123"})
        return client

    def calibrate(self, client):
        client.post("/api/calibration/start")
        end = time.monotonic() + 10
        body = {}
        while time.monotonic() < end:
            body = client.get("/api/calibration/status").json()
            self.detector.ear = 0.10 if body["phase"] == "close" else 0.30
            if not body["active"]:
                break
            time.sleep(0.05)
        return body

    def test_the_posture_baseline_is_saved_with_the_calibration(self):
        client = self.client()
        me = client.get("/api/auth/me").json()
        self.assertFalse(me["posture_calibrated"])
        self.calibrate(client)
        me = client.get("/api/auth/me").json()
        self.assertTrue(me["calibrated"])
        self.assertTrue(me["posture_calibrated"])

    def test_the_saved_baseline_is_used_when_monitoring_starts_again(self):
        client = self.client()
        self.calibrate(client)
        client.post("/api/monitor/stop")
        started = client.post("/api/monitor/start").json()
        self.assertTrue(started["posture_calibrated"])
        self.assertTrue(wait_for(lambda: client.get("/api/monitor/status").json().get("posture", {}).get("state")))
        for key in ("state", "score", "reasons", "message", "note", "events", "disclaimer"):
            self.assertIn(key, client.get("/api/monitor/status").json()["posture"])


class ConfigTests(unittest.TestCase):
    def test_frame_skipping_and_limits_are_sensible(self):
        self.assertGreaterEqual(config.POSE_EVERY_N_FRAMES, 2)
        self.assertGreater(config.POSE_RESULT_MAX_AGE_S, config.POSE_EVERY_N_FRAMES / config.TARGET_FPS)
        self.assertTrue(0 < config.SHOULDER_MIN_VISIBILITY < 1)
        self.assertEqual(set(config.POSTURE_RULE_THRESHOLDS),
                         {"head_tilt_deg", "head_down_deg", "distance_change", "head_drop", "head_shift",
                          "shoulder_tilt_deg", "head_shoulder_gap"})


if __name__ == "__main__":
    unittest.main()
