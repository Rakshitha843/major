"""
Phase 6 tests: fatigue rules, state smoothing, the assessor, the SVM pipeline, the collector, and the
monitor/API integration.

IMPORTANT: everything here uses SIMULATED eye behaviour and SYNTHETIC data. These tests check that the code does
what it is designed to do (for example "a 1.5 second closure is FATIGUED", "training refuses too little data").
They say nothing about how accurate the fatigue assessment is for real people. No real dataset exists yet.

Run from the project root:
    python -m unittest discover -s tests -t . -v
"""

import csv
import importlib.util
import io
import contextlib
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.model_selection import KFold

import joblib

from computer_vision import monitor as monitor_module
from computer_vision.eye_tracker import EyeStateTracker
from ml import dataset, features, make_demo_data, model as model_module, rules, train
from ml.dataset import COLUMNS, DatasetError, append_rows, load_rows
from ml.fatigue import DISCLAIMER, FatigueAssessor, MODE_LABELS, StateSmoother
from ml.features import FEATURE_NAMES
from ml.model import ModelLoadError, describe_model, load_model
from tests.helpers import FakeCamera, FakeDetector, TempDatabaseTestCase, new_monitor, wait_for
from utils import config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import collect_fatigue_data  # noqa: E402

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)

BASELINE = {"ear_open": 0.30, "ear_closed": 0.10}
OPEN, CLOSED = 0.30, 0.10


# ======================================================================
# Safety net: no test in this file can ever touch your real data or model files
# ======================================================================
_module_patches = []
_module_folder = None


def setUpModule():
    """Point every data/model path at a temporary folder for the whole file, so that even a broken safeguard
    can never create (for example) a model in your real models/ folder, which the app would then load."""
    global _module_folder
    _module_folder = tempfile.TemporaryDirectory()
    for name, file_name in (("FATIGUE_MODEL_FILE", "safe_fatigue_svm.joblib"),
                            ("DEMO_FATIGUE_MODEL_FILE", "safe_DEMO_fatigue_svm.joblib"),
                            ("DATASET_FILE", "safe_fatigue_features.csv"),
                            ("DEMO_DATASET_FILE", "safe_DEMO_fatigue_features.csv")):
        patcher = mock.patch.object(config, name, Path(_module_folder.name) / file_name)
        patcher.start()
        _module_patches.append(patcher)


def tearDownModule():
    for patcher in _module_patches:
        patcher.stop()
    _module_patches.clear()
    _module_folder.cleanup()


# ======================================================================
# Helpers
# ======================================================================
def simulate(tracker, assessor, pattern, fps=20.0, start=0.0):
    """Feed a tracker and an assessor a list of (seconds, ear or None). Returns (last snapshot, next time)."""
    t, snapshot = start, None
    for seconds, ear in pattern:
        for _ in range(int(round(seconds * fps))):
            eye = tracker.update(t, ear)
            snapshot = assessor.update(t, eye)
            t += 1.0 / fps
    return snapshot, t


def blinking(seconds, period=4.0, blink_s=0.15):
    """A person blinking once every `period` seconds, each blink lasting `blink_s`."""
    return [(period - blink_s, OPEN), (blink_s, CLOSED)] * int(round(seconds / period))


class StubModel:
    """Pretends to be a trained model that always returns the same probability."""

    def __init__(self, probability):
        self.probability = probability
        self.warning = ""

    def fatigue_probability(self, values):
        return self.probability

    def summary(self):
        return {"rows": 1}


def new_assessor():
    return FatigueAssessor(model=None)


def make_real_flagged_dataset(folder):
    """Synthetic rows re-labelled source='real', ONLY so the tests can exercise the 'trained model' path."""
    demo_path = Path(folder) / "demo.csv"
    make_demo_data.generate(demo_path, sessions_per_class=4, rows_per_session=20, participants=4)
    rows, _ = load_rows([demo_path])
    real_path = Path(folder) / "flagged_real.csv"
    for row in rows:
        row["source"] = "real"
        row["participant"] = row["participant"].replace("DEMO", "T")
        row["session_id"] = row["session_id"].replace("demo_", "test_")
    append_rows(real_path, rows)
    return real_path


# ======================================================================
# Configuration sanity
# ======================================================================
class ConfigTests(unittest.TestCase):
    def test_rule_thresholds_are_ordered(self):
        self.assertEqual(set(config.FATIGUE_RULE_THRESHOLDS), {"perclos", "closure_s", "blink_rate", "blink_duration_s"})
        for warning_level, fatigued_level in config.FATIGUE_RULE_THRESHOLDS.values():
            self.assertLess(0, warning_level)
            self.assertLess(warning_level, fatigued_level)

    def test_fatigue_closure_limit_is_a_long_closure(self):
        warning_level, fatigued_level = config.FATIGUE_RULE_THRESHOLDS["closure_s"]
        self.assertGreaterEqual(warning_level, config.BLINK_MAX_S)      # a normal blink must never trigger it
        self.assertGreater(fatigued_level, warning_level)

    def test_probability_thresholds_are_ordered(self):
        self.assertLess(config.ML_WARNING_PROBABILITY, config.ML_FATIGUED_PROBABILITY)

    def test_demo_and_real_files_are_different(self):
        self.assertNotEqual(config.DEMO_FATIGUE_MODEL_FILE, config.FATIGUE_MODEL_FILE)
        self.assertNotEqual(config.DEMO_DATASET_FILE, config.DATASET_FILE)
        self.assertIn("DEMO", config.DEMO_FATIGUE_MODEL_FILE.name)
        self.assertIn("DEMO", config.DEMO_DATASET_FILE.name)


# ======================================================================
# Rule-based mode
# ======================================================================
class IndicatorScoreTests(unittest.TestCase):
    def test_score_is_half_at_the_warning_level_and_one_at_the_fatigued_level(self):
        score = rules.indicator_score
        self.assertEqual(score(None, 0.1, 0.2), 0.0)
        self.assertEqual(score(0.0, 0.1, 0.2), 0.0)
        self.assertAlmostEqual(score(0.05, 0.1, 0.2), 0.25)
        self.assertAlmostEqual(score(0.1, 0.1, 0.2), 0.5)
        self.assertAlmostEqual(score(0.15, 0.1, 0.2), 0.75)
        self.assertAlmostEqual(score(0.2, 0.1, 0.2), 1.0)
        self.assertAlmostEqual(score(5.0, 0.1, 0.2), 1.0)

    def test_the_score_line_is_not_the_same_when_the_two_levels_are_far_apart(self):
        score = rules.indicator_score
        self.assertAlmostEqual(score(0.05, 0.1, 0.5), 0.25)      # first half: 0 -> 0.5 across 0 .. warning level
        self.assertAlmostEqual(score(0.3, 0.1, 0.5), 0.75)       # second half: 0.5 -> 1.0 across warning .. fatigued
        self.assertAlmostEqual(score(0.09, 4.0, 5.0), 0.01125)   # far below the warning level stays far below 0.5

    def test_state_from_score(self):
        self.assertEqual(rules.state_from_score(0.49), rules.NORMAL)
        self.assertEqual(rules.state_from_score(0.5), rules.WARNING)
        self.assertEqual(rules.state_from_score(0.99), rules.WARNING)
        self.assertEqual(rules.state_from_score(1.0), rules.FATIGUED)


def eye(**overrides):
    """Eye features of a healthy, calibrated person, with some values changed."""
    base = {"calibrated": True, "paused": False, "ear": 0.3, "perclos": 0.03, "longest_closure_s": 0.2,
            "current_closure_s": 0.0, "blinks_per_min": 15.0, "mean_blink_duration_s": 0.15, "window_valid_s": 20.0}
    base.update(overrides)
    return base


class RuleClassifierTests(unittest.TestCase):
    def test_a_healthy_person_is_normal(self):
        result = rules.classify(eye())
        self.assertEqual(result.state, rules.NORMAL)
        self.assertLess(result.score, 0.5)
        self.assertEqual(result.reasons, [])

    def test_each_indicator_can_trigger_warning_and_fatigued(self):
        cases = {
            "perclos": (0.10, 0.20),
            "longest_closure_s": (0.5, 1.0),
            "blinks_per_min": (30.0, 45.0),
            "mean_blink_duration_s": (0.30, 0.45),
        }
        for key, (warning_level, fatigued_level) in cases.items():
            self.assertEqual(rules.classify(eye(**{key: warning_level})).state, rules.WARNING, key)
            self.assertEqual(rules.classify(eye(**{key: fatigued_level})).state, rules.FATIGUED, key)
            self.assertEqual(rules.classify(eye(**{key: warning_level * 0.9})).state, rules.NORMAL, key)

    def test_the_highest_indicator_decides_and_reasons_are_listed_worst_first(self):
        result = rules.classify(eye(perclos=0.11, longest_closure_s=1.4))
        self.assertEqual(result.state, rules.FATIGUED)
        self.assertEqual(len(result.reasons), 2)
        self.assertIn("eye closure", result.reasons[0])            # the fatigued one first
        self.assertIn("Eyes mostly closed", result.reasons[1])

    def test_a_closure_happening_right_now_is_flagged_immediate(self):
        now = rules.classify(eye(current_closure_s=1.05, longest_closure_s=1.05))
        self.assertTrue(now.immediate)
        self.assertEqual(now.state, rules.FATIGUED)
        earlier = rules.classify(eye(current_closure_s=0.0, longest_closure_s=1.4))
        self.assertFalse(earlier.immediate)                        # it happened in the window, not now
        self.assertEqual(earlier.state, rules.FATIGUED)

    def test_missing_values_are_ignored_not_guessed(self):
        result = rules.classify(eye(mean_blink_duration_s=None, perclos=None))
        self.assertEqual(result.state, rules.NORMAL)

    def test_window_only_mode_ignores_the_current_closure(self):
        result = rules.classify(eye(current_closure_s=3.0, longest_closure_s=0.2), include_current=False)
        self.assertEqual(result.state, rules.NORMAL)

    def test_reasons_use_plain_language_with_the_limits(self):
        text = rules.classify(eye(blinks_per_min=50.0)).reasons[0]
        self.assertIn("50 per minute", text)
        self.assertIn("warning 30", text)


# ======================================================================
# State smoothing
# ======================================================================
class StateSmootherTests(unittest.TestCase):
    def test_a_worse_state_must_last_before_it_is_shown(self):
        s = StateSmoother(escalate_hold_s=3.0, recovery_hold_s=10.0)
        self.assertEqual(s.update(0.0, "warning"), "normal")
        self.assertEqual(s.update(2.9, "warning"), "normal")
        self.assertEqual(s.update(3.0, "warning"), "warning")

    def test_a_short_spike_is_ignored(self):
        s = StateSmoother(3.0, 10.0)
        s.update(0.0, "fatigued")
        s.update(1.0, "fatigued")
        s.update(1.5, "normal")                                    # spike over
        self.assertEqual(s.update(5.0, "normal"), "normal")

    def test_a_closure_that_is_already_too_long_is_shown_at_once(self):
        s = StateSmoother(3.0, 10.0)
        self.assertEqual(s.update(0.0, "fatigued", immediate=True), "fatigued")

    def test_only_what_held_the_whole_time_counts(self):
        s = StateSmoother(3.0, 10.0)
        for t, raw in ((0.0, "fatigued"), (1.0, "warning"), (2.0, "fatigued"), (3.1, "fatigued")):
            shown = s.update(t, raw)
        self.assertEqual(shown, "warning")                         # fatigued did not last 3 s, warning did

    def test_a_better_state_takes_longer_to_show(self):
        s = StateSmoother(3.0, 10.0)
        s.update(0.0, "fatigued", immediate=True)
        s.update(1.0, "normal")                                    # the better readings start here
        self.assertEqual(s.update(10.9, "normal"), "fatigued")     # 9.9 s of better readings: not yet
        self.assertEqual(s.update(11.0, "normal"), "normal")       # 10 s: now it is shown

    def test_stepping_down_only_as_far_as_the_readings_held(self):
        s = StateSmoother(3.0, 10.0)
        s.update(0.0, "fatigued", immediate=True)
        for t, raw in ((1.0, "warning"), (5.0, "normal"), (11.0, "warning")):
            shown = s.update(t, raw)
        self.assertEqual(shown, "warning")

    def test_flickering_around_the_shown_state_changes_nothing(self):
        s = StateSmoother(3.0, 10.0)
        s.update(0.0, "warning", immediate=True)
        for i in range(1, 60):
            self.assertEqual(s.update(i * 0.5, "normal" if i % 2 else "warning"), "warning")

    def test_reset(self):
        s = StateSmoother(3.0, 10.0)
        s.update(0.0, "fatigued", immediate=True)
        s.reset()
        self.assertEqual(s.shown, "normal")


# ======================================================================
# Features
# ======================================================================
class FeatureTests(unittest.TestCase):
    def test_feature_names_and_order(self):
        self.assertEqual(FEATURE_NAMES, ["mean_blink_duration_s", "blinks_per_min", "perclos", "longest_closure_s"])

    def test_vector_follows_the_order_and_marks_missing_values(self):
        vector = features.feature_vector({"perclos": 0.1, "blinks_per_min": 12.0, "longest_closure_s": 0.4,
                                          "mean_blink_duration_s": None})
        self.assertTrue(np.isnan(vector[0]))
        self.assertEqual(vector[1:], [12.0, 0.1, 0.4])

    def test_window_readiness(self):
        self.assertTrue(features.window_features_ready(eye()))
        self.assertFalse(features.window_features_ready(eye(calibrated=False)))
        self.assertFalse(features.window_features_ready(eye(paused=True)))
        self.assertFalse(features.window_features_ready(eye(blinks_per_min=None)))
        self.assertFalse(features.window_features_ready({}))

    def test_light_intensity_is_not_a_classifier_input(self):
        self.assertFalse(any("light" in name or "bright" in name for name in FEATURE_NAMES))


# ======================================================================
# The assessor
# ======================================================================
class AssessorUnavailableTests(unittest.TestCase):
    def test_reasons_when_no_assessment_is_possible(self):
        a = new_assessor()
        cases = [
            ({}, "Waiting for eye measurements"),
            (eye(calibrated=False), "calibrate"),
            (eye(paused=True), "Calibration is running"),
            (eye(ear=None), "not visible"),
            (eye(blinks_per_min=None, perclos=None, window_valid_s=3.0), "Collecting data (7 s more)"),
        ]
        for features_dict, expected in cases:
            snapshot = a.update(0.0, features_dict)
            self.assertEqual(snapshot["state"], "unavailable")
            self.assertIn(expected, snapshot["message"])
            self.assertIsNone(snapshot["score"])

    def test_idle_snapshot(self):
        self.assertEqual(new_assessor().idle_snapshot()["state"], "unavailable")


class AssessorScenarioTests(unittest.TestCase):
    """Whole scenarios: simulated eyes -> tracker -> assessor."""

    def setUp(self):
        self.tracker = EyeStateTracker(BASELINE)
        self.assessor = new_assessor()

    def test_the_first_seconds_report_that_data_is_being_collected(self):
        snapshot, _ = simulate(self.tracker, self.assessor, blinking(6))
        self.assertEqual(snapshot["state"], "unavailable")
        self.assertIn("Collecting data", snapshot["message"])

    def test_a_normal_blinker_stays_normal_and_no_episodes_are_counted(self):
        snapshot, _ = simulate(self.tracker, self.assessor, blinking(90))
        self.assertEqual(snapshot["state"], "normal")
        self.assertEqual(snapshot["events"], {"warning": 0, "fatigued": 0})
        self.assertEqual(snapshot["reasons"], [])
        self.assertLess(snapshot["score"], 0.5)
        self.assertEqual(snapshot["mode_label"], "DEMO / RULE-BASED MODE")

    def test_a_long_closure_is_fatigued_at_once_and_counted_once(self):
        _, t = simulate(self.tracker, self.assessor, blinking(40))
        snapshot, t = simulate(self.tracker, self.assessor, [(1.3, CLOSED)], start=t)      # eyes shut for 1.3 s
        self.assertEqual(snapshot["state"], "fatigued")
        self.assertTrue(any("eye closure" in reason for reason in snapshot["reasons"]))
        self.assertEqual(snapshot["events"]["fatigued"], 1)

        snapshot, t = simulate(self.tracker, self.assessor, blinking(12), start=t)          # still in the 20 s window
        self.assertEqual(snapshot["state"], "fatigued")
        snapshot, t = simulate(self.tracker, self.assessor, blinking(36), start=t)          # window and hold have passed
        self.assertEqual(snapshot["state"], "normal")
        self.assertEqual(snapshot["events"]["fatigued"], 1)                                  # one episode, not many

    def test_slow_blinks_give_a_warning_after_the_hold_time_but_not_fatigued(self):
        snapshot, _ = simulate(self.tracker, self.assessor, blinking(60, period=3.0, blink_s=0.4))
        self.assertEqual(snapshot["state"], "warning")
        self.assertEqual(snapshot["events"]["warning"], 1)
        self.assertEqual(snapshot["events"]["fatigued"], 0)
        self.assertTrue(snapshot["reasons"])

    def test_losing_the_face_makes_the_assessment_unavailable_and_it_resumes(self):
        _, t = simulate(self.tracker, self.assessor, blinking(40))
        snapshot, t = simulate(self.tracker, self.assessor, [(2.0, None)], start=t)
        self.assertEqual(snapshot["state"], "unavailable")
        self.assertIn("not visible", snapshot["message"])
        snapshot, _ = simulate(self.tracker, self.assessor, blinking(8), start=t)
        self.assertEqual(snapshot["state"], "normal")

    def test_without_calibration_no_fatigue_is_ever_claimed(self):
        tracker = EyeStateTracker(None)
        snapshot, _ = simulate(tracker, self.assessor, [(30, CLOSED)])
        self.assertEqual(snapshot["state"], "unavailable")
        self.assertIn("calibrate", snapshot["message"])

    def test_every_snapshot_carries_the_disclaimer(self):
        snapshot, _ = simulate(self.tracker, self.assessor, blinking(30))
        self.assertEqual(snapshot["disclaimer"], DISCLAIMER)
        self.assertIn("not a medical diagnosis", DISCLAIMER)


class AssessorModeTests(unittest.TestCase):
    def feed_probability(self, probability, seconds=6.0):
        assessor = FatigueAssessor(model=StubModel(probability))
        snapshot = None
        for i in range(int(seconds * 4)):
            snapshot = assessor.update(i * 0.25, eye())
        return snapshot

    def test_model_probability_is_turned_into_states_with_the_configured_thresholds(self):
        self.assertEqual(self.feed_probability(config.ML_WARNING_PROBABILITY - 0.01)["state"], "normal")
        self.assertEqual(self.feed_probability(config.ML_WARNING_PROBABILITY)["state"], "warning")
        self.assertEqual(self.feed_probability(config.ML_FATIGUED_PROBABILITY - 0.01)["state"], "warning")
        self.assertEqual(self.feed_probability(config.ML_FATIGUED_PROBABILITY)["state"], "fatigued")

    def test_ml_mode_is_labelled_and_reports_the_probability_as_the_score(self):
        snapshot = self.feed_probability(0.55)
        self.assertEqual(snapshot["mode"], "ml")
        self.assertEqual(snapshot["mode_label"], "TRAINED ML MODE")
        self.assertAlmostEqual(snapshot["score"], 0.55)
        self.assertIn("55%", snapshot["reasons"][0])
        self.assertEqual(snapshot["rule_state"], "normal")          # the rules are still reported for comparison

    def test_after_the_eyes_were_not_visible_the_state_starts_again_from_normal(self):
        # A stale "fatigued" must not be shown after the person has been out of view; it has to be confirmed again.
        # (An episode interrupted this way is counted again when it is confirmed again.)
        assessor = FatigueAssessor(model=StubModel(0.9))
        for i in range(24):
            snapshot = assessor.update(i * 0.25, eye())
        self.assertEqual(snapshot["state"], "fatigued")
        gone = assessor.update(6.0, eye(ear=None))
        self.assertEqual(gone["state"], "unavailable")
        back = assessor.update(6.25, eye())
        self.assertEqual(back["state"], "normal")                     # not confirmed yet
        for i in range(1, 14):
            back = assessor.update(6.25 + i * 0.25, eye())
        self.assertEqual(back["state"], "fatigued")                   # confirmed again after the hold time

    def test_without_a_model_the_mode_is_rules(self):
        self.assertEqual(new_assessor().info()["mode_label"], "DEMO / RULE-BASED MODE")
        self.assertEqual(set(MODE_LABELS.values()), {"DEMO / RULE-BASED MODE", "TRAINED ML MODE"})

    def test_a_missing_model_file_gives_rules_mode_and_a_helpful_message(self):
        with tempfile.TemporaryDirectory() as folder:
            assessor = FatigueAssessor.from_files(Path(folder) / "nothing.joblib")
        info = assessor.info()
        self.assertEqual(info["mode"], "rules")
        self.assertIn("DEMO / RULE-BASED MODE", info["mode_message"])
        self.assertIn("python -m ml.train", info["mode_message"])


# ======================================================================
# Dataset files
# ======================================================================
class DatasetTests(unittest.TestCase):
    def row(self, **overrides):
        base = {"participant": "P01", "session_id": "s1", "source": "real", "label": "normal",
                "timestamp": "2026-09-20T10:00:00", "mean_blink_duration_s": 0.15, "blinks_per_min": 16.0,
                "perclos": 0.04, "longest_closure_s": 0.3}
        base.update(overrides)
        return base

    def test_rows_survive_a_round_trip_and_blank_means_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "d.csv"
            append_rows(path, [self.row(), self.row(mean_blink_duration_s=None, label="fatigued")])
            rows, problems = load_rows([path])
        self.assertEqual(problems, [])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["blinks_per_min"], 16.0)
        self.assertTrue(np.isnan(rows[1]["mean_blink_duration_s"]))

    def test_the_file_contains_only_the_documented_columns_no_images(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "d.csv"
            append_rows(path, [self.row()])
            header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertEqual(header, COLUMNS)

    def test_appending_does_not_repeat_the_header(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "d.csv"
            append_rows(path, [self.row()])
            append_rows(path, [self.row()])
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 3)

    def test_bad_rows_are_skipped_and_explained(self):
        bad = [self.row(label="sleepy"), self.row(source="maybe"), self.row(participant=""),
               self.row(perclos="abc"), self.row(perclos=-0.5), self.row(perclos="inf"),
               self.row(mean_blink_duration_s=None, blinks_per_min=None, perclos=None, longest_closure_s=None)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "d.csv"
            append_rows(path, [self.row()] + bad)
            rows, problems = load_rows([path])
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(problems), len(bad))
        self.assertTrue(all("line" in problem for problem in problems))

    def test_missing_file_and_missing_columns_raise_clear_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(DatasetError):
                load_rows([Path(folder) / "nope.csv"])
            broken = Path(folder) / "broken.csv"
            broken.write_text("participant,label\nP01,normal\n", encoding="utf-8")
            with self.assertRaises(DatasetError) as caught:
                load_rows([broken])
        self.assertIn("missing the columns", str(caught.exception))

    def test_fingerprint_changes_when_the_data_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "d.csv"
            append_rows(path, [self.row()])
            before = dataset.dataset_sha256([path])
            append_rows(path, [self.row(perclos=0.5)])
            self.assertNotEqual(before, dataset.dataset_sha256([path]))


# ======================================================================
# Training
# ======================================================================
class TrainingMetricsTests(unittest.TestCase):
    def test_metrics_on_known_answers(self):
        m = train.binary_metrics([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1])
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (2, 1, 2, 3))
        self.assertAlmostEqual(m["accuracy"], 5 / 8)
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))

    def test_undefined_scores_are_none_not_zero(self):
        m = train.binary_metrics([0, 0, 0], [0, 0, 0])              # nobody fatigued, nobody predicted
        self.assertIsNone(m["precision"])
        self.assertIsNone(m["recall"])
        self.assertIsNone(m["f1"])
        self.assertEqual(m["accuracy"], 1.0)


class GroupedValidationTests(unittest.TestCase):
    def test_test_rows_come_from_people_the_model_has_not_seen(self):
        # Every session gets a RANDOM label and its own fingerprint. Rows of a session are near-identical.
        # A model can only score well by recognising the session, which is useless for new sessions.
        rng = np.random.default_rng(1)
        n_sessions, per_session = 30, 20
        labels = rng.integers(0, 2, n_sessions)
        centres = rng.normal(0, 3, (n_sessions, 4))
        X = np.vstack([centres[s] + rng.normal(0, 0.05, (per_session, 4)) for s in range(n_sessions)])
        y = np.repeat(labels, per_session)
        groups = [f"s{s}" for s in range(n_sessions) for _ in range(per_session)]

        grouped_probabilities, _ = train.cross_validate(X, y, groups)
        grouped_accuracy = train.binary_metrics(y, (grouped_probabilities >= 0.5).astype(int))["accuracy"]

        leaky = np.zeros(len(y))                                     # what a careless random split would do
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            m = train.build_pipeline().fit(X[tr], y[tr])
            leaky[te] = m.predict_proba(X[te])[:, list(m.classes_).index(1)]
        leaky_accuracy = train.binary_metrics(y, (leaky >= 0.5).astype(int))["accuracy"]

        self.assertGreater(leaky_accuracy, 0.9, "the careless split should look (falsely) excellent")
        self.assertLess(grouped_accuracy, 0.75, "the grouped test must expose that nothing real was learned")

    def test_groups_are_participants_when_there_are_three_or_more_otherwise_sessions(self):
        many = [{"participant": f"P{i % 3}", "session_id": f"s{i}"} for i in range(9)]
        few = [{"participant": f"P{i % 2}", "session_id": f"s{i}"} for i in range(9)]
        self.assertEqual(train.choose_groups(many)[1], "participant")
        self.assertEqual(train.choose_groups(few)[1], "session")
        self.assertEqual(train.choose_groups(few)[0], [f"s{i}" for i in range(9)])


class TrainingRefusalTests(unittest.TestCase):
    def rows(self, normal=40, fatigued=40, sessions=3):
        out = []
        for label, n in (("normal", normal), ("fatigued", fatigued)):
            for i in range(n):
                out.append({"participant": f"P{i % sessions}", "session_id": f"{label}{i % sessions}", "source": "real",
                            "label": label, "timestamp": "t", "mean_blink_duration_s": 0.2, "blinks_per_min": 15.0,
                            "perclos": 0.05 if label == "normal" else 0.2, "longest_closure_s": 0.3})
        return out

    def train_from(self, rows, folder):
        path = Path(folder) / "d.csv"
        append_rows(path, rows)
        return train.train([path], model_path=Path(folder) / "m.joblib")

    def test_it_trains_when_there_is_enough_data(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.train_from(self.rows(), folder)
            self.assertTrue(result["model_path"].is_file())

    def test_too_few_rows_are_refused_with_the_numbers_and_advice(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(DatasetError) as caught:
                self.train_from(self.rows(normal=10, fatigued=40), folder)
            self.assertFalse((Path(folder) / "m.joblib").exists())
        text = str(caught.exception)
        self.assertIn("'normal'", text)
        self.assertIn("10 rows", text)
        self.assertIn(str(config.ML_MIN_ROWS_PER_CLASS), text)
        self.assertIn("collect_fatigue_data.py", text)

    def test_a_single_class_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(DatasetError):
                self.train_from(self.rows(normal=0, fatigued=60), folder)

    def test_too_few_separate_sessions_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(DatasetError) as caught:
                self.train_from(self.rows(sessions=1), folder)
        self.assertIn("session", str(caught.exception))

    def test_empty_data_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "empty.csv"
            append_rows(path, [])
            path.write_text(",".join(COLUMNS) + "\n", encoding="utf-8")
            with self.assertRaises(DatasetError):
                train.train([path], model_path=Path(folder) / "m.joblib")

    def test_the_command_line_says_why_when_it_refuses(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "d.csv"
            append_rows(path, self.rows(normal=5, fatigued=5))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = train.main(["--data", str(path)])
        self.assertEqual(code, 1)
        self.assertIn("NOT TRAINED", out.getvalue())

    def test_the_command_line_explains_when_there_is_no_data_file(self):
        with tempfile.TemporaryDirectory() as folder:
            out = io.StringIO()
            with contextlib.redirect_stdout(out), mock.patch.object(config, "DATASET_FILE", Path(folder) / "none.csv"):
                code = train.main([])
        self.assertEqual(code, 1)
        self.assertIn("collect_fatigue_data.py", out.getvalue())
        self.assertIn("RULE-BASED", out.getvalue())


class DemoDataIsNeverPassedOffAsRealTests(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        folder = Path(self._folder.name)
        self.demo_csv = folder / "DEMO_data.csv"
        make_demo_data.generate(self.demo_csv, sessions_per_class=4, rows_per_session=20, participants=4)
        self.real_model = folder / "real_model.joblib"
        self.demo_model = folder / "DEMO_model.joblib"
        patches = [mock.patch.object(config, "FATIGUE_MODEL_FILE", self.real_model),
                   mock.patch.object(config, "DEMO_FATIGUE_MODEL_FILE", self.demo_model)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self._folder.cleanup()

    def test_demo_rows_are_marked_source_demo(self):
        rows, _ = load_rows([self.demo_csv])
        self.assertTrue(all(r["source"] == "demo" for r in rows))
        self.assertTrue(all(r["participant"].startswith("DEMO") for r in rows))

    def test_training_on_demo_data_goes_to_the_demo_file_and_is_marked(self):
        result = train.train([self.demo_csv])
        self.assertEqual(result["model_path"], self.demo_model)
        self.assertFalse(self.real_model.exists())
        self.assertTrue(result["metadata"]["is_demo"])
        self.assertIn("DEMO DATA", result["report"].splitlines()[1])
        self.assertIn("NOTHING about real people", result["report"])

    def test_demo_data_cannot_be_saved_as_the_real_model(self):
        with self.assertRaises(DatasetError) as caught:
            train.train([self.demo_csv], model_path=self.real_model)
        self.assertIn("never", str(caught.exception))
        self.assertFalse(self.real_model.exists())

    def test_the_application_refuses_to_load_a_demo_model(self):
        train.train([self.demo_csv])
        with self.assertRaises(ModelLoadError) as caught:
            load_model(self.demo_model)
        self.assertIn("DEMO", str(caught.exception))

    def test_a_demo_model_in_the_real_location_is_still_refused(self):
        train.train([self.demo_csv])
        self.real_model.write_bytes(self.demo_model.read_bytes())          # someone copies it over by hand
        assessor = FatigueAssessor.from_files()
        self.assertEqual(assessor.mode, "rules")
        self.assertIn("DEMO", assessor.info()["mode_message"])

    def test_the_default_real_model_path_stays_empty_after_demo_training(self):
        train.train([self.demo_csv])
        self.assertFalse(config.FATIGUE_MODEL_FILE.exists())
        self.assertEqual(FatigueAssessor.from_files().mode, "rules")


class TrainedModelTests(unittest.TestCase):
    """The 'trained ML mode' path, exercised with synthetic rows re-labelled as real (in tests only)."""

    @classmethod
    def setUpClass(cls):
        cls._folder = tempfile.TemporaryDirectory()
        cls.csv = make_real_flagged_dataset(cls._folder.name)
        cls.model_path = Path(cls._folder.name) / "fatigue_svm.joblib"
        cls.result = train.train([cls.csv], model_path=cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls._folder.cleanup()

    def test_metadata_records_what_the_model_was_trained_on(self):
        meta = self.result["metadata"]
        self.assertFalse(meta["is_demo"])
        self.assertEqual(meta["n_rows"], 160)
        self.assertEqual(meta["class_counts"], {"normal": 80, "fatigued": 80})
        self.assertEqual(meta["n_participants"], 4)
        self.assertEqual(meta["cv_grouping"], "participant")
        self.assertEqual(len(meta["data_sha256"]), 64)
        self.assertEqual(meta["feature_names"], FEATURE_NAMES)

    def test_report_compares_with_the_baselines_and_carries_the_caveats(self):
        report = self.result["report"]
        for phrase in ("always guessing the bigger class", "rule-based mode", "never saw its participant",
                       "not a medical measurement", "0.5"):
            self.assertIn(phrase, report)
        self.assertNotIn("DEMO DATA", report)
        self.assertTrue(self.result["report_path"].is_file())

    def test_the_model_loads_and_reports_facts_not_marketing(self):
        info = describe_model(self.model_path)
        self.assertTrue(info["available"])
        self.assertEqual(info["summary"]["rows"], 160)
        self.assertEqual(info["summary"]["participants"], 4)

    def test_predictions_point_the_right_way_and_missing_values_are_handled(self):
        model = load_model(self.model_path)
        tired = {"mean_blink_duration_s": 0.32, "blinks_per_min": 27.0, "perclos": 0.19, "longest_closure_s": 0.95}
        fresh = {"mean_blink_duration_s": 0.14, "blinks_per_min": 15.0, "perclos": 0.03, "longest_closure_s": 0.22}
        self.assertGreater(model.fatigue_probability(tired), 0.8)
        self.assertLess(model.fatigue_probability(fresh), 0.2)
        no_blink = dict(fresh, mean_blink_duration_s=None)                   # no blink in the window
        self.assertLess(model.fatigue_probability(no_blink), 0.5)

    def test_a_usable_model_switches_the_assessor_to_trained_ml_mode(self):
        assessor = FatigueAssessor.from_files(self.model_path)
        self.assertEqual(assessor.mode, "ml")
        self.assertEqual(assessor.info()["mode_label"], "TRAINED ML MODE")
        self.assertEqual(assessor.info()["model_summary"]["rows"], 160)

    def test_end_to_end_a_tired_pattern_is_flagged_by_the_trained_model(self):
        assessor = FatigueAssessor.from_files(self.model_path)
        tired = eye(mean_blink_duration_s=0.32, blinks_per_min=27.0, perclos=0.19, longest_closure_s=0.95)
        snapshot = None
        for i in range(24):
            snapshot = assessor.update(i * 0.25, tired)
        self.assertEqual(snapshot["state"], "fatigued")
        self.assertEqual(snapshot["mode"], "ml")

    def test_broken_or_incompatible_model_files_fall_back_to_rules_with_a_message(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            garbage = folder / "garbage.joblib"
            garbage.write_bytes(b"not a model at all" * 50)
            with self.assertRaises(ModelLoadError) as caught:
                load_model(garbage)
            self.assertIn("could not be read", str(caught.exception))

            bundle = joblib.load(self.model_path)
            bundle["feature_names"] = ["something", "else"]
            other = folder / "other_features.joblib"
            joblib.dump(bundle, other)
            with self.assertRaises(ModelLoadError) as caught:
                load_model(other)
            self.assertIn("different features", str(caught.exception))

            assessor = FatigueAssessor.from_files(other)
            self.assertEqual(assessor.mode, "rules")
            self.assertIn("different features", assessor.info()["mode_message"])

            bundle = joblib.load(self.model_path)
            bundle["metadata"]["sklearn_version"] = "0.0.1"
            old = folder / "old_version.joblib"
            joblib.dump(bundle, old)
            self.assertIn("scikit-learn", load_model(old).warning)                 # still works, but says so


# ======================================================================
# The recorder
# ======================================================================
class FakeClock:
    def __init__(self, step=0.05):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


class CollectorTests(TempDatabaseTestCase):
    def setUp(self):
        super().setUp()
        self._data_folder = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self._data_folder.name) / "fatigue_features.csv"
        patch = mock.patch.object(config, "DATASET_FILE", self.dataset_path)
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self._data_folder.cleanup()
        super().tearDown()

    def make_user(self, calibrated=True):
        from backend import auth_service
        from database import calibration as calibration_db
        user, _ = auth_service.register(self.connection, "asha", "password123")
        if calibrated:
            calibration_db.save_eye_baseline(self.connection, user["id"], dict(BASELINE))

    def record(self, detector=None, camera=None, label="normal", extra=()):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = collect_fatigue_data.main(
                ["--username", "asha", "--participant", "P01", "--label", label, "--minutes", "1", *extra],
                camera=camera or FakeCamera(), detector=detector or FakeDetector(ear=0.30),
                show_frame=None, clock=FakeClock())
        return code, out.getvalue()

    def test_it_records_labelled_rows_every_five_seconds_after_the_warm_up(self):
        self.make_user()
        code, output = self.record(label="fatigued")
        self.assertEqual(code, 0)
        rows, problems = load_rows([self.dataset_path])
        self.assertEqual(problems, [])
        self.assertTrue(8 <= len(rows) <= 12, len(rows))                     # about (60 - 10) / 5
        self.assertTrue(all(r["label"] == "fatigued" and r["participant"] == "P01" and r["source"] == "real" for r in rows))
        self.assertEqual(len({r["session_id"] for r in rows}), 1)
        self.assertIn("never images", output)

    def test_no_rows_are_written_during_the_warm_up_or_while_the_eyes_are_not_measured(self):
        self.make_user()
        code, _ = self.record(detector=FakeDetector(faces=[], ear=0.30))     # no face at all
        self.assertEqual(code, 0)
        self.assertFalse(self.dataset_path.exists())

    def test_dark_images_are_not_recorded(self):
        self.make_user()
        code, _ = self.record(camera=FakeCamera(gray_level=3))
        self.assertEqual(code, 0)
        self.assertFalse(self.dataset_path.exists())

    def test_the_row_values_are_real_measurements_of_the_simulated_eyes(self):
        self.make_user()
        self.record()
        rows, _ = load_rows([self.dataset_path])
        self.assertTrue(all(r["perclos"] == 0.0 and r["blinks_per_min"] == 0.0 for r in rows))   # eyes never closed

    def test_the_camera_is_released_when_it_is_done(self):
        self.make_user()
        camera = FakeCamera()
        self.record(camera=camera)
        self.assertTrue(camera.released)

    def test_an_uncalibrated_account_is_told_to_calibrate_first(self):
        self.make_user(calibrated=False)
        code, output = self.record()
        self.assertEqual(code, 1)
        self.assertIn("calibration", output)
        self.assertFalse(self.dataset_path.exists())

    def test_an_unknown_account_is_reported(self):
        code, output = self.record()
        self.assertEqual(code, 1)
        self.assertIn("no account named", output)

    def test_only_the_two_known_labels_are_accepted(self):
        self.make_user()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            collect_fatigue_data.main(["--username", "asha", "--participant", "P01", "--label", "sleepy"],
                                      camera=FakeCamera(), detector=FakeDetector(), show_frame=None, clock=FakeClock())

    def test_the_recorder_never_saves_images(self):
        source = Path(collect_fatigue_data.__file__).read_text(encoding="utf-8")
        for forbidden in ("imwrite", "VideoWriter", "write_bytes"):
            self.assertNotIn(forbidden, source)

    def test_the_consent_and_safety_advice_is_in_the_script(self):
        text = Path(collect_fatigue_data.__file__).read_text(encoding="utf-8")
        self.assertIn("agree", text)
        self.assertIn("sleep deprivation", text)
        self.assertIn("never use a real name", text.lower())


# ======================================================================
# Monitor integration
# ======================================================================
class MonitorFatigueTests(unittest.TestCase):
    def setUp(self):
        self._monitors = []

    def tearDown(self):
        for monitor in self._monitors:
            monitor.stop()

    def build(self, ear=0.30, baseline=BASELINE, assessor=None):
        camera, detector = FakeCamera(), FakeDetector(ear=ear)
        monitor = new_monitor(lambda: camera, detector_factory=lambda: detector,
                              assessor_factory=(lambda: assessor) if assessor else (lambda: FatigueAssessor(model=None)))
        self._monitors.append(monitor)
        monitor.start(1, baseline=baseline)
        return monitor, detector

    def fatigue(self, monitor):
        return monitor.get_status(1).get("fatigue", {})

    def test_before_calibration_the_state_is_unavailable_and_says_why(self):
        monitor, _ = self.build(baseline=None)
        self.assertTrue(wait_for(lambda: "calibrate" in self.fatigue(monitor).get("message", "")))
        self.assertEqual(self.fatigue(monitor)["state"], "unavailable")

    def test_the_mode_is_shown_from_the_start(self):
        monitor, _ = self.build()
        self.assertTrue(wait_for(lambda: self.fatigue(monitor).get("mode_label")))
        self.assertEqual(self.fatigue(monitor)["mode_label"], "DEMO / RULE-BASED MODE")

    def test_a_model_assessor_shows_trained_ml_mode(self):
        monitor, _ = self.build(assessor=FatigueAssessor(model=StubModel(0.1)))
        self.assertTrue(wait_for(lambda: self.fatigue(monitor).get("mode") == "ml"))

    def test_normal_then_fatigued_then_recovered_through_the_whole_pipeline(self):
        with mock.patch.object(config, "FEATURE_MIN_COVERAGE_S", 1.0), \
             mock.patch.object(config, "FEATURE_WINDOW_S", 3), \
             mock.patch.object(config, "FATIGUE_RECOVERY_HOLD_S", 0.8):
            monitor, detector = self.build()
            self.assertTrue(wait_for(lambda: self.fatigue(monitor).get("state") == "normal", timeout=6))

            detector.ear = 0.10                                                   # eyes shut
            self.assertTrue(wait_for(lambda: self.fatigue(monitor).get("state") == "fatigued", timeout=6))
            snapshot = self.fatigue(monitor)
            self.assertTrue(any("eye closure" in reason for reason in snapshot["reasons"]))
            self.assertEqual(snapshot["events"]["fatigued"], 1)

            detector.ear = 0.30                                                   # eyes open again
            self.assertTrue(wait_for(lambda: self.fatigue(monitor).get("state") == "normal", timeout=10))
            self.assertEqual(self.fatigue(monitor)["events"]["fatigued"], 1)

    def test_a_broken_assessor_setup_does_not_stop_the_camera_from_working(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / "bad.joblib"
            bad.write_bytes(b"junk" * 100)
            camera, detector = FakeCamera(), FakeDetector()
            monitor = new_monitor(lambda: camera, detector_factory=lambda: detector,
                                  assessor_factory=lambda: FatigueAssessor.from_files(bad))
            self._monitors.append(monitor)
            self.assertEqual(monitor.start(1, baseline=BASELINE)["state"], "running")
            self.assertTrue(wait_for(lambda: self.fatigue(monitor).get("mode") == "rules"))

    def test_another_account_never_sees_the_fatigue_assessment(self):
        monitor, _ = self.build()
        self.assertTrue(wait_for(lambda: self.fatigue(monitor).get("state")))
        self.assertNotIn("fatigue", monitor.get_status(2))


class PrivacyTests(unittest.TestCase):
    def test_the_ml_code_handles_only_numbers(self):
        for path in Path(features.__file__).parent.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for forbidden in ("imwrite", "VideoWriter", "import cv2"):
                self.assertNotIn(forbidden, source, path.name)


# ======================================================================
# API (needs fastapi)
# ======================================================================
@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class FatigueApiTests(TempDatabaseTestCase):
    def setUp(self):
        super().setUp()
        self._data_folder = tempfile.TemporaryDirectory()
        patch = mock.patch.object(config, "FATIGUE_MODEL_FILE", Path(self._data_folder.name) / "fatigue_svm.joblib")
        patch.start()
        self.addCleanup(patch.stop)
        self._original_monitor = monitor_module.monitor
        camera, detector = FakeCamera(), FakeDetector()
        monitor_module.monitor = new_monitor(lambda: camera, detector_factory=lambda: detector)

    def tearDown(self):
        monitor_module.monitor.stop()
        monitor_module.monitor = self._original_monitor
        self._data_folder.cleanup()
        super().tearDown()

    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        client.post("/api/auth/register", json={"username": "asha", "password": "password123"})
        return client

    def test_login_is_required(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        self.assertEqual(TestClient(app).get("/api/fatigue/model").status_code, 401)

    def test_without_a_model_the_rule_based_mode_is_reported_with_advice(self):
        body = self.client().get("/api/fatigue/model").json()
        self.assertEqual(body["mode_label"], "DEMO / RULE-BASED MODE")
        self.assertIsNone(body["summary"])
        self.assertIn("python -m ml.train", body["message"])
        self.assertIn("not a medical diagnosis", body["disclaimer"])

    def test_with_a_trained_model_the_ml_mode_and_training_facts_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            csv_path = make_real_flagged_dataset(folder)
            train.train([csv_path], model_path=config.FATIGUE_MODEL_FILE)
        body = self.client().get("/api/fatigue/model").json()
        self.assertEqual(body["mode_label"], "TRAINED ML MODE")
        self.assertEqual(body["summary"]["rows"], 160)

    def test_the_monitor_status_carries_the_fatigue_block(self):
        client = self.client()
        client.post("/api/monitor/start")
        self.assertTrue(wait_for(lambda: "fatigue" in client.get("/api/monitor/status").json()
                                 and client.get("/api/monitor/status").json()["fatigue"].get("state")))
        fatigue = client.get("/api/monitor/status").json()["fatigue"]
        for key in ("mode_label", "state", "score", "reasons", "message", "events", "disclaimer"):
            self.assertIn(key, fatigue)


if __name__ == "__main__":
    unittest.main()
