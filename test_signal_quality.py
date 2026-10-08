"""
Tests for the signal-quality protections added after a real test showed false blinks when a hand covered
and uncovered the camera lens:

    * EyeMeasurer (computer_vision/eye_measurement.py): eyes are not measured while the picture is dark,
      changing fast, the face box jumps, or during the first 0.8 s after any interruption.
    * Blink confirmation (eye_tracker.py): a blink counts only if the eyes stay measurable 0.7 s afterwards;
      otherwise it is discarded.

The hand-cover scenes below are SIMULATED (brightness ramps, eye values that flap open/closed like a distorted
face would). They check that the protections work on those patterns. Whether they cover EVERYTHING a real hand
does is checked by you with scripts/eye_test.py; the tests cannot promise that.

Run from the project root:
    python -m unittest discover -s tests -t . -v
"""

import time
import unittest
from unittest import mock

import numpy as np

from computer_vision import face_landmarks as fl
from computer_vision.eye_measurement import REASON_TEXT, EyeMeasurer
from computer_vision.eye_tracker import EyeStateTracker
from tests.helpers import FakeCamera, FakeDetector, make_face_landmarks, new_monitor, wait_for
from utils import config

BASELINE = {"ear_open": 0.30, "ear_closed": 0.10}
OPEN, CLOSED = 0.30, 0.10
FPS = 20.0
NO_FACE = fl.analyse_faces([])


def frame_of(level):
    return np.full((480, 640, 3), int(round(level)), dtype=np.uint8)


def face_of(ear=OPEN, cx=0.5, cy=0.5, size=0.4):
    return fl.analyse_faces([make_face_landmarks(cx, cy, size, ear=ear)])


def blink_pattern(t):
    """A normal person: eyes closed for 0.15 s at the end of every 4 s."""
    return CLOSED if (t % 4.0) >= 3.85 else OPEN


def flapping(t):
    """What a distorted face can look like: eye values jumping open/closed every 0.1 s."""
    return CLOSED if int(t / 0.1) % 2 == 0 else OPEN


class Scene:
    """Frames -> EyeMeasurer -> EyeStateTracker, exactly as in the dashboard, with a simulated clock."""

    def __init__(self):
        self.measurer = EyeMeasurer()
        self.tracker = EyeStateTracker(BASELINE)
        self.t = 0.0
        self.features = None
        self.reasons = []                                   # (time, reason) for every frame

    def frame(self, brightness, ear, face=None):
        face = face if face is not None else (NO_FACE if ear is None else face_of(ear))
        m = self.measurer.measure(self.t, frame_of(brightness), face)
        self.features = self.tracker.update(self.t, None if m.values is None else m.values[2])
        self.reasons.append((self.t, m.reason))
        self.t += 1.0 / FPS
        return m

    def segment(self, seconds, brightness, ear):
        """ear: a number, None (no face) or a function of the time within the segment."""
        n = int(round(seconds * FPS))
        for i in range(n):
            local = i / FPS
            b = brightness(local) if callable(brightness) else brightness
            e = ear(local) if callable(ear) else ear
            self.frame(b, e)

    def ramp(self, seconds, start, end, ear):
        n = int(round(seconds * FPS))
        for i in range(n):
            local = i / FPS
            self.frame(start + (end - start) * (i + 1) / n, ear(local) if callable(ear) else ear)

    def measured_between(self, t0, t1):
        return [r for (t, r) in self.reasons if t0 <= t < t1 and r == ""]


def normal(scene, seconds):
    scene.segment(seconds, 100, blink_pattern)


# ======================================================================
# EyeMeasurer
# ======================================================================
class EyeMeasurerTests(unittest.TestCase):
    def test_reasons_all_have_a_dashboard_text(self):
        for reason in ("", "no_face", "too_dark", "unstable_image", "face_jump", "settling", "eyes_unreadable"):
            self.assertIn(reason, REASON_TEXT)

    def test_steady_picture_is_measured_after_the_settle_time(self):
        scene = Scene()
        first = [scene.frame(100, OPEN) for _ in range(int(FPS * 2))]
        self.assertEqual(first[0].reason, "settling")
        self.assertIsNone(first[0].values)
        settle_frames = sum(1 for m in first if m.reason == "settling")
        self.assertAlmostEqual(settle_frames / FPS, config.EYE_SETTLE_S, delta=0.1)
        self.assertEqual(first[-1].reason, "")
        self.assertAlmostEqual(first[-1].values[2], OPEN, places=3)

    def test_no_face_then_the_face_returns_needs_a_new_settle_time(self):
        scene = Scene()
        scene.segment(2, 100, OPEN)
        gone = scene.frame(100, None)
        self.assertEqual(gone.reason, "no_face")
        back = [scene.frame(100, OPEN) for _ in range(int(FPS * 2))]
        self.assertEqual(back[0].reason, "settling")
        self.assertEqual(back[-1].reason, "")

    def test_a_nearly_black_picture_is_not_measured(self):
        scene = Scene()
        scene.segment(2, 100, OPEN)
        m = scene.frame(3, CLOSED)                         # the (fake) model still returns a face
        self.assertEqual(m.reason, "too_dark")
        self.assertIsNone(m.values)

    def test_a_fast_brightness_change_stops_measurement_until_it_settles(self):
        scene = Scene()
        scene.segment(3, 100, OPEN)
        scene.ramp(0.4, 100, 45, OPEN)                     # dimming quickly, still above the "too dark" limit
        self.assertEqual(scene.reasons[-1][1], "unstable_image")
        scene.segment(3, 45, OPEN)                         # picture now steady again
        self.assertEqual(scene.reasons[-1][1], "")
        self.assertTrue(any(r == "settling" for _, r in scene.reasons[-70:]))   # it waited before trusting it

    def test_a_slow_brightness_drift_is_still_measured(self):
        scene = Scene()
        scene.segment(2, 100, OPEN)
        scene.segment(5, lambda t: 100 + 8 * t, OPEN)      # 100 -> 140 over 5 s, like auto-exposure
        self.assertTrue(all(r == "" for _, r in scene.reasons[-80:]))

    def test_a_small_flicker_is_still_measured(self):
        scene = Scene()
        scene.segment(2, 100, OPEN)
        scene.segment(2, lambda t: 100 + 20 * (int(t * 10) % 2), OPEN)      # 100 <-> 120 every 0.1 s
        self.assertTrue(all(r == "" for _, r in scene.reasons[-30:]))

    def test_black_frames_do_not_crash_the_brightness_ratio(self):
        scene = Scene()
        scene.segment(1, 0, OPEN)
        scene.segment(1, 100, OPEN)                        # jump from 0 to 100 (ratio with a zero minimum)

    def test_a_face_box_that_jumps_is_not_trusted(self):
        scene = Scene()
        scene.segment(3, 100, OPEN)
        grown = scene.measurer.measure(scene.t, frame_of(100), face_of(OPEN, size=0.8))       # box area x4
        self.assertEqual(grown.reason, "face_jump")
        moved = EyeMeasurer()
        moved.measure(0.0, frame_of(100), face_of(OPEN, cx=0.35))
        self.assertEqual(moved.measure(0.05, frame_of(100), face_of(OPEN, cx=0.65)).reason, "face_jump")

    def test_the_first_face_after_a_break_is_not_called_a_jump(self):
        m = EyeMeasurer()
        m.measure(0.0, frame_of(100), face_of(OPEN, size=0.4))
        m.measure(0.05, frame_of(100), NO_FACE)
        self.assertEqual(m.measure(0.10, frame_of(100), face_of(OPEN, size=0.7)).reason, "settling")

    def test_normal_head_movement_is_not_a_jump(self):
        scene = Scene()
        scene.segment(2, 100, OPEN)
        for i in range(40):                                # face drifts slowly to one side and grows a little
            scene.measurer.measure(scene.t, frame_of(100), face_of(OPEN, cx=0.5 + i * 0.002, size=0.4 + i * 0.001))
            scene.t += 1 / FPS
        self.assertEqual(scene.measurer.measure(scene.t, frame_of(100), face_of(OPEN, cx=0.58)).reason, "")


# ======================================================================
# Blink confirmation in the tracker
# ======================================================================
def feed(tracker, pattern, start=0.0):
    t, features = start, None
    for seconds, ear in pattern:
        for _ in range(int(round(seconds * FPS))):
            features = tracker.update(t, ear)
            t += 1.0 / FPS
    return features, t


class BlinkConfirmationTests(unittest.TestCase):
    def test_a_blink_is_counted_only_after_the_confirmation_time(self):
        tracker = EyeStateTracker(BASELINE)
        _, t = feed(tracker, [(3, OPEN), (0.15, CLOSED)])
        soon, t = feed(tracker, [(0.6, OPEN)], start=t)
        self.assertEqual(soon["blink_count"], 0)                     # still waiting for confirmation
        later, _ = feed(tracker, [(0.6, OPEN)], start=t)
        self.assertEqual(later["blink_count"], 1)
        self.assertEqual(later["discarded_events"], 0)

    def test_a_blink_followed_by_lost_data_is_discarded_not_counted(self):
        tracker = EyeStateTracker(BASELINE)
        _, t = feed(tracker, [(3, OPEN), (0.15, CLOSED), (0.3, OPEN)])
        features, t = feed(tracker, [(1, None)], start=t)            # the picture goes away 0.3 s later
        self.assertEqual(features["discarded_events"], 1)
        features, _ = feed(tracker, [(3, OPEN)], start=t)
        self.assertEqual(features["blink_count"], 0)
        self.assertEqual(features["discarded_events"], 1)

    def test_a_gap_in_the_data_also_discards_pending_events(self):
        tracker = EyeStateTracker(BASELINE)
        _, t = feed(tracker, [(3, OPEN), (0.15, CLOSED), (0.3, OPEN)])
        features = tracker.update(t + 5.0, OPEN)                     # next measurement arrives 5 s later
        self.assertEqual(features["blink_count"], 0)
        self.assertEqual(features["discarded_events"], 1)

    def test_a_pending_long_closure_still_counts_towards_the_longest_closure(self):
        tracker = EyeStateTracker(BASELINE)
        _, t = feed(tracker, [(12, OPEN), (1.4, CLOSED)])
        features, _ = feed(tracker, [(0.3, OPEN)], start=t)          # closure over, not confirmed yet
        self.assertEqual(features["long_closure_count"], 0)
        self.assertGreaterEqual(features["longest_closure_s"], 1.3)

    def test_a_long_closure_is_confirmed_like_a_blink(self):
        tracker = EyeStateTracker(BASELINE)
        features, _ = feed(tracker, [(3, OPEN), (1.4, CLOSED), (1.6, OPEN)])
        self.assertEqual(features["long_closure_count"], 1)
        self.assertEqual(features["blink_count"], 0)


# ======================================================================
# Simulated hand-cover scenes
# ======================================================================
class HandCoverSceneTests(unittest.TestCase):
    def control_count(self, *normal_seconds):
        """The number of real blinks when the SAME normal stretches happen without any cover."""
        scene = Scene()
        for seconds in normal_seconds:
            normal(scene, seconds)
        normal_tail = Scene  # noqa: F841  (readability)
        scene.segment(1.5, 100, OPEN)                                # let the last blink be confirmed
        return scene.features["blink_count"]

    def test_palm_approaching_and_leaving_adds_no_blinks(self):
        scene = Scene()
        normal(scene, 15)
        scene.ramp(0.6, 100, 4, flapping)                # hand approaches: picture darkens, eye values flap
        scene.segment(3, 3, CLOSED)                      # black picture, model still returns "closed eyes"
        scene.ramp(0.6, 4, 100, flapping)                # hand leaves: picture brightens, eye values flap
        scene.segment(0.6, 100, flapping)                # a blurry hand is still in view, values still flap
        normal(scene, 8)
        scene.segment(1.5, 100, OPEN)

        control = self.control_count(15, 8)
        self.assertEqual(scene.features["blink_count"], control)
        self.assertEqual(scene.features["long_closure_count"], 0)

    def test_the_same_scene_WITHOUT_the_protections_does_produce_false_events(self):
        # Proof that the scene above is a real test: feed the same eye values straight to a tracker.
        tracker = EyeStateTracker(BASELINE)
        t = 0.0
        for seconds, ear in ((15, blink_pattern), (0.6, flapping), (3, CLOSED), (0.6, flapping), (0.6, flapping), (8, blink_pattern), (1.5, OPEN)):
            for i in range(int(round(seconds * FPS))):
                tracker.update(t, ear(i / FPS) if callable(ear) else ear)
                t += 1 / FPS
        control = self.control_count(15, 8)
        events = tracker.blink_count + tracker.long_closure_count
        self.assertGreater(events, control)

    def test_a_blink_like_dip_just_before_the_picture_goes_dark_is_thrown_away(self):
        scene = Scene()
        normal(scene, 12)
        scene.segment(2.0, 100, OPEN)
        scene.segment(0.15, 100, CLOSED)                 # a dip at full brightness, exactly like a blink
        scene.segment(0.3, 100, OPEN)
        scene.segment(3, 3, OPEN)                        # then the lens is covered (an abrupt cover, no ramp)
        scene.segment(1, 100, OPEN)
        scene.segment(0.5, 100, OPEN)
        self.assertGreaterEqual(scene.features["discarded_events"], 1)
        self.assertEqual(scene.features["blink_count"], self.control_count(12))

    def test_flapping_after_the_picture_is_steady_again_is_ignored_during_the_settle_time(self):
        scene = Scene()
        normal(scene, 14)                                # the last real blink is 2 s old when the cover starts
        scene.segment(2, 3, CLOSED)                      # covered
        scene.segment(1.2, 100, lambda t: flapping(t) if t < 0.6 else OPEN)     # uncovered, still flapping for 0.6 s
        normal(scene, 8)
        scene.segment(1.5, 100, OPEN)
        self.assertEqual(scene.features["blink_count"], self.control_count(14, 8))
        self.assertEqual(scene.features["long_closure_count"], 0)

    # ---- the four false blinks in a real recording (timestamps from the user's terminal log) ----
    #   81.7 s BLINK, face lost 82.2 s   (0.5 s before the loss)      137.8 s BLINK, face lost 138.1 s (0.3 s before)
    #   face back 87.2 s, BLINK 87.8 s   (0.6 s after the return)     face back 151.3 s, BLINK 151.8 s (0.5 s after)
    def scene_with_dip_before_face_loss(self, seconds_before_loss):
        scene = Scene()
        normal(scene, 14)
        scene.segment(1.0, 100, OPEN)
        scene.segment(0.15, 100, CLOSED)                 # the blink-like dip ...
        scene.segment(seconds_before_loss, 100, OPEN)    # ... this long before the face is lost
        scene.segment(3.0, 100, None)                    # face lost (like the log: "eyes NOT measured (no face)")
        scene.segment(4.0, 100, OPEN)                    # face back, eyes open, nothing happening
        return scene

    def scene_with_dip_after_face_return(self, seconds_after_return):
        scene = Scene()
        normal(scene, 14)
        scene.segment(3.0, 100, None)                    # face lost
        scene.segment(seconds_after_return, 100, OPEN)   # face found again, eyes open ...
        scene.segment(0.15, 100, CLOSED)                 # ... then a blink-like dip this long after the return
        scene.segment(4.0, 100, OPEN)
        return scene

    def test_the_logged_dips_before_a_face_loss_are_not_counted(self):
        expected = self.control_count(14)
        for gap in (0.3, 0.5, 0.7, 0.9):                 # the log had 0.3 and 0.5; the rest is safety margin
            scene = self.scene_with_dip_before_face_loss(gap)
            self.assertEqual(scene.features["blink_count"], expected, f"dip {gap} s before the loss")
            self.assertGreaterEqual(scene.features["discarded_events"], 1, gap)

    def test_the_logged_dips_after_the_face_returns_are_not_counted(self):
        expected = self.control_count(14)
        for gap in (0.5, 0.6, 0.8, 0.95):                # the log had 0.5 and 0.6
            scene = self.scene_with_dip_after_face_return(gap)
            self.assertEqual(scene.features["blink_count"], expected, f"dip {gap} s after the return")

    def test_a_blink_well_after_the_face_returned_is_counted(self):
        # In the log a blink 1.4 s after the face came back looks real; it must still be counted.
        scene = self.scene_with_dip_after_face_return(1.4)
        self.assertEqual(scene.features["blink_count"], self.control_count(14) + 1)

    def test_a_blink_well_before_a_face_loss_is_counted(self):
        # The log's blinks 2.8 s before a loss look real too.
        scene = self.scene_with_dip_before_face_loss(2.8)
        self.assertEqual(scene.features["blink_count"], self.control_count(14) + 1)

    def test_known_trade_off_a_real_blink_right_before_the_picture_goes_dark_is_lost(self):
        # Documented limit: a genuine blink less than 0.7 s before the data disappears cannot be told apart from
        # a distorted one, so it is discarded rather than risk a false count.
        scene = Scene()
        normal(scene, 12)                                # the last blink ends exactly at 12.0 s
        scene.segment(0.2, 100, OPEN)                    # eyes reopen, the blink is now waiting for confirmation
        scene.segment(2, 3, CLOSED)                      # covered 0.2 s later
        scene.segment(1.5, 100, OPEN)
        self.assertEqual(scene.features["blink_count"], 2)
        self.assertEqual(scene.features["discarded_events"], 1)

    def test_real_blinks_are_still_counted_with_the_protections_on(self):
        scene = Scene()
        normal(scene, 32)                                # blinks at 3.85, 7.85, ... -> 8 blinks
        scene.segment(1.5, 100, OPEN)
        self.assertEqual(scene.features["blink_count"], 8)
        self.assertEqual(scene.features["discarded_events"], 0)
        self.assertEqual(scene.features["long_closure_count"], 0)

    def test_slow_brightness_drift_does_not_suppress_real_blinks(self):
        scene = Scene()
        scene.segment(32, lambda t: 100 + 1.2 * t, blink_pattern)      # lighting slowly brightens
        scene.segment(1.5, 138, OPEN)
        self.assertEqual(scene.features["blink_count"], 8)

    def test_a_real_long_closure_in_a_steady_picture_is_still_seen(self):
        scene = Scene()
        normal(scene, 8)
        scene.segment(1.5, 100, CLOSED)
        scene.segment(1.2, 100, OPEN)
        self.assertEqual(scene.features["long_closure_count"], 1)


# ======================================================================
# Through the whole monitor (fake camera and face)
# ======================================================================
class MonitorSignalQualityTests(unittest.TestCase):
    def setUp(self):
        self._monitors = []

    def tearDown(self):
        for monitor in self._monitors:
            monitor.stop()

    def build(self, camera=None, detector=None, baseline=BASELINE):
        camera, detector = camera or FakeCamera(), detector or FakeDetector(ear=OPEN)
        monitor = new_monitor(lambda: camera, detector_factory=lambda: detector)
        self._monitors.append(monitor)
        monitor.start(1, baseline=baseline)
        return monitor, camera, detector

    def eye(self, monitor):
        return monitor.get_status(1).get("eye", {})

    def test_the_reason_for_not_measuring_is_reported(self):
        monitor, camera, _ = self.build()
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("not_measured_reason") == "settling"))
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("not_measured_reason") == "" and self.eye(monitor).get("ear")))
        camera.gray_level = 3
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("not_measured_reason") == "too_dark"))

    def test_no_face_is_reported_as_the_reason(self):
        monitor, _, _ = self.build(detector=FakeDetector(faces=[], ear=OPEN))
        self.assertTrue(wait_for(lambda: self.eye(monitor).get("not_measured_reason") == "no_face"))

    def test_covering_and_uncovering_the_camera_adds_no_blinks_end_to_end(self):
        with mock.patch.object(config, "TARGET_FPS", 20):
            monitor, camera, detector = self.build()
            self.assertTrue(wait_for(lambda: self.eye(monitor).get("ear") is not None and self.eye(monitor).get("eyes_closed") is False, timeout=6))
            time.sleep(1.0)
            for cycle in range(2):
                started = time.monotonic()
                while (elapsed := time.monotonic() - started) < 0.7:              # hand approaches
                    camera.gray_level = max(3, 100 - 140 * elapsed)
                    detector.ear = CLOSED if int(elapsed / 0.1) % 2 == 0 else OPEN  # eye values flap
                    time.sleep(0.02)
                camera.gray_level, detector.ear = 3, CLOSED                       # black, "closed eyes"
                time.sleep(1.2)
                started = time.monotonic()
                while (elapsed := time.monotonic() - started) < 0.7:              # hand leaves
                    camera.gray_level = min(100, 3 + 140 * elapsed)
                    detector.ear = CLOSED if int(elapsed / 0.1) % 2 == 0 else OPEN
                    time.sleep(0.02)
                camera.gray_level, detector.ear = 100, OPEN
                time.sleep(2.2)                                                    # back to normal
            self.assertTrue(wait_for(lambda: self.eye(monitor).get("not_measured_reason") == "", timeout=4))
            eye = self.eye(monitor)
            self.assertEqual(eye["blink_count"], 0)
            self.assertEqual(eye["long_closure_count"], 0)


if __name__ == "__main__":
    unittest.main()
