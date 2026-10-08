"""
Eye test WITHOUT the web app:

    python scripts/eye_test.py

Opens a window with your webcam and shows the Eye Aspect Ratio (EAR) live.
  * Eyes open, looking at the screen: EAR is usually somewhere around 0.2 - 0.35.
  * Close your eyes: EAR should drop clearly (often below 0.15).
  * Press C to run the personal calibration (about 15 seconds; it beeps to start and to finish
    the closed-eyes step). Afterwards the window counts your blinks and closures.
  * Press Q to quit.
  * Add  --log eye_log.csv  to also save the measured NUMBERS (time, EAR, brightness, blink counts...)
    to a small CSV file for troubleshooting. No images or video are ever saved.
  * The terminal prints a timestamped line for every blink and long closure, every time the eyes stop or
    start being measured (with the reason), and every blink-like dip that was DISCARDED because the eyes
    stopped being measurable right after it. Try covering the lens and uncovering it: NO blink should
    be reported for that. Blinks are reported about 0.7 s after they happen (they are confirmed first).

If EAR does NOT change when you close your eyes, tell the developer: the eye points may not sit
well on your eyes (thick glasses frames or glare can cause this). Nothing is saved to disk.
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from computer_vision import drawing  # noqa: E402  (import this package first: it silences OpenCV's log noise)
from computer_vision.calibrator import Calibrator  # noqa: E402
from computer_vision.camera import Camera, CameraError  # noqa: E402
from computer_vision.eye_measurement import REASON_TEXT, EyeMeasurer  # noqa: E402
from computer_vision.eye_tracker import EyeStateTracker  # noqa: E402
from computer_vision.face_landmarks import FaceDetector, ModelError  # noqa: E402

import cv2  # noqa: E402


def beep(frequency):
    try:
        import winsound                      # Windows only
        winsound.Beep(frequency, 250)
    except ImportError:
        print("\a", end="", flush=True)      # terminal bell elsewhere


LOG_COLUMNS = ["time_s", "phase", "face_found", "brightness", "not_measured_reason", "ear_left", "ear_right",
               "ear_mean", "closure", "eyes_closed", "blinks", "long_closures", "discarded"]


def main(camera=None, detector=None, log_path=None):
    detector = detector or FaceDetector()
    try:
        detector.load()
    except ModelError as error:
        print(error)
        sys.exit(1)

    camera = camera or Camera()
    try:
        camera.open()
    except CameraError as error:
        print(error)
        detector.close()
        sys.exit(1)

    print("Running. Press C to calibrate, Q to quit (click the video window first).")
    log_file = writer = None
    if log_path:
        log_file = open(log_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(log_file)
        writer.writerow(LOG_COLUMNS)
        print(f"Saving measured numbers (no images) to {log_path}")
    rows_written = 0
    tracker = EyeStateTracker(None)
    calibrator = None
    calibration_message = ""
    last_phase = None
    ear_min = ear_max = None
    measurer = EyeMeasurer()
    last_reason = None
    last_break = None                        # (time, reason) of the most recent time the eyes were not measured
    last_counts = (0, 0)
    last_discarded = 0
    start = time.monotonic()
    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Camera stopped sending images.")
                break
            now = time.monotonic() - start
            height, width = frame.shape[:2]
            result = detector.detect(frame, int(now * 1000))
            measurement = measurer.measure(now, frame, result)      # same rules as the dashboard
            values, brightness, reason = measurement.values, measurement.brightness, measurement.reason
            ear = None if values is None else values[2]
            if reason != last_reason:
                if reason:
                    print(f"[{now:6.1f} s] eyes NOT measured: {REASON_TEXT.get(reason, reason)} (brightness {brightness:.0f})")
                    last_break = (now, reason)
                elif last_reason is not None:
                    print(f"[{now:6.1f} s] eyes measured again")
                last_reason = reason
            if ear is not None:
                ear_min = ear if ear_min is None else min(ear_min, ear)
                ear_max = ear if ear_max is None else max(ear_max, ear)

            features = {}
            if calibrator is not None:
                status = calibrator.update(now, ear)
                calibration_message = f"CALIBRATING ({status['seconds_left']} s): {status['message']}"
                if status["phase"] == "close" and last_phase != "close":
                    beep(880)
                if last_phase == "close" and status["phase"] != "close":
                    beep(660)
                last_phase = status["phase"]
                if calibrator.finished:
                    if calibrator.result["ok"]:
                        baseline = calibrator.result["baseline"]
                        tracker = EyeStateTracker(baseline)
                        calibration_message = (f"Calibrated: open EAR {baseline['ear_open']:.2f}, "
                                               f"closed EAR {baseline['ear_closed']:.2f}")
                        print(calibration_message)
                    else:
                        calibration_message = calibrator.result["reason"]
                        print(calibration_message)
                    calibrator, last_phase = None, None
            else:
                features = tracker.update(now, ear)
                if features.get("calibrated"):
                    counts = (features["blink_count"], features["long_closure_count"])
                    since = "" if last_break is None else (
                        f"  | last NOT-measured period began {now - last_break[0]:.1f} s ago ({REASON_TEXT.get(last_break[1], last_break[1])})")
                    if counts[0] > last_counts[0]:
                        print(f"[{now:6.1f} s] BLINK counted (total {counts[0]}){since}")
                    if counts[1] > last_counts[1]:
                        print(f"[{now:6.1f} s] LONG CLOSURE counted (total {counts[1]}){since}")
                    last_counts = counts
                    if features["discarded_events"] > last_discarded:
                        print(f"[{now:6.1f} s] blink-like dip DISCARDED (eyes stopped being measurable right after it) "
                              f"(total discarded {features['discarded_events']})")
                        last_discarded = features["discarded_events"]

            if writer is not None:
                blank = lambda v, digits=4: "" if v is None else (round(v, digits) if isinstance(v, float) else v)
                writer.writerow([
                    round(now, 3), "calibrating" if calibrator is not None else "monitoring",
                    int(result.landmarks is not None), round(brightness, 1), reason,
                    blank(None if values is None else values[0]), blank(None if values is None else values[1]),
                    blank(ear), blank(features.get("closure"), 3), blank(features.get("eyes_closed")),
                    blank(features.get("blink_count")), blank(features.get("long_closure_count")),
                    blank(features.get("discarded_events")),
                ])
                rows_written += 1

            drawing.draw_face_overlay(frame, result, eyes_closed=bool(features.get("eyes_closed")))
            view = cv2.flip(frame, 1)                       # mirror first, THEN write text
            if values is None:
                lines = ["EAR: no face / eyes not measured"]
            else:
                lines = [f"EAR {values[2]:.3f}  (left {values[0]:.3f}, right {values[1]:.3f})"]
            if ear_min is not None:
                lines.append(f"lowest / highest EAR seen: {ear_min:.3f} / {ear_max:.3f}")
            if calibration_message:
                lines.append(calibration_message)
            elif not tracker.baseline:
                lines.append("Not calibrated. Press C to calibrate.")
            if features.get("calibrated"):
                lines.append(f"eyes {'CLOSED' if features['eyes_closed'] else 'open'} | blinks {features['blink_count']} "
                             f"| long closures {features['long_closure_count']}")
                if features["blinks_per_min"] is not None:
                    lines.append(f"blinks/min {features['blinks_per_min']} | PERCLOS-style {features['perclos'] * 100:.1f}% "
                                 f"| longest closure {features['longest_closure_s']} s")
            for i, line in enumerate(lines):
                cv2.putText(view, line, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Eye test (C = calibrate, Q = quit)", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")) and calibrator is None:
                calibrator = Calibrator()
                tracker = EyeStateTracker(tracker.baseline)   # forget old counts
                last_counts, last_discarded = (0, 0), 0
                calibration_message = ""
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()
        if log_file is not None:
            log_file.close()
            print(f"Saved {rows_written} rows to {log_path}")
        print("Camera released.")


if __name__ == "__main__":
    log = sys.argv[sys.argv.index("--log") + 1] if "--log" in sys.argv[:-1] else None
    main(log_path=log)
