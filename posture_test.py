"""
Posture AND attention test WITHOUT the web app:

    python scripts/posture_test.py

Opens a window with your webcam and shows the posture numbers live.
  * Sit the way you WANT to sit, look at the screen, and press C: this records your baseline for 8 seconds
    (the same idea as the dashboard calibration, but for posture only).
  * Then move: tilt your head sideways, nod it down, lean towards or away from the screen, sink lower, shift sideways.
    The numbers show how far you are from your baseline, and the state (GOOD / WARNING / POOR) follows after
    a few seconds. Shoulders are used only when they are clearly visible; the window says whether they are.
  * Press Q to quit.

  * ATTENTION (Phase 8): turn your head sideways about 30-40 degrees, or look well down/up, and hold it. After about 5 seconds the
    attention line changes to DISTRACTED. Turn back and it returns to ATTENTIVE after about 2 seconds. Cover the camera or leave the
    picture for 3 seconds: FACE NOT DETECTED. Attention compares your head direction with where it pointed when you pressed C.

Things to check (please tell the developer what you see):
  1. "head nod" should go POSITIVE when you nod your head DOWN. If it goes negative, the sign is wrong.
  2. "distance" should be positive when you lean CLOSER to the screen.
  3. Do the two shoulder dots (if shown) sit on your shoulders?
  4. Turn your head about 35 degrees sideways: does "turn" in the attention line show roughly that size, and does it become
     DISTRACTED after about 5 seconds? (The angles come from a generic face model, so expect them to be approximate.)
Nothing is saved to disk.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from computer_vision import drawing  # noqa: E402  (import this package first: it silences OpenCV's log noise)
from computer_vision.attention import AttentionAssessor  # noqa: E402
from computer_vision.camera import Camera, CameraError  # noqa: E402
from computer_vision.eye_measurement import REASON_TEXT, EyeMeasurer  # noqa: E402
from computer_vision.face_landmarks import FaceDetector, ModelError  # noqa: E402
from computer_vision.pose_landmarks import PoseDetector  # noqa: E402
from computer_vision.posture import PostureAssessor, build_posture_baseline, deviations, measure_posture  # noqa: E402
from utils import config  # noqa: E402

import cv2  # noqa: E402

BASELINE_SECONDS = 8.0


def main(camera=None, detector=None, pose_detector=None):
    detector = detector or FaceDetector()
    try:
        detector.load()
    except ModelError as error:
        print(error)
        sys.exit(1)
    pose = pose_detector or PoseDetector()
    try:
        pose.load()
    except ModelError as error:
        print(f"{error}  -> continuing WITHOUT shoulders (head position only).")
        pose = None
    camera = camera or Camera()
    try:
        camera.open()
    except CameraError as error:
        print(error)
        detector.close()
        if pose:
            pose.close()
        sys.exit(1)

    print("Running. Sit the way you want to sit and press C to record your baseline. Q to quit.")
    measurer, assessor, attention = EyeMeasurer(), PostureAssessor(None), AttentionAssessor(None)
    samples, recording_since = [], None
    last_pose, last_pose_time, frame_index = None, 0.0, 0
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
            measurement = measurer.measure(now, frame, result)
            metrics = None
            if measurement.values is not None:
                if pose is not None and frame_index % config.POSE_EVERY_N_FRAMES == 0:
                    last_pose, last_pose_time = pose.detect(frame, int(now * 1000)), now
                fresh = last_pose is not None and now - last_pose_time <= config.POSE_RESULT_MAX_AGE_S
                metrics = measure_posture(result.landmarks, result.bbox, last_pose if fresh else None, width, height)
            frame_index += 1

            if recording_since is not None:
                if metrics is not None:
                    samples.append(metrics)
                if now - recording_since >= BASELINE_SECONDS:
                    baseline, problem = build_posture_baseline(samples)
                    recording_since = None
                    if baseline is None:
                        print("Baseline NOT recorded:", problem)
                    else:
                        assessor = PostureAssessor(baseline)
                        attention = AttentionAssessor(baseline)
                        print(f"Baseline recorded from {baseline['samples']} measurements "
                              f"({'with' if baseline['shoulders'] else 'WITHOUT'} shoulders).")

            drawing.draw_face_overlay(frame, result)
            if last_pose is not None and last_pose.visible:            # shoulder dots
                for point in (last_pose.left_point, last_pose.right_point):
                    cv2.circle(frame, (int(point[0] * width), int(point[1] * height)), 8, (255, 0, 255), -1)
            view = cv2.flip(frame, 1)                                  # mirror first, THEN write text
            lines = []
            if metrics is None:
                lines.append("Posture not measured: " + (REASON_TEXT.get(measurement.reason, "") or "head pose unavailable"))
            else:
                lines.append(f"raw: tilt {metrics['roll_deg']:+.1f}  nod {metrics['pitch_deg']:+.1f}  "
                             f"eye distance {metrics['eye_distance']:.3f}  shoulders {'yes' if metrics['shoulder_tilt_deg'] is not None else 'no'}")
            if recording_since is not None:
                lines.append(f"RECORDING BASELINE... {BASELINE_SECONDS - (now - recording_since):.0f} s  (sit normally)")
            elif assessor.baseline is None:
                lines.append("No baseline yet. Sit normally and press C.")
            else:
                snapshot = assessor.update(now, metrics)
                if snapshot["state"] == "unavailable":
                    lines.append(snapshot["message"])
                else:
                    d = snapshot["deviations"]
                    lines.append(f"POSTURE: {snapshot['state'].upper()}  (score {snapshot['score']})")
                    lines.append(f"vs your normal: tilt {d['head_tilt_deg']:+.1f} deg | nod {d['head_nod_deg']:+.1f} deg | "
                                 f"distance {d['distance_pct']:+.0f}% | lower {d['head_drop_pct']:+.0f}% | sideways {d['head_shift_pct']:+.0f}%"
                                 + ("" if d["shoulder_tilt_deg"] is None else f" | shoulders {d['shoulder_tilt_deg']:+.1f} deg"))
                    for reason in snapshot["reasons"][:2]:
                        lines.append("  - " + reason)
                a = attention.update(now, measurement.reason, metrics)
                if a["state"] == "unavailable":
                    lines.append("ATTENTION: " + a["message"])
                else:
                    turn = "" if a["deviations"] is None else (
                        f"  (turn {a['deviations']['head_turn_deg']:+.0f} deg, up/down {a['deviations']['head_up_down_deg']:+.0f} deg)")
                    lines.append(f"ATTENTION: {a['state'].upper()}{turn}")
            for i, line in enumerate(lines):
                cv2.putText(view, line, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Posture test (C = baseline, Q = quit)", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")) and recording_since is None:
                samples, recording_since = [], now
                assessor, attention = PostureAssessor(None), AttentionAssessor(None)
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        detector.close()
        if pose:
            pose.close()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()
