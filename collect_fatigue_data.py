"""
Record labelled fatigue data (NUMBERS ONLY, never images or video):

    python scripts/collect_fatigue_data.py --username asha --participant P01 --label normal --minutes 10
    python scripts/collect_fatigue_data.py --username asha --participant P01 --label fatigued --minutes 10

  --username     the dashboard account whose saved calibration is used (calibrate there first)
  --participant  a pseudonym for the person, e.g. P01. Never use a real name.
  --label        how the participant says they feel during THIS recording: normal or fatigued
  --minutes      how long to record (default 10)

While recording, a window shows the camera with a status line. Press Q to stop early.
One row of eye features is saved every 5 seconds (after the first ~10 seconds) to data/fatigue_features.csv.
Then train with:  python -m ml.train        (see docs/fatigue_ml.md for the recording protocol)

BEFORE YOU RECORD ANYONE ELSE: they must know what is recorded (eye-measurement numbers, no images), agree to
it, and be able to stop at any time. Do not make anyone tired on purpose in any way that could be unsafe
(no sleep deprivation, no driving). Use natural situations, for example a fresh morning versus a late evening
after a long day, and let the participant report honestly how they feel. The label is their report, not a medical test.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from computer_vision import drawing  # noqa: E402  (import this package first: it silences OpenCV's log noise)
from computer_vision.camera import Camera, CameraError  # noqa: E402
from computer_vision.eye_measurement import EyeMeasurer  # noqa: E402
from computer_vision.eye_tracker import EyeStateTracker  # noqa: E402
from computer_vision.face_landmarks import FaceDetector, ModelError  # noqa: E402
from database import calibration as calibration_db  # noqa: E402
from database import connection as db  # noqa: E402
from database import users as users_db  # noqa: E402
from ml import features  # noqa: E402
from ml.dataset import LABELS, append_rows  # noqa: E402
from utils import config  # noqa: E402

import cv2  # noqa: E402


def run_collection(camera, detector, baseline, label, participant, session_id, dataset_path,
                   minutes, interval_s, show_frame=None, key_pressed=None, clock=time.monotonic):
    """
    Record until `minutes` have passed or the user asks to stop. Returns the number of rows saved.
    show_frame(frame, status_text) and key_pressed() are optional (the tests leave them out).
    """
    tracker = EyeStateTracker(baseline)
    measurer = EyeMeasurer()                                        # same "can the eyes be trusted" rules as the dashboard
    start = clock()
    next_row_at = None
    rows_saved = 0
    while True:
        now = clock() - start
        if now >= minutes * 60 or (key_pressed and key_pressed()):
            break
        frame = camera.read()
        if frame is None:
            print("Camera stopped sending images.")
            break

        result = detector.detect(frame, int(now * 1000))
        values = measurer.measure(now, frame, result).values
        eye = tracker.update(now, None if values is None else values[2])

        ready = features.window_features_ready(eye) and eye.get("ear") is not None
        if ready:
            if next_row_at is None:
                next_row_at = now                                   # first row as soon as the window is ready
            if now >= next_row_at:
                row = {"participant": participant, "session_id": session_id, "source": "real", "label": label,
                       "timestamp": datetime.now(timezone.utc).isoformat()}
                row.update(features.feature_dict(eye))
                append_rows(dataset_path, [row])
                rows_saved += 1
                next_row_at = now + interval_s

        if show_frame:
            drawing.draw_face_overlay(frame, result, eyes_closed=bool(eye.get("eyes_closed")))
            state = "recording" if ready else "warming up / eyes not measured"
            show_frame(frame, f"REC '{label}' | {state} | rows saved: {rows_saved} | {int(now)} s of {int(minutes * 60)} s")
    return rows_saved


def main(argv=None, camera=None, detector=None, show_frame="window", clock=time.monotonic):
    parser = argparse.ArgumentParser(description="Record labelled fatigue data (numbers only).")
    parser.add_argument("--username", required=True, help="dashboard account whose calibration is used")
    parser.add_argument("--participant", required=True, help="pseudonym such as P01 (never a real name)")
    parser.add_argument("--label", required=True, choices=LABELS, help="how the participant feels during this recording")
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=config.DATA_SAMPLE_INTERVAL_S, help="seconds between saved rows")
    args = parser.parse_args(argv)

    connection = db.connect()
    try:
        user = users_db.get_user_by_username(connection, args.username)
        baseline = None if user is None else calibration_db.get_eye_baseline(connection, user["id"])
    finally:
        connection.close()
    if user is None:
        print(f"There is no account named '{args.username}'. Register in the dashboard first.")
        return 1
    if baseline is None:
        print(f"'{args.username}' has not calibrated yet. Log in to the dashboard and run the personal calibration first.")
        return 1

    detector = detector or FaceDetector()
    try:
        detector.load()
    except ModelError as error:
        print(error)
        return 1
    camera = camera or Camera()
    try:
        camera.open()
    except CameraError as error:
        print(error)
        detector.close()
        return 1

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Recording participant {args.participant} as '{args.label}' for up to {args.minutes:g} minutes.")
    print("Only measurement numbers are saved, never images. Press Q in the video window to stop.")
    window_name = "Recording (press Q to stop)"

    def default_show(frame, text):
        view = cv2.flip(frame, 1)
        cv2.putText(view, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.imshow(window_name, view)

    def default_key():
        return (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"))

    show = default_show if show_frame == "window" else show_frame
    key = default_key if show_frame == "window" else None
    try:
        saved = run_collection(camera, detector, baseline, args.label, args.participant, session_id,
                               config.DATASET_FILE, args.minutes, args.interval, show, key, clock)
    except KeyboardInterrupt:
        saved = 0
        print("Stopped.")
    finally:
        camera.release()
        detector.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # A "headless" OpenCV build (no GTK/Cocoa/Windows GUI support -- e.g. some Linux
            # servers or containers) cannot open windows in the first place, so there is nothing
            # to destroy. show_frame will already have been a no-op in that case; this just makes
            # sure cleanup never crashes the script on such a machine.
            pass
    print(f"Saved rows this session: {saved}  (file: {config.DATASET_FILE})")
    print("Rows are appended, so you can record more sessions. Train with: python -m ml.train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
