"""
Face detection test WITHOUT the web app:

    python scripts/face_test.py

Opens a window with your webcam. A green box should follow your face and yellow outlines
should sit on your eyes. The message line tells you what the system sees.
Press Q (with the window selected) or Ctrl+C in the terminal to quit.

Try these and watch the message: cover the lens, step out of view, let a second person
in, move so part of your face leaves the image. Nothing is saved to disk.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from computer_vision import drawing  # noqa: E402  (import this package first: it silences OpenCV's log noise)
from computer_vision.camera import Camera, CameraError  # noqa: E402
from computer_vision.face_landmarks import (  # noqa: E402
    FACE_MESSAGES,
    FaceDetector,
    ModelError,
    StatusDebouncer,
)

import cv2  # noqa: E402


def main():
    detector = FaceDetector()
    try:
        detector.load()                       # load the model BEFORE switching the camera on
    except ModelError as error:
        print(error)
        sys.exit(1)

    camera = Camera()
    try:
        camera.open()
    except CameraError as error:
        print(error)
        detector.close()
        sys.exit(1)

    print("Running. Press Q in the video window to quit.")
    debouncer = StatusDebouncer()
    start = time.monotonic()
    milliseconds = 0.0
    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Camera stopped sending images.")
                break
            t0 = time.monotonic()
            result = detector.detect(frame, int((t0 - start) * 1000))
            milliseconds = (time.monotonic() - t0) * 1000 if milliseconds == 0 else 0.9 * milliseconds + 0.1 * (time.monotonic() - t0) * 1000
            shown = debouncer.update(result.status)

            drawing.draw_face_overlay(frame, result)
            view = cv2.flip(frame, 1)         # mirror first, THEN write text so it reads correctly
            message = FACE_MESSAGES[shown] or "Face OK"
            cv2.putText(view, f"{message}   [{milliseconds:.0f} ms/frame, faces: {result.faces_found}]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("Face test (press Q to quit)", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()
