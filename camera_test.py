"""
Quick camera test WITHOUT the web app:

    python scripts/camera_test.py

Opens a window showing your webcam with the frame rate and lighting reading.
Press Q (with the window selected) or Ctrl+C in the terminal to quit.

Use this if the dashboard shows "Camera not detected." - it tells you whether the
problem is the camera itself or the web app. Nothing is saved to disk.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from computer_vision import lighting  # noqa: E402  (import this package first: it silences OpenCV's log noise)
from computer_vision.camera import Camera, CameraError  # noqa: E402

import cv2  # noqa: E402


def main():
    camera = Camera()
    try:
        camera.open()
    except CameraError as error:
        print(error)
        print("Close other apps that use the camera (Teams, Zoom, browser tabs) and try again.")
        print("If you have more than one camera, change CAMERA_INDEX in utils/config.py.")
        sys.exit(1)

    print("Camera opened. Press Q in the video window to quit.")
    smoothed, previous, fps = None, time.monotonic(), 0.0
    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Camera stopped sending images.")
                break
            smoothed = lighting.smooth(smoothed, lighting.mean_brightness(frame))
            now = time.monotonic()
            fps = 1.0 / (now - previous) if fps == 0.0 else 0.9 * fps + 0.1 * (1.0 / (now - previous))
            previous = now
            state, _ = lighting.classify_lighting(smoothed)
            text = f"{fps:.1f} fps | brightness {smoothed:.0f}/255 | lighting: {state}"
            cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Camera test (press Q to quit)", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()
