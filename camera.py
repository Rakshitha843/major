"""
Webcam access.

A small wrapper around OpenCV's VideoCapture so the rest of the project never
has to deal with OpenCV camera details or error messages.
"""

import os
import sys

# OpenCV prints long technical warnings when a camera is missing. We show our own
# clear message instead. This must be set BEFORE cv2 is imported.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2  # noqa: E402

from utils import config  # noqa: E402


class CameraError(Exception):
    """A camera problem. The message is written so it can be shown directly to the user."""


class Camera:
    def __init__(self, index=None, width=None, height=None, fps=None):
        self.index = config.CAMERA_INDEX if index is None else index
        self.width = width or config.FRAME_WIDTH
        self.height = height or config.FRAME_HEIGHT
        self.fps = fps or config.TARGET_FPS
        self._capture = None

    def open(self):
        """Open the webcam and confirm it really delivers an image. Raises CameraError if not."""
        # On Windows the DirectShow backend usually opens faster and more reliably;
        # if it fails we fall back to OpenCV's default choice.
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]

        for backend in backends:
            capture = cv2.VideoCapture(self.index, backend)
            if not capture.isOpened():
                capture.release()
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep only the newest frame (less delay)
            got_frame, _ = capture.read()
            if got_frame:
                self._capture = capture
                return
            capture.release()

        raise CameraError("Camera not detected.")

    def read(self):
        """Return the next frame (a NumPy array, BGR colour) or None if reading failed."""
        if self._capture is None:
            return None
        got_frame, frame = self._capture.read()
        return frame if got_frame else None

    def release(self):
        """Switch the camera off (its light goes out). Safe to call more than once."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
