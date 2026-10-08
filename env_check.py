"""
Environment checks.

These small functions answer: "Is this computer ready to run the project?"
They are used in two places:
  * scripts/check_environment.py  (you run it in the terminal)
  * backend/main.py               (/api/system-check, shown on the dashboard)

Each check returns a dictionary:  {"name": ..., "ok": True/False, "detail": ...}
"""

import importlib.util
import os
import sys

from utils import config

# import-name -> pip package name (they differ for OpenCV and scikit-learn)
REQUIRED_PACKAGES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
}

MINIMUM_PYTHON = (3, 10)


def check_python():
    """Python version must be recent enough for the libraries we use."""
    version = sys.version_info
    ok = (version.major, version.minor) >= MINIMUM_PYTHON
    detail = f"Python {version.major}.{version.minor}.{version.micro}"
    if not ok:
        detail += f" (need {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer)"
    return {"name": "Python version", "ok": ok, "detail": detail}


def check_packages():
    """Check that each required library can be found (without importing it)."""
    results = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        found = importlib.util.find_spec(import_name) is not None
        detail = "installed" if found else f"missing -> pip install {pip_name}"
        results.append({"name": f"Package: {pip_name}", "ok": found, "detail": detail})
    return results


def check_models(model_files=None):
    """Check that the MediaPipe model files have been downloaded."""
    if model_files is None:
        model_files = list(config.MODEL_DOWNLOADS.keys())
    results = []
    for path in model_files:
        present = path.is_file() and path.stat().st_size > 0
        if present:
            detail = f"found ({path.stat().st_size / 1_000_000:.1f} MB)"
        else:
            detail = "missing -> run: python scripts/download_models.py"
        results.append({"name": f"Model file: {path.name}", "ok": present, "detail": detail})
    return results


def check_face_model_loads(model_path=None):
    """
    Try to LOAD the face model (no camera involved). A file can exist but be damaged
    (for example an interrupted download), and this is how we find out.
    """
    from computer_vision.face_landmarks import FaceDetector, ModelError

    detector = FaceDetector(model_path=model_path)
    try:
        detector.load()
    except ModelError as error:
        return {"name": "Face model loads", "ok": False, "detail": str(error)}
    finally:
        detector.close()
    return {"name": "Face model loads", "ok": True, "detail": "loaded correctly"}


def check_pose_model_loads(model_path=None):
    """
    Try to LOAD the shoulder (pose) model (no camera involved). The pose model is OPTIONAL: without it, posture is
    judged from the head position only, so a failure here is a warning, not a blocker.
    """
    from computer_vision.face_landmarks import ModelError
    from computer_vision.pose_landmarks import PoseDetector

    detector = PoseDetector(model_path=model_path)
    try:
        detector.load()
    except ModelError as error:
        return {"name": "Shoulder model loads (optional)", "ok": False, "detail": str(error)}
    finally:
        detector.close()
    return {"name": "Shoulder model loads (optional)", "ok": True, "detail": "loaded correctly"}


def check_camera(camera_index=None):
    """
    Try to grab ONE frame from the webcam, then release it immediately.

    PRIVACY: the frame is only measured (width x height) and then thrown
    away. Nothing is saved or displayed. This function is deliberately NOT
    called from the API, so a web request can never switch the camera on.
    """
    if camera_index is None:
        camera_index = config.CAMERA_INDEX
    # OpenCV prints long technical warnings when no camera exists; hide them,
    # because we show our own clear message instead. Must be set BEFORE import.
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    try:
        import cv2
    except ImportError:
        return {"name": "Camera", "ok": False, "detail": "OpenCV not installed"}

    capture = cv2.VideoCapture(camera_index)
    try:
        if not capture.isOpened():
            return {"name": "Camera", "ok": False, "detail": "Camera not detected."}
        got_frame, frame = capture.read()
        if not got_frame or frame is None:
            return {"name": "Camera", "ok": False,
                    "detail": "Camera opened but returned no image (is another app using it?)"}
        height, width = frame.shape[:2]
        return {"name": "Camera", "ok": True, "detail": f"working ({width}x{height})"}
    finally:
        capture.release()
