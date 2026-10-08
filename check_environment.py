"""
Run this first to see whether your computer is ready:

    python scripts/check_environment.py

Add --no-camera to skip the webcam test.
The camera test switches the webcam on for a fraction of a second,
reads one frame to confirm it works, and discards it (nothing is saved).
"""

import sys
from pathlib import Path

# Allow "from utils import ..." when this script is run from any folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import config, env_check  # noqa: E402


def main():
    skip_camera = "--no-camera" in sys.argv

    results = [env_check.check_python()]
    results += env_check.check_packages()
    results += env_check.check_models()
    if config.FACE_LANDMARKER_FILE.is_file():          # if it is missing, the line above already says so
        results.append(env_check.check_face_model_loads())
    if config.POSE_LANDMARKER_FILE.is_file():
        results.append(env_check.check_pose_model_loads())
    if not skip_camera:
        results.append(env_check.check_camera())

    print("\nEnvironment check")
    print("-" * 60)
    for result in results:
        tag = "[ OK ]" if result["ok"] else "[FAIL]"
        print(f"{tag} {result['name']:<38} {result['detail']}")
    print("-" * 60)

    failures = [r for r in results if not r["ok"]]
    if failures:
        print(f"{len(failures)} problem(s) found. Fix the lines marked [FAIL] above.\n")
        sys.exit(1)
    print("Everything is ready.\n")


if __name__ == "__main__":
    main()
