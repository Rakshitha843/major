"""
Download the MediaPipe model files (one time only, needs internet):

    python scripts/download_models.py

WHY THIS IS NEEDED
Current MediaPipe versions no longer ship their landmark models inside the
pip package. The models are small files that we download once and keep in
the models/ folder. After that the project runs fully offline, and all
webcam processing stays on your computer.
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import config  # noqa: E402

MIN_EXPECTED_BYTES = 1_000_000  # a real model is several MB; smaller means a bad download


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    if temporary.stat().st_size < MIN_EXPECTED_BYTES:
        temporary.unlink()
        raise RuntimeError("downloaded file is too small to be a valid model")
    temporary.replace(destination)


def main():
    failed = False
    for destination, url in config.MODEL_DOWNLOADS.items():
        if destination.is_file() and destination.stat().st_size >= MIN_EXPECTED_BYTES:
            print(f"[skip] {destination.name} already downloaded")
            continue
        print(f"[....] downloading {destination.name}")
        try:
            download(url, destination)
            print(f"[ OK ] saved to {destination}")
        except (urllib.error.URLError, RuntimeError, OSError) as error:
            failed = True
            print(f"[FAIL] {destination.name}: {error}")
            print(f"       Manual option: open this link in a browser and save the file as\n"
                  f"       {destination}\n       {url}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
