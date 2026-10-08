"""
Create SYNTHETIC (invented) fatigue data, ONLY to test the training code end to end.

    python -m ml.make_demo_data

These rows are NOT measurements of anyone. The distributions below are guesses that make "normal" and
"fatigued" look different, so the pipeline has something to learn. Because they are invented, any score
computed from them proves nothing about real people or about this project. The file and the model trained
from it are marked DEMO, and the application refuses to load a demo model.
"""

import sys

import numpy as np

from ml.dataset import append_rows
from ml.features import FEATURE_NAMES
from utils import config

# (mean, spread) of each feature for the two invented classes. Purely made up.
INVENTED_DISTRIBUTIONS = {
    "normal":   {"mean_blink_duration_s": (0.15, 0.03), "blinks_per_min": (16.0, 4.0),
                 "perclos": (0.04, 0.02), "longest_closure_s": (0.25, 0.08)},
    "fatigued": {"mean_blink_duration_s": (0.30, 0.07), "blinks_per_min": (26.0, 8.0),
                 "perclos": (0.18, 0.06), "longest_closure_s": (0.90, 0.35)},
}


def generate(path=None, seed=0, sessions_per_class=6, rows_per_session=40, participants=4):
    """Write synthetic rows (marked source=demo) and return how many were written."""
    path = config.DEMO_DATASET_FILE if path is None else path
    if path.exists():
        path.unlink()                                     # always start from a fresh file
    rng = np.random.default_rng(seed)
    rows = []
    for label, distribution in INVENTED_DISTRIBUTIONS.items():
        for session in range(sessions_per_class):
            participant = f"DEMO{session % participants + 1:02d}"
            session_id = f"demo_{label}_{session + 1}"
            shift = {name: rng.normal(0, 0.15 * spread) for name, (_, spread) in distribution.items()}
            for i in range(rows_per_session):
                row = {"participant": participant, "session_id": session_id, "source": "demo",
                       "label": label, "timestamp": f"DEMO-{session_id}-{i:03d}"}
                for name, (mean, spread) in distribution.items():
                    row[name] = round(max(0.0, rng.normal(mean + shift[name], spread)), 4)
                rows.append(row)
    append_rows(path, rows)
    return len(rows)


def main():
    count = generate()
    print(f"Wrote {count} SYNTHETIC rows to {config.DEMO_DATASET_FILE}")
    print("They are invented: use them only to test the code, e.g.  python -m ml.train --demo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
