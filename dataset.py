"""
The fatigue dataset: a plain CSV file that contains NUMBERS ONLY (no images, no video).

One row = the eye features of one 20-second window, recorded every few seconds during a session.

    participant   a pseudonym you choose, e.g. P01 (never a real name)
    session_id    identifies one continuous recording (e.g. 20260920_143000)
    source        "real" for recordings made with the collector, "demo" for synthetic test data
    label         "normal" or "fatigued": the state the participant reported for that session
    timestamp     when the row was recorded (ISO 8601)
    <features>    mean_blink_duration_s, blinks_per_min, perclos, longest_closure_s
                  (mean_blink_duration_s may be empty when there was no blink in the window)

WHY sessions matter: rows from one session overlap in time and are very similar. To judge a model fairly
it must be tested on sessions (or people) it has NOT seen. ml/train.py does that.
"""

import csv
import hashlib
import math
from pathlib import Path

from ml.features import FEATURE_NAMES

COLUMNS = ["participant", "session_id", "source", "label", "timestamp"] + FEATURE_NAMES
LABELS = ("normal", "fatigued")
SOURCES = ("real", "demo")


class DatasetError(Exception):
    """A problem with the data or the amount of it. The message is written for the user."""


def append_rows(path, rows):
    """Add rows (dicts with the COLUMNS keys) to the CSV, creating it with a header if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: ("" if row.get(column) is None else row.get(column)) for column in COLUMNS})


def load_rows(paths):
    """
    Read and validate one or more CSV files.
    Returns (rows, problems). Each row is a dict with numeric features (NaN if empty).
    Rows that cannot be used are skipped and described in `problems`.
    """
    rows, problems = [], []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise DatasetError(f"Data file not found: {path}")
        with open(path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            missing = [column for column in COLUMNS if column not in (reader.fieldnames or [])]
            if missing:
                raise DatasetError(f"{path.name} is missing the columns: {', '.join(missing)}")
            for line_number, raw in enumerate(reader, start=2):
                row, why = _parse_row(raw)
                if row is None:
                    problems.append(f"{path.name} line {line_number}: {why}")
                else:
                    rows.append(row)
    return rows, problems


def _parse_row(raw):
    if raw["label"] not in LABELS:
        return None, f"label must be one of {LABELS}, not '{raw['label']}'"
    if raw["source"] not in SOURCES:
        return None, f"source must be one of {SOURCES}, not '{raw['source']}'"
    if not raw["participant"] or not raw["session_id"]:
        return None, "participant and session_id must not be empty"
    row = {key: raw[key] for key in ("participant", "session_id", "source", "label", "timestamp")}
    for name in FEATURE_NAMES:
        text = (raw[name] or "").strip()
        try:
            value = float("nan") if text == "" else float(text)
        except ValueError:
            return None, f"{name} is not a number: '{text}'"
        if not math.isnan(value) and (math.isinf(value) or value < 0):
            return None, f"{name} has an impossible value: {value}"
        row[name] = value
    if all(math.isnan(row[name]) for name in FEATURE_NAMES):
        return None, "no feature values"
    return row, ""


def dataset_sha256(paths):
    """A fingerprint of the exact data used, stored in the model so it can be traced later."""
    digest = hashlib.sha256()
    for path in sorted(str(p) for p in paths):
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()
