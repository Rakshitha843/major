"""Database functions for the personal calibration baseline (one row per user).

The eye baseline is stored as JSON text, e.g. {"ear_open": 0.31, "ear_closed": 0.11, ...}.
The posture baseline (head position, tilt, distance, optional shoulders) is stored in its own column; "{}" means
"not calibrated for posture".
"""

import json
from datetime import datetime, timezone


def get_calibration(connection, user_id):
    """Return this user's calibration row, or None if they have not calibrated."""
    return connection.execute(
        "SELECT user_id, eye_baseline_json, posture_baseline_json, calibrated_at "
        "FROM calibrations WHERE user_id = ?",
        (user_id,),
    ).fetchone()


def save_eye_baseline(connection, user_id, eye_baseline):
    """Save (or replace) the user's eye baseline. Their other calibration data is kept."""
    connection.execute(
        """
        INSERT INTO calibrations (user_id, eye_baseline_json, posture_baseline_json, calibrated_at)
        VALUES (?, ?, '{}', ?)
        ON CONFLICT(user_id) DO UPDATE SET
            eye_baseline_json = excluded.eye_baseline_json,
            calibrated_at = excluded.calibrated_at
        """,
        (user_id, json.dumps(eye_baseline), datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()


def save_calibration(connection, user_id, eye_baseline, posture_baseline=None):
    """
    Save a full calibration: the eye baseline and the posture baseline together.
    If the new posture baseline is missing, the OLD one is cleared: a sitting position from an earlier session must not be
    judged against a new eye calibration, and "not calibrated for posture" is safer than a stale baseline.
    """
    connection.execute(
        """
        INSERT INTO calibrations (user_id, eye_baseline_json, posture_baseline_json, calibrated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            eye_baseline_json = excluded.eye_baseline_json,
            posture_baseline_json = excluded.posture_baseline_json,
            calibrated_at = excluded.calibrated_at
        """,
        (user_id, json.dumps(eye_baseline), json.dumps(posture_baseline or {}), datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()


def get_posture_baseline(connection, user_id):
    """The user's posture baseline dict, or None if they have none or the stored data is damaged."""
    row = get_calibration(connection, user_id)
    if row is None:
        return None
    try:
        baseline = json.loads(row["posture_baseline_json"])
        for key in ("roll_deg", "pitch_deg", "eye_distance", "face_x", "face_y"):
            baseline[key] = float(baseline[key])
    except (ValueError, KeyError, TypeError):
        return None
    if not baseline["eye_distance"] > 0:
        return None
    return baseline


def get_eye_baseline(connection, user_id):
    """
    Return the user's eye baseline as a dict with at least ear_open and ear_closed,
    or None if they have not calibrated or the stored data is damaged/unusable.
    """
    row = get_calibration(connection, user_id)
    if row is None:
        return None
    try:
        baseline = json.loads(row["eye_baseline_json"])
        ear_open = float(baseline["ear_open"])
        ear_closed = float(baseline["ear_closed"])
    except (ValueError, KeyError, TypeError):
        return None                                   # damaged data: treat as "not calibrated"
    if not ear_open > ear_closed > 0:
        return None
    baseline["ear_open"], baseline["ear_closed"] = ear_open, ear_closed
    baseline["calibrated_at"] = row["calibrated_at"]
    return baseline
