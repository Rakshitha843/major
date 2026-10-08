"""
Database functions for monitoring sessions (Phase 9).

A "session" is one Start-to-Stop of the camera. The row is written ONCE, when the session ends (see
save_session), using the summary the monitor produces (computer_vision/session.py + the event counts from
the fatigue/posture/attention assessors). Nothing is written while a session is running -- the live numbers
the dashboard shows come straight from the monitor's in-memory status, not from the database.
"""

from datetime import datetime, timedelta, timezone


def should_save_session(summary):
    """
    True if a session is worth a database row. A session that never really started (Stop pressed before the
    first frame was even processed, so duration_s is exactly 0) is not worth cluttering the history with.
    """
    return summary is not None and summary["duration_s"] > 0


def save_session(connection, user_id, summary):
    """
    Save a finished session. `summary` is the dict monitor.take_session_result() returns: start_time (a
    datetime), duration_s, active_duration_s, break_count, break_duration_s, fatigue_events, posture_events,
    attention_events, dhri_events, alert_count (Phase 11; defaults to 0 if missing). Returns the new row's id.
    """
    start_time = summary["start_time"]
    end_time = start_time + timedelta(seconds=summary["duration_s"])
    fatigue, posture, attention = summary["fatigue_events"], summary["posture_events"], summary["attention_events"]
    dhri = summary.get("dhri_events", {})
    alert_count = summary.get("alert_count", 0)
    cursor = connection.execute(
        """
        INSERT INTO monitoring_sessions (
            user_id, start_time, end_time, duration_s, active_duration_s, break_count, break_duration_s,
            fatigue_warning_count, fatigue_event_count, posture_warning_count, posture_poor_count,
            attention_distracted_count, attention_no_face_count, dhri_moderate_count, dhri_high_count, alert_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id, start_time.isoformat(), end_time.isoformat(), summary["duration_s"],
            summary["active_duration_s"], summary["break_count"], summary["break_duration_s"],
            fatigue.get("warning", 0), fatigue.get("fatigued", 0),
            posture.get("warning", 0), posture.get("poor", 0),
            attention.get("distracted", 0), attention.get("no_face", 0),
            dhri.get("moderate", 0), dhri.get("high", 0), alert_count,
        ),
    )
    connection.commit()
    return cursor.lastrowid


def get_recent_sessions(connection, user_id, limit=20, offset=0):
    """The user's most recent finished sessions, newest first. `offset` skips that many (for pagination)."""
    return connection.execute(
        "SELECT * FROM monitoring_sessions WHERE user_id = ? AND end_time IS NOT NULL "
        "ORDER BY start_time DESC LIMIT ? OFFSET ?",
        (user_id, limit, offset),
    ).fetchall()


def count_finished_sessions(connection, user_id):
    """How many finished sessions this user has in total (for the full-history view's pagination)."""
    row = connection.execute(
        "SELECT COUNT(*) AS n FROM monitoring_sessions WHERE user_id = ? AND end_time IS NOT NULL",
        (user_id,),
    ).fetchone()
    return row["n"]


def get_today_summary(connection, user_id, now=None):
    """
    Totals for the user's finished sessions that started today (in UTC), for the "exposure today" figure:
    {"session_count", "duration_s", "active_duration_s", "break_count"}. All zero if there were none.
    """
    now = now or datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    row = connection.execute(
        """
        SELECT COUNT(*) AS session_count,
               COALESCE(SUM(duration_s), 0) AS duration_s,
               COALESCE(SUM(active_duration_s), 0) AS active_duration_s,
               COALESCE(SUM(break_count), 0) AS break_count
        FROM monitoring_sessions
        WHERE user_id = ? AND end_time IS NOT NULL AND start_time >= ?
        """,
        (user_id, start_of_day),
    ).fetchone()
    return dict(row)
