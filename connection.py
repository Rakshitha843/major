"""
Opens the SQLite database and creates the tables.

All SQL in this project lives inside the database/ folder. If the project
ever moves to another database (for example PostgreSQL), this is the folder
that would change.
"""

import sqlite3
from pathlib import Path

from utils import config

SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"


def connect():
    """
    Open a new connection. Each web request opens its own and closes it afterwards.

    check_same_thread=False: FastAPI may set up and finish a request on different
    worker threads. That is safe here because a connection belongs to ONE request
    only and is never shared between requests.
    """
    # config.DATABASE_PATH is read at call time so tests can point it at a temp file.
    config.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DATABASE_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row            # lets us read columns by name
    connection.execute("PRAGMA foreign_keys = ON")  # SQLite ignores foreign keys unless asked
    return connection


def init_db():
    """Create all tables if they do not exist yet, then bring an older database up to date."""
    connection = connect()
    try:
        connection.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
        connection.commit()
        migrate_schema(connection)
    finally:
        connection.close()


# Columns added after monitoring_sessions first shipped (Phase 1). CREATE TABLE IF NOT EXISTS does not add
# columns to a table that already exists, so a database created before Phase 9 needs these added by hand.
# Safe to run every start-up: each ALTER TABLE is skipped if the column is already there.
_SESSION_COLUMNS_ADDED_IN_PHASE_9 = {
    "active_duration_s": "REAL", "break_count": "INTEGER", "break_duration_s": "REAL",
    "fatigue_warning_count": "INTEGER", "fatigue_event_count": "INTEGER",
    "posture_warning_count": "INTEGER", "posture_poor_count": "INTEGER",
    "attention_distracted_count": "INTEGER", "attention_no_face_count": "INTEGER",
    "alert_count": "INTEGER DEFAULT 0",
}

# Columns added in Phase 10 (DHRI). Same idea as the Phase 9 columns above: added by ALTER TABLE so an
# existing database keeps working, skipped if already present so this is safe to run on every start-up.
_SESSION_COLUMNS_ADDED_IN_PHASE_10 = {
    "dhri_moderate_count": "INTEGER",
    "dhri_high_count": "INTEGER",
}


def migrate_schema(connection):
    """Add columns that later phases introduced to a database created by an earlier phase."""
    existing = {row["name"] for row in connection.execute("PRAGMA table_info(monitoring_sessions)")}
    for column, sql_type in {**_SESSION_COLUMNS_ADDED_IN_PHASE_9, **_SESSION_COLUMNS_ADDED_IN_PHASE_10}.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE monitoring_sessions ADD COLUMN {column} {sql_type}")
    connection.commit()
