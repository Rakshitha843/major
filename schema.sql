-- Database schema (SQLite).
--
-- Only `users` and `login_tokens` are used in Phase 2. The other tables are
-- created now (they come from the project brief) but are filled in later phases.
-- Their columns may still change while we build those phases; during development
-- the simple fix is to delete data/wellbeing.db and start the app again.
--
-- Times are stored as text in ISO-8601 UTC, e.g. 2026-09-20T05:44:00+00:00
-- Every table that holds personal data has a user_id, so each user's data stays separate.

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,   -- "Ravi" and "ravi" are the same user
    password_hash TEXT    NOT NULL,                         -- salted hash, never the password
    created_at    TEXT    NOT NULL
);

-- One row per logged-in browser. We store only a HASH of the token, so even if
-- someone copied the database file they could not use it to log in.
CREATE TABLE IF NOT EXISTS login_tokens (
    token_hash TEXT    PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT    NOT NULL,
    expires_at TEXT    NOT NULL
);

-- Personal baseline (one row per user). Values are stored as JSON text so the
-- calibration phases can decide exactly which numbers to keep.
CREATE TABLE IF NOT EXISTS calibrations (
    user_id               INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    eye_baseline_json     TEXT NOT NULL,     -- e.g. normal eye aspect ratio
    posture_baseline_json TEXT NOT NULL,     -- e.g. normal head/shoulder position
    calibrated_at         TEXT NOT NULL
);

-- One row per monitoring session (a "session" = one Start ... Stop of the camera).
-- Phase 9 adds the columns below. If you are upgrading an existing database, connection.py's
-- migrate_schema() adds them with ALTER TABLE, since CREATE TABLE IF NOT EXISTS does not touch
-- a table that already exists.
CREATE TABLE IF NOT EXISTS monitoring_sessions (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    start_time                  TEXT    NOT NULL,
    end_time                    TEXT,                         -- empty while the session is running
    duration_s                  REAL,                          -- wall-clock time from Start to Stop
    active_duration_s           REAL,                          -- duration_s minus time spent on breaks
    break_count                 INTEGER,                       -- absences of at least SESSION_BREAK_MIN_S
    break_duration_s            REAL,
    fatigue_warning_count       INTEGER,                       -- how many times fatigue entered WARNING
    fatigue_event_count         INTEGER,                       -- how many times fatigue entered FATIGUED
    posture_warning_count       INTEGER,
    posture_poor_count          INTEGER,
    attention_distracted_count  INTEGER,
    attention_no_face_count     INTEGER,
    dhri_moderate_count         INTEGER,                       -- how many times the overall DHRI risk entered MODERATE
    dhri_high_count             INTEGER,                       -- ... and HIGH (Phase 10)
    alert_count                 INTEGER DEFAULT 0              -- filled in by Phase 11 (alerts); 0 until then
);

-- Numbers measured during a session (never video frames).
CREATE TABLE IF NOT EXISTS measurements (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id        INTEGER NOT NULL REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
    timestamp         TEXT    NOT NULL,
    ear               REAL,                  -- eye aspect ratio
    blink_count       INTEGER,
    closure_duration_s REAL,
    perclos           REAL,                  -- eye-closure percentage over the time window
    fatigue_score     REAL,
    fatigue_state     TEXT,
    posture_score     REAL,
    posture_state     TEXT,
    attention_score   REAL,
    attention_state   TEXT,
    dhri              REAL,
    risk_level        TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id INTEGER REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
    timestamp  TEXT    NOT NULL,
    alert_type TEXT    NOT NULL,
    message    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_measurements_user_time ON measurements(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON monitoring_sessions(user_id, start_time);
CREATE INDEX IF NOT EXISTS idx_alerts_user_time ON alerts(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_tokens_expiry ON login_tokens(expires_at);
