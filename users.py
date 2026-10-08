"""Database functions for users and login tokens. Contains no password logic."""

import sqlite3
from datetime import datetime, timedelta, timezone


class UsernameTakenError(Exception):
    """Raised when someone registers a username that already exists."""


def utc_now():
    return datetime.now(timezone.utc)


def create_user(connection, username, password_hash):
    """Insert a new user and return the new user id."""
    try:
        cursor = connection.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, utc_now().isoformat()),
        )
        connection.commit()
    except sqlite3.IntegrityError:
        connection.rollback()
        raise UsernameTakenError(username)
    return cursor.lastrowid


def get_user_by_username(connection, username):
    return connection.execute(
        "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
        (username,),  # column is COLLATE NOCASE, so the match ignores upper/lower case
    ).fetchone()


def create_login_token(connection, user_id, token_hash, hours):
    """Remember that this token belongs to this user for the next `hours` hours."""
    now = utc_now()
    connection.execute(
        "INSERT INTO login_tokens (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token_hash, user_id, now.isoformat(), (now + timedelta(hours=hours)).isoformat()),
    )
    connection.commit()


def get_user_by_token_hash(connection, token_hash):
    """Return the user who owns a valid (not expired) token, or None."""
    return connection.execute(
        """
        SELECT users.id, users.username, users.created_at
        FROM login_tokens
        JOIN users ON users.id = login_tokens.user_id
        WHERE login_tokens.token_hash = ? AND login_tokens.expires_at > ?
        """,
        (token_hash, utc_now().isoformat()),
    ).fetchone()


def delete_login_token(connection, token_hash):
    connection.execute("DELETE FROM login_tokens WHERE token_hash = ?", (token_hash,))
    connection.commit()


def delete_expired_tokens(connection):
    connection.execute("DELETE FROM login_tokens WHERE expires_at <= ?", (utc_now().isoformat(),))
    connection.commit()
