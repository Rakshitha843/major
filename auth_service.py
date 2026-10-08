"""
Registration, login and logout logic.

This file knows nothing about web requests or HTTP. The web routes call these
functions and turn their results/errors into HTTP responses. Keeping the logic
separate makes it easy to read and to test.
"""

from computer_vision.attention import baseline_supports_attention
from database import calibration, users
from backend import security
from utils import config


class InvalidCredentialsError(Exception):
    """Wrong username or wrong password (we never say which one)."""


def register(connection, username, password):
    """Create an account and log it in. Returns (user_info, token)."""
    username = security.validate_username(username)
    security.validate_password(password)

    password_hash = security.hash_password(password)
    user_id = users.create_user(connection, username, password_hash)  # may raise UsernameTakenError

    token = _start_login(connection, user_id)
    return {"id": user_id, "username": username}, token


def login(connection, username, password):
    """Check the password. Returns (user_info, token) or raises InvalidCredentialsError."""
    row = users.get_user_by_username(connection, (username or "").strip())

    # Always run one password check, even for unknown usernames (see security.DUMMY_HASH).
    stored_hash = row["password_hash"] if row else security.DUMMY_HASH
    password_ok = security.verify_password(password or "", stored_hash)

    if row is None or not password_ok:
        raise InvalidCredentialsError()

    token = _start_login(connection, row["id"])
    return {"id": row["id"], "username": row["username"]}, token


def logout(connection, token):
    """Forget this login token (safe to call with an unknown token)."""
    if token:
        users.delete_login_token(connection, security.hash_token(token))


def get_user_from_token(connection, token):
    """Return the logged-in user row for a cookie token, or None if missing/expired/unknown."""
    if not token:
        return None
    return users.get_user_by_token_hash(connection, security.hash_token(token))


def describe_user(connection, user):
    """What the dashboard's profile panel needs (Phase 12 will add session status)."""
    baseline = calibration.get_eye_baseline(connection, user["id"])
    posture_baseline = calibration.get_posture_baseline(connection, user["id"])
    return {
        "username": user["username"],
        "calibrated": baseline is not None,
        "calibrated_at": None if baseline is None else baseline["calibrated_at"],
        "ear_open": None if baseline is None else round(baseline["ear_open"], 3),
        "ear_closed": None if baseline is None else round(baseline["ear_closed"], 3),
        "posture_calibrated": posture_baseline is not None,
        # attention needs the head direction, which calibrations made before Phase 8 do not contain
        "attention_calibrated": baseline_supports_attention(posture_baseline),
    }


def _start_login(connection, user_id):
    users.delete_expired_tokens(connection)   # keeps the table small
    token = security.generate_token()
    users.create_login_token(
        connection, user_id, security.hash_token(token), config.LOGIN_TOKEN_HOURS
    )
    return token
