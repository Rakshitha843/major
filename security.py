"""
Passwords and login tokens, using only Python's standard library.

PASSWORDS
We never store a password. We store a salted hash made with scrypt (a
deliberately slow function, so guessing passwords from a stolen database is
expensive). The stored text looks like:

    scrypt$16384$8$1$<salt in hex>$<hash in hex>

The settings are stored inside the text, so we can raise them in future and
old passwords will still verify.

LOGIN TOKENS
After a correct login the browser receives a long random token in a cookie.
The database keeps only the SHA-256 hash of it.
"""

import hashlib
import hmac
import os
import re
import secrets

from utils import config

SCRYPT_N = 2 ** 14   # cost factor (memory/time)
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
HASH_BYTES = 32

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class ValidationError(ValueError):
    """The username or password does not meet the rules. The message is safe to show the user."""


def _scrypt(password, salt, n, r, p):
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=HASH_BYTES)


def hash_password(password):
    salt = os.urandom(SALT_BYTES)
    digest = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password, stored_hash):
    """True only if the password matches. Bad or damaged stored text returns False, never crashes."""
    try:
        scheme, n, r, p, salt_hex, digest_hex = stored_hash.split("$")
        if scheme != "scrypt":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = _scrypt(password, bytes.fromhex(salt_hex), int(n), int(r), int(p))
    except (ValueError, TypeError, AttributeError):
        return False
    return hmac.compare_digest(actual, expected)  # constant-time comparison


# Used when a username does not exist, so a wrong username takes as long as a
# wrong password (stops attackers discovering which usernames exist by timing).
DUMMY_HASH = hash_password("dummy-password-for-timing")


def generate_token():
    return secrets.token_urlsafe(32)


def hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def validate_username(username):
    """Return the cleaned username, or raise ValidationError."""
    username = (username or "").strip()
    if not config.USERNAME_MIN_LENGTH <= len(username) <= config.USERNAME_MAX_LENGTH:
        raise ValidationError(
            f"Username must be {config.USERNAME_MIN_LENGTH}-{config.USERNAME_MAX_LENGTH} characters."
        )
    if not USERNAME_PATTERN.match(username):
        raise ValidationError("Username may only contain letters, numbers, dot, dash and underscore.")
    return username


def validate_password(password):
    if not isinstance(password, str) or len(password) < config.PASSWORD_MIN_LENGTH:
        raise ValidationError(f"Password must be at least {config.PASSWORD_MIN_LENGTH} characters.")
    if len(password) > config.PASSWORD_MAX_LENGTH:
        raise ValidationError(f"Password must be at most {config.PASSWORD_MAX_LENGTH} characters.")
    return password
