"""Small helpers that FastAPI runs before a route (called "dependencies")."""

from fastapi import Cookie, Depends, HTTPException

from backend import auth_service
from database import connection as db
from utils import config


def get_db():
    """Give each request its own database connection and close it afterwards."""
    connection = db.connect()
    try:
        yield connection
    finally:
        connection.close()


def current_user(
    session_token: str | None = Cookie(default=None, alias=config.LOGIN_COOKIE_NAME),
    connection=Depends(get_db),
):
    """Use on any route that needs a logged-in user. Stops the request with 401 otherwise."""
    if not session_token:
        raise HTTPException(status_code=401, detail="Please log in.")
    user = auth_service.get_user_from_token(connection, session_token)
    if user is None:
        raise HTTPException(status_code=401, detail="Your login has expired. Please log in again.")
    return user
