"""Web routes for registration, login, logout and "who am I". Thin wrappers around auth_service."""

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from pydantic import BaseModel

from backend import auth_service
from backend.dependencies import current_user, get_db
from backend.security import ValidationError
from computer_vision import monitor as monitor_module
from database.users import UsernameTakenError
from utils import config

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Credentials(BaseModel):
    username: str
    password: str


def _set_login_cookie(response, token):
    # HttpOnly: JavaScript on the page cannot read the token.
    # SameSite=Lax: other websites cannot make the browser send it along with their requests.
    # (No "Secure" flag because we run on http://127.0.0.1; add it if you ever use https.)
    response.set_cookie(
        key=config.LOGIN_COOKIE_NAME,
        value=token,
        max_age=config.LOGIN_TOKEN_HOURS * 3600,
        httponly=True,
        samesite="lax",
    )


@router.post("/register", status_code=201)
def register(credentials: Credentials, response: Response, connection=Depends(get_db)):
    try:
        user, token = auth_service.register(connection, credentials.username, credentials.password)
    except ValidationError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except UsernameTakenError:
        raise HTTPException(status_code=409, detail="That username is already taken.")
    _set_login_cookie(response, token)
    return {"username": user["username"]}


@router.post("/login")
def login(credentials: Credentials, response: Response, connection=Depends(get_db)):
    try:
        user, token = auth_service.login(connection, credentials.username, credentials.password)
    except auth_service.InvalidCredentialsError:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    _set_login_cookie(response, token)
    return {"username": user["username"]}


@router.post("/logout")
def logout(
    response: Response,
    session_token: str | None = Cookie(default=None, alias=config.LOGIN_COOKIE_NAME),
    connection=Depends(get_db),
):
    user = auth_service.get_user_from_token(connection, session_token)
    if user is not None:
        monitor_module.monitor.stop_if_owner(user["id"])   # never leave the camera on after logout
    auth_service.logout(connection, session_token)
    response.delete_cookie(config.LOGIN_COOKIE_NAME)
    return {"status": "logged out"}


@router.get("/me")
def me(user=Depends(current_user), connection=Depends(get_db)):
    return auth_service.describe_user(connection, user)
