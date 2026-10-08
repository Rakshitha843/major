"""
Backend entry point (Phase 14 version).

  GET  /                  -> dashboard page (frontend/index.html)
  GET  /api/health        -> is the server (and database) running?
  GET  /api/system-check  -> are the libraries and model files present?
  POST /api/auth/register, /login, /logout and GET /api/auth/me  (see routes_auth.py)
  POST /api/monitor/start, /stop and GET /api/monitor/status, /frame, /history, /sessions  (see routes_monitor.py)
  POST /api/calibration/start, /cancel and GET /api/calibration/status  (see routes_calibration.py)
  GET  /api/fatigue/model  (see routes_fatigue.py)

Run from the project root with:
    python -m uvicorn backend.main:app --reload
"""

import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from backend import routes_auth, routes_calibration, routes_fatigue, routes_monitor
from computer_vision import monitor as monitor_module
from database import connection as db
from database import sessions as sessions_db
from utils import config, env_check


@asynccontextmanager
async def lifespan(app):
    """Runs once when the server starts: make sure the database tables exist."""
    try:
        db.init_db()
    except (sqlite3.Error, OSError) as error:
        raise RuntimeError(
            f"Could not open the database at {config.DATABASE_PATH}: {error}. "
            "Check that the data/ folder exists and can be written to."
        ) from error
    yield
    # Server shutting down (Ctrl+C, or --reload auto-restarting after a code change): make sure the
    # camera is off, AND save whatever session was in progress. Read who owns the camera BEFORE
    # stopping it -- stop() has no way to know, since there is no HTTP request during a shutdown, only
    # the monitor's own internal state. Without this, a session running when the server restarts (which
    # happens on every file save while using --reload during development) would be silently discarded:
    # its numbers would exist in memory for a moment and then vanish, never reaching the database.
    owner_id = monitor_module.monitor.get_active_owner()
    monitor_module.monitor.stop()
    if owner_id is not None:
        summary = monitor_module.monitor.take_session_result(owner_id)
        if sessions_db.should_save_session(summary):
            connection = db.connect()
            try:
                sessions_db.save_session(connection, owner_id, summary)
            finally:
                connection.close()


app = FastAPI(title=config.APP_TITLE, version=config.APP_VERSION, lifespan=lifespan)
app.include_router(routes_auth.router)
app.include_router(routes_monitor.router)
app.include_router(routes_calibration.router)
app.include_router(routes_fatigue.router)


@app.exception_handler(sqlite3.Error)
async def database_error_handler(request: Request, error: sqlite3.Error):
    """Any unexpected database problem becomes a clear message instead of a crash."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Database error. Check that the data/ folder can be written to, then try again."},
    )


@app.get("/api/health")
def health():
    try:
        connection = db.connect()
        try:
            connection.execute("SELECT 1")
        finally:
            connection.close()
        database_status = "ok"
    except (sqlite3.Error, OSError):
        database_status = "error"
    return {
        "status": "ok",
        "database": database_status,
        "app": config.APP_TITLE,
        "version": config.APP_VERSION,
        # Reports only whether the camera is on. It never switches the camera on.
        "camera": "on" if monitor_module.monitor.get_status()["camera_active"] else "off",
    }


@app.get("/api/system-check")
def system_check():
    """
    Reports missing libraries / model files.
    It does NOT test the camera, so opening the dashboard never switches it on.
    """
    checks = [env_check.check_python()]
    checks += env_check.check_packages()
    checks += env_check.check_models()
    return {"all_ok": all(c["ok"] for c in checks), "checks": checks}


@app.get("/")
def dashboard():
    # no-store: after an update the browser must load the new page, never an old cached copy
    return FileResponse(config.FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-store"})
