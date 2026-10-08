"""Web routes for switching the camera on/off and reading its status and preview image."""

from fastapi import APIRouter, Depends, HTTPException, Response

from backend.dependencies import current_user, get_db
from computer_vision import monitor as monitor_module
from computer_vision.monitor import CameraBusyError
from database import calibration as calibration_db
from database import sessions as sessions_db

router = APIRouter(prefix="/api/monitor", tags=["monitor"])

# Note: routes use `monitor_module.monitor` (looked up on each request) so tests can swap in a fake camera.


@router.post("/start")
def start_monitoring(user=Depends(current_user), connection=Depends(get_db)):
    baseline = calibration_db.get_eye_baseline(connection, user["id"])   # None = not calibrated yet
    posture_baseline = calibration_db.get_posture_baseline(connection, user["id"])
    try:
        status = monitor_module.monitor.start(user["id"], baseline=baseline, posture_baseline=posture_baseline)
    except CameraBusyError:
        raise HTTPException(
            status_code=409,
            detail="The camera is already being used by another account on this computer.",
        )
    if status["state"] == "error":
        raise HTTPException(status_code=503, detail=status["message"])
    return status


@router.post("/stop")
def stop_monitoring(user=Depends(current_user), connection=Depends(get_db)):
    # Only the account using the camera can stop it; anyone else's Stop does nothing.
    monitor_module.monitor.stop_if_owner(user["id"])
    # stop_if_owner() joins the background thread before returning, so by now the just-finished session's
    # summary (if this account actually owned the camera) is ready to collect and save, exactly once.
    summary = monitor_module.monitor.take_session_result(user["id"])
    if sessions_db.should_save_session(summary):
        sessions_db.save_session(connection, user["id"], summary)
    return monitor_module.monitor.get_status(user["id"])


@router.get("/status")
def monitoring_status(user=Depends(current_user)):
    monitor_module.monitor.touch(user["id"])      # tells the monitor the dashboard is still open
    return monitor_module.monitor.get_status(user["id"])


@router.get("/frame")
def latest_frame(user=Depends(current_user)):
    """The newest camera image as a JPEG (for the live preview). 204 = nothing to show."""
    monitor_module.monitor.touch(user["id"])
    result = monitor_module.monitor.get_frame(user["id"])
    if result is None:
        return Response(status_code=204)
    frame_id, jpeg = result
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "X-Frame-Id": str(frame_id)},
    )


def _session_summary_dict(row):
    """
    The fields the dashboard needs from one finished session row: the trend charts (Phase 12) and the full
    history view (Phase 13) both use this SAME function, so a field added to one is never missing from the
    other -- the earlier Phase 12 bug (alert_count present in the database but missing from one of the two
    JSON responses) can no longer happen this way.
    """
    return {
        "start_time": row["start_time"],
        "duration_s": row["duration_s"],
        "active_duration_s": row["active_duration_s"],
        "break_count": row["break_count"],
        "break_duration_s": row["break_duration_s"],
        "fatigue_warning_count": row["fatigue_warning_count"],
        "fatigue_event_count": row["fatigue_event_count"],
        "posture_warning_count": row["posture_warning_count"],
        "posture_poor_count": row["posture_poor_count"],
        "attention_distracted_count": row["attention_distracted_count"],
        "attention_no_face_count": row["attention_no_face_count"],
        "dhri_moderate_count": row["dhri_moderate_count"],
        "dhri_high_count": row["dhri_high_count"],
        "alert_count": row["alert_count"],
    }


@router.get("/history")
def session_history(user=Depends(current_user), connection=Depends(get_db)):
    """Totals for today, plus the user's most recent finished sessions (newest first), for the trend charts."""
    today = sessions_db.get_today_summary(connection, user["id"])
    recent = sessions_db.get_recent_sessions(connection, user["id"], limit=20)
    return {"today": today, "recent": [_session_summary_dict(row) for row in recent]}


@router.get("/sessions")
def session_list(offset: int = 0, limit: int = 20, user=Depends(current_user), connection=Depends(get_db)):
    """
    A page of the user's full session history, newest first, for the "view full history" list (Phase 13) --
    unlike /history (a fixed snapshot of the last 20, for the trend charts), this supports paging back through
    everything. `total_count` tells the caller whether more remain to load.
    """
    limit = max(1, min(limit, 100))    # a sensible ceiling: nobody needs 10,000 rows in one response
    offset = max(0, offset)
    sessions = sessions_db.get_recent_sessions(connection, user["id"], limit=limit, offset=offset)
    return {
        "sessions": [_session_summary_dict(row) for row in sessions],
        "total_count": sessions_db.count_finished_sessions(connection, user["id"]),
        "offset": offset,
        "limit": limit,
    }
