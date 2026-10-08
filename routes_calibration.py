"""Web routes for personal calibration (eyes now; posture is added in Phase 7)."""

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import current_user, get_db
from computer_vision import monitor as monitor_module
from computer_vision.monitor import CameraBusyError
from database import calibration as calibration_db

router = APIRouter(prefix="/api/calibration", tags=["calibration"])

IDLE_STATUS = {"active": False, "phase": "idle", "message": "", "seconds_left": 0, "progress": 0.0}


@router.post("/start")
def start_calibration(user=Depends(current_user), connection=Depends(get_db)):
    """Switch the camera on (if it is off) and begin the calibration steps."""
    baseline = calibration_db.get_eye_baseline(connection, user["id"])
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
    if not monitor_module.monitor.begin_calibration(user["id"]):
        raise HTTPException(status_code=409, detail="The camera is not available for calibration. Please try again.")
    return monitor_module.monitor.get_calibration_status(user["id"]) or IDLE_STATUS


@router.get("/status")
def calibration_status(user=Depends(current_user), connection=Depends(get_db)):
    """
    Progress of the running calibration. When it has finished successfully, the baseline is
    saved to this user's account here (once), and "saved" is true in that response.
    """
    monitor_module.monitor.touch(user["id"])
    status = monitor_module.monitor.get_calibration_status(user["id"]) or dict(IDLE_STATUS)
    result = monitor_module.monitor.take_calibration_result(user["id"])
    saved = False
    if result is not None and result["ok"]:
        calibration_db.save_calibration(connection, user["id"], result["baseline"], result.get("posture_baseline"))
        saved = True
    return {**status, "result": result, "saved": saved}


@router.post("/cancel")
def cancel_calibration(user=Depends(current_user)):
    monitor_module.monitor.cancel_calibration(user["id"])
    return {"status": "cancelling"}
