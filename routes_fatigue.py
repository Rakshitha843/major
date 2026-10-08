"""Web route that says which fatigue mode is available (Phase 6)."""

from fastapi import APIRouter, Depends

from backend.dependencies import current_user
from ml import fatigue
from ml.model import describe_model

router = APIRouter(prefix="/api/fatigue", tags=["fatigue"])


@router.get("/model")
def model_info(user=Depends(current_user)):
    """DEMO / RULE-BASED MODE or TRAINED ML MODE, and facts about the trained model if there is one."""
    info = describe_model()
    mode = fatigue.MODE_ML if info["available"] else fatigue.MODE_RULES
    return {
        "mode": mode,
        "mode_label": fatigue.MODE_LABELS[mode],
        "message": info["message"],
        "summary": info["summary"],
        "disclaimer": fatigue.DISCLAIMER,
    }
