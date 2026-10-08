"""
Alerts and recommendations (Phase 11), from already-computed fatigue/posture/attention/DHRI snapshots.

TWO DIFFERENT THINGS, BOTH FROM THE PROJECT BRIEF
    ALERTS (brief section 13): point-in-time notifications, each with its OWN COOLDOWN, so the same alert
    does not repeat every frame. An alert fires the moment its condition becomes true, and then will not
    fire again until its cooldown (utils/config.py: ALERT_COOLDOWN_S) has passed -- even if the underlying
    condition never clears in between. This is a repeating reminder ("you are still slouching, 5 minutes
    later"), not a one-shot notification per episode.

    RECOMMENDATIONS (brief section 14): practical, always-current advice. Unlike alerts, these have NO
    cooldown -- they simply reflect whatever is true right now, and disappear the moment the condition
    clears. There can be more than one at a time (for example fatigued AND poor posture at once).

WHERE THE CONDITIONS COME FROM
    All conditions are read from the ALREADY-SMOOTHED state fields (fatigue.state, posture.state,
    attention.state, dhri.level) that Phases 6-10 already compute, plus the session's continuous_active_s
    (Phase 9/10). Nothing here re-implements its own flicker protection -- alerts and recommendations only
    ever see a condition once its own indicator has already decided it is real.

THIS IS NOT MEDICAL ADVICE
    Every recommendation is simple, practical, and deliberately not a treatment -- "take a break", "adjust
    your posture", "rest your eyes". None of this diagnoses or treats anything.
"""

from dataclasses import dataclass

from utils import config

# ---------------------------------------------------------------------------
# Alert definitions: (id, message). The condition for each is decided in _alert_conditions() below, and the
# wording for "take a short break" / "eyes closed" / "sitting posture" / "continuously for a long period" /
# "attention appears reduced" matches the project brief's own examples.
# ---------------------------------------------------------------------------
ALERT_MESSAGES = {
    "take_a_break": "Take a short break.",
    "eye_closure": "Your eyes appear closed for longer than usual.",
    "posture": "Please correct your sitting posture.",
    "exposure": "You have been using the computer continuously for a long period.",
    "attention": "Your attention appears reduced.",
}


@dataclass
class Alert:
    id: str
    message: str
    at_s: float          # seconds since the session started, when this alert fired


def _alert_conditions(fatigue, posture, attention, dhri, continuous_active_s):
    """Which alert conditions are true RIGHT NOW, as {alert_id: bool}. All from already-smoothed states."""
    fatigue_reasons = " ".join((fatigue or {}).get("reasons") or [])
    return {
        "take_a_break": (dhri or {}).get("level") == "high",
        "eye_closure": (fatigue or {}).get("state") == "fatigued" and "eye closure" in fatigue_reasons,
        "posture": (posture or {}).get("state") == "poor",
        "exposure": continuous_active_s is not None and continuous_active_s >= config.DHRI_EXPOSURE_HIGH_MIN * 60.0,
        "attention": (attention or {}).get("state") == "distracted",
    }


def _recommendations(fatigue, posture, attention, dhri, continuous_active_s):
    """Practical, always-current suggestions for whatever is true right now (no cooldown). May be empty."""
    fatigue_state = (fatigue or {}).get("state")
    fatigue_reasons = " ".join((fatigue or {}).get("reasons") or [])
    recommendations = []

    # The brief's own two fatigue examples, kept distinct: a general "increased fatigue" tip, and a more
    # specific one when the cause is repeated/prolonged eye closure specifically.
    if fatigue_state == "fatigued" and "eye closure" in fatigue_reasons:
        recommendations.append("Rest your eyes for a short period.")
    elif fatigue_state in ("warning", "fatigued"):
        recommendations.append("Consider taking a short break.")

    if (posture or {}).get("state") == "poor":
        recommendations.append("Adjust your sitting position and keep your back and neck aligned.")

    if continuous_active_s is not None and continuous_active_s >= config.DHRI_EXPOSURE_WARNING_MIN * 60.0:
        recommendations.append("Consider taking a screen break.")

    # Our own additions (not verbatim from the brief), covering attention and overall risk the same way the
    # brief's own examples cover fatigue/posture/exposure -- simple, practical, not a diagnosis or treatment.
    if (attention or {}).get("state") == "distracted":
        recommendations.append("Try to reduce distractions and refocus on your screen.")
    if (dhri or {}).get("level") == "high":
        recommendations.append("Your combined indicators suggest elevated risk; consider a longer break.")

    return recommendations


class AlertManager:
    def __init__(self):
        self._last_fired_at = {}         # alert_id -> seconds since session start, when it last fired
        self._recent = []                 # most recent fired alerts, newest first (kept short, for the dashboard)
        self.total_count = 0

    def update(self, t, fatigue, posture, attention, dhri, continuous_active_s):
        """t: seconds since the session started. Returns a snapshot dict."""
        conditions = _alert_conditions(fatigue, posture, attention, dhri, continuous_active_s)
        newly_fired = []
        for alert_id, is_true in conditions.items():
            if not is_true:
                continue
            cooldown = config.ALERT_COOLDOWN_S[alert_id]
            last_fired = self._last_fired_at.get(alert_id)
            if last_fired is None or t - last_fired >= cooldown:
                self._last_fired_at[alert_id] = t
                alert = Alert(alert_id, ALERT_MESSAGES[alert_id], round(t, 1))
                newly_fired.append(alert)
                self._recent.insert(0, alert)
                self.total_count += 1
        del self._recent[10:]             # keep only a short recent history

        recommendations = _recommendations(fatigue, posture, attention, dhri, continuous_active_s)
        return {
            "new_alerts": [{"id": a.id, "message": a.message, "at_s": a.at_s} for a in newly_fired],
            "recent_alerts": [{"id": a.id, "message": a.message, "at_s": a.at_s} for a in self._recent],
            "recommendations": recommendations,
            "total_count": self.total_count,
        }

    def idle_snapshot(self):
        return {"new_alerts": [], "recent_alerts": [], "recommendations": [], "total_count": 0}
