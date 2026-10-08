"""
The numbers the fatigue classifier looks at, and how they relate to the base paper.

BASE PAPER (Sun et al. 2020, section 3.3) feeds these eye features to its SVM:
    DEC  duration of eye closure       FB  frequency of blinks       PCT  eyelid closure over time
    LI   light intensity (measured with a separate sensor)
Ours are the same ideas, taken from a 20-second window (the window length the paper also uses):

    feature name              paper idea   meaning
    mean_blink_duration_s     DEC          average length of a blink (empty if there was no blink)
    blinks_per_min            FB           blinks per minute
    perclos                   PCT          share of the window with the eyelids at least 80% closed
    longest_closure_s         DEC          longest single closure in the window

LI (light intensity) is deliberately NOT a classifier input yet. With a small dataset the model could
learn "dark room = tired" from when the recordings happened to be made. The paper uses LI to adapt to
day/night because its data covered both; ours does not yet.
"""

import math

FEATURE_NAMES = ["mean_blink_duration_s", "blinks_per_min", "perclos", "longest_closure_s"]

# eye-feature key (from computer_vision/eye_tracker.py) for each classifier feature
_SOURCE_KEYS = {
    "mean_blink_duration_s": "mean_blink_duration_s",
    "blinks_per_min": "blinks_per_min",
    "perclos": "perclos",
    "longest_closure_s": "longest_closure_s",
}


def window_features_ready(eye):
    """True when the eye tracker has enough valid data (about 10 s) for the window features."""
    return bool(
        eye
        and eye.get("calibrated")
        and not eye.get("paused")
        and eye.get("blinks_per_min") is not None
        and eye.get("perclos") is not None
    )


def feature_dict(eye):
    """The classifier features as {name: float or None}. None = not available (e.g. no blink yet)."""
    return {name: eye.get(key) for name, key in _SOURCE_KEYS.items()}


def feature_vector(values):
    """
    Turn {name: value or None} into a list in FEATURE_NAMES order.
    A missing value becomes NaN; the trained model fills such gaps itself.
    """
    return [float("nan") if values.get(name) is None else float(values[name]) for name in FEATURE_NAMES]


def is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))
