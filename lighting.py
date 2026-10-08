"""
Lighting check.

WHY: the base paper (Sun et al. 2020) treats lighting as a key problem and feeds a
light-intensity value (LI) into its fatigue model, measured with a separate photosensitive
sensor. A normal webcam has no such sensor, so we ESTIMATE lighting from the average
brightness of the camera image instead. This is our own substitute, not the paper's method.

For now we only warn the user when the room is too dark or too bright. A later phase can
also use this number as a feature.
"""

import numpy as np

from utils import config

LIGHTING_OK = "ok"
LIGHTING_TOO_DARK = "too_dark"
LIGHTING_TOO_BRIGHT = "too_bright"

LIGHTING_MESSAGES = {
    LIGHTING_OK: "Lighting looks fine.",
    LIGHTING_TOO_DARK: "Lighting is low. Detection may be less reliable, so please add more light.",
    LIGHTING_TOO_BRIGHT: "The image is very bright (backlight or glare). Detection may be less reliable.",
}

# ITU-R BT.601 weights, applied to OpenCV's blue-green-red channel order
_BGR_TO_GRAY = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def mean_brightness(frame):
    """Average brightness of a frame from 0 (black) to 255 (white). Looks at every 8th pixel for speed."""
    sample = frame[::8, ::8].astype(np.float32)
    return float((sample @ _BGR_TO_GRAY).mean())


def smooth(previous, current, alpha=None):
    """Exponential moving average, so one bright flash does not flip the warning."""
    alpha = config.LIGHT_SMOOTHING if alpha is None else alpha
    if previous is None:
        return current
    return (1 - alpha) * previous + alpha * current


def classify_lighting(brightness):
    """Return (state, message) for a brightness value."""
    if brightness is None:
        return LIGHTING_OK, ""
    if brightness < config.LIGHT_TOO_DARK_BELOW:
        state = LIGHTING_TOO_DARK
    elif brightness > config.LIGHT_TOO_BRIGHT_ABOVE:
        state = LIGHTING_TOO_BRIGHT
    else:
        state = LIGHTING_OK
    return state, LIGHTING_MESSAGES[state]
