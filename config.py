"""
Central configuration for the whole project.

Every path, port and tunable setting lives here so that nothing is
"hard-coded" inside the computer-vision or backend code.

This includes the DHRI weights and risk thresholds, and the alert
cooldowns, added as those parts of the system were built. Values that
are only *our own design choices* are marked as such, so we never
confuse them with values that come from the base paper.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
# utils/config.py -> parent is utils/, parent.parent is the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"          # SQLite database + training data
MODELS_DIR = PROJECT_ROOT / "models"      # MediaPipe .task files + trained ML model
FRONTEND_DIR = PROJECT_ROOT / "frontend"  # dashboard HTML/JS/CSS

# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------
# PRIVACY: 127.0.0.1 means "only this computer". Nobody else on the network
# can open the dashboard or reach the camera stream.
HOST = "127.0.0.1"
PORT = 8000
APP_TITLE = "Adaptive Digital Cognitive & Ergonomic Intelligence Platform"
APP_VERSION = "0.15.6-run-script-race-fix"

# ---------------------------------------------------------------------------
# Database (SQLite for local development)
# ---------------------------------------------------------------------------
DATABASE_PATH = DATA_DIR / "wellbeing.db"

# ---------------------------------------------------------------------------
# Login / accounts (Phase 2). These are our own design choices.
# ---------------------------------------------------------------------------
LOGIN_COOKIE_NAME = "session_token"
LOGIN_TOKEN_HOURS = 12          # after this the user must log in again
USERNAME_MIN_LENGTH = 3
USERNAME_MAX_LENGTH = 30
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128

# ---------------------------------------------------------------------------
# Camera (used from Phase 3 onwards)
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0        # 0 = default webcam
FRAME_WIDTH = 640       # low resolution keeps processing real-time
FRAME_HEIGHT = 480
TARGET_FPS = 20         # our design choice. A blink lasts roughly 0.1-0.4 s, so we need a good
                        # frame rate to see it. 20 fps leaves 50 ms per frame; face detection took
                        # about 32 ms on the developer's test machine.
MIN_RELIABLE_FPS = 12   # below this the dashboard warns that fast blinks may be missed
FPS_WARNING_WARMUP_S = 3   # ... but not during the first seconds after Start, while the rate settles

CAMERA_OPEN_WAIT_S = 10          # how long "Start" waits for the camera to open
CAMERA_MAX_READ_FAILURES = 30    # this many failed reads in a row = camera lost
# PRIVACY: if the dashboard stops contacting the server for this long (tab closed,
# browser crashed) the camera switches itself off. It is 2 minutes, not seconds,
# because browsers slow down background tabs to about one request per minute,
# and monitoring must keep working while the user works in other windows.
CAMERA_IDLE_TIMEOUT_S = 120
PREVIEW_JPEG_QUALITY = 70        # only used for the live preview in the browser

# Lighting check (our own starting values, average brightness 0=black .. 255=white).
# They must be checked against real webcam tests before we claim anything about them.
LIGHT_TOO_DARK_BELOW = 50
LIGHT_TOO_BRIGHT_ABOVE = 210
LIGHT_SMOOTHING = 0.1            # 0..1; smaller = steadier reading

# ---------------------------------------------------------------------------
# Face detection (Phase 4). The three confidence values are MediaPipe's own defaults;
# the rest are our design choices and still need checking with real recordings.
# ---------------------------------------------------------------------------
MAX_FACES = 2                      # we only need to know "one face" or "more than one"
FACE_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MIN_PRESENCE_CONFIDENCE = 0.5
FACE_MIN_TRACKING_CONFIDENCE = 0.5
# A second face counts as "another person" only if it is at least this fraction of the main
# face's size, so a tiny face far in the background does not trigger a warning.
SECONDARY_FACE_MIN_AREA_RATIO = 0.25
# A face touching the image edge is "partly outside the view" (fraction of image size).
FACE_EDGE_MARGIN = 0.01
# A status must be seen this many frames IN A ROW before it is shown (about 0.3 s at 20 fps),
# so one missed detection does not make the message flicker.
STATUS_CHANGE_FRAMES = 6

# ---------------------------------------------------------------------------
# Eye analysis (Phase 5)
# The base paper says only that an eye is open or closed "by setting a threshold of the ratio".
# It gives no numbers, so EVERY value below is our own starting choice and must be checked
# with real recordings before we claim anything about it.
# ---------------------------------------------------------------------------
# "Closure" is 0.0 for the user's own normal open eyes and 1.0 for their own closed eyes
# (both measured during calibration). Thresholds are on that personal 0-1 scale.
EYE_CLOSE_ENTER = 0.5      # eyes count as closed once closure reaches this ...
EYE_CLOSE_EXIT = 0.3       # ... and as open again only when closure falls back below this
PERCLOS_CLOSURE = 0.8      # PERCLOS-style: time with the eyelid at least 80% closed
BLINK_MIN_S = 0.03         # shorter closures are treated as noise
BLINK_MAX_S = 0.5          # closures up to this length are blinks, longer ones are "long closures"
FEATURE_WINDOW_S = 20      # window for blink rate and PERCLOS. The paper also uses 20 s windows.
FEATURE_MIN_COVERAGE_S = 10   # blink rate / PERCLOS are only shown after this much valid data
MAX_SAMPLE_GAP_S = 1.0     # a longer pause between two measurements is treated as a break
# Below this average brightness (0-255) the image is essentially black, for example a hand over the
# lens. The face model can still return a distorted face there, so the eyes are NOT measured.
EYE_ANALYSIS_MIN_BRIGHTNESS = 25

# Signal quality (added after real tests showed false blinks when a hand covered and uncovered the lens).
# While something moves in front of the camera the picture changes fast and the face model can return a
# distorted face, so its eye values are not trusted. All values are our own starting choices.
EYE_UNSTABLE_WINDOW_S = 0.5        # look back this far when judging whether the picture is changing
EYE_BRIGHTNESS_CHANGE_RATIO = 1.5  # brightest / darkest value inside that window above this = "changing too fast"
EYE_FACE_JUMP_AREA_RATIO = 1.6     # the face box growing or shrinking by more than this between two frames = a jump
EYE_FACE_JUMP_SHIFT = 0.15         # ... or its centre moving by more than this fraction of the image
# The two values below come from ONE real recording (4 false blinks): they were counted 0.3 s and 0.5 s BEFORE the
# face was lost, and 0.5 s and 0.6 s AFTER it was found again. Both are set to 1.0 s to leave a margin. That is a
# small sample, so they must be re-checked with the cover test in scripts/eye_test.py.
EYE_SETTLE_S = 1.0                 # after ANY interruption, wait this long of steady picture before measuring again
BLINK_CONFIRM_S = 1.0              # a blink is only counted if the eyes are still measurable this long afterwards

# ---------------------------------------------------------------------------
# Personal calibration (Phase 5: eyes. Phase 7 adds posture)
# ---------------------------------------------------------------------------
CAL_READY_S = 3            # "sit normally" countdown, nothing is recorded
CAL_OPEN_S = 8             # eyes open, looking at the screen
CAL_CLOSE_S = 4            # eyes closed
CAL_MIN_VALID_FRACTION = 0.7   # share of frames in which the eyes must have been measured
CAL_MIN_SAMPLES = 20           # and at least this many measurements per step
CAL_OPEN_EAR_RANGE = (0.12, 0.60)   # a normal open-eye EAR must fall in this range
CAL_MIN_SEPARATION = 0.25      # closed EAR must be at least 25% below open EAR

# ---------------------------------------------------------------------------
# Posture (Phase 7)
# Posture is judged against the user's OWN calibrated sitting position, never against a universal "correct"
# posture. Each indicator has (WARNING, POOR): score 0.5 at the warning value, 1.0 at the poor value (same
# scheme as the fatigue rules); the posture score is the highest indicator score. Every value below is our own
# unvalidated starting choice: nothing in the base paper covers posture.
# ---------------------------------------------------------------------------
POSTURE_RULE_THRESHOLDS = {
    "head_tilt_deg": (8.0, 15.0),          # head tilted sideways, degrees away from your normal
    "head_down_deg": (12.0, 20.0),         # head dropped forward/down, degrees below your normal
    "distance_change": (0.15, 0.30),       # face bigger (leaning in) or smaller (leaning away) by this fraction
    "head_drop": (0.06, 0.12),             # head sunk lower in the picture (fraction of picture height)
    "head_shift": (0.10, 0.20),            # head moved sideways in the picture (fraction of picture width)
    "shoulder_tilt_deg": (5.0, 10.0),      # shoulder line tilted, degrees away from your normal (needs shoulders)
    "head_shoulder_gap": (0.15, 0.30),     # head sinking towards the shoulders, fraction (needs shoulders)
}
POSTURE_ESCALATE_HOLD_S = 5.0     # a worse posture must last this long before it is shown (no alert from one frame)
POSTURE_RECOVERY_HOLD_S = 3.0     # a better posture must last this long before it is shown
POSE_EVERY_N_FRAMES = 4           # the shoulder model is slower, and posture changes slowly: run it every 4th frame
POSE_RESULT_MAX_AGE_S = 1.5       # a shoulder result older than this is not used
SHOULDER_MIN_VISIBILITY = 0.6     # a shoulder point must be at least this confident (0-1) ...
SHOULDER_MIN_WIDTH = 0.15         # ... and the shoulders at least this wide (fraction of picture width)
POSTURE_CAL_MIN_SAMPLES = 20      # posture calibration needs at least this many steady measurements
POSTURE_CAL_MIN_SHOULDER_FRACTION = 0.6   # shoulders go into the baseline only if seen in this share of them

# ---------------------------------------------------------------------------
# Attention (Phase 8)
# Attention is an OBSERVABLE indicator only: is the face visible, and does the head point roughly at the screen?
# "Facing your screen" is the head direction recorded during calibration. It cannot tell where the eyes look or what
# the person is thinking. Nothing in the base paper covers this; all values are our own unvalidated starting choices.
# ---------------------------------------------------------------------------
ATTENTION_YAW_LIMIT_DEG = 25.0      # head turned sideways this far from facing the screen = looking away
ATTENTION_PITCH_LIMIT_DEG = 25.0    # head up or down this far from facing the screen = looking away (phone, ceiling)
ATTENTION_EXIT_RATIO = 0.8          # once "away", the head must come back inside 80% of the limit (stops flickering)
ATTENTION_AWAY_HOLD_S = 5.0         # looking away must last this long before DISTRACTED is shown (glances are normal)
ATTENTION_NO_FACE_HOLD_S = 3.0      # no face for this long before FACE NOT DETECTED is shown
ATTENTION_RECOVERY_HOLD_S = 2.0     # facing the screen again for this long before ATTENTIVE is shown
ATTENTION_WINDOW_S = 60             # the "attention over the last minute" figure looks at this many seconds
ATTENTION_MIN_OBSERVED_S = 10       # ... and is only shown after this much observation
ATTENTION_EXTENDED_ABSENCE_S = 60   # no face for this long counts as an extended absence
ATTENTION_UNKNOWN_LIMIT_S = 5.0     # if the picture stays unsteady this long, say attention cannot be judged

# ---------------------------------------------------------------------------
# Session / exposure monitoring (Phase 9)
# A "session" is one Start-to-Stop of the camera. "Active" time excludes breaks. A short absence (a person
# glancing away, or the settle time after a hand blocks the lens) is not a break; only a sustained absence
# is, because that is when someone has actually stepped away from the computer. Our own starting value.
# ---------------------------------------------------------------------------
SESSION_BREAK_MIN_S = 30.0          # an absence at least this long counts as a break, not just a glance away

# ---------------------------------------------------------------------------
# DHRI -- Digital Health Risk Index (Phase 10)
# The project brief asks for "a transparent and understandable scoring method" combining fatigue, posture,
# attention and exposure into ONE risk score, with documented weights and CONFIGURABLE thresholds. All the
# numbers below are our own unvalidated starting choices -- DHRI is a project-specific indicator, not a
# medically validated score. See docs/architecture.md for the full reasoning.
# ---------------------------------------------------------------------------
# How much each already-computed 0..1 "how bad right now" score counts towards DHRI. Must sum to 1.0 when
# all four are available; when some are unavailable (not calibrated yet, no data yet), DHRI re-normalises
# over whichever ARE available rather than guessing a value for the missing ones.
DHRI_WEIGHTS = {"fatigue": 0.40, "posture": 0.25, "attention": 0.20, "exposure": 0.15}

# Exposure input: how long someone has been continuously active without a qualifying break (see
# computer_vision/session.py). Scored 0.5 at the warning length, 1.0 at the high length, like the
# fatigue/posture rules. In minutes, since that is the natural unit for a break reminder.
DHRI_EXPOSURE_WARNING_MIN = 45.0
DHRI_EXPOSURE_HIGH_MIN = 90.0

# Risk levels on the reported 0-100 scale. Configurable, as the brief asks.
DHRI_LOW_MAX = 35.0        # below this: LOW RISK
DHRI_MODERATE_MAX = 65.0   # this and above: HIGH RISK; in between: MODERATE RISK
DHRI_ESCALATE_HOLD_S = 5.0   # a worse risk level must last this long before it is shown
DHRI_RECOVERY_HOLD_S = 5.0   # a better risk level must last this long before it is shown

# ---------------------------------------------------------------------------
# Alerts and recommendations (Phase 11)
# ALERTS are point-in-time notifications, each with its own COOLDOWN: an alert fires the moment its
# condition becomes true, then will not fire again for that same condition until its cooldown has passed --
# even if the condition never clears in between (this is a repeating reminder, not a one-shot-per-episode
# notification). Alert conditions are read from the ALREADY-SMOOTHED fatigue/posture/attention/DHRI states,
# so alerts never fire from single-frame flicker.
# RECOMMENDATIONS are different: they always reflect the CURRENT condition (no cooldown), and disappear the
# moment the condition clears. See ml/alerts.py for exactly how each is decided; all wording, conditions and
# cooldowns below are our own choices -- the brief gives example alert/recommendation TEXT but not the exact
# trigger conditions or cooldown lengths.
# ---------------------------------------------------------------------------
ALERT_COOLDOWN_S = {
    "take_a_break": 600.0,    # overall risk (DHRI) reaching HIGH -- "Take a short break."
    "eye_closure": 300.0,     # fatigue reaching FATIGUED specifically via a long eye closure
    "posture": 300.0,         # posture reaching POOR
    "exposure": 900.0,        # continuous active time reaching the "high" exposure length (DHRI_EXPOSURE_HIGH_MIN)
    "attention": 300.0,       # attention reaching DISTRACTED
}
# The exposure RECOMMENDATION ("consider a screen break") appears earlier, at the "warning" exposure length
# (DHRI_EXPOSURE_WARNING_MIN); the exposure ALERT above ("...for a long period") only fires at the "high" length.

# ---------------------------------------------------------------------------
# MediaPipe model files (downloaded once by scripts/download_models.py)
# ---------------------------------------------------------------------------
# Face Landmarker -> eye landmarks (EAR), head pose (attention, head tilt)
# Pose Landmarker -> shoulder positions (posture)
FACE_LANDMARKER_FILE = MODELS_DIR / "face_landmarker.task"
POSE_LANDMARKER_FILE = MODELS_DIR / "pose_landmarker_lite.task"

MODEL_DOWNLOADS = {
    FACE_LANDMARKER_FILE: (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    POSE_LANDMARKER_FILE: (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}

# ---------------------------------------------------------------------------
# Fatigue assessment (Phase 6)
# There are two modes:
#   DEMO / RULE-BASED MODE   simple, readable rules (below). Used when no trained model exists.
#   TRAINED ML MODE          an SVM trained on data collected with scripts/collect_fatigue_data.py.
# NO real dataset exists yet, so every number below is our own unvalidated starting value.
# ---------------------------------------------------------------------------
# Each indicator has (WARNING level, FATIGUED level). Its 0-1 score is: 0.5 exactly at the warning
# level and 1.0 at the fatigued level (in between it is a straight line). The overall fatigue score
# is the highest score of any indicator.
FATIGUE_RULE_THRESHOLDS = {
    "perclos": (0.10, 0.20),            # share of the last 20 s with the eyelids at least 80% closed
    "closure_s": (0.5, 1.0),            # longest single eye closure (seconds)
    "blink_rate": (30.0, 45.0),         # blinks per minute (frequent blinking is a fatigue sign in the paper)
    "blink_duration_s": (0.30, 0.45),   # average blink duration (slow blinks)
}
FATIGUE_ESCALATE_HOLD_S = 3.0    # a worse state must last this long before it is shown (not for a long closure)
FATIGUE_RECOVERY_HOLD_S = 10.0   # a better state must last this long before it is shown

# Trained-model mode: the model outputs a probability of "fatigued" (0-1).
ML_WARNING_PROBABILITY = 0.4
ML_FATIGUED_PROBABILITY = 0.7

# Data collection and training
DATASET_FILE = DATA_DIR / "fatigue_features.csv"                 # real recordings (numbers only)
DEMO_DATASET_FILE = DATA_DIR / "DEMO_fatigue_features.csv"       # SYNTHETIC, only for testing the code
DATA_SAMPLE_INTERVAL_S = 5       # one row of features every 5 s while collecting
ML_MIN_ROWS_PER_CLASS = 30       # refuse to train with less than this per class ...
ML_MIN_SESSIONS_PER_CLASS = 2    # ... or from fewer separate recording sessions
ML_MAX_CV_SPLITS = 5

# Our own trained fatigue classifier (created by ml/train.py, not downloaded)
FATIGUE_MODEL_FILE = MODELS_DIR / "fatigue_svm.joblib"
# A model trained on the SYNTHETIC demo data is saved here and is NEVER loaded by the application.
DEMO_FATIGUE_MODEL_FILE = MODELS_DIR / "DEMO_fatigue_svm.joblib"
