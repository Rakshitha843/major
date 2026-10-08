# Real-Time Fatigue and Posture Monitoring System Using Computer Vision
*(implemented as the Adaptive Digital Cognitive & Ergonomic Intelligence Platform)*

A webcam-based digital-wellbeing monitor for students, office workers and home users. It
watches eye behaviour, posture, attention and usage time, combines them into a Digital Health
Risk Index (DHRI), and gives alerts and recommendations. It is a monitoring aid, **not** a
medical device, and DHRI is a project-specific indicator, **not** a validated health score.

Based on the fatigue-detection ideas in: Sun et al., *Driver Fatigue Detection System Based on
Colored and Infrared Eye Features Fusion*, CMC 63(3), 2020, adapted from drivers to everyday
computer use. See `docs/architecture.md` for exactly what comes from the paper and what does not.

**Status: Phase 15 of 16 (login, camera, face detection, eye analysis, calibration, fatigue, posture, attention, session tracking, DHRI, alerts/recommendations, history/trend graphs, full history browsing, multi-user hardening, end-to-end testing, and final integration done), plus a colourful tile-based Home hub, a Dashboard, and a batch of submission-readiness extras (CSV export, weekly summary, desktop notifications, accessibility, one-click run scripts) -- see `docs/architecture.md` sections 20-20d. Only the consolidated final report (Phase 16) remains. All 540 automated tests pass, confirmed on real hardware with every optional dependency (fastapi, httpx, mediapipe) installed.**

## Screenshots

| | |
|---|---|
| ![Home hub](docs/screenshots/00-home.png) | ![Live monitoring](docs/screenshots/02-live-monitoring.png) |
| Home: one colourful tile per feature | Live monitoring (camera, eye measurements) |
| ![Fatigue warning](docs/screenshots/03-fatigue.png) | ![DHRI breakdown](docs/screenshots/04-dhri.png) |
| Fatigue assessment | Digital Health Risk Index breakdown, with breadcrumb back to Home |
| ![Dashboard](docs/screenshots/05-dashboard.png) | ![Mobile view](docs/screenshots/06-mobile.png) |
| Dashboard: today/week summary + history | Home hub, responsive down to mobile width |

Log in / register is unchanged from before -- see `docs/screenshots/01-login.png`.

(Rendered from the real `frontend/index.html` with sample data standing in for a live camera feed,
so the screenshots stay current without needing a webcam in front of a screenshot tool.)

## Setup

You need Python 3.10 or newer.

```bash
# 1. open a terminal in the project folder, then create a virtual environment
python -m venv .venv

# 2. activate it
#    Windows (PowerShell):   .venv\Scripts\Activate.ps1
#    Windows (cmd):          .venv\Scripts\activate.bat
#    macOS / Linux:          source .venv/bin/activate

# 3. install libraries
pip install -r requirements.txt

# 4. one-time download of the MediaPipe model files (needs internet)
python scripts/download_models.py

# 5. check that everything is ready
python scripts/check_environment.py
```

## Run

```bash
python -m uvicorn backend.main:app --reload
```
Then open http://127.0.0.1:8000 in your browser.

## Test

```bash
pip install httpx                      # only needed for the API tests
python scripts/verify_all.py           # compiles everything, checks your setup, runs all tests
```
This is the one command to run before treating the project as "done": it byte-compiles every
file, re-runs `scripts/check_environment.py`, then runs the full automated test suite, and prints
a plain OK/FAIL summary (see `scripts/verify_all.py`'s docstring for details). To run just the
test suite on its own: `python -m unittest discover -s tests -t . -v`. Either way, this only
exercises the code with a simulated camera and face -- see `docs/testing_checklist.md` for the
manual walkthrough that needs your real webcam.

## Accounts and multi-user support
Open the page, register an account, log out, log in again. Each user has a separate account,
their own calibration, their own camera session, and their own history -- nothing is shared
between users. The profile panel shows the username, login status and calibration status. Use a
private/incognito window to log in as a second user at the same time and confirm the accounts
stay separate.

| Endpoint | Purpose |
|---|---|
| `POST /api/auth/register` | Create account and log in (JSON: `username`, `password`) |
| `POST /api/auth/login` | Log in |
| `POST /api/auth/logout` | Log out |
| `GET /api/auth/me` | Current user (401 if not logged in) |
| `GET /api/health`, `GET /api/system-check` | Server, database and library status |

Rules: username 3-30 characters (letters, numbers, `.` `-` `_`, not case sensitive);
password 8-128 characters. Login lasts 12 hours. All of these are set in `utils/config.py`.

## Camera preview (Phase 3)
After logging in, the **Live monitoring** card has a **Start camera** button. The camera stays off
until you press it. While it is on, a red **CAMERA ON** badge is shown, with a mirrored live preview,
frame rate and a lighting reading (it warns if the room is too dark or too bright).

The camera switches off when you press **Stop camera**, log out, stop the server, if the camera is
unplugged, or if the dashboard has not contacted the server for 2 minutes (for example the browser
tab was closed). Only one account can use the camera at a time. Nothing is saved to disk.

| Endpoint | Purpose |
|---|---|
| `POST /api/monitor/start` / `stop` | Camera on / off (login required) |
| `GET /api/monitor/status` | State, frame rate, lighting (also tells the server the dashboard is open) |
| `GET /api/monitor/frame` | Newest preview image (JPEG); only for the account using the camera |

**Camera troubleshooting:** run `python scripts/camera_test.py`. It opens a plain window with your
webcam and shows whether the camera itself works, without the web app. If it says "Camera not
detected", close Teams/Zoom/other browser tabs that may be using the camera. With more than one
camera, change `CAMERA_INDEX` in `utils/config.py`.

## Face detection (Phase 4)
While the camera is on, the preview shows a **green box** around your face and **yellow outlines**
on your eyes. The card also shows whether a face is detected and how long processing takes per frame
(measured on your own computer). Messages you may see:

| Situation | Message |
|---|---|
| No face in view | Please position your face inside the camera view. |
| Two people of similar size | More than one face detected. Only the largest face is monitored. |
| Face touching the image edge | Part of your face is outside the camera view. Please move back a little. |
| Face model missing | Face model not found. Run this once (needs internet): python scripts/download_models.py |

To check face detection without the web app, run `python scripts/face_test.py` (Q to quit).
`python scripts/check_environment.py` now also confirms that the face model file loads properly.
The face model is loaded BEFORE the camera is switched on, so a setup problem never turns the camera on.

## Eye analysis and calibration (Phase 5)
**Calibrate first.** In the **Personal calibration** card press **Start calibration** (about 15 seconds).
Sit normally, look at the screen and keep your face visible; keep looking and blink normally for 8 seconds;
then close your eyes when you hear the first beep and open them at the second beep. This measures YOUR
normal open-eye and closed-eye values. It is saved to your account, and you can repeat it any time
("Calibrate again", for example after changing your seat, lighting or glasses).
Without calibration the dashboard shows the raw EAR but no blink or closure analysis.

What the **Live monitoring** card then shows:

| Item | Meaning | Relation to the base paper |
|---|---|---|
| Eye aspect ratio (EAR) | Eye height divided by eye width; falls when the eye closes | Paper's eye aspect ratio (its eq. 2) |
| Eyes: open / CLOSED (closure %) | 0% = your normal open eyes, 100% = your closed eyes | Paper thresholds the ratio; we use your own calibrated range |
| Blinks counted | Closures shorter than 0.5 s | Paper's blink frequency (FB) input |
| Current eye-closure time | How long the eyes have been closed right now | Paper's duration of eye closure (DEC) |
| Blinks per minute (last 20 s) | Blink rate in a 20-second window | Paper's FB, same 20 s window |
| Eyes-closed share, PERCLOS-style (last 20 s) | Share of time the eyelid is at least 80% closed | Paper's PCT/PERCLOS-type feature (approximation, see docs) |
| Longest closure (last 20 s) | Longest single closure in the window | DEC-related |

Blink rate and the closed-share appear after about 10 seconds of valid data. The eye outline in the preview
turns red while the eyes count as closed. All thresholds are our own starting values (see `docs/architecture.md`)
and have NOT yet been validated against ground truth, so treat the numbers as indicators, not measurements.

To check EAR without the web app: `python scripts/eye_test.py` (C = calibrate, Q = quit). The terminal prints
a timestamped line for every blink, long closure, and every time the eyes stop or start being measured.
For troubleshooting, `python scripts/eye_test.py --log eye_log.csv` also saves the measured numbers (not images) to a CSV file.

| Endpoint | Purpose |
|---|---|
| `POST /api/calibration/start` | Switch the camera on (if needed) and begin the steps |
| `GET /api/calibration/status` | Progress; saves the result to your account when finished |
| `POST /api/calibration/cancel` | Stop a running calibration |

## Fatigue assessment (Phase 6)
The **Fatigue assessment** card shows **NORMAL**, **WARNING** or **FATIGUED**, a score, and *why* (which eye indicator
caused it). It needs calibration and about 10 seconds of eye data first, and says so while it waits.
**No real dataset exists yet, so no trained model exists and no accuracy figure exists.** Until you record data and train, the badge reads
**DEMO / RULE-BASED MODE** (readable rules with our own unvalidated thresholds). After you record data and train, it reads
**TRAINED ML MODE**. This is a monitoring indication, not a medical diagnosis.

| Command | What it does |
|---|---|
| `python scripts/collect_fatigue_data.py --username U --participant P01 --label normal` | Record labelled eye-feature numbers (never images) |
| `python -m ml.train` | Train the SVM on your recordings (refuses if there is too little data) |
| `python -m ml.make_demo_data` then `python -m ml.train --demo` | Test the pipeline on SYNTHETIC data (results are meaningless and the model is never used) |

Full guide, recording ethics and how to read the training report: **`docs/fatigue_ml.md`**.

| Endpoint | Purpose |
|---|---|
| `GET /api/fatigue/model` | Which mode is available, and facts about the trained model if there is one |

## Posture (Phase 7)
The **Posture** card shows **GOOD**, **WARNING** or **POOR**, the reasons, and how far you are from YOUR normal: head tilt, head nod
(down is +), distance to the screen, head lower/sideways, and shoulders when they are clearly visible. Posture is judged against the
sitting position you had **during calibration**, so calibrate sitting the way you want to sit. **If you calibrated before Phase 7, calibrate
again once** (the calibration now also records your sitting position); until then the card says posture is not calibrated.
A warning only appears after the posture has lasted about 5 seconds, so a single moment never triggers it.
The shoulder model is optional: without it (or when your shoulders are not clearly in view) posture is judged from the head alone, and the card says so.
**All posture limits are our own unvalidated starting values; nothing in the base paper covers posture.**

To check the numbers on your own camera: `python scripts/posture_test.py` (C = record baseline, Q = quit). Nod your head DOWN: "nod" must go positive.

## Attention (Phase 8)
The **Attention** card shows one of three states: **ATTENTIVE**, **DISTRACTED / LOOKING AWAY**, or **FACE NOT DETECTED**, with the reason, how far your
head is turned from facing your screen, the share of the last minute you were attentive, and how long you have been away.
It is an OBSERVABLE indicator only: it uses whether your face is visible and where your head points compared with where it pointed
**when you calibrated**. It cannot see where your eyes look and cannot know what you are thinking. Looking away must last about **5 seconds**
before DISTRACTED shows (a glance at the keyboard never triggers it), and a missing face about **3 seconds**. "Face not detected" means you are away
OR turned so far that the camera lost your face; the camera cannot tell those apart.
**Calibrate again once:** calibrations made before Phase 8 do not record your head direction, and the card says so until you do.
To see the real head-direction numbers on your camera, use `python scripts/posture_test.py` (turn your head about 35 degrees and watch the ATTENTION line).
All limits are our own unvalidated starting values; the base paper does not cover attention.

## Testing (Phase 14)
**A real bug was found and fixed in this phase**: the server's shutdown handling stopped the camera but never
saved the session that was running -- meaning any session active when the server stopped (including every
`--reload` auto-restart during development, which is how this project's own instructions have told you to
run it) was silently discarded. This is now fixed: shutting down the server (`Ctrl+C`, or `--reload`
restarting after a file change) saves whatever session was in progress, the same as clicking Stop would.

This phase also added `docs/testing_checklist.md`, a complete walkthrough of the project brief's own 20-item
testing list (registration through multiple users), organized so you can run it end to end on your real
system -- the one thing that can never be verified without a real camera, a real face, and a real browser.
Run it after any update, and definitely before a demo or submission.

Several new automated tests also simulate realistic situations no earlier phase specifically tried together:
recalibrating mid-session while fatigue/posture/attention/DHRI/alerts all have accumulated state, stopping
the camera while a calibration is still running, and how alert cooldowns behave across a stop/start cycle
(a design choice -- cooldowns reset with each new session -- that is now deliberate and documented rather
than an accident someone might discover later). See `docs/architecture.md` for the full story on the
shutdown bug and why it stayed hidden for three phases.

## Full history and multi-user hardening (Phase 13)
Below the trend charts, a **View full history** button reveals a paginated table of every finished session
(not just the last 20 used for charting): start time, duration, active time, breaks, and the same per-
category event counts, with a **Load more** button that fetches further back. Logging out always clears
this table, so a second account logging in on the same browser tab never sees a previous account's history
even for a moment.

This phase also added an end-to-end integration test that runs two accounts through a realistic shared-
computer sequence together (both register, one calibrates and uses the camera while the other is correctly
refused, alerts fire, the first logs out, the second then calibrates and uses the camera, and both check
their own history) -- going beyond the many individual "two users stay separate" tests already written
throughout Phases 2-12, each of which checks one subsystem in isolation. See `docs/architecture.md` for why
this kind of whole-pipeline test matters (it is the same technique that found the Phase 8 posture bug and
the Phase 10 DHRI smoothing bug).

## History and trend graphs (Phase 12)
The **History & trends** card shows six small bar charts, one bar per finished session (oldest on the left,
most recent on the right, up to your last 20): **Session duration**, **Fatigue events**, **Posture events**,
**Attention events**, **DHRI events**, and **Alerts** -- matching the six trend types your brief lists.
Each chart counts how many times that condition occurred during a session (for example fatigue's warning
count plus its fatigued count), not how severe any single moment was. Nothing appears until you have
finished at least one session (pressed Stop). The charts are drawn as plain SVG bars with no external
charting library, so the dashboard keeps working fully offline after the one-time setup -- consistent with
the project's design (see `docs/architecture.md` for why this was a deliberate choice, not an oversight).

**A bug found and fixed while building this phase:** `alert_count` was added to the database in Phase 11,
but the `/api/monitor/history` endpoint never actually included it in its response -- a Phase 11 test
asserted it would be there, but that assertion could only fail on a real FastAPI installation, which is not
available in the sandbox used to build this project. It is fixed now; see `docs/architecture.md` for the
full story and the test added to catch this class of gap in one place going forward.

## Alerts and recommendations (Phase 11)
The **Alerts & recommendations** card has two different things, matching the project brief's two sections:

**Recommendations** always reflect your CURRENT state (no cooldown -- they appear the moment a condition is
true and disappear the moment it clears): fatigue warning/fatigued suggests a short break; fatigue reaching
FATIGUED specifically from prolonged eye closure suggests resting your eyes instead; poor posture suggests
adjusting your position; continuous use past 45 minutes without a break suggests a screen break; being
distracted or reaching HIGH overall risk each get their own practical suggestion. None of these are medical
advice or a treatment.

**Alerts** are point-in-time notifications, each with its OWN cooldown (`utils/config.py: ALERT_COOLDOWN_S`),
so the same alert will not repeat too often even while its condition continues -- it works like a repeating
reminder (for example, a posture alert every 5 minutes while you keep slouching), not a one-time notice:
"Take a short break." (overall risk reaching HIGH), "Your eyes appear closed for longer than usual." (fatigue
from prolonged eye closure specifically), "Please correct your sitting posture." (posture reaching POOR),
"You have been using the computer continuously for a long period." (90+ minutes without a break), "Your
attention appears reduced." (attention reaching DISTRACTED). The card shows a running count and a short
recent history. How many total alerts fired is saved with each session (`alert_count`, in the session's
row in the database, alongside the other episode counts from Phases 6-10).

## Digital Health Risk Index -- DHRI (Phase 10)
The **Digital Health Risk Index (DHRI)** card sits near the top of the dashboard and shows a single number from
0 to 100 combining Fatigue (40%), Posture (25%), Attention (20%) and continuous Exposure without a break (15%)
into one overall risk level: **LOW RISK** (below 35), **MODERATE RISK** (35-65) or **HIGH RISK** (65 and above).
The card shows a bar for each input's contribution to the score, so you can see WHY the number is what it is.
If an input is not available yet (for example you have not calibrated), DHRI is calculated from whichever
inputs ARE ready, and the card says which ones are still missing rather than guessing a value for them.

**This is a project-specific digital wellbeing indicator, not a medically validated health score and not a
diagnosis.** The weights (40/25/20/15) and the two risk thresholds are our own starting choices -- see
`docs/architecture.md` for the full reasoning, including a bug found and fixed before this phase shipped
(a fast-rising score could take twice as long as intended to reach HIGH RISK; fixed by giving DHRI its own
escalation logic, separate from the one fatigue/posture/attention use). Both thresholds and all four weights
are configurable in `utils/config.py`.

## Session and exposure tracking (Phase 9)
The **Session** card shows the current session's total time, its "active" time (total time minus breaks), and how many
breaks were taken. A **break** is a sustained absence: no face visible (or the camera covered) for at least **30 seconds** --
a brief glance away is not a break, since that is already reflected in the Attention card. Nothing is written to the
database until you press **Stop**; the "Today so far" line and the recent-sessions list are then refreshed automatically.
Each saved session also records how many times fatigue, posture and attention entered a warning/worse state during it,
which feeds into the DHRI trend and the history page (see below). All numbers are wall-clock time your computer measured.
Alerts and recommendations are a separate feature, covered later in this document.

## Fixed since Phase 8: turning your head no longer looks like bad posture
A real test showed that turning your head to look away (correctly flagged by Attention as DISTRACTED) ALSO shrank the
measured face size enough to trigger Posture's "leaning away" warning, because a turned head looks narrower to the
camera. This is now corrected (a turn of 40 degrees now changes the measured distance by about 4%, not 20%). Details
and the regression test: `docs/architecture.md`, section 11.

## Blink counting: what changed after the cover test
A hand covering and uncovering the lens produced false blinks. Blinks are now counted about **1 second after they happen**, and only if the
eyes stayed measurable in that second. Eyes are also not measured while the picture is dark or changing fast, when the face box jumps, or during the
first second after any break. The terminal of `scripts/eye_test.py` says why the eyes were not measured and reports discarded blink-like dips.
The 1-second values come from one recording and should be re-checked with the cover test. Details: `docs/architecture.md`, section 9.

## Limitations and future work

**What this project does not claim.** DHRI, fatigue, posture and attention are all monitoring
indications, not medical devices or validated health measures. None of the thresholds in
`utils/config.py` (eye closure, posture angles, attention hold times, DHRI's 35/65 risk cut-offs,
alert cooldowns) have been tuned against real data -- they are reasonable starting values picked
by hand, documented as such everywhere they appear, and should be re-checked before anyone relies
on them. Posture and attention are both measured relative to the user's own calibration, not an
absolute "correct" posture, so calibrating while slouching makes slouching the new normal. Attention
can tell that you looked away, not where you looked or what you were thinking. The base paper (Sun
et al. 2020) covers only eye-based fatigue in drivers with dual colour/infrared cameras; posture,
attention, DHRI, accounts, alerts and history are this project's own design -- see
`docs/architecture.md` section 9's comparison table for exactly which pieces come from the paper.

**Known rough edges**, each already called out in detail where it is implemented (see
`docs/architecture.md`): the 30-second break threshold and 0.5-second long-closure cut-off are
single-author guesses, not measured; the blink/closure timing comes from one recording on one
person and should be re-checked with `scripts/eye_test.py`'s cover test on other people and
lighting; the "This week" dashboard figure reads only the most recent 20 sessions, so it
undercounts for anyone running more than 20 sessions in a week (the CSV export has the real total).

**Future work**, in roughly the order it would add the most value:

1. **Validate the fatigue ML model against real data.** The pipeline to train and cross-validate an
   SVM classifier already exists (`ml/train.py`, `ml/model.py`) and the UI already has a place to
   show its accuracy/precision/recall once trained (`/api/fatigue/model`) -- what's missing is
   recorded data. Running `scripts/collect_fatigue_data.py` across several real sessions (ideally
   more than one person) and then training would let the fatigue page report real, not rule-based,
   numbers.
2. **A small real user study.** Even 2-3 people running a normal work session with the monitor open
   and comparing what it flagged against what they actually felt would be far stronger evidence than
   any of the current unvalidated thresholds. See `docs/user_study_template.md` for a ready protocol.
3. **Re-tune thresholds from that data** rather than by hand -- particularly the DHRI weights
   (40/25/20/15) and risk cut-offs (35/65), which currently have no evidence behind them at all.
4. **Extend history beyond "last 20 sessions."** The trend charts, "This week" summary and recent-
   sessions list are all capped at the last 20 for simplicity; a real longitudinal view (weekly/
   monthly aggregates server-side) would make the "digital wellbeing over time" story stronger.
5. **Multi-camera or external-sensor light intensity**, matching the base paper's photosensitive
   sensor instead of approximating it from webcam brightness (Phase 5's `LI` substitution).

## Folder guide

| Folder | Purpose |
|---|---|
| `frontend/` | Dashboard web page |
| `backend/` | FastAPI server: API, login, sessions |
| `computer_vision/` | Webcam, landmarks, eye / posture / attention analysis |
| `ml/` | Fatigue rules, features, dataset format, SVM training and prediction |
| `database/` | SQLite storage code |
| `models/` | Downloaded MediaPipe files and our trained model |
| `data/` | SQLite database file and training data |
| `utils/` | Configuration and environment checks |
| `scripts/` | Command-line helpers (environment check, model download, camera test, face test, eye test, posture and attention test, fatigue data collector) |
| `tests/` | Automated tests |
| `docs/` | Architecture notes |

## Privacy
The server only listens on your own computer (127.0.0.1). Webcam frames are analysed in memory
and never saved or uploaded; only numbers such as EAR or posture angle are stored.
