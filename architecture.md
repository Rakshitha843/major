# Architecture (updated for Phase 8)

## 1. What runs where

Everything runs on ONE computer, in ONE Python process:

```
 Browser (dashboard)  <----HTTP---->  FastAPI backend  (127.0.0.1:8000)
                                          |
                                          |-- database/      SQLite file in data/
                                          |-- computer_vision/  background thread:
                                          |       webcam -> landmarks -> eye/posture/attention features
                                          |-- ml/            fatigue classifier (SVM) or rule-based fallback
                                          '-- DHRI + alerts + recommendations
```

* The webcam is opened by Python (OpenCV), **not** by the browser. Frames are
  analysed in memory and thrown away. Only numbers (EAR, posture angle, DHRI...)
  are stored.
* The dashboard is a normal web page that asks the backend for the latest numbers
  and shows the live camera preview (added in Phase 3).
* The server listens on `127.0.0.1` only, so nothing outside your computer can reach it.

**Consequence to be aware of:** one computer has one webcam, so only one logged-in user
is *actively monitored* at a time. "Multi-user" means every user has a separate account,
calibration and history on that computer. Serving many simultaneous live cameras from one
central server is out of scope for this version (listed under future work).

## 2. Workflow from the project brief -> where it is built

| Workflow step | Folder / phase |
|---|---|
| Login / user identification | `backend/`, `database/` (Phase 2) |
| Personal calibration | `computer_vision/`, `database/` (Phases 5-7) |
| Webcam, face + landmarks | `computer_vision/` (Phases 3-4) |
| Eye analysis, fatigue | `computer_vision/`, `ml/` (Phases 5-6) |
| Posture, attention | `computer_vision/` (Phases 7-8) |
| Exposure / session time | `backend/` (Phase 9) |
| DHRI, risk level | `backend/` (Phase 10) |
| Alerts, recommendations | `backend/` (Phase 11) |
| Dashboard, history, trends | `frontend/` (Phases 12-13) |

## 3. Technology choices and why

| Choice | Why |
|---|---|
| Python | Required by the brief; all CV/ML libraries are Python-first. |
| OpenCV | Webcam access and image handling. Free, standard. |
| MediaPipe (Tasks API) | One pip install gives eye landmarks, head pose and shoulder landmarks, fast enough for CPU. |
| scikit-learn SVM | The base paper uses an SVM on pre-extracted features; scikit-learn provides it. |
| FastAPI + uvicorn | Lightweight, runs locally, automatic API docs at `/docs`. |
| SQLite via Python's built-in `sqlite3` | No server to install. All SQL will be kept inside `database/` so a later move to PostgreSQL only changes that folder. |
| Plain HTML + JavaScript dashboard | No build tools for a student to install. |
| `unittest` | Built into Python, no extra install. |

### Why MediaPipe and not dlib (which the base paper uses)
The base paper detects landmarks with dlib. dlib usually needs a C++ compiler/CMake to install
on Windows and a separate ~100 MB landmark file. MediaPipe installs with pip and also provides
head-pose and shoulder landmarks we need for attention and posture. The Eye Aspect Ratio
formula is the same; only the landmark numbers used differ (handled in Phase 5).
This is **our design choice, not something from the paper.**

### Why model files must be downloaded
Newer MediaPipe releases no longer bundle landmark models in the pip package (the old
`mp.solutions` API is gone; this was confirmed on MediaPipe 0.10.33). The two `.task` files are
downloaded once by `scripts/download_models.py`; after that the project works offline.

## 4. Where each idea comes from (traceability)

| Feature | Source |
|---|---|
| Eye Aspect Ratio to decide open/closed eye | **Base paper** (eq. 2, threshold on the ratio) |
| Duration of Eye Closure (DEC), Frequency of Blinks (FB), PCT (PERCLOS-type) | **Base paper**, sec. 3.3 |
| Recompute FB and PCT over short windows (paper: every 20 s instead of 1 min) | **Base paper**, sec. 3.3. We start from a 20 s window. |
| Combine several eye features in an SVM classifier | **Base paper**, sec. 3.4 |
| Light intensity as an extra input | **Base paper** (measured with a photosensitive sensor). We will *approximate* it from average webcam-frame brightness: **our substitute, not the paper's method.** |
| Colored + infrared cameras | **Base paper only. Deliberately NOT used** (webcam only, per the synopsis). |
| Posture monitoring | **Synopsis / abstract**, not the paper |
| Attention monitoring | **Synopsis / abstract**, not the paper |
| Personal calibration | **Synopsis** (objective 2), not the paper |
| DHRI single risk score | **Synopsis** (objective 3), not the paper |
| Login and multi-user | **Synopsis** (objective 4) |
| Dashboard with live data and trends | **Synopsis** (objective 5) |

The base paper studies **drivers** with dual cameras. Its reported results do not validate this
system for computer users. Any accuracy figure in our documentation will come only from our own
tests, and will be labelled as such.

## 5. Differences between the source documents (please confirm)

* **SDG list:** the abstract lists SDG 3, 4, 8, 9 **and 12**. The synopsis slides and your build
  brief list only 3, 4, 8, 9. I am following the brief (3, 4, 8, 9). Tell me if SDG 12 should be included.
* **Slide 2 says "classical image processing":** the synopsis describes the approach as classical
  image processing plus a full-stack system. MediaPipe's landmark models are learned models, so
  strictly it is "classical feature engineering on top of pretrained landmark detection". If
  your reviewers expect only hand-crafted image processing, we should discuss wording before
  documentation (Phase 16).
* **Synopsis mentions "desktop processing, backend, dashboard":** implemented as one local app
  where the Python process does the desktop-side processing (see section 1).

## 6. Accounts and security (Phase 2)

Source: synopsis objective 4 (secure login, multiple independent users). The base paper has no
login or multi-user component, so nothing here is "from the paper".

* **Passwords** are never stored. We store a salted scrypt hash (Python standard library).
* **Login** gives the browser a random token in an HttpOnly, SameSite=Lax cookie. The database keeps
  only the SHA-256 hash of the token, and each token expires after 12 hours.
* **Wrong username and wrong password give the same message**, and take similar time.
* **Separation of users:** every data table has a `user_id`; every query filters by the logged-in user's id.
* **Database errors** are turned into a clear message rather than a crash.
* **Deliberately not included** (keeps the academic project simple): password reset, email
  verification, login-attempt lockout, https. The server only listens on your own computer.
  These would be needed before any real multi-machine deployment.

## 7. Camera and privacy (Phase 3)

How it runs: pressing Start creates one background thread (`computer_vision/monitor.py`). It opens
the webcam, then repeats: read a frame -> measure lighting -> keep only the newest frame as a small
JPEG in memory -> wait for the next frame time (15 fps, our choice). The browser asks for the newest
image about 10 times a second to draw the live preview. Phases 4-8 add their analysis at the marked
place inside this same loop.

Privacy rules and how each is enforced (all covered by tests):

| Rule | Where |
|---|---|
| Camera never starts on its own | Only `POST /api/monitor/start` starts it |
| No images saved | Frames live in memory only; a test checks the source has no file-writing code |
| Last image deleted when monitoring stops | `_reset()` clears it |
| Other accounts cannot see the picture or numbers | Owner check in `get_frame` / `get_status` |
| Other accounts cannot switch it off | `stop_if_owner` |
| Off at logout, server shutdown, camera loss | `routes_auth.logout`, `lifespan`, capture loop |
| Off if the dashboard disappears | Idle timeout, `CAMERA_IDLE_TIMEOUT_S` (120 s) |

Why the idle timeout is 2 minutes and not shorter: browsers slow down background tabs to roughly one
request per minute, and the whole point is that monitoring keeps working while you use other windows.
The trade-off is that after a browser crash the camera can stay on for up to 2 minutes. You can lower
the value in `utils/config.py`; going below about 60 seconds risks switching monitoring off while the
tab is in the background.

**Relation to the base paper:** the paper measures light intensity (LI) with a separate photosensitive
sensor and uses it as a model input, and it uses a dual colour + infrared camera. We use one ordinary
webcam and only ESTIMATE lighting from the average image brightness (`computer_vision/lighting.py`).
That is our own substitute; the paper's results for its sensor-based method do not carry over to it.
The dark/bright thresholds (50 and 210 out of 255) are starting values that still need to be checked
with real webcam recordings.

## 8. Face detection (Phase 4)

`computer_vision/face_landmarks.py` wraps MediaPipe's Face Landmarker in video mode (it tracks the face
from frame to frame). For each camera frame it returns up to 2 faces with 478 landmarks each; x and y
are fractions of the image width/height. Phases 5, 7 and 8 read the eye and head points from this.
The monitor runs it on every frame, draws the box and eye outlines (`drawing.py`, lines only, because
the mirrored preview would reverse text), and reports the face status.

Rules (our design choices, each in `utils/config.py`, none from the paper):

| Rule | Value |
|---|---|
| Person monitored | the LARGEST face in view |
| A second face counts as "another person" | only if its area is at least 25% of the main face |
| "Partly outside the view" | face box within 1% of an image edge |
| A status must last this many frames in a row before it is shown | 6 (about 0.4 s at 15 fps) |
| Faces looked for at most | 2 (so "3 people" also shows as "more than one face") |

**Relation to the base paper:** the paper finds facial landmarks with dlib and then works on the eye
region. We use MediaPipe for the landmarks (own choice, for easy installation and extra points) and keep
the same idea: locate the eye landmarks first, then measure them (Phase 5). The paper's cameras were
mounted in a car and recorded drivers, so nothing in it tells us how well landmark detection works
for a person at a desk.

**What is and is not verified.** Automated tests use synthetic landmarks and a fake detector, so they check
our logic (largest face, warnings, debouncing, drawing, camera-off-on-setup-error), not how well the
real model finds faces. The real model was not available when this phase was written. Whether it finds
YOUR face with your glasses, lighting and camera angle is checked by `scripts/face_test.py` and in the
dashboard. The "processing time" shown on the dashboard is measured on your computer; we do not claim
any speed figure of our own.

## 9. Eye analysis and calibration (Phase 5)

Files: `eye_features.py` (EAR), `eye_tracker.py` (blinks, closures, windows), `calibrator.py` (personal
calibration), `database/calibration.py` (saving), and the calibration card in the dashboard.

### How each base-paper idea is used

| Base paper (Sun et al. 2020) | Our implementation | Difference |
|---|---|---|
| Eye aspect ratio, eq. 2 with dlib points P37-P42 | Same formula with the equivalent six MediaPipe points per eye, averaged over both eyes, measured in pixels | Different landmark library and numbering. The mapping is ours; tests check the geometry, not the paper's code |
| Eye open/closed by "setting a threshold of the ratio" (no value given) | Personal scale: 0% = the user's calibrated open EAR, 100% = calibrated closed EAR. Closed at 50%, open again at 30% (hysteresis) | Our design. The paper gives no numbers |
| DEC, duration of eye closure | Blink duration, current closure time, longest closure, count of "long closures" (over 0.5 s) | Our 0.5 s limit is an assumption |
| FB, frequency of blinks, updated every 20 s | Blinks per minute over a sliding 20 s window | Same window length; shown after 10 s of valid data (our choice) |
| PCT, eyelid closure over time | PERCLOS-style: share of the 20 s window with closure of at least 80% | Approximation: derived from EAR, not from measuring the eyelid over the pupil |
| Light intensity (LI) as a model input | Not used yet. Brightness is only measured for the lighting warning (Phase 3) | Could become a feature in Phase 6 |
| Colour + infrared camera fusion | Not used (webcam only, per the synopsis) | Deliberate |
| SVM classifier on the features | Phase 6 | - |

### Our thresholds (all in `utils/config.py`, all unvalidated starting values)

| Setting | Value | Meaning |
|---|---|---|
| EYE_CLOSE_ENTER / EYE_CLOSE_EXIT | 0.5 / 0.3 | Closure at which eyes count as closed / open again |
| PERCLOS_CLOSURE | 0.8 | "80% closed" for the PERCLOS-style share |
| BLINK_MIN_S / BLINK_MAX_S | 0.03 s / 0.5 s | Shorter = noise; longer = long closure, not a blink |
| FEATURE_WINDOW_S / FEATURE_MIN_COVERAGE_S | 20 s / 10 s | Window and minimum data before showing blink rate and PERCLOS |
| TARGET_FPS | 20 | Raised from 15 in Phase 5: a blink lasts only a fraction of a second |

### Calibration (synopsis objective 2)

Steps: 3 s get ready, 8 s eyes open, 4 s eyes closed. The open baseline is the 60th percentile of the
EAR values while the eyes are open (blinks are short, so they do not pull it down). The closed baseline is
the average of the lowest quarter of the EAR values in the closed step. The result is REJECTED, and
nothing is saved, if: the eyes were measured in fewer than 70% of the frames (or under 20 frames), the
open value is outside 0.12-0.60, or the closed value is not at least 25% below the open value. Posture is
added to this same procedure in Phase 7, so users will re-run calibration once then.

### Signal quality (added after a real cover test)
A real recording showed 4 false blinks when a hand covered and uncovered the lens: two were counted 0.3 and 0.5 s BEFORE the face was lost, two
0.5 and 0.6 s AFTER it was found again (the face model returns a distorted face while a hand is close). Two protections, both with a 1.0 s margin:

| Protection | Where | Effect |
|---|---|---|
| Eyes are not measured while the picture is dark, changing fast (brightness moves by more than 50% within 0.5 s), the face box jumps, or for 1.0 s after any break | `computer_vision/eye_measurement.py` (one shared decision for the dashboard, `eye_test.py` and the data recorder) | Transients after a hand leaves are ignored |
| A blink is only counted once the eyes stay measurable 1.0 s more; otherwise it is discarded and counted as `discarded_events` | `computer_vision/eye_tracker.py` | Blink-like dips just before the picture disappears are dropped |

Trade-offs: counts appear about 1 s late, and a REAL blink in the second before the data is lost is dropped too. The 1.0 s values are from ONE recording
(4 events), so re-check with `python scripts/eye_test.py --log eye_log.csv`. The simulated scenes in `tests/test_signal_quality.py` reproduce the logged
timings and would produce false events without the protections; they cannot prove that no real hand movement gets through.

### While the eyes cannot be seen, and unreliable measurements
* **No face measured:** any closure in progress is thrown away, and a gap longer than 1 s in the data is treated
  as a break. The system never guesses what unseen eyes were doing.
* **Nearly black image** (for example a hand over the lens, average brightness under 25 of 255): the eyes are not
  measured even if the face model still returns a face, because that face can be distorted.
* **Open eyes first (added after a real test showed false blinks when the camera was covered):** after monitoring
  starts and after every break in the data, nothing is counted, and no time is added to the 20 s window, until the
  eyes have been seen clearly open once. A blink is "open, closed, open", so the unreliable first frames after the
  face reappears can no longer become blinks.
* During calibration, normal blink counting is paused.

### Known limits (please read before trusting the numbers)
* **No validation yet.** All logic is tested with synthetic data. There are no accuracy figures for real
  users, and the paper's results (drivers, dual camera) do not transfer to this setting. Phase 14 should
  compare the system's blink counts with a manual count on short recordings, with and without glasses.
* Looking down at a keyboard lowers EAR and can look like partial closure; head turning and glasses frames
  or glare also affect EAR. Calibrate in your normal working position.
* A low frame rate can miss fast blinks. The dashboard warns if the measured rate drops below 12 fps.
* Blink and closure timing is accurate to about one frame (50 ms at 20 fps).
* A hand held in front of the EYES only (face still visible) can look like closed eyes. It cannot be told apart
  from real closure with the current method.
* One person is monitored, and one calibration is stored per account.

## 10. Fatigue assessment (Phase 6)

Files: `ml/features.py`, `ml/rules.py`, `ml/fatigue.py` (assessor + smoothing), `ml/dataset.py`, `ml/train.py`, `ml/model.py`,
`ml/make_demo_data.py`, `scripts/collect_fatigue_data.py`, and the fatigue card in the dashboard. Details, thresholds, recording
ethics and the validation plan are in **docs/fatigue_ml.md**.

Data flow: eye features (Phase 5, 20 s window) -> assessor -> either the RULES or the trained SVM -> raw state and score ->
time-based smoothing -> shown state (normal / warning / fatigued, or "unavailable" with a reason) -> dashboard.

| Base paper | Here |
|---|---|
| DEC, FB, PCT features, 20 s windows | Same ideas, same window length (Phase 5 computes them) |
| SVM on those features, fatigue vs normal | `ml/train.py`: SVM (RBF) on four window features; trained only on data YOU record |
| Light intensity as an input | Not used by the classifier (shortcut risk with small data) |
| Results on its driver data | Not transferable. No result exists yet for computer users |

Our own design choices: the two-mode structure with a permanent mode label, the readable rule set and its thresholds, the
0.5/1.0 score scale, smoothing times (3 s worse, 10 s better), the probability thresholds (0.4 / 0.7) and the safeguards:
training refuses too little data; cross-validation keeps whole people or sessions together; every report compares with two
baselines; synthetic data is marked DEMO throughout and its model can never be loaded by the app.

What is verified: the logic with simulated eye behaviour and synthetic data (268 automated tests, including deliberate-break checks).
What is NOT: the accuracy of either mode on real people. That needs the recordings and the plan in docs/fatigue_ml.md, section 7.

## 11. Posture (Phase 7)

Files: `computer_vision/head_pose.py`, `pose_landmarks.py`, `posture.py`, the posture part of `calibrator.py`, `database/calibration.py`, the posture card.

**Source: the synopsis only.** The base paper is about eye-based fatigue and contains nothing on posture, so nothing here is "from the paper".

| Signal | How it is measured | Needs |
|---|---|---|
| Head tilt (sideways) | Angle of the line between the outer eye corners | face landmarks |
| Head nod (down = +) | 3D pose fit of six face points (OpenCV solvePnP) against a generic face model | face landmarks |
| Distance | Eye-to-eye width: bigger = leaning towards the screen | face landmarks |
| Head lower / sideways | Centre of the face box | face landmarks |
| Shoulder tilt, head sinking towards the shoulders | Two shoulder points from MediaPipe Pose (run on every 4th frame) | the optional pose model, shoulders clearly visible |

Everything is a DEVIATION from the baseline recorded during calibration (median of the steady measurements in the 8 s "eyes open" step). A generic face
model and an assumed lens make absolute angles inaccurate by several degrees; a constant error cancels out against the personal baseline.
Each indicator has a warning and a poor value (`POSTURE_RULE_THRESHOLDS`), scored like the fatigue rules (0.5 at warning, 1.0 at poor, highest wins), and a worse
posture must last 5 s before it is shown (`POSTURE_ESCALATE_HOLD_S`), so one bad frame never triggers a warning; a better one must last 3 s.

| Limit | Value (warning / poor) |
|---|---|
| Head tilt | 8 / 15 degrees |
| Head dropped down | 12 / 20 degrees |
| Face bigger or smaller (leaning in / away) | 15% / 30% |
| Head sunk lower in the picture | 6% / 12% of picture height |
| Head moved sideways | 10% / 20% of picture width |
| Shoulders uneven | 5 / 10 degrees |
| Head sinking towards the shoulders | 15% / 30% |

All are our own starting values, not validated.

**Calibration** now also stores the sitting position (in `calibrations.posture_baseline_json`). A recalibration without a usable posture baseline
CLEARS the old one (a stale sitting position is worse than none) and the result says so; the eye calibration is still saved. Users calibrated before Phase 7 see
"posture not calibrated" until they calibrate again.

**Honest limits:** "good" only means "close to how you sat when you calibrated" (calibrating while slouching makes slouching normal); a front-facing webcam cannot see
slouching directly, so the cues are indirect; the sign convention of head nod and the real accuracy of the angles were verified only on SIMULATED faces built
from the same model, so `scripts/posture_test.py` (nod down: value must go positive) is the real check; the pose model was not available when this phase was
written, so shoulder detection on real people is untested; only one person is measured.

## 12. Attention (Phase 8)

Files: `computer_vision/attention.py`, the head-direction part of `posture.py` and `calibrator.py`, the attention card.

**Source: the synopsis only.** The base paper does not cover attention (its comparison table lists other authors' driver systems that use head pose,
which we neither implemented nor drew from).

What it is: an OBSERVABLE indicator from two things the camera sees, whether a face is visible and where the head points. "Facing the screen" is the head
direction (yaw and pitch from the 3D pose fit used for posture) recorded during calibration, stored as `yaw_deg` inside the posture baseline.

| State | Meaning | Shown after |
|---|---|---|
| ATTENTIVE | face visible, head within 25 degrees sideways and 25 degrees up/down of facing the screen | returns after 2 s of facing the screen |
| DISTRACTED / LOOKING AWAY | face visible, head turned beyond a limit | 5 s (`ATTENTION_AWAY_HOLD_S`) |
| FACE NOT DETECTED | no face visible (away, OR turned so far the model lost the face). A black picture also counts | 3 s (`ATTENTION_NO_FACE_HOLD_S`) |

A hysteresis (once away, the head must come back inside 80% of the limit) stops flicker at the limit. Moments when the picture is unsteady (a hand or a light
change, the first second after a break) are skipped rather than guessed; if that lasts more than 5 s the card says attention cannot be judged. Extra outputs for later
phases: the share of the last 60 s not attentive (`score`, shown once 10 s have been observed), the current time away, an extended-absence flag (60 s without a face)
and episode counts.

**Honest limits:** it cannot see where the EYES look (a second monitor, or reading beside the screen with a small head turn, is not detected) and cannot know what
anyone is thinking; the head direction comes from a generic face model, so absolute angles are approximate and only the deviation from your own calibrated direction
is used (verified on SIMULATED faces only; `scripts/posture_test.py` shows the real numbers); people who habitually look down at the keyboard will need a baseline that
reflects that or a larger pitch limit; the limits and hold times are unvalidated starting values; one person is measured. Calibrations saved before Phase 8 lack
the head direction and are reported as "please calibrate again".

## 13. Bug fix: turning the head was misread as leaning away (found after the Phase 8 real-camera test)

A scripted end-to-end run (simulated camera, real monitoring engine) showed that turning the head 40 degrees away from
the screen -- correctly flagged by Attention as DISTRACTED -- ALSO pushed Posture into WARNING. Cause: `measure_posture`
computes "distance to the screen" from the eye-to-eye pixel width, but a turned head is narrower in the picture purely
from the camera's viewing angle (foreshortening), regardless of actual distance. At 40 degrees the measured width
dropped to about 80% of normal -- comfortably past the 15% "leaning away" warning threshold, even though the person had
not moved at all.

Fix: `computer_vision/posture.py`'s `measure_posture` divides the measured eye distance by `cos(yaw)` (the yaw already
computed for the 3D head-pose fit), capped at a 60-degree turn so the correction cannot blow up as cos approaches zero.
After the fix, the same 40-degree turn changes the measured distance by about 4-5%, well inside the warning threshold.
A regression test (`tests/test_phase7.py::MeasurePostureTests::test_turning_the_head_does_not_look_like_leaning_away`)
checks this holds at 20-50 degrees, and was confirmed to fail without the fix (removing the correction caught by the
test) before being accepted.

This was found by scripting a full session through the real monitor with a simulated camera and simulated head
movements (no webcam or web server needed) -- see the "how to reproduce the demo" note in the Phase 9 section below.
It highlights that indicators built independently (Attention from yaw, Posture from face width) can interact in ways
that are only visible once they run together on the same input, which is why an end-to-end scripted run was added to
this project's own testing process going forward.

## 14. Session and exposure tracking (Phase 9)

Files: `computer_vision/session.py` (`SessionTracker`), `database/sessions.py`, the migration in `database/connection.py`,
the `/api/monitor/history` route and the additions to `/api/monitor/stop` in `backend/routes_monitor.py`, the session card.

**Source: the synopsis only** ("session start time, duration, active monitoring duration, break periods, number of
alerts, fatigue events, posture warnings" -- section 11 of the original project brief). Nothing in the base paper covers
session or exposure tracking.

A **session** is one Start-to-Stop of the camera, tracked in memory by `SessionTracker` while it runs:
* **duration_s** -- wall-clock time from Start to Stop.
* **A break** is a sustained absence (no face, or the camera covered) of at least `SESSION_BREAK_MIN_S` (30 s, our own
  starting value). A brief glance away or a settle period after a hand blocks the lens is NOT a break -- that is already
  reflected in Attention's DISTRACTED/FACE NOT DETECTED states, which use much shorter hold times (5 s / 3 s). A break
  is only counted once it ends (or the session ends), so its final length is known; the LIVE summary still subtracts
  time an absence has run so far, even before it is confirmed as a "real" break, so the live "active time" figure
  does not run ahead of reality while someone is away.
* **active_duration_s** = duration_s minus confirmed break time. This is presence, not attentiveness: someone can be
  present but DISTRACTED, which still counts as active time.
* **Event counts** (fatigue/posture/attention entering a worse state) are read from each assessor's own `events`
  counter (already built in Phases 6-8) at the moment the session ends -- Phase 9 added no new counting logic for these,
  only a place to persist them.

**Persistence and safety, all covered by tests:**
* The session row is written EXACTLY ONCE, from `POST /api/monitor/stop`, using a one-shot value
  (`monitor.take_session_result()`) that -- unlike the calibration result -- must survive the monitor's own internal
  state reset, because a session is only known to be finished at the moment that reset happens. This needed its own
  code path (`_finish_session`, called from every exit point of the capture loop, storing the result keyed to the
  session's owner rather than the monitor's current owner) rather than reusing the calibration result's pattern outright.
* Calling Stop twice, or a second account calling Stop without owning the camera, saves nothing extra (checked by
  `take_session_result`'s owner check and by a dedicated test).
* A session that ends before a single frame was processed (duration_s exactly 0) is not saved -- `should_save_session`
  in `database/sessions.py` is a small, independently testable function so this rule is checked without needing FastAPI
  installed (a real gap found during mutation testing: the same rule inline in the route was invisible to a mutation
  check in an environment without FastAPI).
* An existing database (from an earlier phase) gets the new columns added automatically and safely by
  `database/connection.migrate_schema()` (`ALTER TABLE ... ADD COLUMN`, guarded by checking `PRAGMA table_info` first,
  so it can run on every start-up without erroring on a column that is already there). Existing rows are preserved.

**Reproducing the scripted demo used to test this (and to find the posture bug above), without a webcam or web server:**
a `FakeCamera`/`FakeDetector`/`FakePoseDetector` (from `tests/helpers.py`) can drive the real `CameraMonitor` end to
end, including calibration, in a plain Python script -- see the test classes in `tests/test_phase9.py::MonitorSessionTests`
for the pattern. This is how the head-turn/posture interaction bug was actually found: not by a unit test of either
module in isolation, but by scripting a whole session through the real engine.

**Honest limits:** the 30-second break threshold is our own unvalidated starting guess; a break is not the same as a
"good" or "healthy" break, and this module makes no judgement about that (that is the recommendations module's job,
section 16 below). "Active time" counts presence, not engagement -- a distracted but present person is fully "active"
by this measure. As of Phase 11, `alert_count` reflects the real number of alerts fired during the session.

## 15. DHRI -- Digital Health Risk Index (Phase 10)

Files: `ml/dhri.py` (the whole thing), the wiring in `computer_vision/monitor.py`, the DHRI columns and
migration in `database/connection.py`/`schema.sql`, `database/sessions.py`, the DHRI card.

**Source: the synopsis's brief (section 12).** It asks for "a single overall risk score" combining fatigue,
eye closure, posture, attention and exposure/session duration, with "a transparent and understandable
scoring method", documented inputs/normalisation/weights/thresholds, and configurable risk levels
(LOW/MODERATE/HIGH). Nothing in the base paper covers a combined score across modalities.

### How the four inputs are combined

Each of fatigue, posture and attention already produces its own 0..1 "how bad right now" score (Phases 6-8);
DHRI adds a fourth input, **exposure**, scored the same way (0.5 at a configurable "warning" length, 1.0 at a
"high" length) from how long the person has been continuously active without a qualifying break -- itself a
new field added to `computer_vision/session.py`'s `SessionTracker` in this phase (`continuous_active_s`,
reset only when a real break is confirmed, not by every brief glance away).

DHRI = a **weighted average of whichever inputs are currently available**, re-normalised so the available
weights always sum to 1: `score = sum(weight_i * value_i for available i) / sum(weight_i for available i)`.
An input counts as available only when its OWN card is not "unavailable" (not calibrated, calibrating, or no
data yet); a missing input is never treated as 0 (that would understate risk) or skipped silently (the card
names which inputs are missing). If NONE are available, DHRI itself is "unavailable".

**Weights** (`DHRI_WEIGHTS`, must sum to 1.0, checked by a test): fatigue 0.40, posture 0.25, attention 0.20,
exposure 0.15 -- our own choice. Fatigue gets the largest weight because it is the phase built most directly
on the base paper's own concept (PERCLOS/eye-closure); giving it the most weight means the paper's original
idea is already the single biggest influence on DHRI, through the fatigue component, rather than counting a
separate "eye closure" input a second time (the brief lists "eye closure behaviour" and "fatigue" as
separate inputs; we treat eye closure as already fully absorbed into the fatigue score to avoid double-
counting the same underlying signal).

**Risk levels** (`DHRI_LOW_MAX` = 35, `DHRI_MODERATE_MAX` = 65, both configurable as the brief asks):
LOW below 35, MODERATE 35-65, HIGH at 65 and above, on the 0-100 scale.

### Bug found and fixed before shipping: the "stairstep" delay

A scripted end-to-end run (simulated camera driving the real monitor, no webcam needed -- the same technique
used to find the Phase 8/9 posture bug) showed that when fatigue's raw score ramped smoothly from 0 up to 1.0
over about a second (as `current_closure_s` accumulated past the fatigue thresholds) while posture was
already fully bad, DHRI's score correctly reached 65.1 (HIGH) within about a second -- but the SHOWN risk
level took roughly **10-11 seconds** to reach HIGH, not the intended 5.

Cause: DHRI's first implementation reused the shared `utils/smoothing.StateSmoother` (the same one
fatigue/posture/attention use). That smoother's rule -- "only what held for the WHOLE waiting window counts"
-- is deliberately conservative for a single indicator that can genuinely DIP mid-escalation (its own test,
`test_only_what_held_the_whole_time_counts`, locks this in for exactly that case). But DHRI's combined score
does not dip here; it RAMPS monotonically through MODERATE on its way to a settled HIGH, because it is built
from an underlying quantity (eye-closure duration) that accumulates continuously. Applying the same
conservative rule meant the brief MODERATE reading at the very start of the ramp became the "ceiling" for the
whole 5-second window, so DHRI could only reach MODERATE after 5 seconds -- and then had to run a SECOND
full 5-second window (now escalating from MODERATE to HIGH) before finally reaching HIGH. Each level crossed
during a fast ramp added another full hold period.

Fix: DHRI does **not** reuse `utils/smoothing.py`. It has its own `_SustainedLevelSmoother` in `ml/dhri.py`
that tracks how long the *current* raw level has held, restarting the clock whenever the raw level changes at
all (up, down, or to a different level). A value that ramps through several levels and then settles reaches
its settled level about one hold period after it stops changing, however it got there; a genuine single-frame
flicker still cannot cause a promotion, since the raw level changing again resets the clock before the hold
time is reached. The shared `StateSmoother` used by fatigue/posture/attention was **not modified** -- its
existing tests correctly want the conservative behaviour for a single dipping indicator, and changing it
would have silently altered three already-shipped, tested behaviours to fix a problem specific to combining
multiple already-smoothed scores into one. `tests/test_phase10.py` includes a regression test built from
this exact scenario.

### What is saved, and what is not

Each session's `monitoring_sessions` row gets two new columns (added by the Phase 10 migration, safe for an
existing database from any earlier phase): `dhri_moderate_count` and `dhri_high_count`, mirroring the
episode-count pattern already used for fatigue/posture/attention. DHRI itself is not written as a per-second
time series (the `measurements` table's `dhri`/`risk_level` columns from the original Phase 1 schema are not
used yet) -- only the session-level counts, matching this phase's brief scope ("DHRI CALCULATION -> RISK
LEVEL"); a detailed history-with-graphs is Phase 12/13's job.

### Honest limits

DHRI is a project-specific summary number, not a validated health score, and never claims to be one (the
card and the API both carry a fixed disclaimer). The 40/25/20/15 weighting and the 35/65 thresholds are
starting guesses with no evidence behind them; they were not tuned against any data, because none exists.
Re-normalising over available inputs means DHRI computed from only fatigue (say, if posture/attention/
exposure are all unavailable) is not comparable to DHRI computed from all four -- the card always lists which
inputs went into the number so this is visible, but it also means the SAME underlying fatigue level can
produce a different DHRI score depending on what else happens to be available at that moment.

## 16. Alerts and recommendations (Phase 11)

Files: `ml/alerts.py` (the whole thing), the wiring in `computer_vision/monitor.py`, `database/sessions.py`
(now actually persists `alert_count`, previously hardcoded to 0 since Phase 9), the alerts card.

**Source: the brief's sections 13 (alerts) and 14 (recommendations).** These are deliberately kept as two
separate mechanisms with different rules, matching how the brief describes them:

| | Alerts (section 13) | Recommendations (section 14) |
|---|---|---|
| Timing | Point-in-time, fires once per cooldown | Always reflects the current state |
| Repeats while the condition persists? | Yes, once per cooldown (a repeating reminder) | Continuously shown, not a "firing" concept |
| Disappears when the condition clears? | No -- it already fired; the cooldown is unaffected | Yes, immediately |
| Purpose | A notification you might have missed | A live summary of practical next steps |

**Where the conditions come from:** every alert/recommendation condition reads the ALREADY-SMOOTHED
`state`/`level` fields fatigue (Phase 6), posture (Phase 7), attention (Phase 8) and DHRI (Phase 10) already
compute, plus `continuous_active_s` from the session tracker (Phase 9/10). `ml/alerts.py` adds no new
flicker-avoidance logic of its own -- by design, it only ever sees a condition once that condition's own
indicator has already decided it is real, so there was no risk of reproducing the Phase 10 "stairstep" bug
here (there is nothing left to smooth: the inputs are the shown, not raw, values).

**Alert conditions** (`_alert_conditions`, all our own choices; the brief gives example wording but not
the exact trigger or cooldown): `take_a_break` from DHRI reaching HIGH; `eye_closure` from fatigue reaching
FATIGUED **and** its listed reasons mentioning "eye closure" specifically (so a fatigue alert caused by,
say, a fast blink rate does not use the "eyes appear closed" wording); `posture` from posture reaching POOR
(not WARNING); `exposure` from continuous active time reaching `DHRI_EXPOSURE_HIGH_MIN` (90 minutes, the
same length DHRI's own exposure component uses for its "1.0" score, reused rather than inventing a third
threshold for the same underlying idea); `attention` from attention reaching DISTRACTED.

**Cooldown mechanism** (`utils/config.py: ALERT_COOLDOWN_S`, one value per alert id): purely time-based from
when an alert last fired, regardless of whether the underlying condition ever cleared in between. A test
(`test_the_condition_clearing_and_returning_does_not_reset_the_cooldown`) locks this in explicitly, since it
is the one place the "repeating reminder" reading of the brief's wording ("remain inactive for a configurable
period before the SAME alert can appear again") could reasonably have been implemented the other way
(reset-on-clear); we chose the reading that also naturally suppresses alert spam from a condition flickering
near a boundary.

**Recommendations** (`_recommendations`): four follow the brief's own worked examples almost verbatim
("Consider taking a short break.", "Adjust your sitting position and keep your back and neck aligned.",
"Consider taking a screen break.", "Rest your eyes for a short period."); two are our own additions in the
same spirit, covering attention and overall DHRI risk the way the brief's examples cover fatigue/posture/
exposure. The closure-specific and general fatigue recommendations are mutually exclusive (closure-caused
FATIGUED gets only the eye-rest tip, not both), to avoid two overlapping "take a break"-shaped messages for
the same underlying condition. The exposure recommendation appears earlier than the exposure alert
(`DHRI_EXPOSURE_WARNING_MIN`, 45 minutes, vs the alert's 90 minutes) -- a two-stage response to the same
underlying signal, gentle suggestion first, firmer alert if it continues.

**What is saved:** each session's `alert_count` (how many alerts fired in total during it) is now taken from
the real count instead of the placeholder 0 the column has held since Phase 9. No per-alert-type breakdown is
stored yet (unlike fatigue/posture/attention/DHRI, which each get their own event-count columns) -- that is
left for Phase 13's history work if it turns out to be useful; this phase only needed the single total the
brief's data model (section 18, "ALERTS" table concept) implies.

**Honest limits:** the alert wording and cooldown lengths are our own choices with no evidence behind them;
"take a short break" and "eyes appear closed" are plain-language notices, never a diagnosis or instruction to
do anything beyond a normal break; recommendations can show more than one at once (for example fatigued AND
poor posture together) and the card lists all of them without ranking which matters more.

## 17. History and trend graphs (Phase 12)

Files: the six-field projection in `backend/routes_monitor.py`'s `/api/monitor/history` (limit raised from 10
to 20 sessions), and the chart code entirely in `frontend/index.html` (`computeTrendPoints`,
`buildBarChartSVG`, `renderOneTrendChart`, `renderTrendCharts`).

**Source: the brief's section 15 ("HISTORY / TRENDS")**, which lists exactly six trend types: fatigue,
posture, attention, DHRI, session duration, and number of alerts. All six are built, one bar chart each.

**Why plain SVG, not a charting library:** the brief's own section 23 says "do not add unnecessary
technologies merely to make the project appear advanced." A charting library would either need a CDN
(breaking the project's fully-offline-after-setup design once the models are downloaded) or a build step
(which the project deliberately has none of, per its "no build tools for a student to install" design from
Phase 1). Six small bar charts are simple enough to draw directly: `buildBarChartSVG` is a pure function that
takes a list of `{label, value}` points and returns an SVG string with one `<rect>` per point, scaled to the
tallest value in the set; `computeTrendPoints` reads the already-fetched history rows (which come back
newest-first from the database, matching how `ORDER BY start_time DESC` naturally reads) and reverses them
so the chart draws oldest-to-newest, left-to-right, the way a trend is normally read.

**What each chart counts** (all from columns already saved by Phases 6-11, nothing new computed): fatigue =
`fatigue_warning_count + fatigue_event_count`; posture = `posture_warning_count + posture_poor_count`;
attention = `attention_distracted_count + attention_no_face_count`; DHRI = `dhri_moderate_count +
dhri_high_count`; session duration = `duration_s` in minutes; alerts = `alert_count`. These are EVENT COUNTS
per session (how many times a condition was entered), not a continuous score over time -- a true within-
session time series would need the `measurements` table from the original Phase 1 schema, which no phase has
populated yet (deliberately out of this phase's scope; the brief's own example list is itself session-level:
"session duration" and "number of alerts" are not continuous signals either).

### A real gap found and fixed: `alert_count` was missing from the history API response

While building this phase, checking exactly what the charts needed to read led to comparing the `/history`
route's projection against `database/sessions.py`'s save path -- and `alert_count`, added to the database and
to `monitor.take_session_result()` in Phase 11, was never added to the route's per-row dictionary that turns
a database row into the JSON the dashboard reads. A Phase 11 test,
`test_alert_count_appears_in_saved_history`, explicitly asserts `self.assertIn("alert_count", row)` against
that exact response -- and would have failed the moment it ran with FastAPI actually installed. It could not
be caught earlier because FastAPI is not available in the sandbox this project is built in, so that test (and
every other one requiring a real HTTP round-trip) has been skipped in every phase's own verification here; it
only runs on the person's own machine. This is the same category of blind spot as the Phase 9 "zero-duration
session" gap: a route-level behavior that only a real FastAPI test can exercise.

Fixed by adding `alert_count` to the route's projection (now returns all twelve fields the charts and the
recent-sessions list use). To reduce the chance of a similar single-field omission going unnoticed again,
`tests/test_phase12.py` adds `test_every_field_the_trend_charts_need_is_present_in_the_database_row` (and its
FastAPI-requiring twin), which checks the FULL set of fields the charts depend on in one assertion against a
named set (`FIELDS_THE_CHARTS_NEED`), rather than relying on remembering to add one `assertIn` per field as
each new column is introduced. Confirmed this catches the exact class of bug by mutating a throwaway copy of
`get_recent_sessions` to only select a few columns and verifying the test fails with a clear, specific
message naming every missing field.

### What is and is not tested here

The chart-drawing functions (`computeTrendPoints`, `buildBarChartSVG`) are pure JavaScript with no DOM
dependency, so they are tested directly with Node (chronological reordering, proportional bar heights, zero-
value styling, empty/single-point edge cases). The orchestration (`renderTrendCharts`, show/hide of the
empty-state message) is tested against the project's existing "fake browser" harness used for every other
dashboard card, checking bar counts and per-session values via each bar's embedded tooltip text. What is
**not** tested by any of this: whether six small bar charts are actually a good way to show these six kinds
of information to a real user, or whether 20 sessions is the right amount of history to keep on screen at
once -- those are usability judgements, not something a test can settle.

## 18. Full history and multi-user hardening (Phase 13)

Files: `database/sessions.py` (`get_recent_sessions` gained an `offset` parameter, plus
`count_finished_sessions`), the new `GET /api/monitor/sessions` route and the `_session_summary_dict` helper
it now shares with `/history` in `backend/routes_monitor.py`, and the full-history table in
`frontend/index.html`.

**Source: the brief's phase list itself** ("Phase 13: History and multi-user support"), read as two related
but distinct pieces of remaining work: Phase 12 built the six trend charts from the brief's dashboard section
but was capped at the last 20 sessions; Phase 13 adds genuine pagination so ALL history is reachable, and
hardens multi-user behaviour with a whole-pipeline test rather than only per-feature ones.

**Pagination:** `GET /api/monitor/sessions?offset=N&limit=M` (limit capped to 100, offset clamped to at least
0) returns a page of sessions plus `total_count`, so the dashboard's "Load more" button knows when to stop
offering more. `/history` (Phase 12, used only for the six charts) and `/sessions` (this phase, for the full
table) now both call the SAME `_session_summary_dict(row)` function to build each session's JSON -- this is a
direct, deliberate fix for the way the two could previously drift apart (see the Phase 12 `alert_count` bug
below); a field added to one can no longer silently be missing from the other.

**Multi-user hardening:** `tests/test_phase13.py`'s `ConcurrentMultiUserTests` runs a realistic sequence --
both accounts register, one calibrates and starts monitoring, the other is refused the camera AND refused
calibration AND sees no leaked data, the first triggers a real alert and finishes, logs out, the second then
calibrates and uses the camera, and finally both check their own history and see only their own session. This
is a genuinely different kind of test from the many "two users stay separate" tests already in every earlier
phase (12 of them by Phase 12): those each check ONE subsystem's isolation with two accounts touching it
briefly; this one runs the WHOLE stack -- auth, calibration, camera ownership, fatigue, alerts, and history --
together, in the order a real shared computer would see them. This is the same technique (a full pipeline run
rather than per-module tests) that found the Phase 8 posture/attention interaction bug and the Phase 10 DHRI
smoothing bug; here it did not find a new defect, which itself is useful evidence that the many individual
isolation tests across Phases 2-12 were not missing an interaction-level problem.

**A note on what could and could not be verified while building this:** `ConcurrentMultiUserTests` and the
`/api/monitor/sessions` route's own tests all require FastAPI, which is not available in the sandbox this
project is built in -- so, as with every route-level test since Phase 2, they could not be run here and are
confirmed only by the person running the project's own test suite. To get SOME direct verification before
shipping, the same two-account sequence was run manually at the CameraMonitor/database layer directly
(bypassing the HTTP routes, calling the same underlying functions the routes call) and passed; this confirms
the core logic works, though it is not a substitute for the real route-level test actually running.

### The Phase 12 `alert_count` bug, and what it changed about how this project is built

While preparing this phase's history endpoint, comparing `/history`'s field list against what the database
actually stores surfaced that `alert_count` -- added to the database in Phase 11 -- had never been added to
`/history`'s response, despite a Phase 11 test asserting it would be there
(`test_alert_count_appears_in_saved_history`). That test could only fail on a machine with FastAPI installed;
in the sandbox used to build this project, it was silently skipped in every verification run from Phase 11
through Phase 12, so the gap went unnoticed until Phase 12's own field-completeness check
(`tests/test_phase12.py`) was written specifically to compare the full set of fields the dashboard needs
against what a session row actually contains, and caught it. It was fixed in Phase 12 (see that phase's notes
above) and is the direct reason `/history` and `/sessions` now share one projection function in this phase,
rather than each route maintaining its own field list that could drift out of sync again.

## 19. Testing (Phase 14)

Files: `backend/main.py` (the shutdown fix), `computer_vision/monitor.py` (`get_active_owner`),
`tests/test_phase14.py`, `docs/testing_checklist.md`.

### The shutdown data-loss bug

**What was wrong:** `backend/main.py`'s `lifespan` shutdown handler called `monitor_module.monitor.stop()`
and nothing else. `stop()` correctly releases the camera and finalizes the session INSIDE the monitor
(`_finish_session` runs, setting `self._session_result`), but nothing ever CALLED `take_session_result()`
and `sessions_db.save_session()` for that pending result -- those two steps only ever happened inside the
`/api/monitor/stop` HTTP route (`backend/routes_monitor.py`), which a bare shutdown function, with no HTTP
request and no database dependency, never goes through. The result: any session running when the SERVER
stopped -- not when the CAMERA was stopped via the button -- vanished. Since this project's own instructions
throughout Phases 3-13 told you to run the server with `--reload`, and `--reload` restarts the whole process
on every saved file change, this bug would trigger constantly during normal development, and identically on
a plain `Ctrl+C`.

**Why it stayed hidden for three phases:** Phase 9 built session saving, Phase 10 added DHRI event counts to
it, Phase 11 added alert counts to it -- all three phases tested that stopping the camera VIA THE API ROUTE
saves a session correctly (and their tests pass). None of them tested what happens when the *server itself*
stops rather than the camera being stopped through the normal button/route, because nothing about building
DHRI or alerts pointed at server shutdown as a relevant scenario. It took deliberately asking "what are all
the ways a session can end?" as part of this dedicated testing phase to notice that "the server process
ends" is a distinct path from "the /stop route runs," and that only one of the two was ever wired to save.

**The fix:** a new `CameraMonitor.get_active_owner()` method reports who is using the camera (or `None`),
read BEFORE `stop()` is called (since after `_reset()` runs, the monitor no longer remembers who the owner
was). The shutdown handler now reads this, stops the camera, and if there was an owner, collects and saves
their session -- reusing `take_session_result`, `should_save_session` and `save_session` exactly as the
`/stop` route does, so there is only one save code path's worth of logic to trust, just two ways of reaching it.

**Verification:** reproduced first (ran a session, ran the OLD shutdown code, confirmed 0 rows in the
database where there should have been 1), then confirmed the fix resolves it, both as a direct unit test and
by running the ACTUAL FastAPI `lifespan` context manager end to end (`ServerShutdownViaRealAppTests`, which
needs FastAPI and is therefore one of the tests that has never run in the sandbox this project was built in
-- only on the user's own machine, same limitation as every FastAPI-dependent test throughout this project).
This is a real, still-open gap: the unit tests that CAN run here test the shutdown LOGIC via an inline
reimplementation, not the actual wiring in `backend/main.py`; a regression in the real wiring would only be
caught by the one test that requires FastAPI.

### Other whole-pipeline scenarios tested this phase

Following the same technique that found the Phase 8 posture bug and the Phase 10 DHRI bug (script a
simulated camera through the REAL monitor engine and see what happens when several already-correct pieces
run together), this phase also tried:

* **Recalibrating mid-session** after fatigue, posture, attention, DHRI and alerts all have real accumulated
  state -- confirmed this does not crash, does not lose the camera, and continues using the NEW baseline
  while keeping the OLD session's event counts (recalibrating does not start a new session).
* **Stopping while a calibration is actively running** (equivalent to logging out mid-calibration) -- the
  camera releases cleanly, and a fresh Start-and-calibrate afterwards works normally.
* **Alert cooldowns across a stop/start cycle** -- a fresh `AlertManager` is created every time the camera
  starts, so cooldowns reset with each new session rather than persisting for the account indefinitely. This
  was already true of the code before this phase, just never deliberately tested or written down; it is now
  both, so it is a documented design choice rather than something to be surprised by later.

### The brief's 20-item checklist (section 25)

Two forms: `tests/test_phase14.py::BriefTestingChecklistTests` runs all 20 items as one continuous automated
session against the real FastAPI app with a simulated camera and face (needs FastAPI; not run in this
sandbox); `docs/testing_checklist.md` is the same 20 items as a manual walkthrough for a real camera, real
face, and real browser -- the parts no amount of simulation can verify. Both cover the same list so a gap in
one is easy to notice by comparing against the other.

## 20. Final integration (Phase 15)

Every phase from 2 onward was built directly inside the one running application rather than as
separate modules glued together at the end, so most of what "integration" would normally mean had
already happened by the time Phase 14 finished. This phase's work was therefore narrower and more
honest about what was actually still missing:

**1. A full-codebase audit for stale references to "future" work that later phases had since
completed.** Several docstrings, the frontend's subtitle text, and two lines in `README.md` still
said things like "calibration is built in later phases" or "Phase 11 has not been built yet" --
true when first written, false by Phase 14. Found with:

```bash
grep -rniE "later phase|not yet built|coming soon|will be added" --include="*.py" --include="*.html" --include="*.md" .
```

Fixed in `utils/config.py`, `computer_vision/session.py`, `computer_vision/attention.py`,
`frontend/index.html` (the page subtitle and the camera-privacy note), `README.md` (the status
line and the "what works so far" section), and this file's Phase 9 write-up (section 14). Left
alone: the "(Phase N)" labels on the dashboard's own section headers and in code comments --
those say which phase built a feature, which is accurate and useful build history, not a promise
about something missing.

**2. One portability bug found while re-running the full test suite for this phase**, unrelated to
any of the above: `scripts/collect_fatigue_data.py` called `cv2.destroyAllWindows()`
unconditionally in its cleanup path. On an OpenCV build with no GUI backend (a "headless" build --
some Linux servers and containers, not the Windows build this project has actually been tested on)
that call raises `cv2.error` instead of doing nothing, which meant the script's own cleanup could
crash on such a machine even though it had already finished its real work. Wrapped in a
`try/except cv2.error` with a comment explaining why. This was caught by five of
`tests/test_phase6.py`'s `CollectorTests` failing when the full suite was run on a machine whose
OpenCV happened to be a headless build -- a good example of why "run the whole suite," not just
the tests for whatever you just changed, is worth doing before calling a phase finished.

**3. `scripts/verify_all.py`, a one-command final check**, meant to be run once before treating the
project as done (and rerun after any later change): confirms it is being run from the project
root, byte-compiles every `.py` file (catching a plain typo before wasting time on tests that
would fail for a trivial reason), runs `scripts/check_environment.py`, then runs the full
automated test suite, and prints a plain OK/FAIL summary of all four steps with a non-zero exit
code if anything failed. It does not replace `docs/testing_checklist.md`'s manual walkthrough --
automated tests use a simulated camera and a simulated face; only a human in front of a real
webcam can confirm the calibration beeps are audible or that a real blink is actually counted.

**4. A final closing run**, using `verify_all.py` itself: 68 files compiled cleanly; the full
suite ran 540 tests with 0 failures and 0 errors (51 skipped, all because `fastapi`/`httpx` are
not installed in the sandbox this project is built in -- see the note repeated throughout this
document about what can and cannot be verified without a real FastAPI install). The environment
check step correctly reported the sandbox's own missing `fastapi` package and un-downloaded model
files as `[FAIL]` -- exactly the behaviour it is supposed to have, and exactly why this project has
always relied on the user re-running the suite on their own machine, where those packages and
files are actually present, for the final word.

**What Phase 15 did NOT do, and why:** it did not rewrite `README.md`'s per-phase build-log
sections (Phase 2's "what works so far", Phase 9's session write-up, and so on) into a single
polished description of the finished system -- that consolidation, plus the full documentation
list in the original brief (problem statement, methodology, results, limitations, SDG relevance,
etc.), is Phase 16's job by the original plan, and doing it here would duplicate that work rather
than complete it early. It also did not attempt to build or validate a trained ML model, or to
run any test that needs a real webcam, real FastAPI, or a real browser -- those remain genuinely
outside what this sandbox can do, same as every phase before it; see `docs/testing_checklist.md`
for the manual steps that close that gap on the user's own machine.

### Real bugs found only once the user ran the suite with FastAPI actually installed

The gap named throughout this document -- that `ServerShutdownViaRealAppTests`,
`BriefTestingChecklistTests`, and every other FastAPI-gated test had literally never executed
anywhere, because this project is built in a sandbox without FastAPI -- turned out to be real. The
user installed the project's full dependencies (as every phase's instructions have told them to)
and ran `python scripts/verify_all.py`, and three of those never-before-run tests failed. All three
turned out to be bugs in the TESTS, not in the application -- but two of them were subtle enough
that they are worth recording here so a future change does not reintroduce the same mistake.

1. **`test_two_people_sharing_one_computer_end_to_end` (`tests/test_phase13.py`)** logged the first
   user out, then later tried to use her (now invalid) session to check her history -- correctly
   rejected with 401 by `current_user` in `backend/dependencies.py`, which the test then choked on
   (`KeyError: 'recent'`, since a 401 body has no `"recent"` key). The app was right; the test
   called logout too early. Fixed by moving her logout to the very end of the test, after every
   check that needs her still logged in, and added an explicit assertion that the now-logged-out
   session really does get 401 -- so logout itself stays tested, just at the right moment.

2. **`test_the_real_lifespan_shutdown_saves_an_active_session` (`tests/test_phase14.py`)** --
   *the* test for the Phase 14 shutdown-data-loss fix, the single most severe bug found in this
   project, and one this document already flagged as never having run anywhere. It failed with
   `IndexError: No item with that key` from `row["id"]` -- because the query two lines above had
   selected the `username` column, not `id`. A plain copy-paste mistake in the test's own SQL, not
   the shutdown code. Fixed the query to select `id`. With that fixed, the test PASSES: the Phase
   14 shutdown fix is now confirmed correct by the one test that can actually prove it, for the
   first time.

3. **`test_the_full_20_item_checklist_end_to_end` (`tests/test_phase14.py`)** failed to ever reach
   "fatigued", and turned out to have two compounding problems, found in that order:
   - Its `wait_for(..., timeout=8)` for "fatigued" was simply too short: reaching "fatigued" needs
     `FEATURE_MIN_COVERAGE_S` (10s of accumulated window data) plus `FATIGUE_ESCALATE_HOLD_S` (3s)
     -- about 13s minimum with the REAL, unmocked config values this test deliberately uses. Under
     the load of running many camera-simulation threads back to back (this project's own full test
     suite), even a generous 30s timeout was not always enough -- through no fault of the fatigue
     logic, which was independently confirmed to measure window coverage from real elapsed time,
     not frame count (`computer_vision/eye_tracker.py`), so it is not itself rate-sensitive. Fixed
     by shrinking the window for just this test (`FEATURE_WINDOW_S`/`FEATURE_MIN_COVERAGE_S`,
     mocked the same way `tests/test_phase6.py`'s fatigue pipeline test already does) -- this test's
     actual point is exercising the real FastAPI/lifespan wiring, not the window's real duration,
     which is already covered elsewhere.
   - Underneath that timing issue was a second, more interesting bug: `calibrate_via_api`'s polling
     loop leaves the simulated eyes CLOSED once calibration finishes, because calibration's own
     LAST step is "close your eyes" (`computer_vision/calibrator.py`'s phase order is
     ready -> open -> close -> done). The eye tracker's `_seen_open` guard
     (`computer_vision/eye_tracker.py`) -- which deliberately waits to see the eyes clearly OPEN
     before trusting any closure, so a closure already in progress when tracking resumes is never
     miscounted -- then never arms, and `window_valid_s` stays 0 forever: fatigue can never be
     assessed no matter how long the test waits. A real person naturally opens their eyes again the
     moment calibration ends; a script has to be told to. Fixed by explicitly setting the simulated
     eyes open again right after calibration and waiting for the tracker to confirm it has seen
     them open, before simulating closure for the fatigue check -- which is also exactly the
     pattern every other passing eye/fatigue test in this project already follows
     (`tests/test_phase5.py`'s `open_eyes_measured` helper). This was not a shrinking-timeout
     problem in disguise; no timeout, however long, would have fixed it.

**The honest takeaway:** the `_seen_open` design itself is correct and was not changed -- a real
user's own eyes reopening after calibration is exactly what makes it work in practice, and this
was invisible to every test in this project until one specifically drove the real calibration
phase sequence through the real FastAPI app end to end. It is a good example of why "run with the
real dependencies actually installed," not just "run whatever the sandbox can," was worth insisting
on even at Phase 15.

**Confirmed fixed:** with all three fixes applied, the user re-ran the full suite on their own
Windows machine, with `fastapi`/`httpx` actually installed (not skipped) -- **540 tests, 0
failures, 0 errors, 0 skipped.** This is the first time in the whole project that every
FastAPI-gated test has actually executed anywhere, and the first real confirmation (not just a
sandbox-only unit-test reconstruction) that the Phase 14 shutdown fix genuinely works. This closes
the residual verification gap this document has flagged since Phase 14.

## 20a. UI restructuring: button-based navigation and a Dashboard page

After Phase 15 shipped, the single page (`frontend/index.html`) had grown into one long scroll:
every feature card (profile, DHRI, calibration, live monitoring, fatigue, posture, attention,
session, alerts, history/trends) stacked one after another, so finding one meant scrolling past
all the others. This was a display-only problem -- nothing about the underlying monitoring,
calibration, alerting or history logic needed to change -- so the fix is a display-only change.

**What changed.** `frontend/index.html` gained:

- A `<nav id="main-nav">` bar with one button per feature (Live monitoring, Fatigue, Posture,
  Attention, DHRI, Alerts & tips, Session, Calibration, Dashboard, Profile), shown only while
  logged in.
- Each existing feature card was wrapped, unchanged, in its own `<div class="page">` container
  (`page-live`, `page-fatigue`, ... `page-profile`). A new `switchPage(pageId)` function shows
  exactly one `.page` at a time and highlights the matching nav button; clicking a button calls
  it. The default page after login is Live monitoring, and logging out and back in on the same
  browser tab always returns there rather than remembering the last page.
- A small always-visible DHRI badge (`#dhri-mini`) in the top bar, fed by the same `dhri` object
  already returned from `/api/monitor/status`, so the headline risk level stays visible no matter
  which page is open, without needing to duplicate the full DHRI card everywhere.
- The former "History & trends" card (Phase 12) is now reached through the **Dashboard** nav
  button. It is otherwise the same card -- same trend charts, same "View full history" table and
  pagination (Phase 13), same `/api/monitor/history` and `/api/monitor/sessions` endpoints -- with
  one small addition: a one-line "Today" summary (`#dash-today`) built from the `today` object the
  page already fetches, so the day's totals are visible without switching to the Session page.
  No new API endpoints, and no changes to the database schema or any backend module, were needed
  for either the navigation or the Dashboard.
- The privacy note and the "System check" details block, previously always visible at the very
  bottom of the long scroll, now live inside the Profile page (same markup and ids, just moved).

**What did not change.** Every element id, every `renderX(...)` function, every event listener,
every API call, the computer-vision pipeline, the DHRI calculation, the alert/cooldown logic and
the database layer are untouched -- the new nav is a pure display layer added on top of the
existing `show(id, visible)` pattern the page already used to gate content on login state. This
was checked by walking the whole file end to end while restructuring, so nothing was silently
dropped.

**How it was verified.** This sandbox has no real browser and the Node.js "fake browser" harness
built during earlier phases did not survive the Phase 15 environment reset, so a fresh one was
built: a jsdom-based script loads the real `frontend/index.html`, mocks the same `fetch` calls the
page makes, and drives it end to end -- confirms the page starts logged out, logs in, checks that
exactly one `.page` is visible at a time, clicks every nav button in turn and checks that button
alone becomes active and its page alone becomes visible, confirms the Dashboard page still
contains the Phase 12/13 history markup and shows the correct "Today" summary, confirms the
Profile page still contains the profile card, privacy note and system check, logs out, and logs
back in to confirm the page resets to Live monitoring. All of this passed. `node --check` was also
run over the extracted `<script>` block to confirm it is syntactically valid, and the Python test
suite (`scripts/verify_all.py`'s compile and unit-test steps) was re-run to confirm the untouched
backend was not affected by this frontend-only change.

## 20b. Visual design pass: colour, hierarchy and two real bugs found by actually rendering the page

After the navigation restructuring above, the user asked for the UI to be more colourful, clean
and user-friendly. `frontend/index.html` gained a small, deliberate token system -- CSS custom
properties for one brand colour family (a calming teal, used for the header, nav and primary
actions) and a single warm accent (coral, spent only on the header gradient and the "Start camera"
button, the page's one most important action) -- plus a reusable inline SVG icon sprite (`<symbol>`
defs at the top of `<body>`, referenced with `<use>`) so every nav button and its matching page
heading carry the same icon, reinforcing which feature is open. The existing green/amber/red state
colours for fatigue, posture, attention and DHRI were deliberately left untouched throughout: they
already carry real meaning (good/warning/bad) and must stay visually distinct from the decorative
palette rather than be absorbed into it.

This sandbox has Chromium and Playwright pre-installed, so instead of only reasoning about the CSS,
the page was actually served locally and screenshotted at several states (logged out, live
monitoring, fatigue warning, DHRI breakdown, dashboard, and a 390px mobile width). This caught two
real, pre-existing-pattern bugs that a purely textual review would have missed:

1. **`.hidden` could be silently overridden.** `.hidden { display: none; }` and several new chrome
   classes (`.topbar-right`, `.nav`) both set `display` at equal CSS specificity (one class
   selector each); because the new flex rules happened to come later in the stylesheet, they won
   the cascade, and the nav bar and DHRI badge would have been visible even while logged out. Fixed
   by making `.hidden` authoritative with `display: none !important` -- documented inline as a
   deliberate, narrow use of `!important` for a utility class whose entire job is to force-hide
   regardless of what else targets the same element.
2. **The DHRI contribution bars never showed their width.** `.dhri-bar-fill` is a `<span>`, and the
   `width` CSS property has no effect on inline elements -- so the fill bars (Fatigue/Posture/
   Attention/Exposure) had been invisible since Phase 10, with only the numeric value next to each
   bar giving any information. This was not introduced by this UI pass; it was only caught because
   the DHRI page was actually rendered and looked at rather than inferred from the CSS source. Fixed
   with `display: inline-block` on `.dhri-bar-fill`.

Both fixes were confirmed by re-screenshotting the affected pages (nav/badge correctly hidden when
logged out; the four DHRI bars render with their proportional, colour-coded widths) and by
re-running the Phase 15 jsdom navigation test and `node --check`, which both still pass.

## 20c. Submission extras: export, weekly summary, notifications, accessibility, one-click run

A small batch of additions aimed specifically at making this a stronger project to submit and
demo, on top of the already-complete 16-phase build. None of these required touching the
computer-vision pipeline, the DHRI calculation, the database schema, or any API route except by
calling an endpoint that already existed.

- **CSV export** (Dashboard page, "Download CSV" button): walks every page of the existing
  `/api/monitor/sessions` endpoint client-side and builds a CSV of the full session history, so the
  data behind the charts can be opened in Excel/Sheets or attached to a report. No new backend code.
- **"This week" summary** (Dashboard, next to "Today"): a quick aggregate computed client-side from
  the up-to-20 sessions `/api/monitor/history` already returns. It is a glance stat, not the
  authoritative total -- past 20 sessions in a week it says so and the CSV export has the real
  numbers.
- **Desktop notification on HIGH DHRI**: in addition to the existing in-page alert, the browser can
  show an OS-level notification when DHRI enters HIGH risk (permission requested on the "Start
  camera" click, a real user gesture; declining it leaves the in-page alert working exactly as
  before). Debounced to fire once per entry into HIGH, and at most once every 5 minutes if it stays
  there, so it cannot spam.
- **Accessibility**: the feature nav and pages now carry the standard ARIA tabs pattern
  (`role="tablist"/"tab"/"tabpanel"`, `aria-selected`, `aria-controls`/`aria-labelledby`), on top of
  the visible focus ring added in the Phase 15 visual pass.
- **`run.bat` / `run.sh`**: a one-command launcher (creates the venv on first run if missing,
  installs dependencies, starts uvicorn, opens the browser) so a demo or evaluator does not need the
  multi-step PowerShell sequence this project's own development needed more than once.
  **Fixed a real race condition** (caught by a real user hitting it, not by review): the first
  version fired `start "http://127.0.0.1:8000"` (Windows) / opened the browser after a fixed
  `sleep 1.5` (macOS/Linux) and *then* started uvicorn -- `start` does not wait for the command it
  launches, so the browser could try to load the page before uvicorn had bound the port, showing
  "This site can't be reached" even though the server was starting up fine moments later. Both
  scripts now poll port 8000 in the background (a hidden PowerShell loop on Windows, a `curl`-based
  loop on macOS/Linux; up to 20 seconds) and only open the browser once the server actually answers.

All of the above were verified with the same jsdom approach as the Phase 15 navigation test:
`/tmp/dom_export_test.js` (not checked into the repo -- a throwaway verification script) drives a
real login, switches to the Dashboard, asserts the "This week" total correctly includes only the
in-week session and excludes a 10-day-old one, asserts the ARIA attributes update on tab switch, and
clicks the real "Download CSV" button, capturing the generated filename and CSV content to confirm
both sessions are present with the right header row.

**What was intentionally not added**, and why:

- **A trained ML fatigue model's validation metrics.** The fatigue page already has the code to
  display cross-validated accuracy/precision/recall once a model exists (`describeModel()` in
  `frontend/index.html`, fed by `/api/fatigue/model`) -- but no model has been trained yet
  (`models/` is empty), because that requires real recorded fatigue data from `collect_fatigue_data.py`,
  which only a person with a webcam can provide. Running that collection script for a few sessions
  and then `ml/train.py` would light this up with real numbers; it cannot be done from this
  environment.
- **A real user study.** Equally, this needs real people using the real app; see
  `docs/user_study_template.md` for a ready-to-run short protocol, instead of fabricating data here.

## 20d. Home hub: tile-based navigation instead of a plain tab bar

The Phase 15 nav bar (20a) already turned the single long scroll into one page per feature, but it
was still a flat row of small text buttons -- functional, not inviting. This pass replaces the "land
on Live monitoring" behaviour with a dedicated **Home** page: a welcome banner plus a grid of ten
large, colourful tiles, one per feature, each showing an icon, a title and a one-line description.
Clicking a tile (or a nav-bar button, or the new breadcrumb's "Home" link) calls the same
`switchPage()` as before -- this is a visual and navigational layer on top of the existing pages, not
a rewrite of them.

What changed in `frontend/index.html`:

- **A new `page-home` page**, inserted first in the page list and shown by default after login (and
  again after every logout/re-login). Its tiles reuse the `data-page` attribute convention the nav
  bar already used, so one shared `document.addEventListener("click", ...)` listener (matching any
  `button[data-page]`, anywhere in the page) now drives all three ways to navigate: the nav bar, the
  Home tiles, and the breadcrumb's "Home" link -- replacing the old nav-bar-only click listener.
- **A breadcrumb bar** (`Home / <Page name>`) shown above every feature page except Home itself, so
  there is always a one-click way back to the hub without relying on the nav bar alone. Its text is
  kept in sync with the current page inside `switchPage()`.
- **A decorative tile colour palette** (`--tile-accent` / `--tile-bg` custom properties on classes
  like `.tile-teal`, `.tile-indigo`, `.tile-coral`, ...), deliberately a *different* set of colours
  from the red/amber/green STATE colours used inside the feature pages (`.f-warning`, `.dhri-high`,
  etc.), so a colourful tile is never mistaken for a live status reading.
- **A page-switch fade/rise animation** (`@keyframes pageIn`, applied via a `.page-anim` class that
  `switchPage()` adds and restarts on every navigation), wrapped in
  `@media (prefers-reduced-motion: no-preference)` so it is skipped entirely for anyone whose OS asks
  for reduced motion.
- One real bug, again caught only by actually rendering the page with Playwright, not by reading the
  code: the global `h2 { display: flex; gap: 9px; }` rule (added in the Phase 15 visual pass for
  icon+text headings) was also being applied to the Home hero's plain-text `<h2>Welcome back<span
  id="home-username">...`, inserting a visible 9px gap before the comma ("Welcome back , asha!").
  Fixed with a scoped `.home-hero h2 { display: block; }` override.

Verification: `/tmp/dom_nav_test.js` (the same Phase 15 navigation test, extended to include `"home"`
in its list of pages) still passes in full -- home is the default page on login and after
logout/re-login, exactly one page is visible at a time for every one of the eleven pages including
Home, and every existing nav button still works. `/tmp/dom_export_test.js` (CSV export, "This week",
ARIA) also still passes unchanged. Six new/replaced screenshots in `docs/screenshots/` (`00-home.png`
plus refreshed `02` through `06`) were taken with Playwright against the real file, including a real
simulated click on a Home tile and on the breadcrumb's "Home" link (not just calling the JS function
directly) to prove the shared click listener actually works end-to-end, and a mobile-width capture
showing the tile grid collapse to one column.

## 21. Phase plan

1 Architecture (this) -> 2 Database + login -> 3 Webcam -> 4 Landmarks -> 5 Eye features ->
6 Fatigue + SVM -> 7 Posture -> 8 Attention -> 9 Session monitoring -> 10 DHRI ->
11 Alerts + recommendations -> 12 Dashboard -> 13 History + multi-user -> 14 Testing ->
15 Integration (done -- see section 20) -> 16 Documentation
