# Testing checklist (Phase 15)

This is the project brief's own 20-item testing list (section 25), organized into one ordered walkthrough
you can run on your real computer. Automated tests already cover the LOGIC behind every item (see
`tests/test_phase14.py::BriefTestingChecklistTests` for the same 20 items run against a simulated camera) --
this checklist covers the parts that only a real webcam, a real face, and a real browser can actually verify.

Run this after any update, and definitely before treating the project as "done" for a demo or submission.
Expect it to take about 20 minutes if everything works on the first try.

## Before you start

```powershell
cd $HOME\Desktop\wellbeing_monitor
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python scripts/verify_all.py
```
This runs the compile check, the environment check, and the full automated test suite in one go
(see `scripts/verify_all.py`'s own docstring for what each step does), and ends with an OK/FAIL
summary. The test-suite step should report `Ran 540 tests ... OK`. Do not continue past a failing
step -- fix that first (paste the failure and ask). If you only want the test suite on its own:
`python -m unittest discover -s tests -t .`

---

## 1. Registration
Start the server (`python -m uvicorn backend.main:app --reload --no-use-colors`), open
http://127.0.0.1:8000, `Ctrl+F5`. Click **Register**, choose a username and an 8+ character password.
**Expect:** the profile card appears immediately, showing "Not calibrated yet".

## 2. Login
Click **Log out**, then log back in with the same username and password.
**Expect:** you land back on the profile card. Try a wrong password once: **Expect:** "Invalid username or
password," and you are NOT let in.

## 3. Calibration
Click **Start calibration**. Sit normally, look at the screen, and follow the on-screen countdown; close
your eyes at the first beep, open them at the second.
**Expect:** a success message showing your open/closed eye values, and the profile's calibration status
updates to "Calibrated" (posture and attention too, if this is a fresh calibration).

## 4. Camera access
Click **Start camera**.
**Expect:** your webcam's light turns on, and the mirrored preview appears within a second or two. If it
says "Camera not detected," close other apps that might be using the camera (Teams, Zoom, other browser
tabs) and try again.

## 5. Face detection
Look at the camera.
**Expect:** a green box tracks your face in the preview. Move out of frame: **Expect:** the message changes
to "Please position your face inside the camera view" within about a second.

## 6. Eye detection
Look at the camera with your eyes open.
**Expect:** the "Eye aspect ratio (EAR)" row shows a number (typically 0.2-0.35) and updates continuously.
Yellow outlines should sit on your eyes in the preview.

## 7. Blink detection
Blink normally 10 times, at a natural pace.
**Expect:** "Blinks counted" rises by approximately 10 (within one or two). The eye outline briefly turns red
during each blink.

## 8. Eye closure
Close your eyes and hold them shut for about 2 seconds, three separate times.
**Expect:** each time, "Eyes" shows CLOSED, "Current eye-closure time" counts up, and afterwards "Blinks
counted" shows "(+ 3 long closures)" without the main blink count increasing for those three.

## 9. Fatigue detection
Wait about 10 seconds after starting the camera for the Fatigue card to leave "collecting data." Then close
your eyes for 1.5+ seconds.
**Expect:** the Fatigue card shows FATIGUED with a reason mentioning "eye closure," then returns to NORMAL
after you stop.

## 10. Posture detection
Note your posture card's status while sitting normally (should be GOOD). Tilt your head sideways about 20
degrees and hold it for 5+ seconds.
**Expect:** the card changes to WARNING or POOR, naming "tilted" as the reason. Sit up straight again:
**Expect:** it returns to GOOD after a few seconds.

## 11. Attention detection
Turn your head about 35-40 degrees away from the screen and hold it for 5+ seconds.
**Expect:** the Attention card shows DISTRACTED / LOOKING AWAY. Turn back: **Expect:** ATTENTIVE returns
after about 2 seconds. Then cover the camera or step out of frame for 3+ seconds: **Expect:** FACE NOT
DETECTED.

## 12. Session timer
Watch the Session card while the camera runs.
**Expect:** "This session" counts up continuously. Cover the camera for 30+ seconds, then uncover it:
**Expect:** "Breaks taken" increases to 1.

## 13. DHRI calculation
With fatigue, posture and attention all showing real data (not "collecting..."), check the DHRI card.
**Expect:** a score from 0-100 and a LOW/MODERATE/HIGH label, with four bars (Fatigue, Posture, Attention,
Exposure) that roughly match what the individual cards are showing.

## 14. Alerts
Trigger a clear fatigue or posture problem (steps 9 or 10) and keep it going for at least 10 seconds.
**Expect:** the Alerts card's count increases, and the matching message ("Your eyes appear closed for
longer than usual." or "Please correct your sitting posture.") appears in the recent list.

## 15. Recommendations
While a condition from step 14 is still active, look at the Alerts card's recommendations area.
**Expect:** a matching practical suggestion appears (for example "Rest your eyes for a short period.") and
disappears once you correct the condition.

## 16. Database storage
Click **Stop camera**.
**Expect:** no error. In a second PowerShell window (same folder, environment activated), run:
```powershell
python -c "import sqlite3; c = sqlite3.connect('data/wellbeing.db'); print(c.execute('SELECT COUNT(*) FROM monitoring_sessions WHERE end_time IS NOT NULL').fetchone())"
```
**Expect:** a count of at least 1.

## 17. History
Scroll to the **History & trends** card.
**Expect:** six small charts appear showing your session(s). Click **View full history**: **Expect:** a
table listing your session(s) with duration, breaks, and event counts. If you have more than 20 sessions,
try **Load more**.

## 18. Dashboard graphs
Look at the six trend charts.
**Expect:** each bar's tooltip (hover over it) shows a date and a number; the numbers should roughly match
what you saw during your session (for example, if you triggered several fatigue alerts, the Fatigue chart's
most recent bar should be taller than 0).

## 19. Logout
Click **Log out** while the camera is running (do this deliberately, not after already stopping it).
**Expect:** the webcam light turns off immediately, and you return to the login screen.

## 20. Multiple users
Open a private/incognito browser window at the same address. Register a second account there.
**Expect:** the new account shows "Not calibrated yet" and an empty history -- nothing from your first
account is visible. Back in your original window, confirm your own history and calibration are unaffected.

---

## Extra checks specific to this project (beyond the brief's list)

These aren't in the brief's numbered list, but are worth checking because real bugs were found in exactly
these situations during development (see `docs/architecture.md` for the full stories):

- **Server restart mid-session (Phase 14 fix).** Start the camera, let a session run for a few seconds,
  then save any Python file in the project (triggering `--reload` to restart the server) OR press `Ctrl+C`
  in the server window. Re-run the database check from step 16. **Expect:** the session that was running IS
  saved, even though you never clicked Stop. (Before this phase, it was silently lost.)
- **Turning your head does not look like bad posture (Phase 8/12 fix).** Turn your head 35-40 degrees
  sideways (as in step 11) and watch the Posture card at the same time as Attention. **Expect:** Posture
  stays GOOD while Attention shows DISTRACTED -- turning to look away should not also trigger a posture
  warning.
- **Covering the camera does not create false blinks (Phase 5 fix).** With the camera running and blinks
  being counted normally, cover the lens with your palm for 3 seconds, then uncover it. Repeat twice more.
  **Expect:** "Blinks counted" does not increase from the covering/uncovering itself.
- **DHRI reaches HIGH within about 5 seconds of conditions settling, not 10+ (Phase 10 fix).** Trigger both
  bad fatigue and bad posture together (steps 9 and 10 at the same time) and time how long DHRI takes to
  reach HIGH after both cards show their worst state. **Expect:** roughly 5 seconds, not 10 or more.

## If something fails

Note exactly which numbered item failed and what you saw (a screenshot of the relevant card is ideal). Most
issues so far have been either: (a) a stale browser cache showing an old version of the page (fix: `Ctrl+F5`,
or close all tabs and reopen), or (b) a version mismatch between the server and the files on disk (check the
System check section at the bottom of the page shows the version you expect). If neither explains it, it may
be a genuine new finding -- exactly what this phase exists to catch.
