# Informal user study template

A short, practical protocol for getting real evidence about whether this project's monitoring
feels accurate -- something no amount of code review can provide. Aim for 2-3 participants
(classmates, housemates, anyone willing to sit at a computer for 20-30 minutes). This is informal:
it will not produce a validated accuracy number, but a few honest data points are worth far more in
a viva or a report than none.

## Before each session

1. Make sure the app is running (`run.bat` / `run.sh`, or the usual `uvicorn` command) and the
   participant can see the screen.
2. Have them register their own account (`username_1`, `username_2`, ...) so sessions don't mix.
3. Run through **Personal calibration** with them -- sitting the way they'd normally sit, looking
   at the screen the way they normally would.
4. Explain, briefly, what the system watches for (eye closure/blinking, posture vs. their
   calibrated position, whether they're facing the screen) and that nothing is recorded on video --
   only the numbers.

## During the session (15-20 minutes of normal work)

Ask the participant to just do their normal work -- reading, typing, whatever they'd usually do --
while you (or they) keep a simple paper/notes log of what they **actually experienced**, independent
of what the app shows:

| Time | What I actually felt/did | What the app showed (check after) |
|---|---|---|
| e.g. 0:04 | "got distracted by my phone for a bit" | |
| e.g. 0:11 | "eyes felt tired, rubbed them" | |
| e.g. 0:15 | "slouched on purpose to test posture" | |

Deliberately include a few moments that *should* trigger something (look away for 10+ seconds,
slouch on purpose, let your eyes get tired if you can) so there's something to compare against, not
just quiet agreement.

## After the session

1. Press **Stop**, then open the **Dashboard** page and look at that session's row in the full
   history table (or export it with **Download CSV**).
2. Fill in the third column of the table above from what the app actually logged (fatigue/posture/
   attention event counts, alerts).
3. Ask the participant two questions and write down their answers verbatim:
   - "Did anything the app flagged NOT match what you were actually doing?" (false positives)
   - "Did anything you noticed in yourself NOT get flagged?" (false negatives/misses)

## Writing it up

A short paragraph per participant is enough for a report:

> Participant 2 ran a 20-minute session. The system correctly flagged 2 of 2 deliberate "looking
> away" moments (Attention: DISTRACTED) within about 5-6 seconds, matching the configured 5-second
> hold time. It also flagged one posture warning during a moment the participant said they were
> "just stretching," which they felt was a false positive. No fatigue warnings fired (participant
> reported not feeling tired), consistent with the log.

Collecting 2-3 of these, with both the matches and the mismatches named honestly, is exactly what
turns "we built a monitor" into "we built a monitor and checked whether it works."
