# Fatigue assessment: modes, recording data, training

**Status: there is NO real dataset yet, so there is no trained model, and no accuracy figure exists for this
project.** Until you record data and train, the app runs in **DEMO / RULE-BASED MODE**. Nothing below has been
validated on real users. The result is a monitoring indication, not a medical diagnosis.

## 1. The two modes (the dashboard always shows which is active)

| | DEMO / RULE-BASED MODE | TRAINED ML MODE |
|---|---|---|
| When | No usable trained model exists (default) | `models/fatigue_svm.joblib` exists, loads, and is not a demo model |
| How it decides | Readable rules on four eye features (section 3) | An SVM trained on YOUR recordings |
| Validated? | No. The thresholds are our starting values | Only as far as your own cross-validation report shows (section 5) |

A model trained on the synthetic demo data (section 6) is saved under a different name and is refused by the app.

## 2. How this relates to the base paper (Sun et al. 2020)

| Base paper | This project |
|---|---|
| Eye features DEC, FB, PCT computed over 20 s windows | The same three ideas over a 20 s window: blink duration and longest closure (DEC), blinks per minute (FB), PERCLOS-style closure share (PCT) |
| SVM classifier, fatigue vs normal | `ml/train.py`: SVM (RBF kernel) on those features, fatigue vs normal |
| Light intensity (LI) as a fifth input, from a light sensor | **Not used** by the classifier: with a small dataset it could become a shortcut ("dark = tired") |
| Data: 8 drivers, 15 hours awake, driving, colour + infrared camera | Data: you record it, computer users, ordinary webcam |
| Reported precision/recall on its own test set | **Not transferable.** The paper's numbers say nothing about this system. Only your own report does |

The paper's SVM settings are not given, so ours (RBF, C=1, balanced classes) are common defaults, not tuned.

## 3. Rule-based mode

Four indicators over the last 20 s, each with a WARNING and a FATIGUED level (`FATIGUE_RULE_THRESHOLDS` in `utils/config.py`):

| Indicator | Warning | Fatigued |
|---|---|---|
| Share of time eyelids at least 80% closed (PERCLOS-style) | 10% | 20% |
| Longest single eye closure | 0.5 s | 1.0 s |
| Blinks per minute (frequent blinking) | 30 | 45 |
| Average blink duration (slow blinks) | 0.30 s | 0.45 s |

Each indicator scores 0.5 at its warning level and 1.0 at its fatigued level (a straight line between). The fatigue
score is the highest indicator score: below 0.5 = NORMAL, 0.5 up to 1.0 = WARNING, 1.0 = FATIGUED. The dashboard lists
which indicator caused the state. **All of these numbers are our own guesses.**

**Smoothing:** a worse state must last 3 s before it is shown (an eye closure already longer than the fatigue limit
is shown at once), and a better state must last 10 s. The state becomes "not available" (with the reason) when you
are not calibrated, are calibrating, your eyes are not visible, or fewer than about 10 s of data exist.
"Episodes this session" counts how often WARNING or FATIGUED was entered; an episode interrupted because your eyes
left the view is counted again when it is confirmed again.

## 4. Recording data (needed for TRAINED ML MODE)

**Before recording another person:** they must know what is recorded (eye-measurement numbers; no images or video are
ever saved), agree, and be free to stop at any time. Use a pseudonym (P01, P02, ...), never a real name.
**Do not make anyone tired on purpose** in any way that could be unsafe (no sleep deprivation, no driving). Record natural
situations, for example a fresh morning and a late evening after a long day, and let the person report honestly how they feel.
The label is their own report, not a medical test.

Prerequisite: the account has been calibrated in the dashboard. Then, with the server stopped or running:

```powershell
python scripts/collect_fatigue_data.py --username asha --participant P01 --label normal   --minutes 10
python scripts/collect_fatigue_data.py --username asha --participant P01 --label fatigued --minutes 10
```

* One row of four numbers is saved every 5 s (after the first ~10 s) to `data/fatigue_features.csv`. 10 minutes is about 110 rows.
* Rows are appended, so record as many sessions as you can. Press **Q** in the video window to stop early.
* Nothing is recorded while your eyes are not measured (no face, or a nearly black image).

**How much?** The trainer refuses to run with fewer than 30 rows per class from fewer than 2 sessions per class. That is a
floor to prevent nonsense, **not a target**. For a result worth reporting, aim for several sessions per class from at least
3 different people (so testing can hold out whole people), record BOTH labels for the same people (otherwise the model may
learn who is who instead of who is tired), and vary time of day, lighting and glasses.

## 5. Training and reading the report

```powershell
python -m ml.train
```

It reads `data/fatigue_features.csv`, refuses if there is too little data (telling you what is missing), then:

1. **Cross-validates keeping whole people together** (or whole sessions if there are fewer than 3 people), so every test row
   comes from someone the model never saw. (Rows of one session are near-duplicates; a random split would give falsely high scores.)
2. Prints accuracy, precision, recall and F1 for the "fatigued" class, plus the confusion counts.
3. Compares with two baselines on the same rows: always guessing the bigger class, and the rule-based mode.
4. Trains the final model on all data and saves `models/fatigue_svm.joblib` and `models/fatigue_svm.report.txt`.

How to read it: the SVM is only worth using if it **clearly beats both baselines**. The score only says how it does on people/sessions
like those in your data. With fewer than 3 people it says little about new people. Then restart the camera in the dashboard; the
assessor loads the model when the camera starts, and the badge changes to TRAINED ML MODE with the training facts.
In the app the states use probability thresholds (0.4 warning, 0.7 fatigued in `utils/config.py`); the report scores use 0.5.
Only load model files you trained yourself (joblib files can contain code).

## 6. Testing the pipeline without data (SYNTHETIC)

```powershell
python -m ml.make_demo_data        # writes data/DEMO_fatigue_features.csv (invented rows)
python -m ml.train --demo          # trains and prints a report marked DEMO DATA
```

The invented rows are trivially different between the two classes, so the demo report will look excellent. **That is meaningless**:
it only shows the code runs. Never quote it. The demo model is saved as `models/DEMO_fatigue_svm.joblib` and the app will not load it.

## 7. Suggested validation for computer users (Phase 14)

The paper validates drivers; nothing validates this system for students or office users yet. A simple, honest plan:

1. Record 5 or more volunteers, both labels, at different times of day, some with glasses.
2. Train with `python -m ml.train` and keep the report (it holds hold-out-people results).
3. Report precision and recall for "fatigued" (as the paper does) together with the number of people, sessions and rows.
4. Compare with the rule-based mode on the same recordings; report both.
5. Separately, count blinks by hand in a few short recordings and compare with the system's count (validates the eye analysis itself).

## 8. Known limits

* Everything depends on the eye analysis, which itself has no accuracy figure yet (see docs/architecture.md, section 9).
* Self-reported labels are subjective. Sleepiness is not the only reason for slow or frequent blinking (dry eyes, screen glare,
  concentration, contact lenses).
* Person-to-person differences are large; a small dataset cannot cover them.
* The rules' thresholds and the probability thresholds are unvalidated starting values.
