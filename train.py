"""
Train the fatigue classifier (an SVM, like the base paper's).

    python -m ml.train                    train on data/fatigue_features.csv (real recordings)
    python -m ml.train --data a.csv b.csv train on other files
    python -m ml.train --demo             train on the SYNTHETIC demo data (only to test this code)

WHAT IT DOES
    1. Reads and checks the CSV (ml/dataset.py). Bad rows are skipped and reported.
    2. REFUSES to continue if there is too little data (utils/config.py: ML_MIN_ROWS_PER_CLASS,
       ML_MIN_SESSIONS_PER_CLASS). A model trained on a handful of rows would only look accurate.
    3. Cross-validation that keeps whole participants (or sessions) together, so every test row comes from
       a person/session the model has not seen. Rows of one session are near-duplicates; splitting them
       randomly would give falsely good scores.
    4. Reports accuracy, precision, recall and F1 for "fatigued" (the paper reports precision and recall),
       and compares with two baselines on the SAME rows: always guessing the bigger class, and the
       rule-based mode.
    5. Trains the final model on all data and saves it together with facts about the training.

THE MODEL: imputer (fills a missing mean blink duration) -> standard scaling -> SVC with an RBF kernel,
C=1, balanced class weights, probability outputs. These are common defaults, NOT tuned; the base paper does
not give its SVM settings either.

Numbers from data that is not real (source = demo) are marked DEMO everywhere and the model is saved under a
different name, which the application refuses to load.
"""

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

# scikit-learn >= 1.9 warns that SVC's `probability=True` will move to CalibratedClassifierCV
# in a future release. We still need predict_proba() for the fatigue-probability score (see
# ml/model.py), and switching estimators now would change the trained-model format without any
# real benefit at this project's scale, so this is a deliberate, narrow, cosmetic silence of that
# one message -- not a blanket "ignore all warnings". Revisit if/when upgrading past sklearn 1.10.
warnings.filterwarnings("ignore", message=r".*`probability` parameter was deprecated.*",
                         category=FutureWarning)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import joblib

from ml import rules
from ml.dataset import DatasetError, dataset_sha256, load_rows
from ml.features import FEATURE_NAMES
from utils import config


def build_pipeline():
    return Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                    probability=True, random_state=0)),
    ])


def binary_metrics(y_true, y_pred):
    """Counts and scores for class 1 = fatigued. A score is None when it is undefined (no such cases)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision, "recall": recall, "f1": f1,
    }


def check_enough_data(rows):
    """Raise DatasetError (with numbers and advice) if the data is too small to train on."""
    for label in ("normal", "fatigued"):
        of_class = [r for r in rows if r["label"] == label]
        sessions = {r["session_id"] for r in of_class}
        if len(of_class) < config.ML_MIN_ROWS_PER_CLASS or len(sessions) < config.ML_MIN_SESSIONS_PER_CLASS:
            raise DatasetError(
                f"Not enough '{label}' data to train a model that means anything: you have {len(of_class)} rows "
                f"from {len(sessions)} session(s); at least {config.ML_MIN_ROWS_PER_CLASS} rows from at least "
                f"{config.ML_MIN_SESSIONS_PER_CLASS} separate sessions are required. Keep recording with "
                "scripts/collect_fatigue_data.py (see docs/fatigue_ml.md). Until then the DEMO / RULE-BASED MODE is used."
            )


def choose_groups(rows):
    """Keep whole participants together if there are at least 3, otherwise whole sessions."""
    participants = {r["participant"] for r in rows}
    if len(participants) >= 3:
        return [r["participant"] for r in rows], "participant"
    return [r["session_id"] for r in rows], "session"


def cross_validate(X, y, groups, max_splits=None):
    """Out-of-sample predictions: every row is predicted by a model that never saw its person/session."""
    n_groups = len(set(groups))
    n_splits = min(max_splits or config.ML_MAX_CV_SPLITS, n_groups)
    if n_splits < 2:
        raise DatasetError("At least 2 separate sessions (or people) are needed for a fair test.")
    probabilities = np.zeros(len(y))
    for train_index, test_index in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(set(y[train_index])) < 2:
            raise DatasetError(
                "One test round would train without any 'normal' or without any 'fatigued' rows, because the "
                "sessions are unevenly spread. Record 'normal' and 'fatigued' sessions with more than one person/session each."
            )
        try:
            model = build_pipeline().fit(X[train_index], y[train_index])
        except ValueError as error:
            raise DatasetError(f"Training in a test round failed ({error}). You probably need more sessions.")
        probabilities[test_index] = model.predict_proba(X[test_index])[:, list(model.classes_).index(1)]
    return probabilities, n_splits


def rule_predictions(rows):
    """What the rule-based mode would say for each row (window features only)."""
    strict, lenient = [], []
    for row in rows:
        eye = {
            "perclos": _none_if_nan(row["perclos"]),
            "longest_closure_s": _none_if_nan(row["longest_closure_s"]),
            "blinks_per_min": _none_if_nan(row["blinks_per_min"]),
            "mean_blink_duration_s": _none_if_nan(row["mean_blink_duration_s"]),
        }
        result = rules.classify(eye, include_current=False)
        strict.append(1 if result.state == rules.FATIGUED else 0)
        lenient.append(1 if result.state != rules.NORMAL else 0)
    return np.array(strict), np.array(lenient)


def _none_if_nan(value):
    return None if value != value else value


def train(dataset_paths, model_path=None, max_splits=None):
    """
    Train and save the model. Returns {"model_path", "report_path", "metadata", "report"}.
    Raises DatasetError if the data is not good enough.
    """
    rows, problems = load_rows(dataset_paths)
    if not rows:
        raise DatasetError("The data files contain no usable rows.")
    is_demo = any(r["source"] == "demo" for r in rows)
    if model_path is None:
        model_path = config.DEMO_FATIGUE_MODEL_FILE if is_demo else config.FATIGUE_MODEL_FILE
    model_path = Path(model_path)
    if is_demo and model_path == config.FATIGUE_MODEL_FILE:
        raise DatasetError("Synthetic DEMO data must never be saved as the real model file.")

    check_enough_data(rows)

    X = np.array([[r[name] for name in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "fatigued" else 0 for r in rows])
    groups, grouping = choose_groups(rows)

    probabilities, n_splits = cross_validate(X, y, groups, max_splits)
    cv_metrics = binary_metrics(y, (probabilities >= 0.5).astype(int))
    majority = int(y.mean() >= 0.5)
    majority_accuracy = float((y == majority).mean())
    strict, lenient = rule_predictions(rows)
    rules_strict = binary_metrics(y, strict)
    rules_lenient = binary_metrics(y, lenient)

    final = build_pipeline().fit(X, y)
    metadata = {
        "is_demo": is_demo,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "class_counts": {"normal": int((y == 0).sum()), "fatigued": int((y == 1).sum())},
        "n_sessions": len({r["session_id"] for r in rows}),
        "n_participants": len({r["participant"] for r in rows}),
        "skipped_rows": len(problems),
        "cv_grouping": grouping,
        "cv_splits": n_splits,
        "cross_validation": cv_metrics,
        "majority_class_accuracy": majority_accuracy,
        "rules_fatigued_only": rules_strict,
        "rules_warning_or_worse": rules_lenient,
        "sklearn_version": sklearn.__version__,
        "data_sha256": dataset_sha256(dataset_paths),
        "feature_names": FEATURE_NAMES,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "feature_names": FEATURE_NAMES, "metadata": metadata}, model_path)

    report = make_report(metadata, problems)
    report_path = model_path.with_suffix(".report.txt")
    report_path.write_text(report, encoding="utf-8")
    return {"model_path": model_path, "report_path": report_path, "metadata": metadata, "report": report}


def _fmt(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def make_report(meta, problems):
    cv, counts = meta["cross_validation"], meta["class_counts"]
    lines = []
    if meta["is_demo"]:
        lines += ["=" * 74, "DEMO DATA: these rows are SYNTHETIC (invented by ml/make_demo_data.py).",
                  "Every number below only shows that the code runs. It says NOTHING about real people.",
                  "=" * 74, ""]
    lines += [
        "Fatigue model training report",
        f"trained at:          {meta['trained_at']}",
        f"data fingerprint:    sha256 {meta['data_sha256'][:16]}...",
        f"rows:                {meta['n_rows']}  (normal {counts['normal']}, fatigued {counts['fatigued']}), "
        f"{meta['n_sessions']} sessions, {meta['n_participants']} participant(s), {meta['skipped_rows']} rows skipped",
        f"features:            {', '.join(meta['feature_names'])}",
        "",
        f"Cross-validation: {meta['cv_splits']} rounds, keeping whole {meta['cv_grouping']}s together "
        "(every row is predicted by a model that never saw its " + meta["cv_grouping"] + ").",
        f"  SVM      accuracy {_fmt(cv['accuracy'])}  precision {_fmt(cv['precision'])}  recall {_fmt(cv['recall'])}  "
        f"F1 {_fmt(cv['f1'])}   (confusion: TP {cv['tp']}, FP {cv['fp']}, FN {cv['fn']}, TN {cv['tn']})",
        f"  always guessing the bigger class:     accuracy {_fmt(meta['majority_class_accuracy'])}",
    ]
    for key, title in (("rules_fatigued_only", "rule-based mode, 'fatigued' only"),
                       ("rules_warning_or_worse", "rule-based mode, 'warning or worse'")):
        r = meta[key]
        lines.append(f"  {title:<38}accuracy {_fmt(r['accuracy'])}  precision {_fmt(r['precision'])}  recall {_fmt(r['recall'])}")
    lines += [
        "",
        "How to read this:",
        f"  * The SVM is only worth using if it clearly beats both baselines above on data like yours.",
        f"  * The test only shows how well it works on {meta['cv_grouping']}s like those in the data."
        + (" With fewer than 3 participants it says little about NEW people." if meta["cv_grouping"] == "session" else ""),
        "  * Labels are what participants reported, not a medical measurement. This is a monitoring aid, not a diagnosis.",
        f"  * In the app, states use probability thresholds ({config.ML_WARNING_PROBABILITY} warning, "
        f"{config.ML_FATIGUED_PROBABILITY} fatigued); the scores above use 0.5.",
    ]
    if problems:
        lines += ["", f"Skipped rows ({len(problems)}), first few:"] + [f"  {p}" for p in problems[:5]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the fatigue SVM (see the module docstring).")
    parser.add_argument("--data", nargs="+", help="CSV file(s) to train on (default: data/fatigue_features.csv)")
    parser.add_argument("--demo", action="store_true", help="train on the SYNTHETIC demo data (code test only)")
    args = parser.parse_args(argv)

    if args.demo:
        paths = [config.DEMO_DATASET_FILE]
        if not paths[0].is_file():
            print(f"Demo data not found. Create it first with: python -m ml.make_demo_data")
            return 1
    else:
        paths = [Path(p) for p in args.data] if args.data else [config.DATASET_FILE]
        if not paths[0].is_file():
            print(f"No data file found at {paths[0]}.")
            print("Record data first with scripts/collect_fatigue_data.py (see docs/fatigue_ml.md).")
            print("Until a model exists, the app uses the DEMO / RULE-BASED MODE.")
            return 1
    try:
        result = train(paths)
    except DatasetError as error:
        print(f"\nNOT TRAINED: {error}\n")
        return 1
    print(result["report"])
    print(f"Model saved to:  {result['model_path']}")
    print(f"Report saved to: {result['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
