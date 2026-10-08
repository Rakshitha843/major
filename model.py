"""
Loading and using the TRAINED fatigue model (an SVM, as in the base paper).

SECURITY NOTE: model files are read with joblib, which can run code stored inside a file.
Only ever load model files that YOU trained (ml/train.py) on your own computer.

A model trained on the synthetic DEMO data is saved under a different file name and is refused here,
so the application can never present a demo model as a real one.
"""

from dataclasses import dataclass, field

import joblib
import numpy as np
import sklearn

from ml.features import FEATURE_NAMES, feature_vector
from utils import config


class ModelLoadError(Exception):
    """The trained model is missing or unusable. The message is written for the user."""


@dataclass
class LoadedModel:
    pipeline: object
    metadata: dict
    warning: str = ""
    _fatigue_column: int = field(init=False, default=1)

    def __post_init__(self):
        self._fatigue_column = list(self.pipeline.classes_).index(1)      # class 1 = "fatigued"

    def fatigue_probability(self, values):
        """values: {feature name: number or None}. Returns the model's probability (0-1) of 'fatigued'."""
        vector = np.array([feature_vector(values)], dtype=float)         # missing values become NaN
        return float(self.pipeline.predict_proba(vector)[0][self._fatigue_column])

    def summary(self):
        """Facts about the training, taken from the model file. Contains no invented numbers."""
        meta = self.metadata
        cv = meta.get("cross_validation") or {}
        return {
            "trained_at": meta.get("trained_at"),
            "rows": meta.get("n_rows"),
            "class_counts": meta.get("class_counts"),
            "sessions": meta.get("n_sessions"),
            "participants": meta.get("n_participants"),
            "cv_grouping": meta.get("cv_grouping"),
            "cv_accuracy": cv.get("accuracy"),
            "cv_precision": cv.get("precision"),
            "cv_recall": cv.get("recall"),
        }


def load_model(path=None):
    """Load the trained model or raise ModelLoadError with an explanation."""
    path = config.FATIGUE_MODEL_FILE if path is None else path
    if not path.is_file():
        raise ModelLoadError(
            "No trained model found, so DEMO / RULE-BASED MODE is used. To create one: collect data with "
            "scripts/collect_fatigue_data.py, then run: python -m ml.train  (see docs/fatigue_ml.md)."
        )
    try:
        bundle = joblib.load(path)
        pipeline, metadata, names = bundle["pipeline"], bundle["metadata"], bundle["feature_names"]
    except Exception as error:
        raise ModelLoadError(
            f"The trained model file could not be read ({error}). Retrain it with: python -m ml.train"
        )

    if metadata.get("is_demo"):
        raise ModelLoadError("This model was trained on synthetic DEMO data and is never used by the application.")
    if list(names) != FEATURE_NAMES:
        raise ModelLoadError("The trained model uses different features than this version of the program. Retrain it with: python -m ml.train")
    try:
        classes = list(pipeline.classes_)
    except AttributeError:
        classes = []
    if 0 not in classes or 1 not in classes:
        raise ModelLoadError("The trained model file is not a two-class fatigue model. Retrain it with: python -m ml.train")

    warning = ""
    trained_with = metadata.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        warning = (f"The model was trained with scikit-learn {trained_with} but {sklearn.__version__} is installed; "
                   "if results look wrong, retrain it.")
    return LoadedModel(pipeline=pipeline, metadata=metadata, warning=warning)


def describe_model(path=None):
    """What the dashboard shows about the model: {"available", "message", "summary"}."""
    try:
        model = load_model(path)
    except ModelLoadError as error:
        return {"available": False, "message": str(error), "summary": None}
    return {"available": True, "message": model.warning, "summary": model.summary()}
