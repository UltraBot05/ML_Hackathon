"""Loading contract for trusted Phase 3 candidate classifiers."""

import io
import json
from pathlib import Path
from zipfile import ZipFile

import joblib


TEAM_MODELS = ("logistic", "xgboost", "extra_trees", "lightgbm", "catboost", "random_forest")


def load_classifier(name, archive_paths, expected_features):
    """Load a configured model plus its canonical ordered feature columns."""
    if name not in TEAM_MODELS:
        raise ValueError(f"Unknown team model {name!r}")
    if name not in archive_paths:
        raise FileNotFoundError(f"No artifact is configured for {name}")
    with ZipFile(Path(archive_paths[name])) as archive:
        report = json.loads(archive.read(f"{name}/results.json"))
        features = report["feature_order"]
        if features != list(expected_features):
            raise ValueError(f"{name} does not use the canonical Phase 2 feature order")
        estimator = joblib.load(io.BytesIO(archive.read(f"{name}/model.joblib")))
    if list(getattr(estimator, "feature_names_in_", features)) != features:
        raise ValueError(f"{name} artifact feature order disagrees with its report")
    if not callable(getattr(estimator, "predict_proba", None)) or list(estimator.classes_) != [0, 1]:
        raise ValueError(f"{name} must supply class 0/1 probabilities")
    return estimator, features
