"""PathSnap API while the team's route decoder is being integrated."""

import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


MODEL_NAMES = ("logistic", "xgboost", "extra_trees", "lightgbm", "random_forest", "catboost")


class PredictionRequest(BaseModel):
    trip_id: str = Field(pattern=r"^[0-9]{8}$")


def load_classifier(path):
    """Load a trusted experiment archive and its ordered feature contract."""
    with ZipFile(path) as archive:
        features = json.loads(archive.read("results.json"))["features"]
        model = joblib.load(io.BytesIO(archive.read("model.joblib")))
    if (not features or any(not isinstance(name, str) for name in features)
            or len(set(features)) != len(features)):
        raise ValueError(f"Invalid feature contract in {path}")
    if list(getattr(model, "feature_names_in_", features)) != features:
        raise ValueError(f"Feature order does not match the fitted model in {path}")
    if not callable(getattr(model, "predict_proba", None)) or list(model.classes_) != [0, 1]:
        raise ValueError(f"Expected a binary probability classifier in {path}")
    return model, features


def create_app():
    root = Path(os.environ.get("PATHSNAP_DATA_ROOT", "Student folder/Student folder"))
    split = os.environ.get("PATHSNAP_SPLIT", "evaluator_test")
    if split not in {"evaluator_test", "student_test"}:
        raise ValueError("PATHSNAP_SPLIT must be evaluator_test or student_test.")
    partition = root / split
    configured = json.loads(os.environ.get("PATHSNAP_MODELS_JSON", "{}"))
    if not isinstance(configured, dict) or configured.keys() - set(MODEL_NAMES):
        raise ValueError("PATHSNAP_MODELS_JSON must map supported model names to artifact paths.")
    registry = {name: Path(configured[name]) if name in configured else None for name in MODEL_NAMES}
    app = FastAPI(title="PathSnap")
    origins = [origin.strip() for origin in os.environ.get("PATHSNAP_CORS_ORIGINS", "").split(",")
               if origin.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins,
                           allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "trip_partition_available": partition.is_dir(),
            "prediction_ready": False,
            "models": {
                name: {"artifact_available": path is not None and path.is_file()}
                for name, path in registry.items()
            },
        }

    @app.get("/trips")
    def trips():
        if not partition.is_dir():
            raise HTTPException(status_code=503, detail="Configured trip partition is unavailable.")
        return {"trip_ids": sorted(
            path.name for path in partition.iterdir()
            if path.is_dir() and len(path.name) == 8 and path.name.isascii() and path.name.isdecimal()
            and (path / f"{path.name}.track").is_file()
        )}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        if not partition.is_dir():
            raise HTTPException(status_code=503, detail="Configured trip partition is unavailable.")
        if not (partition / request.trip_id / f"{request.trip_id}.track").is_file():
            raise HTTPException(status_code=404, detail="Unknown trip ID.")
        raise HTTPException(status_code=503, detail="Route prediction pipeline is not ready.")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host=os.environ.get("PATHSNAP_HOST", "0.0.0.0"),
        port=int(os.environ.get("PATHSNAP_PORT", "8000")),
    )
