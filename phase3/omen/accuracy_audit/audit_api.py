"""Serve PathSnap routes from raw, unlabeled trips and a trained candidate classifier."""

from functools import cache
import logging
import math
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from zipfile import BadZipFile, ZipFile

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field

REPO_ROOT = Path(os.environ.get("PATHSNAP_REPO_ROOT", Path(__file__).resolve().parents[3])).resolve()
sys.path[:0] = [str(REPO_ROOT / "phase2/scripts"), str(REPO_ROOT / "phase3/omen")]

from audit_hmm import MLEmissionHMMMatcher
from candidate_generation import CandidateGenerator
from data_loader import DataLoader, TripData
from feature_engineering import FeatureEngineer
from model_registry import TEAM_MODELS, load_classifier
from road_graph import RoadNetworkGraph
from route_reconstruction import RouteReconstructor
from spatial_index import RoadSpatialIndex
from trajectory_processor import TrajectoryProcessor


TRIP_ID_PATTERN = r"^[0-9]{8}$"
RAW_EXTENSIONS = ("track", "nodes", "arcs")


class PredictionRequest(BaseModel):
    trip_id: str = Field(pattern=TRIP_ID_PATTERN)
    model: str | None = None


def has_raw_files(path: Path) -> bool:
    return path.is_dir() and all((path / f"{path.name}.{ext}").is_file() for ext in RAW_EXTENSIONS)


def discover_models(archive_path: Path) -> list[str]:
    try:
        with ZipFile(archive_path) as archive:
            members = set(archive.namelist())
    except (OSError, BadZipFile):
        return []
    return [name for name in TEAM_MODELS if {
        f"{name}/model.joblib", f"{name}/results.json",
    } <= members]


def run_prediction(path: Path, estimator, feature_order: list[str]) -> dict:
    start = perf_counter()
    trip_id = path.name
    loader = DataLoader(path.parent)
    try:
        trip = TripData(
            trip_id=trip_id,
            partition=path.parent.name,
            track=loader.load_track(path / f"{trip_id}.track"),
            nodes=loader.load_nodes(path / f"{trip_id}.nodes"),
            arcs=loader.load_arcs(path / f"{trip_id}.arcs"),
            route=[],
        )
    except (OSError, ValueError, OverflowError):
        raise HTTPException(422, "Unable to parse the raw trip files.") from None

    if not trip.track or not trip.nodes or not trip.arcs or not (
        all(math.isfinite(p.timestamp) and math.isfinite(p.lon) and math.isfinite(p.lat)
            for p in trip.track)
        and all(math.isfinite(n.lon) and math.isfinite(n.lat) for n in trip.nodes)
    ):
        raise HTTPException(422, "Trip has missing or invalid raw observations or road data.")

    observations = TrajectoryProcessor().process_trip_track(trip_id, trip.track)
    graph = RoadNetworkGraph(trip_id, trip.nodes, trip.arcs)
    if not graph.arcs_dict:
        raise HTTPException(422, "Trip has no usable road arcs.")
    candidates, _ = CandidateGenerator().generate_candidates_for_trip(
        trip, observations, graph, RoadSpatialIndex(graph),
    )
    frame = FeatureEngineer().compute_candidate_features(candidates, observations, graph)
    features = frame[feature_order].astype(np.float32) if not frame.empty else frame
    features_end = perf_counter()

    probs = estimator.predict_proba(features)[:, 1] if not frame.empty else np.empty(0)
    classifier_end = perf_counter()

    # Feature rows define probability order. The HMM expects dense observation positions.
    positions = {obs.obs_index: position for position, obs in enumerate(observations)}
    ordered_candidates = [
        {"obs_index": positions[idx], "candidate_arc_id": arc_id}
        for idx, arc_id in zip(frame.get("obs_index", []), frame.get("candidate_arc_id", []))
    ]
    matches = MLEmissionHMMMatcher().match_trip(ordered_candidates, probs, observations, graph)
    hmm_end = perf_counter()

    reconstructor = RouteReconstructor()
    collapsed = reconstructor.collapse_observation_sequence(matches)
    route = reconstructor.reconstruct_and_bridge_route(collapsed, graph)
    continuity_pct, _, _ = reconstructor.evaluate_continuity(route, graph)
    segments = []
    for arc_id in route:
        arc = graph.arcs_dict[arc_id]
        start_node = graph.nodes_dict[arc.from_node]
        end_node = graph.nodes_dict[arc.to_node]
        segments.append({"arc_id": arc_id, "coordinates": [
            [start_node.lon, start_node.lat], [end_node.lon, end_node.lat],
        ]})
    gps = [[p.lon, p.lat] for p in trip.track]
    end = perf_counter()

    return {
        "observation_arc_ids": matches,
        "route_arc_ids": route,
        "route_segments": segments,
        "gps_coordinates": gps,
        "continuity_pct": continuity_pct,
        "timings_ms": {
            "preprocessing": round((features_end - start) * 1000, 3),
            "classifier": round((classifier_end - features_end) * 1000, 3),
            "hmm": round((hmm_end - classifier_end) * 1000, 3),
            "reconstruction": round((end - hmm_end) * 1000, 3),
            "total": round((end - start) * 1000, 3),
        },
    }


def create_app() -> FastAPI:
    trip_root = Path(os.environ.get("PATHSNAP_TRIP_ROOT", REPO_ROOT / "Dataset/train")).resolve()
    archive_path = Path(os.environ.get(
        "PATHSNAP_MODEL_ARCHIVE", REPO_ROOT / "phase3/omen/review1/experiment.zip",
    )).resolve()
    available_models = discover_models(archive_path)
    preferred_model = "xgboost" if "xgboost" in available_models else next(iter(available_models), None)
    selected_model = os.environ.get("PATHSNAP_MODEL", preferred_model)
    default_model = selected_model if selected_model in available_models else None

    # The Phase 2 parsers put filenames in warnings about malformed rows.
    logging.getLogger("DataLoader").setLevel(logging.ERROR)

    @cache
    def get_classifier(name: str):
        return load_classifier(name, {name: archive_path}, FeatureEngineer.FEATURE_COLUMNS)

    app = FastAPI(title="PathSnap Accuracy Audit")
    origins = [origin.strip() for origin in os.environ.get("PATHSNAP_CORS_ORIGINS", "").split(",")
               if origin.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins,
                           allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "available_models": available_models,
            "default_model": default_model,
            "trip_root_available": trip_root.is_dir(),
        }

    @app.get("/trips")
    def trips():
        if not trip_root.is_dir():
            raise HTTPException(503, "Configured trip root is unavailable.")
        return {"trip_ids": sorted(
            path.name for path in trip_root.iterdir()
            if re.fullmatch(TRIP_ID_PATTERN, path.name) and has_raw_files(path)
        )}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        if not trip_root.is_dir():
            raise HTTPException(503, "Configured trip root is unavailable.")
        trip_path = trip_root / request.trip_id
        if not has_raw_files(trip_path):
            raise HTTPException(404, "Unknown trip ID or incomplete raw files.")
        model = request.model if request.model is not None else default_model
        if model is None:
            raise HTTPException(503, "No default model is available.")
        if model not in available_models:
            raise HTTPException(404, "Requested model is unavailable.")
        try:
            estimator, feature_order = get_classifier(model)
        except Exception:
            raise HTTPException(503, "Unable to load the configured model artifact.") from None
        prediction = run_prediction(trip_path, estimator, feature_order)
        return {"trip_id": request.trip_id, "model": model, **prediction}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("PATHSNAP_HOST", "127.0.0.1"),
                port=int(os.environ.get("PATHSNAP_PORT", "8000")), access_log=False)
