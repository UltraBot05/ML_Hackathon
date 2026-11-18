"""Local API adapter for the frozen TA map-matching service."""

import json
import math
import os
from http.client import HTTPConnection, HTTPException as ReferenceProtocolError
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictBool


ENGINE = "ta-reference-lightgbm-24f"
MODEL = {
    "id": "best",
    "name": "TA reference LightGBM + HMM",
    "engine": ENGINE,
    "feature_schema": "ta-reference-24f",
    "feature_count": 24,
    "supports_no_hmm": False,
}
TRIP_ID_PATTERN = r"^[0-9A-Za-z_-]+$"
TRIP_FIELDS = ("trip_id", "n_obs", "n_nodes", "n_arcs", "duration_s")

app = FastAPI(title="PathSnap Adapter")


class PredictionRequest(BaseModel):
    trip_id: str = Field(pattern=TRIP_ID_PATTERN, max_length=64)
    model: Literal["best"] = "best"
    use_hmm: StrictBool = True


def _partition() -> str:
    partition = os.getenv("PATHSNAP_PARTITION", "student_test")
    if partition not in {"student_test", "evaluator_test"}:
        raise HTTPException(503, "PATHSNAP_PARTITION must be student_test or evaluator_test.")
    return partition


def _trip_root() -> Path:
    root = os.getenv("PATHSNAP_TRIP_ROOT")
    if not root or not Path(root).is_dir():
        raise HTTPException(503, "PATHSNAP_TRIP_ROOT must point to the selected partition directory.")
    return Path(root).resolve()


def _reference_request(path: str, payload: dict | None = None) -> dict:
    try:
        url = urlsplit(os.getenv("PATHSNAP_REFERENCE_URL", "http://127.0.0.1:8001"))
        port = url.port
        host = url.hostname
        if (
            url.scheme != "http"
            or host not in {"127.0.0.1", "localhost", "::1"}
            or url.username is not None
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
            or port == 0
        ):
            raise ValueError
    except ValueError as exc:
        raise HTTPException(503, "PATHSNAP_REFERENCE_URL must be a loopback HTTP origin.") from exc

    try:
        timeout = float(os.getenv("PATHSNAP_REFERENCE_TIMEOUT_SECONDS", "120"))
        if not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError
    except ValueError as exc:
        raise HTTPException(503, "PATHSNAP_REFERENCE_TIMEOUT_SECONDS must be between 0 and 3600.") from exc

    # Direct HTTP keeps this request on loopback and ignores proxy and redirect settings.
    connection = HTTPConnection("127.0.0.1" if host == "localhost" else host, port or 80, timeout=timeout)
    try:
        if payload is None:
            connection.request("GET", path)
        else:
            connection.request("POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status == 404 and payload is not None:
            raise HTTPException(404, "Trip is unavailable from the reference service.")
        if response.status != 200:
            raise HTTPException(502, f"Reference service returned HTTP {response.status}.")
        result = json.loads(response.read())
    except TimeoutError as exc:
        raise HTTPException(504, "Reference service timed out.") from exc
    except (OSError, ReferenceProtocolError) as exc:
        raise HTTPException(502, "Could not reach the reference service.") from exc
    except ValueError as exc:
        raise HTTPException(502, "Reference service returned invalid JSON.") from exc
    finally:
        connection.close()

    if not isinstance(result, dict):
        raise HTTPException(502, "Reference service returned an unexpected response.")
    return result


def _read_trip_rows(path: Path, *converters: type) -> list[tuple]:
    try:
        # Require the named input itself, even when the trip root contains symlinks.
        if path.resolve() != path:
            raise HTTPException(404, "Trip input files are unavailable.")
        with path.open(encoding="utf-8") as input_file:
            return [
                tuple(convert(value) for convert, value in zip(converters, line.split(), strict=True))
                for line in input_file
                if line.strip()
            ]
    except FileNotFoundError as exc:
        raise HTTPException(404, "Trip input files are unavailable.") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(503, "Could not read the configured trip input files.") from exc


def _valid_arc_ids(ids: object, n_arcs: int, *, minimum: int = 0) -> bool:
    return isinstance(ids, list) and all(type(arc_id) is int and minimum <= arc_id < n_arcs for arc_id in ids)


@app.get("/health")
def health():
    partition = _partition()
    _trip_root()
    reference = _reference_request("/api/health")
    partitions = reference.get("partitions")
    if reference.get("status") != "ok" or not isinstance(partitions, list) or partition not in partitions:
        raise HTTPException(503, "Reference service is not ready for the configured partition.")
    return {"status": "ok", "engine": ENGINE, "model": MODEL["id"], "partition": partition}


@app.get("/models")
def models():
    return {"models": [MODEL], "default_model": MODEL["id"]}


@app.get("/trips")
def trips():
    partition = _partition()
    reference = _reference_request(f"/api/trips?partition={partition}")
    trips = reference.get("trips")
    if reference.get("partition") != partition or not isinstance(trips, list) or any(
        not isinstance(trip, dict) or any(key not in trip for key in TRIP_FIELDS) for trip in trips
    ):
        raise HTTPException(502, "Reference service returned an invalid trip catalog.")
    return {"partition": partition, "trips": [{key: trip[key] for key in TRIP_FIELDS} for trip in trips]}


@app.post("/predict")
def predict(request: PredictionRequest):
    if not request.use_hmm:
        raise HTTPException(422, "The TA reference engine requires use_hmm=true.")

    partition = _partition()
    trip_id = request.trip_id
    trip_path = _trip_root() / trip_id / trip_id
    nodes = _read_trip_rows(trip_path.with_suffix(".nodes"), float, float)
    arcs = _read_trip_rows(trip_path.with_suffix(".arcs"), int, int)
    track = _read_trip_rows(trip_path.with_suffix(".track"), float, float, float)
    if not nodes or not arcs or not track or any(
        not math.isfinite(value) for rows in (nodes, track) for row in rows for value in row
    ) or any(not 0 <= node < len(nodes) for arc in arcs for node in arc):
        raise HTTPException(503, "Configured trip inputs contain invalid coordinates or node IDs.")

    reference = _reference_request("/api/predict", {
        "partition": partition,
        "trip_id": trip_id,
        "generate_map": False,
    })
    route = reference.get("reconstructed_route")
    inference_s = reference.get("inference_time_s")
    metrics = reference.get("metrics")
    if (
        reference.get("trip_id") != trip_id
        or reference.get("partition") != partition
        or reference.get("n_obs", len(track)) != len(track)
        or not _valid_arc_ids(route, len(arcs))
        or type(inference_s) not in (int, float)
        or not math.isfinite(inference_s * 1000)
        or inference_s < 0
        or (metrics is not None and not isinstance(metrics, dict))
    ):
        raise HTTPException(502, "Reference service returned an invalid prediction for the trip inputs.")

    result = {
        "trip_id": trip_id,
        "partition": partition,
        "model": request.model,
        "engine": ENGINE,
        "feature_schema": MODEL["feature_schema"],
        "feature_count": MODEL["feature_count"],
        "use_hmm": request.use_hmm,
        "n_obs": len(track),
        "gps_coordinates": [(lon, lat) for lon, lat, _ in track],
        "predicted_arc_ids": route,
        "route_segments": [(nodes[arcs[arc_id][0]], nodes[arcs[arc_id][1]]) for arc_id in route],
        "inference_ms": inference_s * 1000,
        "metrics": metrics,
        "warnings": [],
    }
    if "observation_arc_ids" in reference:
        observations = reference["observation_arc_ids"]
        if not _valid_arc_ids(observations, len(arcs), minimum=-1) or len(observations) != len(track):
            raise HTTPException(502, "Reference service returned invalid per-observation arc IDs.")
        result["observation_arc_ids"] = observations
    else:
        result["warnings"].append("The reference service does not expose per-observation arc IDs.")
    return result
