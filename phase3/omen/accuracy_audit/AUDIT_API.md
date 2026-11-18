# PathSnap accuracy audit API

Run from the PathSnap checkout root. Use the same Python environment as the fitted classifier, with FastAPI and Uvicorn available.

```powershell
$env:PATHSNAP_TRIP_ROOT = (Resolve-Path 'Dataset/train').Path
$env:PATHSNAP_MODEL_ARCHIVE = (Resolve-Path 'phase3/omen/accuracy_audit/results/xgboost_baseline.zip').Path
$env:PATHSNAP_MODEL = 'xgboost'
python -B phase3/omen/accuracy_audit/audit_api.py
```

The API binds to `127.0.0.1:8000` by default. Set `PATHSNAP_HOST` to the Omen’s ZeroTier IPv4 address for Aadi’s Mac, and use that address as the base URL. `PATHSNAP_PORT` defaults to 8000. `PATHSNAP_CORS_ORIGINS` accepts comma-separated origins for direct browser clients. A local evaluator can set `PATHSNAP_TRIP_ROOT` to a different partition. Each trip is an eight-digit directory containing `<trip_id>.track`, `<trip_id>.nodes`, and `<trip_id>.arcs`.

`PATHSNAP_MODEL_ARCHIVE` selects a trusted experiment ZIP. Each selectable model has `<model>/model.joblib` and `<model>/results.json` in the archive. The report's `feature_order` must match the canonical Phase 2 features and the estimator must support `predict_proba` for classes `[0, 1]`. Use `PATHSNAP_MODEL` to set the default. With no preference configured, the API selects XGBoost when present or the first available model in the team registry. Restart after switching archives. `PATHSNAP_REPO_ROOT` overrides the checkout path when running the API files from another directory.

## `GET /health`

```json
{
  "status": "ok",
  "available_models": ["xgboost"],
  "default_model": "xgboost",
  "trip_root_available": true
}
```

`available_models` lists paired artifact and report entries for known team models. The server validates and loads a model on its first prediction request. A `null` default means callers need an available explicit model or a configuration change.

## `GET /trips`

```json
{"trip_ids": ["00000005"]}
```

Trip IDs are strings sorted in ascending order. Use `.track`, `.nodes`, and `.arcs` from the folder of the same name.

## `POST /predict`

Send JSON with a `trip_id` and an optional `model` from `/health`.

```json
{"trip_id": "00000005", "model": "xgboost"}
```

| Response field | Meaning |
| --- | --- |
| `trip_id`, `model` | Requested trip and classifier used. |
| `observation_arc_ids` | One HMM arc ID per parsed GPS row, in file order. `-1` denotes an unmatched observation. |
| `route_arc_ids` | Directed reconstructed arc IDs in traversal order after collapsing repeats and inserting short graph bridges. |
| `route_segments` | Entries in the same order as `route_arc_ids`, shaped as `{"arc_id": 42, "coordinates": [[lon, lat], [lon, lat]]}`. The first pair is the road arc's source node and the second is its destination. |
| `gps_coordinates` | `[lon, lat]` coordinates in `.track` file order, aligned with `observation_arc_ids`. |
| `continuity_pct` | Share of adjacent reconstructed arcs that connect in the directed graph, as a percentage. The Phase 2 reconstructor reports 100 for sequences with fewer than two arcs. |
| `timings_ms` | `preprocessing`, `classifier`, `hmm`, `reconstruction`, and `total` wall-clock durations in milliseconds. These exclude one-time artifact loading and HTTP overhead. |

Draw each arc as a separate line segment to preserve gaps. Convert `[lon, lat]` to the coordinate order required by the map library.

Errors use FastAPI's `detail` field. `404` means the trip, its raw files, or the requested model is unavailable. `422` means invalid request data or unusable raw trip data. `503` means the configured trip root or default model is unavailable, or the artifact could not be loaded.

Prediction uses the canonical Phase 2 route-free features, the ML-emission Viterbi decoder from `768a80b6100ca4332ce7d27b5736b64ed64cbac4`, and Phase 2 route reconstruction. Continuity measures predicted arcs against the road graph. The API exposes only raw GPS and generated predictions for the selected trip.

Live HTTP verification from the Omen to its own ZeroTier listener returned 200 on a public trip with 1,045 observations and 105 reconstructed arcs at 100% directed continuity. A separate loopback check against the private partition discovered ten trips and returned 200 for one raw-trip prediction. It used only the three inference files and did not score or retain the response.

From the repository root, run the API contract tests with `python -B -m unittest discover -s phase3/omen/accuracy_audit -p test_audit_api.py -v`.
