# PathSnap final integration

The team backend calls a separately provisioned, local TA reference engine. The engine keeps its frozen 24-feature LightGBM, hybrid HMM, and predicted-route boundary realignment. The adapter is team-owned. It validates the response and resolves directed arc IDs against the selected trip's road network. The model and its training provenance belong to the TA reference package.

## Run the two processes

Start the authorized reference engine from its own checkout and Python environment. Bind that process to loopback.

```powershell
python -B -m uvicorn src.api.server:app --host 127.0.0.1 --port 8001
```

Start this project's adapter from the team checkout root in an environment with FastAPI and Uvicorn. The selected directory must contain the same trip files as the engine's selected partition. The adapter reads `.track`, `.nodes`, and `.arcs` for geometry. In the reference engine, raw predictive inference runs on those three files and uses the reference route only for subsequent scoring.

```powershell
$env:PATHSNAP_REFERENCE_URL = 'http://127.0.0.1:8001'
$env:PATHSNAP_PARTITION = 'student_test'
$env:PATHSNAP_TRIP_ROOT = (Resolve-Path 'Dataset/student_test').Path
python -B -m uvicorn server.app:app --host 10.92.151.151 --port 8000
```

For a locally authorized lecturer run, set `PATHSNAP_PARTITION=evaluator_test` and point `PATHSNAP_TRIP_ROOT` to its matching external directory. Use loopback for local verification. The reference URL accepts only a loopback HTTP origin. `PATHSNAP_REFERENCE_TIMEOUT_SECONDS` defaults to 120. Start Aadi's client with `python -m streamlit run evaluator_ui/app.py` on the Mac and enter the Omen's ZeroTier URL.

## API contract

| Request | Response |
| --- | --- |
| `GET /health` | Readiness, engine ID, and configured partition. |
| `GET /models` | Available selector IDs, engine, and 24-feature schema. |
| `GET /trips` | Trip catalog for the server's configured partition. |
| `POST /predict` | A real route prediction. Body is `{"trip_id":"00000000","model":"best","use_hmm":true}`. |

`/predict` returns the actual `engine` and `feature_schema`, `predicted_arc_ids` in reconstructed order, endpoint pairs in `route_segments`, and original `gps_coordinates`. Both coordinate arrays use `[lon, lat]`. Streamlit converts these to Folium's `[lat, lon]`. Returned `metrics` are the engine's post-inference scores, and `inference_ms` is measured engine inference time. The 24-feature engine requires HMM. `observation_arc_ids` is returned only if supplied by the engine. The current reference HTTP endpoint exposes reconstructed route IDs, so the response carries a warning and omits per-observation IDs.

Run the adapter tests with `python -B -m unittest -v server.test_app`. Production deployment also needs the separately authorized reference runtime and assets provisioned outside this repository. A self-contained replacement needs team-owned training artifacts, a matching feature extractor and decoder, and fresh independent verification on both datasets.
