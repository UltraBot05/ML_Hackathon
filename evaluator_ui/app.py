"""PathSnap Hackathon — Streamlit evaluator for the Omen prediction API."""

import json
import math
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_API_URL = os.environ.get("PATHSNAP_API_URL", "http://10.92.151.151:8000")

st.set_page_config(
    page_title="PathSnap — Evaluator UI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .status-badge-ok {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .status-badge-err {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


def api_json(api_url, endpoint, *, payload=None):
    """Fetch an API response with bounded connection and read timeouts."""
    if payload is None:
        response = requests.get(f"{api_url}{endpoint}", timeout=(2, 5))
    else:
        response = requests.post(f"{api_url}{endpoint}", json=payload, timeout=(2, 120))
    response.raise_for_status()
    return response.json()


def request_error(error):
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return f"HTTP {error.response.status_code}"
    if isinstance(error, requests.Timeout):
        return "the server timed out"
    if isinstance(error, requests.ConnectionError):
        return "could not connect to the server"
    return "the server returned an invalid response"


def catalog_ids(payload, envelope_keys, item_keys):
    """Accept a bare ID list, an envelope, or a name-keyed model catalog."""
    if isinstance(payload, dict):
        for key in envelope_keys:
            if key in payload:
                payload = payload[key]
                break

    if isinstance(payload, dict):
        return list(payload)
    if not isinstance(payload, list):
        raise ValueError("Expected a list or catalog from the API.")

    ids = []
    for item in payload:
        if isinstance(item, dict):
            item = next((item[key] for key in item_keys if key in item), None)
        if isinstance(item, str) and item:
            ids.append(item)
    return list(dict.fromkeys(ids))


def lat_lon(point):
    """Convert a valid API longitude-latitude pair for Folium."""
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    lon, lat = point
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) for value in point):
        return None
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None
    return (lat, lon)


def prediction_geometry(prediction):
    gps = []
    for point in prediction.get("gps_coordinates") or []:
        coord = lat_lon(point)
        if coord is not None:
            gps.append(coord)

    segments = []
    for segment in prediction.get("route_segments") or []:
        if isinstance(segment, dict):
            segment = segment.get("coordinates")
        if isinstance(segment, (list, tuple)) and len(segment) >= 2:
            coords = [lat_lon(point) for point in segment]
            if None not in coords:
                segments.append(coords)
    return gps, segments


def show_prediction_map(prediction):
    gps, segments = prediction_geometry(prediction)
    bounds = gps + [point for segment in segments for point in segment]
    if not bounds:
        st.info("The server returned no GPS points or route geometry for this trip.")
        return

    try:
        import folium
    except (ImportError, ModuleNotFoundError):
        st.error("The map requires Folium from the project dependencies. Install via: pip install folium streamlit-folium")
        return

    if prediction.get("partition") == "evaluator_test":
        m = folium.Map(location=bounds[0], zoom_start=15, tiles=None)
    else:
        m = folium.Map(
            location=bounds[0],
            zoom_start=15,
            tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            attr="© OpenStreetMap contributors © CARTO",
        )

    fg_gps = folium.FeatureGroup(name="Raw GPS Observations (Blue)", show=True)
    for point in gps:
        folium.CircleMarker(
            location=point, radius=3, color="#1f77b4", fill=True, fill_opacity=0.8,
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    fg_pred = folium.FeatureGroup(name="Predicted Route (Red)", show=True)
    for segment in segments:
        folium.PolyLine(
            locations=segment, color="#d62728", weight=5, opacity=0.85,
            tooltip="Predicted Route Segment",
        ).add_to(fg_pred)
    fg_pred.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    if len(bounds) > 1:
        m.fit_bounds(bounds)

    try:
        from streamlit_folium import st_folium
        st_folium(m, height=550, use_container_width=True)
    except Exception:
        import streamlit.components.v1 as components
        components.html(m.get_root().render(), height=550)


def show_prediction_metrics(prediction):
    st.subheader("Performance Metrics Summary")
    elapsed = prediction.get("inference_ms")
    elapsed_text = "Unavailable"
    if isinstance(elapsed, (int, float)) and math.isfinite(elapsed):
        elapsed_text = f"{elapsed:,.2f} ms"
    matches = prediction.get("observation_arc_ids")
    route = prediction.get("predicted_arc_ids") or []
    gps = prediction.get("gps_coordinates") or []

    columns = st.columns(4)
    columns[0].metric("GPS Observations", len(gps))
    columns[1].metric(
        "Matched Observations",
        sum(arc_id not in (None, -1) for arc_id in matches) if isinstance(matches, list) else "Not exposed by API",
    )
    columns[2].metric("Predicted Route Arcs", len(route))
    columns[3].metric("Inference Latency", elapsed_text)

    metrics = prediction.get("metrics")
    if isinstance(metrics, dict) and metrics:
        headline = (
            ("Directed Top-1", "top1_acc_pct"),
            ("Route Continuity", "seq_continuity_pct"),
            ("Directed Route F1", "route_arc_f1_pct"),
        )
        cols = st.columns(3)
        for column, (label, key) in zip(cols, headline):
            value = metrics.get(key)
            column.metric(label, f"{value:.2f}%" if type(value) in (int, float) and math.isfinite(value)
                          else "Unavailable")
        with st.expander("Full Server Metrics"):
            st.json(metrics)
    else:
        st.info("Scoring metrics are unavailable for this trip.")
    if prediction.get("feature_schema"):
        with st.expander("Feature Schema"):
            st.write(prediction["feature_schema"])
    with st.expander("Predicted Route Arc IDs"):
        st.json(route)


def main():
    st.title("🗺️ PathSnap — Map-Matching & Route Reconstruction Evaluator")
    st.caption("PES University PathSnap Hackathon | Abhigyan Dutta · Aadi Basal · Adarsh Prakash")

    st.sidebar.header("⚙️ System Architecture & Connectivity")
    api_url = st.sidebar.text_input(
        "Omen ZeroTier Server URL",
        value=DEFAULT_API_URL,
        help="ZeroTier endpoint for remote model serving.",
    ).strip().rstrip("/")

    trip_ids = []
    try:
        api_json(api_url, "/health")
    except (requests.RequestException, ValueError) as error:
        st.sidebar.markdown(
            '<span class="status-badge-err">🔴 OMEN SERVER UNREACHABLE</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.error(f"Health check failed because {request_error(error)}.")
    else:
        st.sidebar.markdown(
            '<span class="status-badge-ok">🟢 OMEN SERVER ONLINE</span>',
            unsafe_allow_html=True,
        )
        try:
            trip_ids = catalog_ids(api_json(api_url, "/trips"), ("trip_ids", "trips"), ("trip_id", "id"))
            if not trip_ids:
                st.sidebar.info("The server has no trips available.")
        except (requests.RequestException, ValueError) as error:
            st.sidebar.error(f"Trip catalog request failed ({request_error(error)}).")

        try:
            models = catalog_ids(
                api_json(api_url, "/models"), ("models", "available_models"), ("name", "id", "model"),
            )
            if models:
                st.sidebar.caption("Server model catalog · " + ", ".join(models))
        except (requests.RequestException, ValueError) as error:
            st.sidebar.error(f"Model catalog request failed ({request_error(error)}).")

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Evaluation Target")
    selected_trip = st.sidebar.selectbox("Select Trip ID", trip_ids, disabled=not trip_ids)
    st.sidebar.caption("The server selects its best model and runs HMM decoding.")
    run_btn = st.sidebar.button(
        "🚀 Run Map-Matching Evaluation",
        type="primary",
        width="stretch",
        disabled=not selected_trip,
    )

    prediction = None
    selection = (api_url, selected_trip)
    if st.session_state.get("prediction_selection") == selection:
        prediction = st.session_state.get("prediction")
    else:
        st.session_state.pop("prediction", None)

    if run_btn and selected_trip:
        prediction = None
        st.session_state.pop("prediction", None)
        try:
            with st.spinner(f"Running map matching on Omen for trip {selected_trip}…"):
                result = api_json(
                    api_url, "/predict",
                    payload={"trip_id": selected_trip, "model": "best", "use_hmm": True},
                )
                if (
                    not isinstance(result, dict)
                    or result.get("trip_id") != selected_trip
                    or any(not isinstance(result.get(key), list)
                           for key in ("gps_coordinates", "predicted_arc_ids", "route_segments"))
                ):
                    raise ValueError("Invalid prediction response.")
        except (requests.RequestException, ValueError) as error:
            st.error(f"Prediction failed ({request_error(error)}).")
        else:
            prediction = result
            st.session_state["prediction"] = prediction
            st.session_state["prediction_selection"] = selection

    if prediction is not None:
        st.success(f"Evaluated trip {prediction['trip_id']} using {prediction.get('model', 'the server model')}.")
        if prediction.get("engine"):
            st.caption(f"Inference engine · {prediction['engine']}")
        for warning in prediction.get("warnings") or []:
            st.warning(warning)

    tab_map, tab_metrics, tab_compare, tab_viva = st.tabs([
        "📍 Interactive Trajectory Map",
        "📊 Trip Evaluation Metrics",
        "🏆 Model Leaderboard Comparison",
        "💡 Viva & Architecture Specs",
    ])

    with tab_map:
        if prediction is None:
            st.info("Select an available trip and run map matching to see the route.")
        else:
            st.subheader(f"Trip `{selected_trip}` — Trajectory & Predicted Route")
            show_prediction_map(prediction)

    with tab_metrics:
        if prediction is None:
            st.info("Trip metrics will appear after a successful prediction.")
        else:
            show_prediction_metrics(prediction)

    with tab_compare:
        st.subheader("Six-model leaderboard · Phase 2 validation")
        st.caption("Team-trained 22-feature models on the 16 validation trips and original Phase 2 proxy.")
        audit_path = PROJECT_ROOT / "phase3/omen/accuracy_audit/results/audit_team_results.json"
        if audit_path.exists():
            audited = json.loads(audit_path.read_text(encoding="utf-8"))["summary"]
            models = {
                "Extra Trees (Adarsh Prakash)": "aadi_extra_trees",
                "Logistic Regression (Abhigyan Dutta)": "aadi_logistic_regression",
                "CatBoost (Aadi Basal)": "aadi_catboost",
                "Random Forest (Aadi Basal)": "aadi_random_forest",
                "XGBoost (Abhigyan Dutta)": "omen_xgboost",
                "LightGBM (Adarsh Prakash)": "adarsh_lightgbm",
            }
            rows = [
                {
                    "Model": name,
                    "Directed micro Top-1 (%)": round(audited[key]["micro_coverage_top1_pct"], 2),
                    "Conditional Top-1 (%)": round(audited[key]["micro_conditional_top1_pct"], 2),
                    "Correct observations": audited[key]["correct"],
                    "Total observations": audited[key]["observations"],
                }
                for name, key in models.items()
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.info("The audited team scores are unavailable in this checkout.")

    with tab_viva:
        st.subheader("M1 Viva Reference Guide")
        viva_file = PROJECT_ROOT / "docs" / "MAC_VIVA.md"
        if viva_file.exists():
            st.markdown(viva_file.read_text(encoding="utf-8"))
        else:
            st.info("Viva guide at `docs/MAC_VIVA.md`.")


if __name__ == "__main__":
    main()
