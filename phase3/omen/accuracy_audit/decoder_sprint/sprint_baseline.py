"""Replay the published Phase 2 baseline and audit the public validation trip 84."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


TRIP = "00000084"


def quantiles(values):
    return dict(zip(("p25", "p50", "p75", "p90"), np.quantile(values, [.25, .5, .75, .9]).tolist())) if len(values) else {}


def diagnose(trip, observations, graph, reference, predictions):
    from pathsnap_context import motion_window

    truth = reference.true_arc_id.to_numpy()
    covered = reference.has_positive.to_numpy()
    near_tie = reference.geometric_near_tie.to_numpy()
    step = np.asarray([o.step_distance_m for o in observations])
    gap = np.asarray([o.dt for o in observations])
    instant_speed = np.asarray([o.speed_mps for o in observations])
    stationary = np.asarray([o.is_stationary for o in observations])
    points = np.asarray([graph.project_point(o.lon, o.lat) for o in observations])
    seconds = np.asarray([o.timestamp for o in observations])
    before, after, motion, elapsed = motion_window(points, seconds, 3)
    net = np.linalg.norm(motion, axis=1)
    speed = np.divide(net, elapsed, out=np.zeros_like(net), where=elapsed > 0)
    cumulative_step = np.cumsum(step)
    local_path = cumulative_step[after] - cumulative_step[before]
    cancellation = np.divide(net, local_path, out=np.ones_like(net), where=local_path > 0)
    masks = {
        "all": np.ones(len(truth), dtype=bool),
        "step_stationary": stationary,
        "window_speed_lt_0_8_mps": speed < .8,
        "window_speed_lt_0_8_mps_and_step_ge_2_m": (speed < .8) & (step >= 2),
        "window_speed_lt_0_8_mps_and_net_path_ratio_lt_0_35": (speed < .8) & (cancellation < .35),
        "window_speed_ge_3_mps": speed >= 3,
        "geometric_near_tie": near_tie,
        "no_geometric_near_tie": ~near_tie,
    }
    arcs = graph.arcs_dict
    endpoints = {a: (arc.from_node, arc.to_node) for a, arc in arcs.items()}
    route = set(trip.route)

    def classify(prediction):
        hit = prediction == truth
        reverse = np.asarray([
            a != b and a in arcs and b in arcs and endpoints[a] == endpoints[b][::-1]
            for a, b in zip(prediction, truth, strict=True)
        ])
        same_direction_duplicate = np.asarray([
            a != b and a in arcs and b in arcs and endpoints[a] == endpoints[b]
            for a, b in zip(prediction, truth, strict=True)
        ]) & ~reverse
        on_route = np.isin(prediction, list(route))
        other = ~(hit | reverse | same_direction_duplicate)
        totals = {}
        for name, mask in masks.items():
            valid = mask & covered
            categories = {
                "observations": mask,
                "retrieval_miss": mask & ~covered,
                "exact": valid & hit,
                "exact_opposite_endpoints": valid & reverse,
                "same_direction_duplicate": valid & same_direction_duplicate,
                "other_arc_on_reference_route": valid & other & on_route,
                "other_arc_off_reference_route": valid & other & ~on_route,
            }
            totals[name] = {key: int(value.sum()) for key, value in categories.items()}
        return totals

    def transitions(prediction):
        known = np.isin(prediction, list(arcs))
        valid_next = known[:-1] & known[1:] & (gap[1:] <= 30)
        positions = np.flatnonzero(valid_next) + 1
        prev_ids, curr_ids = prediction[positions - 1], prediction[positions]
        changed = prev_ids != curr_ids
        reverse = np.asarray([a != b and endpoints[a] == endpoints[b][::-1]
                              for a, b in zip(prev_ids, curr_ids, strict=True)])
        a1 = [arcs[a] for a in prev_ids]
        a2 = [arcs[a] for a in curr_ids]

        def fractions(xy, selected):
            starts = np.array([(arc.x1, arc.y1) for arc in selected])
            vectors = np.array([(arc.x2 - arc.x1, arc.y2 - arc.y1) for arc in selected])
            length_squared = (vectors * vectors).sum(axis=1)
            along = ((xy - starts) * vectors).sum(axis=1)
            return np.clip(np.divide(along, length_squared, out=np.zeros_like(along),
                                     where=length_squared > 0), 0, 1)

        prev_fraction = fractions(points[positions - 1], a1)
        curr_fraction = fractions(points[positions], a2)
        legacy = np.array([graph.candidate_transition_distance(a, b, 1500.)
                           for a, b in zip(prev_ids, curr_ids, strict=True)])
        lengths_before = np.array([a.length_m for a in a1])
        lengths_after = np.array([a.length_m for a in a2])
        partial = legacy + (1 - prev_fraction) * lengths_before - (1 - curr_fraction) * lengths_after
        same_arc_backtracks = ~changed & (curr_fraction < prev_fraction)
        partial[~changed] = (curr_fraction[~changed] - prev_fraction[~changed]) * lengths_after[~changed]
        positive_backtracks = same_arc_backtracks & (partial < -1)
        true_reverse = np.asarray([
            a != b and a in arcs and b in arcs and endpoints[a] == endpoints[b][::-1]
            for a, b in zip(truth[positions - 1], truth[positions], strict=True)
        ])
        same_as_ref = reverse & true_reverse & (prediction[positions - 1] == truth[positions - 1])
        finite = np.isfinite(legacy)
        legacy_residual = np.abs(legacy - step[positions])
        partial_residual = np.abs(partial - step[positions])
        moving = speed[positions] >= 3
        return {
            "pairs_with_known_arcs_and_gap_at_most_30_s": len(positions),
            "changed_arc_pairs": int(changed.sum()),
            "legacy_infinite_or_beyond_cutoff": int((~finite).sum()),
            "legacy_changed_finite_residual_m": quantiles(legacy_residual[finite & changed]),
            "projection_aware_changed_finite_residual_m": quantiles(partial_residual[finite & changed]),
            "legacy_finite_residual_gt_50_m": int((finite & (legacy_residual > 50)).sum()),
            "projection_aware_changed_finite_residual_gt_50_m": int((finite & changed & (partial_residual > 50)).sum()),
            "same_arc_backward_projection_over_1m": int(positive_backtracks.sum()),
            "exact_opposite_arc_switches": int(reverse.sum()),
            "opposite_switches_at_window_speed_lt_0_8_mps": int((reverse & (speed[positions] < .8)).sum()),
            "opposite_switches_at_window_speed_ge_3_mps": int((reverse & moving).sum()),
            "opposite_switches_matching_route_first_proxy_pair": int(same_as_ref.sum()),
            "opposite_switch_legacy_residual_m": quantiles(legacy_residual[finite & reverse]),
            "opposite_switch_projection_aware_residual_m": quantiles(partial_residual[finite & reverse]),
        }

    return {
        "observations": len(truth),
        "sampling_dt_s_excluding_first": quantiles(gap[1:]),
        "gaps_gt_30_s": int((gap > 30).sum()),
        "one_step_gps_speed_mps": quantiles(instant_speed),
        "gap_aware_window_gps_net_speed_mps": quantiles(speed),
        "window_lt_0_8_mps_one_step_gps_speed_mps": quantiles(instant_speed[speed < .8]),
        "window_lt_0_8_mps_one_step_displacement_m": quantiles(step[speed < .8]),
        "window_lt_0_8_mps_net_over_raw_gps_path": quantiles(cancellation[speed < .8]),
        "opposite_pair_is_swapped_graph_endpoints": True,
        "interpretation": "Observation labels and low-motion classifications are GPS-derived proxies. Opposite-arc switches are graph events; neither they nor negative projections establish a physical U-turn.",
        "score_breakdowns": {name: classify(pred) for name, pred in predictions.items()},
        "directed_transitions": {
            name: transitions(pred) for name, pred in {"route_first_proxy": truth, **predictions}.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--xgb-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trip-84-only", action="store_true")
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    if not args.geometry_only and args.xgb_archive is None:
        parser.error("--xgb-archive is required for the trip 84 model diagnosis")
    sys.path[:0] = [str(Path(p).resolve()) for p in (
        "phase2/scripts", "phase3/omen/accuracy_audit", "phase3/omen",
        "phase3/omen/preliminary/source",
    )]
    from audit_hmm import MLEmissionHMMMatcher
    from audit_team import MAC, KEYS, archived_baseline, blob, catboost_probabilities, load_model, observe, route_metrics, summarize
    from baseline_matcher import GraphConstrainedHMMMatcher
    from candidate_generation import CandidateGenerator
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    from label_generation import LabelGenerator
    from model_registry import load_classifier
    from road_graph import RoadNetworkGraph
    from route_reconstruction import RouteReconstructor
    from spatial_index import RoadSpatialIndex
    from trajectory_processor import TrajectoryProcessor

    manifest = json.loads((args.cache / "dataset_manifest.json").read_text(encoding="utf-8"))
    ids = json.loads(Path("phase2/results/dataset_split_manifest.json").read_text())["val_trips"]
    assert ids == manifest["split"]["val_trips"] and len(set(ids)) == 16 and TRIP in ids
    trips = [TRIP] if args.trip_84_only else [TRIP, *(t for t in ids if t != TRIP)]
    loader, processor, generator, engineer = DataLoader(Path("Dataset")), TrajectoryProcessor(), CandidateGenerator(), FeatureEngineer()
    reconstructor = RouteReconstructor()
    geom = GraphConstrainedHMMMatcher()
    learned = MLEmissionHMMMatcher()
    legacy, route_first = {}, {}
    diagnostics = {}
    for trip_id in trips:
        path = f"observations/val/{trip_id}.parquet"
        reference_path = args.cache / path
        assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == manifest["file_sha256"][path]
        reference = pd.read_parquet(reference_path)
        trip = loader.load_trip(trip_id, "train")
        obs = processor.process_trip_track(trip_id, trip.track)
        graph = RoadNetworkGraph(trip_id, trip.nodes, trip.arcs)
        candidates, _ = generator.generate_candidates_for_trip(trip, obs, graph, RoadSpatialIndex(graph))
        archived = pd.read_parquet(f"phase2/features/val/{trip_id}_features.parquet")
        labels, _ = LabelGenerator().align_trip_observations(trip, obs, candidates)
        # Phase 2 warms single-source distances at 2 km before decoding at 1.5 km.
        frame = engineer.compute_candidate_features(LabelGenerator().assign_candidate_labels(candidates, labels),
                                                    obs, graph)
        assert len(graph._source_dist_cache) > 0 and frame[KEYS].equals(archived[KEYS])
        warm_cache = graph._source_dist_cache.copy()
        geom_prediction = np.asarray(geom.match_trip(candidates, obs, graph))
        actual = np.fromiter((labels.get(i) if labels.get(i) is not None else -1 for i in range(len(obs))), dtype=int)
        np.testing.assert_array_equal(archived.drop_duplicates("obs_index").true_arc_id, actual)
        np.testing.assert_array_equal(reference.obs_index, np.arange(len(obs)))
        reference_truth = reference.true_arc_id.to_numpy()
        have_positive = reference.has_positive.to_numpy()
        matched_candidate_obs = frame.loc[frame.candidate_arc_id.eq(reference_truth[frame.obs_index]), "obs_index"]
        np.testing.assert_array_equal(have_positive, np.isin(np.arange(len(obs)), matched_candidate_obs))
        route = route_metrics(geom_prediction, graph, trip.route, reconstructor)
        legacy[trip_id] = {**observe(geom_prediction, actual), **route}
        route_first[trip_id] = {**observe(geom_prediction, reference_truth, have_positive), **route}
        assert legacy[trip_id]["correct"] == archived_baseline([trip_id], {trip_id: (len(obs), int((actual >= 0).sum()))})[trip_id]["correct"]
        print(trip_id, "legacy", legacy[trip_id]["correct"], "route first", route_first[trip_id]["correct"], flush=True)

        if trip_id != TRIP or args.geometry_only:
            continue
        x = archived[engineer.FEATURE_COLUMNS].astype(np.float32)
        raw_x = frame[engineer.FEATURE_COLUMNS].astype(np.float32)
        xgb_report = json.loads(Path("phase3/omen/accuracy_audit/results/xgboost_hmm_validation.json").read_text())
        assert hashlib.sha256(args.xgb_archive.read_bytes()).hexdigest() == xgb_report["artifact_sha256"]
        xgb, features = load_classifier("xgboost", {"xgboost": args.xgb_archive}, engineer.FEATURE_COLUMNS)
        xgb.n_jobs = 2
        probabilities = {
            "aadi_catboost_hmm": catboost_probabilities(blob(MAC, "phase3/models/catboost_classifier.cbm"), x.to_numpy()),
            "aadi_extra_trees_hmm": load_model(MAC, "extra_trees_classifier").predict_proba(x.to_numpy())[:, 1],
            "omen_frozen_xgboost_hmm": xgb.predict_proba(raw_x[features])[:, 1],
        }
        predictions = {"original_phase2_geometry_hmm": geom_prediction}
        ordered = frame[KEYS].to_dict(orient="records")
        for name, probability in probabilities.items():
            graph._source_dist_cache = warm_cache.copy()
            predictions[name] = np.asarray(learned.match_trip(ordered, probability, obs, graph))
        counts = {name: observe(prediction, reference_truth, have_positive)
                  for name, prediction in predictions.items()}
        team = json.loads(Path("phase3/omen/accuracy_audit/results/audit_team_results.json").read_text())
        for name in ("aadi_catboost_hmm", "aadi_extra_trees_hmm"):
            assert counts[name]["correct"] == team["route_first"]["per_trip"][name][TRIP]["correct"]
        xgb_expected = next(r["route_first_hits"] for r in xgb_report["per_trip"] if r["trip_id"] == TRIP)
        assert counts["omen_frozen_xgboost_hmm"]["correct"] == xgb_expected
        graph._source_dist_cache = warm_cache.copy()
        diagnostics = diagnose(trip, obs, graph, reference, predictions)
        diagnostics["replayed_top1"] = counts
        print(json.dumps(diagnostics, allow_nan=False), flush=True)

    results = {"dataset_version": manifest["dataset_version"], "validation_trips": trips,
               "observations_total": sum(r["observations"] for r in legacy.values()),
               "geometry_hmm_legacy": {"summary": summarize(legacy), "per_trip": legacy},
               "geometry_hmm_route_first": {"summary": summarize(route_first), "per_trip": route_first},
               "trip_00000084_diagnosis": diagnostics}
    if len(trips) == 16:
        assert results["geometry_hmm_legacy"]["summary"]["correct"] == 33707
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: results[k]["summary"] for k in ("geometry_hmm_legacy", "geometry_hmm_route_first")}), flush=True)


if __name__ == "__main__":
    main()
