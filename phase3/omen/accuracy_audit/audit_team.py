"""Replay the team models on the fixed Phase 2 validation trips and proxy labels."""

import argparse
import collections
from io import BytesIO
import json
from pathlib import Path
import struct
import subprocess
import sys
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import log_loss, roc_auc_score


MAC = "9ab819e0c88d079cc7361a2c4c1d047746d946bc"
ADARSH = "768a80b6100ca4332ce7d27b5736b64ed64cbac4"
KEYS = ["obs_index", "candidate_arc_id"]


def blob(commit, file):
    return subprocess.check_output(["git", "show", f"{commit}:{file}"])


def load_model(commit, name):
    model = joblib.load(BytesIO(blob(commit, f"phase3/models/{name}.joblib")))
    model.n_jobs = 2
    return model


def catboost_probabilities(model, features):
    # Numerical symmetric trees from the team's CBM1 artifact. The model and
    # feature tables are defined in CatBoost's model.fbs and features.fbs.
    assert model[:4] == b"CBM1" and struct.unpack_from("<I", model, 4)[0] == len(model) - 8
    data = model[8:]

    def field(table, index):
        vtable = table - struct.unpack_from("<i", data, table)[0]
        offset = 4 + index * 2
        if offset >= struct.unpack_from("<H", data, vtable)[0]:
            return None
        relative = struct.unpack_from("<H", data, vtable + offset)[0]
        return table + relative if relative else None

    def indirect(position):
        return position + struct.unpack_from("<I", data, position)[0]

    def scalar(table, index, format, default):
        position = field(table, index)
        return struct.unpack_from(format, data, position)[0] if position is not None else default

    def vector(table, index, dtype):
        position = field(table, index)
        if position is None:
            return np.array([], dtype=dtype)
        start = indirect(position)
        return np.frombuffer(data, dtype=dtype, count=struct.unpack_from("<I", data, start)[0], offset=start + 4)

    root = indirect(0)
    trees = indirect(field(root, 1))
    assert scalar(trees, 0, "<i", 0) == 1
    assert all(not vector(trees, i, "<u4").size for i in (4, 6, 7, 10, 11, 12, 13, 18))
    float_features = indirect(field(trees, 5))
    num_features = struct.unpack_from("<I", data, float_features)[0]
    assert num_features == features.shape[1] and np.isfinite(features).all()
    splits = []
    for idx in range(num_features):
        feature = indirect(float_features + 4 + idx * 4)
        raw_idx = scalar(feature, 2, "<i", -1)
        assert 0 <= raw_idx < features.shape[1]
        splits.extend((raw_idx, border) for border in vector(feature, 3, "<f4"))

    tree_splits = vector(trees, 1, "<i4")
    depths = vector(trees, 2, "<i4")
    offsets = vector(trees, 3, "<i4")
    values = vector(trees, 8, "<f8")
    scores = np.zeros(len(features))
    leaf_offset = 0
    for depth, offset in zip(depths, offsets, strict=True):
        leaf = np.zeros(len(features), dtype=np.int32)
        for bit, split in enumerate(tree_splits[offset:offset + depth]):
            feature, border = splits[split]
            leaf |= (features[:, feature] > border).astype(np.int32) << bit
        scores += values[leaf_offset + leaf]
        leaf_offset += 1 << int(depth)
    assert leaf_offset == len(values) and len(depths) == 37
    bias = vector(trees, 16, "<f8")
    assert len(bias) <= 1
    return expit(scores * scalar(trees, 14, "<d", 1.0) +
                 (bias[0] if len(bias) else scalar(trees, 15, "<d", 0.0)))


def select(frame, scores):
    ranked = frame.assign(score=scores).sort_values(
        ["obs_index", "score", "distance_to_arc_m"], ascending=[True, False, True],
    )
    return ranked.drop_duplicates("obs_index").candidate_arc_id.to_numpy()


def route_metrics(prediction, graph, route, reconstructor):
    collapsed = reconstructor.collapse_observation_sequence(prediction)
    raw = reconstructor.evaluate_continuity(collapsed, graph)
    reconstructed = reconstructor.reconstruct_and_bridge_route(collapsed, graph)
    bridged = reconstructor.evaluate_continuity(reconstructed, graph)
    similarity = reconstructor.compare_with_ground_truth(reconstructed, route)
    return {
        "raw_continuity_pct": raw[0], "bridged_continuity_pct": bridged[0],
        "bridged_connected_transitions": bridged[1], "bridged_transitions": bridged[2],
        "route_f1": similarity["route_f1"], "route_jaccard": similarity["route_jaccard"],
    }


def rate(correct, count):
    return 100 * correct / count


def observe(prediction, truth, covered=None):
    if covered is None:
        covered = truth >= 0
    hits = int(((prediction == truth) & covered).sum())
    total = len(truth)
    valid = int(np.count_nonzero(covered))
    return {"correct": hits, "observations": total, "covered": valid,
            "coverage_top1_pct": rate(hits, total), "conditional_top1_pct": rate(hits, valid)}


def archived_baseline(trips, denominators):
    rows = pd.read_csv("phase2/results/baseline_trip_metrics.csv", dtype={"trip_id": str})
    rows = rows[rows.matcher.eq("graph_constrained_hmm")].set_index("trip_id")
    results = {}
    for trip in trips:
        total, valid = denominators[trip]
        row = rows.loc[trip]
        hits, = (h for h in range(valid + 1) if round(rate(h, valid), 2) == row.top1_accuracy)
        results[trip] = {"correct": hits, "observations": total, "covered": valid,
                         "coverage_top1_pct": rate(hits, total), "conditional_top1_pct": rate(hits, valid),
                         "bridged_continuity_pct": float(row.route_continuity_pct),
                         "route_f1": float(row.route_f1), "route_jaccard": float(row.route_jaccard)}
    return results


def summarize(trips):
    rows = list(trips.values())
    totals = {metric: sum(row[metric] for row in rows) for metric in ("correct", "observations", "covered")}
    averages = {f"macro_{metric}": sum(row[metric] for row in rows) / len(rows) for metric in
                ("coverage_top1_pct", "conditional_top1_pct", "raw_continuity_pct",
                 "bridged_continuity_pct", "route_f1", "route_jaccard") if metric in rows[0]}
    worst = min(trips, key=lambda trip: trips[trip]["coverage_top1_pct"])
    return {**totals, "micro_coverage_top1_pct": rate(totals["correct"], totals["observations"]),
            "micro_conditional_top1_pct": rate(totals["correct"], totals["covered"]), **averages,
            "worst_coverage_trip": worst, "worst_coverage_pct": trips[worst]["coverage_top1_pct"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trip", help="Replay one of the archived validation trips for diagnosis")
    parser.add_argument("--route-first", type=Path, help="Also score frozen winners against this cache's alternate proxy")
    args = parser.parse_args()
    sys.path.insert(0, str(Path("phase2/scripts").resolve()))
    from data_loader import DataLoader
    from audit_hmm import MLEmissionHMMMatcher
    from trajectory_processor import TrajectoryProcessor
    from road_graph import RoadNetworkGraph
    from feature_engineering import FeatureEngineer
    from candidate_generation import CandidateGenerator
    from label_generation import LabelGenerator
    from spatial_index import RoadSpatialIndex
    from route_reconstruction import RouteReconstructor

    ids = json.loads(Path("phase2/results/dataset_split_manifest.json").read_text())["val_trips"]
    assert len(ids) == len(set(ids)) == 16
    trips = [args.trip] if args.trip else ids
    assert set(trips).issubset(ids)
    archive = pd.read_parquet(BytesIO(blob(ADARSH, "phase3/results/val_predictions.parquet")))
    with ZipFile("phase3/omen/review1/experiment.zip") as omen:
        xgb = pd.read_parquet(BytesIO(omen.read("xgboost/validation_predictions.parquet")))
    mac_et = load_model(MAC, "extra_trees_classifier")
    mac_lr = load_model(MAC, "logistic_regression_classifier")
    mac_rf = load_model(MAC, "random_forest_classifier")
    adarsh_et = load_model(ADARSH, "extra_trees_model")
    lightgbm = load_model(ADARSH, "lightgbm_model")
    cat_model = blob(MAC, "phase3/models/catboost_classifier.cbm")
    decoder = MLEmissionHMMMatcher()
    loader = DataLoader(Path("Dataset"))
    trajectory = TrajectoryProcessor()
    generator = CandidateGenerator()
    engineer = FeatureEngineer()
    reconstructor = RouteReconstructor()
    scores = collections.defaultdict(dict)
    alternate = collections.defaultdict(dict)
    provenance = {}
    candidate_scores = []
    denominators = {}

    for trip_id in trips:
        archived = pd.read_parquet(f"phase2/features/val/{trip_id}_features.parquet")
        frozen = archive[archive.trip_id.eq(trip_id)].reset_index(drop=True)
        assert archived[KEYS + ["true_arc_id", "is_correct"]].equals(frozen[KEYS + ["true_arc_id", "is_correct"]])
        truth = archived.drop_duplicates("obs_index").true_arc_id.to_numpy()
        denominators[trip_id] = len(truth), int(np.count_nonzero(truth >= 0))
        reference = None
        if args.route_first:
            reference = pd.read_parquet(args.route_first / f"observations/val/{trip_id}.parquet")
            np.testing.assert_array_equal(reference.obs_index, np.arange(len(truth)))
        predictions = {}

        trip = loader.load_trip(trip_id, "train")
        obs = trajectory.process_trip_track(trip_id, trip.track)
        assert len(truth) == len(obs) and archived.obs_index.nunique() == len(obs)
        graph = RoadNetworkGraph(trip_id, trip.nodes, trip.arcs)
        candidates, _ = generator.generate_candidates_for_trip(trip, obs, graph, RoadSpatialIndex(graph))
        regenerated = engineer.compute_candidate_features(candidates, obs, graph)
        cache = graph._source_dist_cache.copy()

        columns = engineer.FEATURE_COLUMNS
        archived_x = archived[columns].astype(np.float32)
        regenerated_x = regenerated[columns].astype(np.float32)
        raw_lgbm = lightgbm.predict_proba(regenerated_x)[:, 1]
        raw_candidates = pd.DataFrame(candidates)
        assert regenerated[KEYS].equals(raw_candidates[KEYS])
        raw_labels, _ = LabelGenerator().align_trip_observations(trip, obs, candidates)
        raw_truth = np.asarray([raw_labels.get(i, -1) for i in range(len(obs))])
        assert np.array_equal(np.where(pd.isna(raw_truth), -1, raw_truth), truth)
        merged = archived[KEYS].merge(regenerated[KEYS], on=KEYS, validate="one_to_one")
        provenance[trip_id] = {
            "archived_candidates": len(archived), "regenerated_candidates": len(regenerated),
            "intersecting_candidates": len(merged), "same_candidate_order": archived[KEYS].equals(regenerated[KEYS]),
            "same_feature_values": archived_x.equals(regenerated_x),
        }

        # The archived branch evaluates regenerated features. Its evaluator uses
        # conditional accuracy, with first maximum in generated order as tie break.
        adarsh_pos = regenerated.assign(score=raw_lgbm).groupby("obs_index").score.idxmax()
        pred_raw = regenerated.loc[adarsh_pos].candidate_arc_id.to_numpy()
        pred_raw_hmm = np.asarray(decoder.match_trip(candidates, raw_lgbm, obs, graph))
        predictions.update(adarsh_regenerated_lightgbm=pred_raw, adarsh_regenerated_lightgbm_hmm=pred_raw_hmm)

        mac_prob = mac_et.predict_proba(archived_x.to_numpy())[:, 1]
        adarsh_prob = adarsh_et.predict_proba(archived_x)[:, 1]
        lightgbm_prob = lightgbm.predict_proba(archived_x)[:, 1]
        cat_prob = catboost_probabilities(cat_model, archived_x.to_numpy())
        xgb_trip = archived[KEYS].merge(xgb.loc[xgb.trip_id.eq(trip_id), KEYS + ["probability"]],
                                       on=KEYS, how="left", validate="one_to_one")
        assert xgb_trip.probability.notna().all()
        predictions.update(
            aadi_logistic_regression=select(archived, mac_lr.predict_proba(archived_x.to_numpy())[:, 1]),
            aadi_random_forest=select(archived, mac_rf.predict_proba(archived_x.to_numpy())[:, 1]),
            omen_xgboost=select(archived, xgb_trip.probability.to_numpy()),
        )
        candidate_scores.append(pd.DataFrame({"label": archived.is_correct, "catboost": cat_prob}))
        np.testing.assert_array_equal(np.round(adarsh_prob, 5), frozen.prob_extra_trees)
        np.testing.assert_array_equal(np.round(lightgbm_prob, 5), frozen.prob_lightgbm)
        archived_candidates = archived[KEYS].to_dict(orient="records")

        # Feature generation primes the graph's source-node cache at the 2 km
        # feature cutoff before Adarsh's decoder queries it at its 1.5 km cutoff.
        # Reuse that initial state for all models and leave decoder order irrelevant.
        for model, probability in (("aadi_extra_trees", mac_prob), ("adarsh_extra_trees", adarsh_prob),
                                   ("aadi_catboost", cat_prob), ("adarsh_lightgbm", lightgbm_prob)):
            graph._source_dist_cache = cache.copy()
            pointwise = select(archived, probability)
            if model == "adarsh_lightgbm" and provenance[trip_id]["same_candidate_order"] and provenance[trip_id]["same_feature_values"]:
                hmm = pred_raw_hmm
            else:
                hmm = np.asarray(decoder.match_trip(archived_candidates, probability, obs, graph))
            predictions[model] = pointwise
            predictions[f"{model}_hmm"] = hmm

        for name, prediction in predictions.items():
            scores[name][trip_id] = {**observe(prediction, truth),
                                    **route_metrics(prediction, graph, trip.route, reconstructor)}
            if reference is not None:
                alternate[name][trip_id] = {
                    **scores[name][trip_id],
                    **observe(prediction, reference.true_arc_id.to_numpy(), reference.has_positive.to_numpy()),
                }

        print(trip_id, provenance[trip_id], {name: records[trip_id]["correct"]
                                             for name, records in scores.items()}, flush=True)

    scores["archived_phase2_graph_hmm"] = archived_baseline(trips, denominators)
    summary = {name: summarize(records) for name, records in scores.items()}
    evidence = pd.concat(candidate_scores, ignore_index=True)
    booster_check = {"log_loss": log_loss(evidence.label, evidence.catboost),
                     "roc_auc": roc_auc_score(evidence.label, evidence.catboost)}
    artifact = {"commits": {"aadi": MAC, "adarsh": ADARSH}, "validation_trips": trips,
                "candidate_check": booster_check, "regeneration": provenance,
                "summary": summary, "per_trip": scores}
    if alternate:
        artifact["route_first"] = {"summary": {name: summarize(records) for name, records in alternate.items()},
                                   "per_trip": alternate}
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
