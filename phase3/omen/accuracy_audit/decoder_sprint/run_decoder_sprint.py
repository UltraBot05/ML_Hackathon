"""Replay fixed PathSnap decoders without training classifiers or changing the API."""

import argparse
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from sprint_decoder import Config, TripDecoder


DEVELOPMENT = ("00000002", "00000003", "00000012", "00000033")
CALIBRATION = "00000012"
META = ["obs_index", "candidate_arc_id", "candidate_rank", "projection_t", "distance_to_arc_m",
        "arc_bearing_deg", "arc_length_m", "arc_from_node", "arc_to_node"]
BASES = {"geometry": "geometry", "catboost": "catboost", "extra_trees": "extra_trees",
         "xgboost": "xgboost", "lightgbm": "lightgbm"}


def dependencies(repo):
    sys.path[:0] = [str(repo / "phase2/scripts"), str(repo / "phase3/omen/preliminary/source"),
                   str(repo / "phase3/omen/accuracy_audit")]
    from candidate_generation import CandidateGenerator
    from data_loader import DataLoader, TripData
    from feature_engineering import FeatureEngineer
    from road_graph import RoadNetworkGraph
    from route_reconstruction import RouteReconstructor
    from spatial_index import RoadSpatialIndex
    from trajectory_processor import TrajectoryProcessor
    from pathsnap_context import motion_window
    from pathsnap_quickstart import ordered_proxy

    return (DataLoader, TripData, TrajectoryProcessor, RoadNetworkGraph, RoadSpatialIndex,
            CandidateGenerator, FeatureEngineer, RouteReconstructor, motion_window, ordered_proxy)


def load_models(repo, names):
    from audit_team import ADARSH, MAC, catboost_probabilities

    def artifact(commit, path):
        return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=repo)

    result = {}
    if "catboost" in names:
        binary = artifact(MAC, "phase3/models/catboost_classifier.cbm")
        result["catboost"] = lambda x: catboost_probabilities(binary, x.to_numpy())
    for name, commit, path in (
        ("extra_trees", MAC, "phase3/models/extra_trees_classifier.joblib"),
        ("lightgbm", ADARSH, "phase3/models/lightgbm_model.joblib"),
    ):
        if name in names:
            estimator = joblib.load(io.BytesIO(artifact(commit, path)))
            estimator.n_jobs = 2
            result[name] = lambda matrix, clf=estimator, key=name: clf.predict_proba(
                matrix.to_numpy() if key == "extra_trees" else matrix,
            )[:, 1]
    if "xgboost" in names:
        with ZipFile(repo / "phase3/omen/accuracy_audit/results/xgboost_baseline.zip") as archive:
            estimator = joblib.load(io.BytesIO(archive.read("xgboost/model.joblib")))
        result["xgboost"] = lambda matrix, clf=estimator: clf.predict_proba(matrix)[:, 1]
    return result


def prepare(root, trip_id, components, rows=None):
    DataLoader, TripData, Processor, Graph, Index, Generator, Engineer, *_ = components
    root = root.resolve()
    loader = DataLoader(root)
    path = root / trip_id / trip_id
    gps = loader.load_track(path.with_suffix(".track"))
    nodes = loader.load_nodes(path.with_suffix(".nodes"))
    arcs = loader.load_arcs(path.with_suffix(".arcs"))
    if not gps or not nodes or not arcs or [p.index for p in gps] != list(range(len(gps))):
        raise ValueError("Invalid raw trip data or observation indices")
    graph = Graph(trip_id, nodes, arcs)
    obs = Processor().process_trip_track(trip_id, gps)
    if rows is None:
        trip = TripData(trip_id, root.name, gps, nodes, arcs, [])
        records, _ = Generator().generate_candidates_for_trip(trip, obs, graph, Index(graph))
        rows = Engineer().compute_candidate_features(records, obs, graph)
    rows = rows.sort_values(["obs_index", "candidate_rank"]).reset_index(drop=True)
    lonlat = np.array([(p.lon, p.lat) for p in gps])
    seconds = np.array([p.timestamp for p in gps])
    xy = np.array([graph.project_point(lon, lat) for lon, lat in lonlat])
    dec = TripDecoder(graph, len(obs), np.array([p.step_distance_m for p in obs]), xy, seconds, rows)
    return dec, rows, lonlat, seconds, nodes, arcs, path


def independent_truth(prepared, components):
    dec, _, gps, seconds, nodes, arcs, path = prepared
    window, align = components[-2:]
    route = np.loadtxt(path.with_suffix(".route"), dtype=int, ndmin=1)
    center = gps.mean(axis=0)
    scale = np.pi*6_371_008.8/180*np.array([np.cos(np.deg2rad(center[1])), 1])
    points = (gps-center)*scale
    network = (np.array([(n.lon, n.lat) for n in nodes])-center)*scale
    ab = np.array([(a.from_node, a.to_node) for a in arcs], dtype=int)
    origins = network[ab[:, 0]]
    _, _, motion, elapsed = window(points, seconds, 3)
    magnitude = np.linalg.norm(motion, axis=1)
    speed = np.divide(magnitude, elapsed, out=np.zeros(len(gps)), where=elapsed > 0)
    direction = np.divide(motion, magnitude[:, None], out=np.zeros_like(motion), where=magnitude[:, None] > 0)
    reliability = np.clip((speed-.8)/2.5, 0, 1)*np.clip(magnitude/5, 0, 1)*(speed <= 55)
    return align(points, origins, network[ab[:, 1]]-origins, route,
                 motion=direction, reliability=reliability)[0], route.tolist()


def temperature(decoder, probs, truth):
    padded = np.full(decoder.arcs.shape, -np.inf)
    p = np.clip(probs, 1e-6, 1-1e-6)
    padded[decoder.index] = np.log(p)-np.log1p(-p)
    target = decoder.arcs == truth[:, None]
    valid = target.any(axis=1)
    values, classes = padded[valid], target[valid].argmax(axis=1)
    return min((.5, .75, 1., 1.5, 2., 3., 5.),
               key=lambda t: float(np.mean(logsumexp(values/t, axis=1)-values[np.arange(len(classes)), classes]/t)))


def candidates_for(temp):
    raw = Config()
    fixed = replace(raw, transition="cutoff")
    projected = replace(fixed, transition="projection")
    robust = replace(projected, emission="robust")
    stopped = replace(robust, stationary=True)
    headed = replace(stopped, heading=2.0)
    return {
        "geometry": (raw, None),
        **{name: (replace(raw, emission="ml"), name) for name in BASES if name != "geometry"},
        "cutoff_only": (fixed, None),
        "partial_arcs": (projected, None),
        "robust_geometry": (robust, None),
        "stationary_and_gaps": (stopped, None),
        "smoothed_heading": (headed, None),
        "heading_tighter_transition": (replace(headed, beta=18), None),
        "catboost_geometry_blend": (replace(headed, emission="hybrid", temperature=temp), "catboost"),
        "catboost_geometry_blend_tighter": (
            replace(headed, emission="hybrid", beta=18, temperature=temp), "catboost",
        ),
        "bounded_projection_jitter": (replace(stopped, transition="projected_jitter", heading=2.0), None),
        "cutoff_with_heading": (replace(fixed, heading=2.0), None),
        "cutoff_robust_stationary": (replace(fixed, emission="robust", stationary=True), None),
        "cutoff_geometry_catboost": (replace(fixed, emission="hybrid", heading=1.5, temperature=temp), "catboost"),
    }


def score(decoder, pred, truth, seconds, duration, route=None, components=None):
    station = decoder.stationary
    covered = (decoder.arcs == truth[:, None]).any(axis=1)
    hit = pred == truth
    miss = ~hit & covered & (pred >= 0)
    graph = decoder.graph
    reverse = np.zeros(len(truth), dtype=bool)
    for i in np.flatnonzero(miss):
        a, b = graph.get_arc(int(pred[i])), graph.get_arc(int(truth[i]))
        reverse[i] = a is not None and b is not None and a.from_node == b.to_node and a.to_node == b.from_node
    raw_pairs = [(a.from_node, a.to_node) for a in graph.raw_arcs]
    data = dict(
        hits=int(hit.sum()), n=len(truth), covered=int(covered.sum()),
        kept_stationary=int(station.sum()), stationary_hits=int((hit & station).sum()),
        moving_hits=int((hit & ~station).sum()), reverse_errors=int(reverse.sum()),
        wrong_nonreverse_when_retrieved=int(miss.sum()-reverse.sum()),
        unmatched_when_retrieved=int((covered & (pred < 0)).sum()),
        gaps_gt_30s=int((np.diff(seconds)>30).sum()), duplicate_directed_edges=len(raw_pairs)-len(set(raw_pairs)),
        decode_seconds=duration,
    )
    if route is not None:
        rebuild = components[7]()
        reconstructed = rebuild.reconstruct_and_bridge_route(rebuild.collapse_observation_sequence(pred.tolist()), graph)
        directed = rebuild.compare_with_ground_truth(reconstructed, route)
        data.update(route_f1=directed["route_f1"],
                    continuity_pct=rebuild.evaluate_continuity(reconstructed, graph)[0])
    return data, hit


def summary(rows):
    total = {f: sum(row[f] for row in rows.values()) for f in (
        "hits", "n", "covered", "kept_stationary", "stationary_hits", "moving_hits",
        "reverse_errors", "wrong_nonreverse_when_retrieved", "decode_seconds",
    )}
    n = total["n"]
    worst = min(rows, key=lambda trip: rows[trip]["hits"]/rows[trip]["n"])
    total.update(micro_top1=total["hits"]/n, macro_trip_top1=float(np.mean([x["hits"]/x["n"] for x in rows.values()])),
                 conditional_top1=total["hits"]/total["covered"], ceiling=total["covered"]/n,
                 worst_trip=worst, worst_top1=rows[worst]["hits"]/rows[worst]["n"],
                 stationary_top1=total["stationary_hits"]/total["kept_stationary"] if total["kept_stationary"] else None,
                 moving_top1=total["moving_hits"]/(n-total["kept_stationary"]) if total["kept_stationary"] < n else None)
    if "route_f1" in next(iter(rows.values())):
        total["mean_route_f1"] = float(np.mean([x["route_f1"] for x in rows.values()]))
        total["mean_continuity_pct"] = float(np.mean([x["continuity_pct"] for x in rows.values()]))
    return total


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--stage", choices=("dev", "val", "heldout"), required=True)
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--trip", help="Optional one-trip validation diagnosis")
    p.add_argument("--private", action="store_true")
    p.add_argument("--full-ablation", action="store_true", help="Score each preset on validation.")
    p.add_argument("--only-frozen", action="store_true", help="Measure the frozen decoder alone on validation.")
    args = p.parse_args()
    if args.output.exists() or (args.stage == "dev" and args.frozen.exists()):
        p.error("Choose fresh output names.")
    start = time.perf_counter()
    repo = args.repo.resolve()
    manifest = json.loads((args.cache/"dataset_manifest.json").read_text(encoding="utf-8"))
    components = dependencies(repo)
    features = manifest["feature_columns"]
    root = repo/"Dataset/train" if args.root is None else args.root
    if args.stage == "heldout" and not args.root:
        p.error("Provide the raw heldout root.")
    partition = "train" if args.stage == "dev" else "val"
    trips = (DEVELOPMENT if args.stage == "dev" else manifest["split"]["val_trips"])
    if args.stage == "heldout":
        trips = sorted(x.name for x in root.iterdir() if x.is_dir() and x.name.isascii() and x.name.isdecimal())
    if args.trip:
        if args.stage != "val" or args.trip not in trips:
            p.error("--trip is for a one-trip validation diagnosis only.")
        trips = [args.trip]

    def public_cache(trip, phase):
        rows = pd.read_parquet(args.cache/"features"/phase/f"{trip}.parquet", columns=list(dict.fromkeys(META+features)))
        labels = pd.read_parquet(args.cache/"observations"/phase/f"{trip}.parquet")
        return rows, labels.sort_values("obs_index").true_arc_id.to_numpy(dtype=int)

    if args.stage == "dev":
        assert set(DEVELOPMENT).issubset(manifest["split"]["train_trips"])
        models = load_models(repo, BASES)
        cat_rows, cat_truth = public_cache(CALIBRATION, "train")
        cal, ordered, *_ = prepare(root, CALIBRATION, components, cat_rows)
        temp = temperature(cal, models["catboost"](ordered[features].astype(np.float32)), cat_truth)
        del cal, ordered, cat_rows, cat_truth
        configs = candidates_for(temp)
        trips = [trip for trip in trips if trip != CALIBRATION]
    else:
        frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
        fixed = Config(**frozen["config"]), frozen["model"]
        configs = {"frozen": fixed}
        if args.stage == "val" and not args.only_frozen:
            configs = {name: value for name, value in candidates_for(frozen["temperature"]).items()
                       if name in BASES or args.trip or args.full_ablation}
            configs["frozen"] = fixed
        models = load_models(repo, {model for _, model in configs.values() if model})

    records = {name: {} for name in configs}
    joint_counts = {name: 0 for name in configs if name in BASES}
    oracle_base_hits = 0
    for trip in trips:
        trip_start = time.perf_counter()
        from_cache = args.stage != "heldout"
        rows, truth = public_cache(trip, partition) if from_cache else (None, None)
        data = prepare(root, trip, components, rows)
        decoder, table, _, seconds, *_ = data
        if any(cfg.transition == "legacy" for cfg, _ in configs.values()):
            decoder.prime_legacy()
        matrix = table[features].astype(np.float32)
        probabilities = {name: model(matrix) for name, model in models.items()}
        predictions = {}
        for name, (config, model) in configs.items():
            tick = time.perf_counter()
            pred = decoder.decode(config, probabilities.get(model))
            predictions[name] = pred, time.perf_counter()-tick
        route = None
        if not from_cache:
            # Reference data enters only after the completed predictions exist.
            truth, route = independent_truth(data, components)
        elif args.stage == "val":
            route = np.loadtxt(data[-1].with_suffix(".route"), dtype=int, ndmin=1).tolist()
        hit_masks = {}
        for name, (pred, duration) in predictions.items():
            route_for_method = route if args.stage == "heldout" or name in BASES or name == "frozen" else None
            stat, hit = score(decoder, pred, truth, seconds, duration, route_for_method, components)
            stat["trip_total_seconds"] = time.perf_counter()-trip_start
            records[name][trip] = stat
            hit_masks[name] = hit
        if joint_counts:
            other_hits = np.stack([hit_masks[name] for name in joint_counts]).sum(axis=0)
            oracle_base_hits += int((other_hits > 0).sum())
            for name in joint_counts:
                joint_counts[name] += int((hit_masks[name] & (other_hits == 1)).sum())
        if not args.private:
            print(trip, {n: s[trip]["hits"] for n,s in records.items()}, flush=True)
    results = {name: {**summary(rows), "only_baseline_method_correct": joint_counts.get(name)}
               for name, rows in records.items()}
    if args.stage == "dev":
        choice = max(results, key=lambda name: results[name]["micro_top1"])
        config, model = configs[choice]
        freeze = dict(name=choice, config=asdict(config), model=model, temperature=temp,
                      calibration_trip=CALIBRATION, development_trips=trips, dataset=manifest["dataset_version"])
        write(args.frozen, freeze)
    result = dict(stage=args.stage, dataset=manifest["dataset_version"], frozen=freeze if args.stage == "dev" else frozen,
                  elapsed_seconds=time.perf_counter()-start, summary=results, per_trip=records,
                  any_baseline_correct=oracle_base_hits if joint_counts else None)
    write(args.output, result)
    print(json.dumps({n: {k: v for k,v in row.items() if not args.private or k != "worst_trip"}
                      for n,row in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
