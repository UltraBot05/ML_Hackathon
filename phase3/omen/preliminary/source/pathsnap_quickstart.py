"""Provisional map-matching classifier experiments on the PathSnap training partition."""

import argparse
import hashlib
import io
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "distance_m", "heading_difference_rad", "segment_length_m",
    "gps_speed_mps", "timestamp_gap_s", "candidate_rank",
]
KEYS = ["trip_id", "observation_idx"]
RADIUS_M = 75
TOP_K = 8
AMBIGUITY_M = 2
SEED = 42
CACHE_VERSION = 1


def segment_distance(points, starts, vectors):
    offset = points - starts
    squared_length = np.sum(vectors**2, axis=-1)
    projection = np.sum(offset * vectors, axis=-1)
    fraction = np.divide(
        projection, squared_length, out=np.zeros_like(projection), where=squared_length > 0,
    )
    return np.linalg.norm(offset - np.clip(fraction, 0, 1)[..., None] * vectors, axis=-1)


def candidate_pairs(points, starts, vectors, radius=RADIUS_M, top_k=TOP_K):
    lengths = np.linalg.norm(vectors, axis=1)
    midpoints = starts + vectors / 2
    batches = []
    # Within each length bucket, the triangle inequality bounds midpoint distance.
    buckets = np.ceil(np.log2(np.maximum(lengths, 1)))
    for bucket in np.unique(buckets):
        arc_ids = np.flatnonzero(buckets == bucket)
        neighbors = cKDTree(midpoints[arc_ids]).query_ball_point(
            points, radius + lengths[arc_ids].max() / 2 + 1e-8,
        )
        sizes = np.fromiter(map(len, neighbors), dtype=int, count=len(points))
        if not sizes.any():
            continue
        obs = np.repeat(np.arange(len(points)), sizes)
        arc = arc_ids[np.concatenate(neighbors.tolist()).astype(int)]
        distance = segment_distance(points[obs], starts[arc], vectors[arc])
        within = distance <= radius
        batches.append((obs[within], arc[within], distance[within]))
    if not batches:
        return np.array([], int), np.array([], int), np.array([], float), np.array([], int)

    obs, arc, distance = (np.concatenate(column) for column in zip(*batches))
    order = np.lexsort((arc, distance, obs))
    obs, arc, distance = obs[order], arc[order], distance[order]
    if not len(obs):
        return obs, arc, distance, np.array([], int)
    row = np.arange(len(obs))
    rank = row - np.maximum.accumulate(np.where(np.r_[True, obs[1:] != obs[:-1]], row, 0)) + 1
    keep = rank <= top_k
    return obs[keep], arc[keep], distance[keep], rank[keep]


def ordered_proxy(points, starts, vectors, route, *, motion=None, reliability=None):
    """Choose a nondecreasing sequence of route positions by total geometric distance."""
    route_starts, route_vectors = starts[route], vectors[route]
    parents = np.empty((len(points), len(route)), dtype=np.int32)
    ambiguous = np.empty(len(points), dtype=bool)
    positions = np.arange(len(route))
    previous = np.zeros(len(route))
    if motion is not None:
        lengths = np.linalg.norm(route_vectors, axis=1)
        unit = np.divide(
            route_vectors, lengths[:, None], out=np.zeros_like(route_vectors), where=lengths[:, None] > 0,
        )
    for i, point in enumerate(points):
        distance = segment_distance(point, route_starts, route_vectors)
        close = distance <= distance.min() + AMBIGUITY_M
        ambiguous[i] = len(np.unique(route[close])) > 1
        if motion is not None:
            # Penalize opposing arc directions only when the smoothed GPS motion is reliable.
            direction_cost = 8 * reliability[i] * (1 - np.clip(np.sum(unit * motion[i], axis=1), -1, 1))
            distance = distance + direction_cost
        prefix = np.minimum.accumulate(previous)
        new_minimum = np.r_[True, prefix[1:] < prefix[:-1]]
        parents[i] = np.maximum.accumulate(np.where(new_minimum, positions, 0))
        previous = prefix + distance

    chosen = np.empty(len(points), dtype=int)
    state = int(np.argmin(previous))
    for i in range(len(points) - 1, -1, -1):
        chosen[i] = route[state]
        state = parents[i, state]
    return chosen, ambiguous


def load_trip_features(folder, *, context=False, bridges=False, radius=RADIUS_M, top_k=TOP_K):
    def read(suffix, *, dtype=float, ndmin=2):
        return np.loadtxt(folder / f"{folder.name}.{suffix}", dtype=dtype, ndmin=ndmin)

    track = read("track")
    nodes = read("nodes")
    arcs = read("arcs", dtype=int)
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError(f"Unexpected file schema for {folder.name}")
    if not len(track) or not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError(f"Invalid coordinates or timestamps in {folder.name}")
    if not len(arcs) or arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError(f"Invalid arc endpoints in {folder.name}")

    origin = track[:, :2].mean(axis=0)
    metres_per_degree = np.pi * 6_371_008.8 / 180
    # Local equirectangular projection, centered on each trip.
    scale = metres_per_degree * np.array([np.cos(np.deg2rad(origin[1])), 1.0])
    points = (track[:, :2] - origin) * scale
    nodes = (nodes - origin) * scale
    starts = nodes[arcs[:, 0]]
    vectors = nodes[arcs[:, 1]] - starts
    lengths = np.linalg.norm(vectors, axis=1)
    gap = np.diff(track[:, 2], prepend=track[0, 2])
    if (gap[1:] <= 0).any():
        raise ValueError(f"Non-increasing timestamps in {folder.name}")
    motion = np.diff(points, axis=0, prepend=points[:1])
    displacement = np.linalg.norm(motion, axis=1)
    speed = np.divide(displacement, gap, out=np.zeros_like(gap), where=gap > 0)
    obs, arc, distance, rank = candidate_pairs(points, starts, vectors, radius=radius, top_k=top_k)
    heading = np.arctan2(motion[obs, 1], motion[obs, 0])
    direction = np.arctan2(vectors[arc, 1], vectors[arc, 0])
    angle = np.abs((heading - direction + np.pi) % (2 * np.pi) - np.pi)
    angle[(displacement[obs] == 0) | (lengths[arc] == 0)] = np.pi / 2
    rows = pd.DataFrame({
        "trip_id": folder.name, "observation_idx": obs, "arc_id": arc,
        "distance_m": distance, "heading_difference_rad": angle,
        "segment_length_m": lengths[arc], "gps_speed_mps": speed[obs],
        "timestamp_gap_s": gap[obs], "candidate_rank": rank,
    })
    if context:
        from pathsnap_context import add_context

        rows = add_context(rows, points, track[:, 2], arcs, starts, vectors, lengths, top_k, bridges=bridges)
    return points, starts, vectors, rows


def build_cache(root, trips, *, context=False, bridges=False, radius=RADIUS_M, top_k=TOP_K,
                directional_labels=False):
    candidates, observations = [], []
    for trip in trips:
        folder = root / trip
        points, starts, vectors, rows = load_trip_features(
            folder, context=context, bridges=bridges, radius=radius, top_k=top_k,
        )
        route = np.loadtxt(folder / f"{trip}.route", dtype=int, ndmin=1)
        if route.ndim != 1 or not len(route) or route.min() < 0 or route.max() >= len(starts):
            raise ValueError(f"Invalid route arc IDs in {trip}")
        proxy, ambiguous = ordered_proxy(points, starts, vectors, route)
        obs = rows["observation_idx"].to_numpy()
        rows["label"] = (rows["arc_id"].to_numpy() == proxy[obs]).astype(np.int8)
        if directional_labels:
            from pathsnap_context import motion_window

            seconds = np.loadtxt(folder / f"{trip}.track", usecols=2, ndmin=1)
            _, _, motion, elapsed = motion_window(points, seconds, 3)
            length = np.linalg.norm(motion, axis=1)
            speed = np.divide(length, elapsed, out=np.zeros_like(length), where=elapsed > 0)
            unit_motion = np.divide(
                motion, length[:, None], out=np.zeros_like(motion), where=length[:, None] > 0,
            )
            reliability = np.clip((speed - .8) / 2.5, 0, 1) * np.clip(length / 5, 0, 1)
            directional, _ = ordered_proxy(
                points, starts, vectors, route, motion=unit_motion, reliability=reliability,
            )
            rows["training_label"] = (rows["arc_id"].to_numpy() == directional[obs]).astype(np.int8)
            print(f"{trip}: direction-aware proxy differs at {(directional != proxy).sum()} observations", flush=True)
        present = np.zeros(len(points), dtype=bool)
        present[obs[rows["label"].to_numpy() == 1]] = True
        observations.append(pd.DataFrame({
            "trip_id": trip, "observation_idx": np.arange(len(points)),
            "proxy_arc_id": proxy, "ambiguous_geometry": ambiguous, "candidate_present": present,
        }))
        candidates.append(rows)
        print(f"{trip}: {len(points)} observations, {len(rows)} candidates, "
              f"{(~present).sum()} misses, {ambiguous.sum()} ambiguous", flush=True)
    return pd.concat(candidates, ignore_index=True), pd.concat(observations, ignore_index=True)


def evaluate(rows, obs, probabilities):
    y = rows["label"].to_numpy()
    predicted = probabilities >= 0.5
    scored = rows[[*KEYS, "arc_id", "label"]].assign(probability=probabilities)
    winners = scored.loc[scored.groupby(KEYS, sort=False)["probability"].idxmax()]
    correct = int(winners["label"].sum())
    covered = int(obs["candidate_present"].sum())
    return {
        "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "macro_f1": f1_score(y, predicted, average="macro", labels=[0, 1], zero_division=0),
        "log_loss": log_loss(y, probabilities, labels=[0, 1]),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "top1_conditional": correct / covered if covered else None,
        "top1_coverage_aware": correct / len(obs),
        "candidate_miss_rate": 1 - covered / len(obs),
        "candidate_miss_count": len(obs) - covered,
        "observations_without_candidates": len(obs) - len(winners),
        "ambiguous_geometry_count": int(obs["ambiguous_geometry"].sum()),
        "observations": len(obs), "candidate_rows": len(rows),
    }, scored


def fit_model(name, train, val, val_obs, features, *, save_model=True):
    if name == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    else:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="binary:logistic", tree_method="hist", n_estimators=150, max_depth=5,
            learning_rate=0.08, subsample=0.85, colsample_bytree=0.85,
            random_state=SEED, n_jobs=4, device="cpu",
        )
    target = train["training_label"] if "training_label" in train else train["label"]
    if target.nunique() != 2 or val.empty:
        raise ValueError("The selected split needs both training classes and validation candidates.")
    start = time.perf_counter()
    model.fit(train[features], target)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probabilities = model.predict_proba(val[features])[:, 1]
    inference_seconds = time.perf_counter() - start
    metrics, scored = evaluate(val, val_obs, probabilities)
    metrics.update(
        training_seconds=fit_seconds, inference_seconds=inference_seconds,
        training_candidate_rows=len(train), training_positive_rows=int(target.sum()),
        configuration={**model.get_params(), "missing": "NaN"} if name == "xgboost" else {
            "scaler": model.named_steps["standardscaler"].get_params(),
            "classifier": model.named_steps["logisticregression"].get_params(),
        },
    )
    if not save_model:
        return metrics, None, None
    binary = io.BytesIO()
    joblib.dump(model, binary)
    binary.seek(0)
    reloaded = joblib.load(binary)
    np.testing.assert_allclose(reloaded.predict_proba(val[features])[:, 1], probabilities)
    return metrics, scored, binary.getvalue()


def trip_rows(frame, trip_ids):
    return frame[frame["trip_id"].isin(trip_ids)]


def team_summary(report):
    metrics = report["metrics"]
    student_test = bool(report.get("student_test_path"))
    xgboost = report["model"] == "xgboost"
    config = metrics["configuration"] if xgboost else metrics["configuration"]["classifier"]
    parameter_names = (
        "objective", "tree_method", "n_estimators", "max_depth", "learning_rate",
        "subsample", "colsample_bytree", "random_state", "n_jobs", "device",
    ) if xgboost else ("C", "solver", "max_iter", "class_weight", "random_state")
    return {
        "model": "XGBoost" if xgboost else "Logistic Regression",
        "owner": "omen",
        "seed": report["random_seed"],
        "training_trip_count": len(report["train_trip_ids"]),
        "cv_macro_f1": report.get("cv_mean_macro_f1"),
        "student_test_top1_accuracy": metrics["top1_coverage_aware"] if student_test else None,
        "student_test_macro_f1": metrics["macro_f1"] if student_test else None,
        "student_test_log_loss": metrics["log_loss"] if student_test else None,
        "route_continuity": None,
        "training_seconds": metrics["training_seconds"],
        "inference_seconds": metrics["inference_seconds"],
        "hyperparameters": {name: config[name] for name in parameter_names},
        "run_completed": True,
    }


def cross_validate(name, rows, observations, trips, folds, features):
    results = []
    for fold, (train, val) in enumerate(GroupKFold(folds).split(trips, groups=trips), 1):
        train_ids, val_ids = trips[train].tolist(), trips[val].tolist()
        val_obs = trip_rows(observations, val_ids)
        metrics, _, _ = fit_model(
            name, trip_rows(rows, train_ids), trip_rows(rows, val_ids), val_obs, features, save_model=False,
        )
        metrics.pop("configuration")
        metrics["training_observations"] = len(trip_rows(observations, train_ids))
        results.append(dict(fold=fold, train_trip_ids=train_ids, validation_trip_ids=val_ids, **metrics))
        print(f"{name} fold {fold}: macro F1 {metrics['macro_f1']:.4f}, "
              f"Top-1 {metrics['top1_coverage_aware']:.4f}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Student folder/Student folder/train"))
    parser.add_argument("--output", type=Path, default=Path("pathsnap_preliminary"))
    parser.add_argument("--trips", type=int, default=6)
    parser.add_argument("--context", action="store_true", help="Add GPS-window and network context features.")
    parser.add_argument("--bridges", action="store_true", help="Include short directed road connections.")
    parser.add_argument("--directional-labels", action="store_true",
                        help="Use motion-aware proxies for training and the original proxies for evaluation.")
    parser.add_argument("--radius", type=float, default=RADIUS_M)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--student-test", type=Path, help="Refit on all chosen training trips and evaluate here.")
    parser.add_argument("--model", choices=["logistic", "xgboost", "both"], default="both")
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args()
    root = args.data.resolve()
    if root.name != "train" or args.trips < 3:
        parser.error("Use a training partition named train and at least three trips.")
    if not np.isfinite(args.radius) or args.radius <= 0 or args.top_k < 1:
        parser.error("Set a positive finite candidate radius and a positive candidate limit.")
    if args.bridges and not args.context:
        parser.error("--bridges requires --context.")
    trips = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdecimal())[:args.trips]
    if len(trips) != args.trips:
        parser.error(f"The training partition contains only {len(trips)} usable trip directories.")
    if args.cv_folds != 0 and not 2 <= args.cv_folds <= len(trips):
        parser.error("--cv-folds must be zero or between two and the number of training trips.")
    test_root = args.student_test.resolve() if args.student_test else None
    if test_root and (test_root.name != "student_test" or test_root.parent != root.parent):
        parser.error("Use the student_test directory beside the training partition.")
    models = ("logistic", "xgboost") if args.model == "both" else (args.model,)
    for name in models:
        if (args.output / f"{name}.zip").exists():
            parser.error(f"The {name} archive already exists. Pass a fresh --output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    cache_file = args.output / "candidate_features.joblib"
    source = {
        "cache_version": CACHE_VERSION, "dataset_path": str(root), "trip_ids": trips,
        "radius_m": args.radius, "top_k": args.top_k, "geometry_ambiguity_margin_m": AMBIGUITY_M,
        "source_files": {
            f"{trip}/{suffix}": (root / trip / f"{trip}.{suffix}").stat().st_mtime_ns
            for trip in trips for suffix in ("track", "nodes", "arcs", "route")
        },
    }
    features = FEATURES
    if args.context:
        from pathsnap_context import BRIDGE_FEATURES, CONTEXT_FEATURES

        features = [*FEATURES, *CONTEXT_FEATURES]
        source["context_feature_sha256"] = hashlib.sha256(
            Path(__file__).with_name("pathsnap_context.py").read_bytes()
        ).hexdigest()
        if args.bridges:
            features = [*features, *BRIDGE_FEATURES]
            source["directed_bridge_arc_limit"] = 2
    if args.directional_labels:
        source["training_label_method"] = (
            "Ordered route alignment with 8 m scaled penalty for disagreement with smoothed GPS motion. "
            "Scoring still uses the geometric proxy."
        )
    if cache_file.exists():
        cache_source, rows, observations = joblib.load(cache_file)
        if cache_source != source:
            parser.error("Cache inputs changed. Pass a fresh --output directory.")
    else:
        rows, observations = build_cache(
            root, trips, context=args.context, bridges=args.bridges, radius=args.radius, top_k=args.top_k,
            directional_labels=args.directional_labels,
        )
        joblib.dump((source, rows, observations), cache_file, compress=3)

    if test_root:
        train_ids = trips
        val_ids = sorted(p.name for p in test_root.iterdir() if p.is_dir() and p.name.isdecimal())
        if len(val_ids) != 10 or set(trips) & set(val_ids):
            parser.error("Expected 10 student-test trips disjoint from the training trip IDs.")
        print("Preparing student-test data for evaluation only.", flush=True)
        val_rows, val_obs = build_cache(
            test_root, val_ids, context=args.context, bridges=args.bridges, radius=args.radius, top_k=args.top_k,
        )
    else:
        train, val = next(GroupShuffleSplit(
            n_splits=1, test_size=1 / 3, random_state=SEED,
        ).split(trips, groups=trips))
        train_ids, val_ids = np.asarray(trips)[train].tolist(), np.asarray(trips)[val].tolist()
        val_rows, val_obs = trip_rows(rows, val_ids), trip_rows(observations, val_ids)
    train_rows = trip_rows(rows, train_ids)
    metadata = {
        **source, "source_commit": args.source_commit, "preliminary": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in ("joblib", "numpy", "pandas", "scipy", "scikit-learn")},
        "features": features, "random_seed": SEED,
        "protocol": (
            "Refit on training trips, evaluate on separate student-test trips" if test_root
            else "Single group-held-out split of training partition"
        ),
        "student_test_path": str(test_root) if test_root else None,
        "train_trip_ids": train_ids, "validation_trip_ids": val_ids,
        "label_method": "PROXY labels: ordered route alignment minimizing total segment distance "
                        "with nondecreasing route positions. Ambiguity counts distinct route arcs "
                        "within 2 m of the locally closest route arc.",
    }
    for name in models:
        archive = args.output / f"{name}.zip"
        cv = cross_validate(
            name, rows, observations, np.asarray(trips), args.cv_folds, features,
        ) if args.cv_folds else []
        metrics, scored, model = fit_model(name, train_rows, val_rows, val_obs, features)
        metrics["training_observations"] = len(trip_rows(observations, train_ids))
        report = {**metadata, "model": name, "metrics": metrics, "group_kfold": cv}
        if name == "xgboost":
            report["package_versions"] = {**report["package_versions"], "xgboost": version("xgboost")}
        if cv:
            report["cv_mean_macro_f1"] = float(np.mean([fold["macro_f1"] for fold in cv]))
        report_json = json.dumps(report, indent=2, allow_nan=False)
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as output:
            output.writestr("model.joblib", model)
            output.writestr("results.json", report_json)
            output.writestr("team_summary.json", json.dumps(team_summary(report), indent=2, allow_nan=False))
            output.writestr("validation_probabilities.csv", scored.to_csv(index=False))
            output.writestr("validation_observations.csv", val_obs.to_csv(index=False))
        print(report_json, flush=True)
        print(f"Artifact: {archive.resolve()}", flush=True)


if __name__ == "__main__":
    main()
