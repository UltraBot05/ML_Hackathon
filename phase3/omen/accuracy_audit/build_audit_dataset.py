"""Compare candidate pools and freeze a route-first PathSnap feature cache."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


VERSION = "omen-phase2-route-first-v1"
BASE_COMMIT = "37099cee2786d8168a53444a785cdea299e671c2"
GAP_S = 30
KEYS = ["trip_id", "obs_index"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dependencies(repo):
    sys.path[:0] = [str(repo / "phase2/scripts"), str(repo / "phase3/omen/preliminary/source")]
    from feature_engineering import FeatureEngineer
    from label_generation import LabelGenerator
    from pathsnap_context import motion_window
    from pathsnap_quickstart import load_trip_features, ordered_proxy

    return FeatureEngineer.FEATURE_COLUMNS, LabelGenerator(), motion_window, load_trip_features, ordered_proxy


def compare_pool(obs, arc, rank, proxy):
    n = len(proxy)
    count = np.bincount(obs, minlength=n)
    retrieved = int((arc == proxy[obs]).sum())
    hits = int(((arc == proxy[obs]) & (rank == 1)).sum())
    return {
        "raw_observations": n, "with_candidates": int((count > 0).sum()),
        "proxy_assigned": int((proxy >= 0).sum()), "proxy_present": retrieved,
        "candidate_rows": len(arc), "mean_candidates": float(count.mean()),
        "median_candidates": float(np.median(count)), "nearest_hits": hits,
        "nearest_coverage_top1": hits / n, "nearest_conditional_top1": hits / retrieved if retrieved else None,
    }


def pairwise_alignment(arcs, first, second):
    overlap = (first >= 0) & (second >= 0)
    different = overlap & (first != second)
    reversed_pair = (arcs[first[different]] == arcs[second[different]][:, ::-1]).all(axis=1)
    return {
        "both_assigned": int(overlap.sum()), "agree": int((overlap & (first == second)).sum()),
        "exact_reverse": int(reversed_pair.sum()), "one_unassigned": int(((first >= 0) ^ (second >= 0)).sum()),
    }


def backwards_count(route, labels):
    positions = {}
    for i, arc in enumerate(route):
        positions.setdefault(arc, []).append(i)
    positions = {arc: np.array(indices) for arc, indices in positions.items()}
    pointer = count = 0
    for arc in labels[labels >= 0]:
        available = positions[arc]
        index = np.searchsorted(available, pointer)
        if index < len(available):
            pointer = available[index]
        else:
            count += 1
    return count


def build_trip(repo, trip, split, features, labeler, window, load_trip, align):
    folder = repo / "Dataset/train" / trip
    points, starts, vectors, old = load_trip(folder)
    route = np.loadtxt(folder / f"{trip}.route", dtype=int, ndmin=1)
    endpoints = np.loadtxt(folder / f"{trip}.arcs", dtype=int, ndmin=2)
    seconds = np.loadtxt(folder / f"{trip}.track", usecols=2, ndmin=1)
    if len(points) != len(seconds) or (np.diff(seconds) <= 0).any():
        raise ValueError(f"Timestamp contract failed for {trip}")
    geometric, ambiguity = align(points, starts, vectors, route)
    # Motion windows reset at >30 s; the global route position can advance but never reverse.
    _, _, motion, elapsed = window(points, seconds, 3)
    length = np.linalg.norm(motion, axis=1)
    speed = np.divide(length, elapsed, out=np.zeros_like(length), where=elapsed > 0)
    direction = np.divide(motion, length[:, None], out=np.zeros_like(motion), where=length[:, None] > 0)
    reliability = np.clip((speed - .8) / 2.5, 0, 1) * np.clip(length / 5, 0, 1) * (speed <= 55)
    independent, _ = align(points, starts, vectors, route, motion=direction, reliability=reliability)
    assert backwards_count(route, independent) == 0

    columns = [*KEYS, "candidate_arc_id", "true_arc_id", "is_correct", "arc_from_node", "arc_to_node", *features]
    phase2 = pd.read_parquet(repo / "phase2/features" / split / f"{trip}_features.parquet", columns=columns)
    if phase2.empty or set(phase2.trip_id) != {trip} or not np.isfinite(phase2[features]).all().all():
        raise ValueError(f"Invalid canonical features for {trip}")
    ref_window = phase2.groupby("obs_index").true_arc_id.first().reindex(range(len(points)), fill_value=-1).to_numpy()
    old = old.rename(columns={"observation_idx": "obs_index", "arc_id": "candidate_arc_id",
                              "distance_m": "distance_to_arc_m"})
    old.distance_to_arc_m = old.distance_to_arc_m.round(3)
    old_mapping, _ = labeler.align_trip_observations(
        SimpleNamespace(route=route.tolist(), trip_id=trip), range(len(points)),
        old[["obs_index", "candidate_arc_id", "distance_to_arc_m"]].to_dict("records"),
    )
    old_window = np.array([old_mapping[i] if old_mapping[i] is not None else -1 for i in range(len(points))])

    comparisons = []
    for name, pool, window_labels in (("omen-75m-k8", old, old_window), ("phase2-60-120m-k10", phase2, ref_window)):
        indices = pool.obs_index.to_numpy()
        arcs = pool.candidate_arc_id.to_numpy()
        ranks = pool.candidate_rank.to_numpy()
        for label, reference in (("candidate-window", window_labels), ("ordered-geometry", geometric),
                                 ("route-first-gap-aware-v1", independent)):
            comparisons.append(dict(
                trip_id=trip, split=split, candidate_generator=name, label_alignment=label,
                **compare_pool(indices, arcs, ranks, reference),
                **{f"vs_frozen_{key}": value for key, value in pairwise_alignment(endpoints, reference, independent).items()},
                chronological_conflicts=backwards_count(route, reference),
                geometric_near_ties=int(ambiguity.sum()), timestamp_gaps_gt30=int((np.diff(seconds) > GAP_S).sum()),
                repeated_route_positions=len(route)-len(set(route)),
            ))

    # Preserve the Phase 2 variables byte-for-value. Replace only evaluation metadata/targets in the new cache.
    pos = phase2.obs_index.to_numpy()
    phase2["true_arc_id"] = independent[pos]
    phase2["is_correct"] = phase2.candidate_arc_id.eq(phase2.true_arc_id).astype(np.int8)
    labels = pd.DataFrame({
        "trip_id": trip, "obs_index": np.arange(len(points)), "true_arc_id": independent,
        "has_candidates": np.isin(np.arange(len(points)), pos),
        "has_positive": np.isin(np.arange(len(points)), pos[phase2.is_correct.to_numpy() == 1]),
        "geometric_near_tie": ambiguity,
    })
    return phase2, labels, comparisons, {
        "arc_ids": route.tolist(), "length_m": float(np.linalg.norm(vectors[route], axis=1).sum()),
    }


def aggregate(audit):
    summary = []
    columns = ["split", "candidate_generator", "label_alignment"]
    sums = ["raw_observations", "with_candidates", "proxy_assigned", "proxy_present",
            "candidate_rows", "nearest_hits", "vs_frozen_both_assigned", "vs_frozen_agree",
            "vs_frozen_exact_reverse", "chronological_conflicts"]
    for keys, rows in audit.groupby(columns, sort=True):
        values = {key: int(rows[key].sum()) for key in sums}
        summary.append(dict(
            zip(columns, keys), **values,
            nearest_coverage_top1=values["nearest_hits"]/values["raw_observations"],
            nearest_conditional_top1=values["nearest_hits"]/values["proxy_present"],
        ))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Verify and reuse files from an interrupted build.")
    args = parser.parse_args()
    repo = args.repo.resolve()
    features, labeler, window, load_trip, align = dependencies(repo)
    manifest_path = repo / "phase2/results/dataset_split_manifest.json"
    split = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (len(set(split["train_trips"])), len(set(split["val_trips"]))) == (64, 16)
    assert not set(split["train_trips"]) & set(split["val_trips"])
    if (args.output / "dataset_manifest.json").exists():
        parser.error("This cache is already frozen. Use a fresh output directory.")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    records, routes, files = [], {}, {}
    for partition in ("train", "val"):
        for family in ("features", "observations"):
            (args.output / family / partition).mkdir(parents=True, exist_ok=args.resume)
        for number, trip in enumerate(split[f"{partition}_trips"], 1):
            rows, obs, info, route = build_trip(repo, trip, partition, features, labeler, window, load_trip, align)
            for family, frame in (("features", rows), ("observations", obs)):
                rel = Path(family) / partition / f"{trip}.parquet"
                target = args.output / rel
                if target.exists():
                    pd.testing.assert_frame_equal(pd.read_parquet(target), frame)
                else:
                    frame.to_parquet(target, index=False)
                files[rel.as_posix()] = digest(target)
            records.extend(info)
            routes[trip] = route
            if number % 8 == 0:
                print(f"{partition}: wrote {number}/{len(split[f'{partition}_trips'])} trips", flush=True)
    audit = pd.DataFrame(records)
    audit.to_csv(args.output / "candidate_audit.csv", index=False)
    (args.output / "reference_routes.json").write_text(json.dumps(routes), encoding="utf-8")
    for name in ("candidate_audit.csv", "reference_routes.json"):
        files[name] = digest(args.output / name)
    output = dict(
        dataset_version=VERSION, base_commit=BASE_COMMIT, manifest_sha256=digest(manifest_path),
        generator_sha256=digest(Path(__file__)), split=split, feature_columns=features,
        candidate_generator="Phase 2 archived 60/120 m KD-tree, nearest-sample fallback, max 10",
        label_alignment="Global monotone DP over all ordered route positions, independent of candidates. "
                        "Uses segment distance and an 8 m directional penalty weighted by GPS reliability. "
                        "Motion windows of +/-3 observations stop across >30 s gaps. Progress remains monotone "
                        "through gaps; repeated arc IDs occupy distinct route positions.",
        scoring="Highest candidate probability per observation; exact directed arc ID; ties by spatial rank then arc ID. "
                "Coverage-aware denominator is every raw observation. Conditional denominator has the proxy in the pool.",
        seed=split["random_seed"], file_sha256=files, controlled_comparison=aggregate(audit),
    )
    (args.output / "dataset_manifest.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve() / "dataset_manifest.json", flush=True)


if __name__ == "__main__":
    main()
