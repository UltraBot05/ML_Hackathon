"""Evaluate the deployed route matcher on the fixed public validation trips."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate(rows, suffix):
    n = sum(r["observations"] for r in rows)
    covered = sum(r[f"{suffix}_covered"] for r in rows)
    hits = sum(r[f"{suffix}_hits"] for r in rows)
    worst = min(rows, key=lambda r: r[f"{suffix}_hits"] / r["observations"])
    return dict(
        observations=n, covered=covered, hits=hits, coverage_top1=hits/n, conditional_top1=hits/covered,
        macro_trip_coverage_top1=float(np.mean([r[f"{suffix}_hits"]/r["observations"] for r in rows])),
        macro_trip_conditional_top1=float(np.mean([r[f"{suffix}_hits"]/r[f"{suffix}_covered"] for r in rows])),
        worst_trip_id=worst["trip_id"], worst_trip_top1=worst[f"{suffix}_hits"]/worst["observations"],
        mean_bridged_continuity_pct=float(np.mean([r["continuity_pct"] for r in rows])),
        mean_directed_route_set_f1=float(np.mean([r["route_set_f1"] for r in rows])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path.")

    repo = args.repo.resolve()
    os.environ["PATHSNAP_REPO_ROOT"] = str(repo)
    from audit_api import FeatureEngineer, load_classifier, run_prediction
    from build_audit_dataset import digest

    cache = args.cache.resolve()
    data = cache / "dataset_manifest.json"
    manifest = json.loads(data.read_text(encoding="utf-8"))
    assert manifest["feature_columns"] == FeatureEngineer.FEATURE_COLUMNS
    model, features = load_classifier(args.model, {args.model: args.archive}, manifest["feature_columns"])
    routes = json.loads((cache / "reference_routes.json").read_text(encoding="utf-8"))
    rows = []
    for trip in manifest["split"]["val_trips"]:
        # The HTTP route and this scorer call the same prediction function. Load labels only after inference.
        result = run_prediction(repo / "Dataset/train" / trip, model, features)
        pred = np.asarray(result["observation_arc_ids"])
        route = set(result["route_arc_ids"])
        reference = set(routes[trip]["arc_ids"])
        key = f"observations/val/{trip}.parquet"
        if digest(cache / key) != manifest["file_sha256"][key]:
            raise ValueError(f"Observation ledger changed for {trip}")
        corrected = pd.read_parquet(cache / key).sort_values("obs_index")
        canonical = pd.read_parquet(
            repo / "phase2/features/val" / f"{trip}_features.parquet", columns=["obs_index", "true_arc_id"],
        ).groupby("obs_index").true_arc_id.first().reindex(range(len(pred)), fill_value=-1)
        assert len(corrected) == len(pred)
        np.testing.assert_array_equal(corrected.obs_index, np.arange(len(pred)))
        v1_hits = corrected.true_arc_id.eq(pred) & corrected.has_positive
        legacy_hits = canonical.ge(0) & canonical.eq(pred)
        row = dict(
            trip_id=trip, observations=len(pred), route_first_covered=int(corrected.has_positive.sum()),
            route_first_hits=int(v1_hits.sum()), legacy_covered=int(canonical.ge(0).sum()),
            legacy_hits=int(legacy_hits.sum()), continuity_pct=result["continuity_pct"],
            route_set_f1=2*len(route&reference)/(len(route)+len(reference)),
            decoded_route_arcs=len(result["route_arc_ids"]), timings_ms=result["timings_ms"],
        )
        rows.append(row)
        print(f"{trip}: {row['route_first_hits']}/{len(pred)} v1 directed hits, "
              f"{row['continuity_pct']:.2f}% reconstructed continuity", flush=True)
    report = dict(
        dataset_version=manifest["dataset_version"], dataset_manifest_sha256=digest(data),
        artifact_sha256=digest(args.archive), model=args.model, per_trip=rows,
        route_first=aggregate(rows, "route_first"), phase2_legacy=aggregate(rows, "legacy"),
    )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ("route_first", "phase2_legacy")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
