"""Fit Omen classifiers on the published PathSnap Phase 2 features."""

import argparse
import io
import json
from pathlib import Path
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


KEYS = ["trip_id", "obs_index"]
TARGET = "is_correct"
METADATA = [*KEYS, "candidate_arc_id", "true_arc_id", TARGET]
SOURCE_COMMIT = "59f573e2fdd76d546069bac0b204f1c64590e502"
MODELS = {"logistic": "Logistic Regression", "xgboost": "XGBoost"}


def feature_contract(root):
    sys.path.insert(0, str(root / "phase2" / "scripts"))
    from feature_engineering import FeatureEngineer

    features = list(FeatureEngineer.FEATURE_COLUMNS)
    forbidden = {*METADATA, "timestamp", "gps_lon", "gps_lat", "arc_from_node", "arc_to_node"}
    if len(features) != 22 or len(set(features)) != 22 or forbidden.intersection(features):
        raise ValueError("Unexpected Phase 2 feature contract")
    return features


def load_partition(root, split, trips, features, recorded_coverage):
    directory = root / "phase2" / "features" / split
    files = {p.stem.removesuffix("_features"): p for p in directory.glob("*_features.parquet")}
    if set(files) != set(trips) or len(list(directory.glob("*.parquet"))) != len(trips):
        raise ValueError(f"{split} Parquet trip IDs differ from the manifest")
    parts, observations, per_trip = [], [], []
    for trip in trips:
        frame = pd.read_parquet(files[trip], columns=[*METADATA, *features])
        if frame.empty or set(frame.trip_id) != {trip}:
            raise ValueError(f"{files[trip]} has unexpected trip IDs or no candidates")
        if frame[METADATA].isna().any().any() or frame.duplicated([*KEYS, "candidate_arc_id"]).any():
            raise ValueError(f"{trip} has null metadata or duplicate candidates")
        correct = (frame.true_arc_id >= 0) & (frame.true_arc_id == frame.candidate_arc_id)
        if not frame[TARGET].eq(correct.astype(int)).all():
            raise ValueError(f"{trip} has labels inconsistent with its proxy arc IDs")
        by_obs = frame.groupby("obs_index", sort=False)
        positives = by_obs[TARGET].sum()
        if positives.gt(1).any() or by_obs["true_arc_id"].nunique().gt(1).any():
            raise ValueError(f"{trip} has inconsistent reference labels within an observation")
        expected = recorded_coverage.loc[trip]
        total = int(expected.total_observations)
        if (len(frame) != expected.total_candidates_generated or len(positives) != expected.points_with_candidates
                or positives.index.min() < 0 or positives.index.max() >= total):
            raise ValueError(f"{trip} does not match the Phase 2 coverage ledger")
        x = frame[features].to_numpy(dtype=np.float32)
        nan_count = int(np.isnan(x).sum())
        infinity_count = int(np.isinf(x).sum())
        if nan_count or infinity_count:
            raise ValueError(f"{trip} has {nan_count} NaNs and {infinity_count} infinities")
        obs = pd.DataFrame({"trip_id": trip, "obs_index": np.arange(total)})
        obs["has_candidates"] = obs.obs_index.isin(positives.index)
        obs["has_positive"] = obs.obs_index.isin(positives[positives.eq(1)].index)
        obs["true_arc_id"] = obs.obs_index.map(by_obs["true_arc_id"].first()).fillna(-1).astype(np.int32)
        observations.append(obs)
        per_trip.append(dict(trip_id=trip, observations=total, candidate_rows=len(frame),
                             no_candidates=total-len(positives), no_positive=total-int(positives.sum())))
        meta = frame[METADATA].astype({c: np.int32 for c in METADATA if c != "trip_id"})
        parts.append(pd.concat([meta, pd.DataFrame(x, columns=features)], axis=1))

    rows = pd.concat(parts, ignore_index=True)
    obs = pd.concat(observations, ignore_index=True)
    audit = {
        "parquet_files": len(trips), "trip_ids": trips, "observations": len(obs),
        "candidate_rows": len(rows), "positive_candidates": int(rows[TARGET].sum()),
        "positive_candidate_fraction": float(rows[TARGET].mean()),
        "observations_without_candidates": int((~obs.has_candidates).sum()),
        "observations_without_positive": int((~obs.has_positive).sum()),
        "represented_groups_without_positive": int((obs.has_candidates & ~obs.has_positive).sum()),
        "feature_nans": 0, "feature_infinities": 0, "per_trip": per_trip,
    }
    return rows, obs, audit


def make_model(name, seed):
    if name == "logistic":
        params = dict(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=seed)
        return make_pipeline(StandardScaler(), LogisticRegression(**params)), params
    from xgboost import XGBClassifier

    params = dict(objective="binary:logistic", tree_method="hist", device="cpu", n_estimators=150,
                  max_depth=5, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85,
                  random_state=seed, n_jobs=4)
    return XGBClassifier(**params), params


def score(rows, observations, probabilities):
    target = rows[TARGET].to_numpy()
    binary = probabilities >= 0.5
    # Parquet candidate rank (then arc ID) breaks ties without using reference information.
    scored = rows[[*METADATA, "candidate_rank"]].assign(probability=probabilities)
    order = scored.sort_values(["probability", "candidate_rank", "candidate_arc_id"],
                               ascending=[False, True, True])
    winners = order.drop_duplicates(KEYS)
    correct = int(winners[TARGET].sum())
    aligned = int(observations.has_positive.sum())
    return dict(
        accuracy=accuracy_score(target, binary),
        precision=precision_score(target, binary, zero_division=0),
        recall=recall_score(target, binary, zero_division=0),
        macro_f1=f1_score(target, binary, average="macro", labels=[0, 1], zero_division=0),
        log_loss=log_loss(target, probabilities, labels=[0, 1]),
        confusion_matrix=confusion_matrix(target, binary, labels=[0, 1]).tolist(),
        top1_conditional=correct / aligned if aligned else None,
        top1_coverage_aware=correct / len(observations),
        observations_without_positive=len(observations)-aligned,
        observations_without_candidates=int((~observations.has_candidates).sum()),
        observations=len(observations), candidates=len(rows), top1_correct=correct,
    ), scored


def fit(name, seed, x_train, y_train, x_val):
    model, params = make_model(name, seed)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    training_time = time.perf_counter() - start
    start = time.perf_counter()
    scores = model.predict_proba(x_val)[:, 1]
    return model, params, scores, training_time, time.perf_counter() - start


def training_cv(name, seed, x, y, groups, folds):
    if not folds:
        return []
    results = []
    for fold, (tr, va) in enumerate(GroupKFold(folds).split(x, y, groups), 1):
        _, _, p, _, _ = fit(name, seed, x.iloc[tr], y.iloc[tr], x.iloc[va])
        f1 = f1_score(y.iloc[va], p >= 0.5, average="macro", labels=[0, 1], zero_division=0)
        results.append(dict(fold=fold, macro_f1=f1, train_trips=sorted(set(groups.iloc[tr])),
                            validation_trips=sorted(set(groups.iloc[va]))))
        print(f"{name} training CV fold {fold}: macro F1 {f1:.4f}", flush=True)
    return results


def chart(reports):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    names = ["Candidate macro F1", "Conditional Top-1", "Coverage-aware Top-1"]
    fields = ["macro_f1", "top1_conditional", "top1_coverage_aware"]
    for i, (name, report) in enumerate(reports.items()):
        values = [report["validation"][field] * 100 for field in fields]
        bars = ax.bar(np.arange(3) + .34 * (i - .5), values, .34, label=MODELS[name])
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xticks(range(3), names)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percent")
    ax.set_title("PathSnap Review 1 · Phase 2 validation (16 trips)")
    ax.legend(loc="lower right")
    data = io.BytesIO()
    fig.savefig(data, dpi=160, format="png")
    return data.getvalue()


def json_bytes(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()
    root = args.repo.resolve()
    if args.output.exists():
        parser.error("The output already exists. Choose a new path.")
    features = feature_contract(root)
    results_root = root / "phase2" / "results"
    manifest = json.loads((results_root / "dataset_split_manifest.json").read_text(encoding="utf-8"))
    training, validation = manifest["train_trips"], manifest["val_trips"]
    if (len(set(training)) != len(training) or len(set(validation)) != len(validation)
            or (len(training), len(validation)) != (64, 16) or set(training).intersection(validation)):
        raise ValueError("Expected the canonical 64/16 disjoint trip split")
    if args.cv_folds != 0 and not 2 <= args.cv_folds <= len(training):
        parser.error("Use zero or between two and 64 CV folds.")
    coverage = pd.read_csv(results_root / "candidate_coverage_summary.csv", dtype={"trip_id": str})
    if set(coverage.trip_id) != set(training + validation):
        raise ValueError("The Phase 2 coverage ledger has unexpected trips")
    coverage = coverage.set_index("trip_id", verify_integrity=True)
    train, _, train_audit = load_partition(root, "train", training, features, coverage)
    val, val_obs, val_audit = load_partition(root, "val", validation, features, coverage)
    print(f"Audit passed: {len(train):,} training and {len(val):,} validation candidates", flush=True)
    reports = {}
    temp = args.output.with_suffix(".partial")
    with ZipFile(temp, "x", compression=ZIP_DEFLATED) as archive, threadpool_limits(limits=4):
        archive.writestr("audit.json", json_bytes(dict(
            source_commit=SOURCE_COMMIT, manifest=manifest, features=features, trip_overlap=[],
            train=train_audit, validation=val_audit,
            label_method="Phase 2 route-window alignment selects a candidate arc from the supplied reference route; "
                         "an observation without a route candidate has no positive. Labels are proxies.",
            feature_provenance="GPS track, candidate geometry and previous GPS-only rank-1 candidate. "
                               "Reference labels are copied into metadata and excluded from all 22 inputs.",
        )))
        x_train, x_val = train[features], val[features]
        for name in MODELS:
            model, params, prob, fit_s, infer_s = fit(
                name, manifest["random_seed"], x_train, train[TARGET], x_val,
            )
            metrics, predictions = score(val, val_obs, prob)
            print(f"{MODELS[name]} validation: {metrics['macro_f1']:.4f} macro F1, "
                  f"{metrics['top1_coverage_aware']:.2%} coverage-aware Top-1", flush=True)
            fitted = io.BytesIO()
            joblib.dump(model, fitted)
            fitted.seek(0)
            np.testing.assert_allclose(joblib.load(fitted).predict_proba(x_val.iloc[:256])[:, 1], prob[:256])
            archive.writestr(f"{name}/model.joblib", fitted.getvalue())
            if name == "xgboost":
                archive.writestr(f"{name}/model.ubj", bytes(model.get_booster().save_raw(raw_format="ubj")))
            del model
            grouped_cv = training_cv(name, manifest["random_seed"], x_train, train[TARGET],
                                     train["trip_id"], args.cv_folds)
            team_summary = dict(
                model=MODELS[name], owner="omen", seed=manifest["random_seed"], training_trip_count=len(training),
                cv_macro_f1=float(np.mean([f["macro_f1"] for f in grouped_cv])) if grouped_cv else None,
                student_test_top1_accuracy=None, student_test_macro_f1=None, student_test_log_loss=None,
                route_continuity=None, training_seconds=fit_s, inference_seconds=infer_s,
                hyperparameters=params, run_completed=True,
            )
            report = dict(**team_summary, phase2_source_commit=SOURCE_COMMIT, feature_order=features,
                          train_trips=training, validation_trips=validation, validation=metrics,
                          group_cv=grouped_cv)
            reports[name] = report
            archive.writestr(f"{name}/results.json", json_bytes(report))
            archive.writestr(f"{name}/team_summary.json", json_bytes(team_summary))
            data = io.BytesIO()
            predictions.to_parquet(data, index=False)
            archive.writestr(f"{name}/validation_predictions.parquet", data.getvalue())
        data = io.BytesIO()
        val_obs.to_parquet(data, index=False)
        archive.writestr("validation_observations.parquet", data.getvalue())
        archive.writestr("comparison.png", chart(reports))
    temp.rename(args.output)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
