"""Train comparable CPU boosters and tune the training-CV winner on the frozen cache."""

import argparse
import gc
from importlib.metadata import version
import io
import json
from pathlib import Path
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from build_audit_dataset import KEYS, VERSION, digest


def load_split(cache, manifest, split):
    frames = {}
    ids = manifest["split"][f"{split}_trips"]
    for family in ("features", "observations"):
        parts = []
        for trip in ids:
            rel = f"{family}/{split}/{trip}.parquet"
            path = cache / rel
            if digest(path) != manifest["file_sha256"][rel]:
                raise ValueError(f"Cache integrity check failed for {rel}")
            rows = pd.read_parquet(path)
            if set(rows.trip_id) != {trip}:
                raise ValueError(f"Wrong trip ID in {rel}")
            parts.append(rows)
        frames[family] = pd.concat(parts, ignore_index=True)
    return frames["features"], frames["observations"]


def model_for(name, seed, overrides=None):
    if name == "xgboost":
        from xgboost import XGBClassifier

        params = dict(objective="binary:logistic", tree_method="hist", device="cpu", n_estimators=150,
                      max_depth=5, learning_rate=.08, subsample=.85, colsample_bytree=.85,
                      random_state=seed, n_jobs=4)
        cls = XGBClassifier
    else:
        from lightgbm import LGBMClassifier

        params = dict(objective="binary", device_type="cpu", n_estimators=150, learning_rate=.08,
                      num_leaves=31, max_depth=-1, min_child_samples=20, colsample_bytree=.85,
                      subsample=.85, subsample_freq=1, random_state=seed, n_jobs=4,
                      deterministic=True, force_col_wise=True, verbosity=-1)
        cls = LGBMClassifier
    params.update(overrides or {})
    return cls(**params), params


def group_winners(frame, probabilities):
    groups = pd.factorize(pd.MultiIndex.from_frame(frame[KEYS]))[0]
    order = np.lexsort((frame.candidate_arc_id, frame.candidate_rank, -probabilities, groups))
    ordered = groups[order]
    return order[np.r_[True, ordered[1:] != ordered[:-1]]]


def fold_sets(train, folds):
    trips = np.array(sorted(train.trip_id.unique()))
    return [(trips[a].tolist(), trips[b].tolist())
            for a, b in GroupKFold(n_splits=folds).split(trips, groups=trips)]


def cross_validate(name, seed, params, fold_ids, train, observations, features):
    scores = []
    for fit_ids, score_ids in fold_ids:
        a = train.trip_id.isin(fit_ids)
        b = train.trip_id.isin(score_ids)
        model, _ = model_for(name, seed, params)
        model.fit(train.loc[a, features], train.loc[a, "is_correct"])
        matrix = train.loc[b, features]
        probs = model.predict_proba(matrix)[:, 1]
        labels = train.loc[b, [*KEYS, "candidate_arc_id", "candidate_rank", "is_correct"]]
        hits = int(labels.iloc[group_winners(labels, probs)].is_correct.sum())
        count = int(observations.trip_id.isin(score_ids).sum())
        scores.append(dict(train_trips=fit_ids, evaluation_trips=score_ids, correct=hits,
                           observations=count, coverage_aware_top1=hits/count))
        del model, matrix, probs, labels
        gc.collect()
    return sum(f["correct"] for f in scores) / sum(f["observations"] for f in scores), scores


def route_metrics(winners, ledger, routes):
    by_trip = []
    for trip, obs in ledger.groupby("trip_id", sort=True):
        part = winners[winners.trip_id.eq(trip)].sort_values("obs_index")
        collapsed = part[part.candidate_arc_id.ne(part.candidate_arc_id.shift())]
        reference = set(routes[trip]["arc_ids"])
        predicted = set(collapsed.candidate_arc_id)
        hits = int(part.is_correct.sum())
        reversed_hits = 0
        if "true_arc_id" in part:
            reversed_hits = int(part.exact_reverse.sum())
        from_node = collapsed.arc_from_node.to_numpy()
        to_node = collapsed.arc_to_node.to_numpy()
        edges = len(collapsed)-1
        by_trip.append(dict(
            trip_id=trip, observations=len(obs), missing_candidates=int((~obs.has_candidates).sum()),
            missing_positive=int((~obs.has_positive).sum()), hits=hits, errors=len(obs)-hits,
            reverse_errors=reversed_hits, coverage_top1=hits/len(obs),
            classifier_collapsed_continuity=float((to_node[:-1] == from_node[1:]).sum()/edges) if edges > 0 else None,
            directed_arc_set_f1=2*len(reference & predicted)/(len(reference)+len(predicted)),
            directed_arc_set_jaccard=len(reference & predicted)/len(reference | predicted),
            predicted_collapsed_arcs=len(collapsed), supplied_route_arcs=len(routes[trip]["arc_ids"]),
            predicted_collapsed_length_m=float(collapsed.arc_length_m.sum()),
            supplied_route_length_m=routes[trip]["length_m"],
        ))
    return by_trip


def final_fit(name, seed, overrides, train, val, observations, features, routes, manifest_sha, score):
    model, params = model_for(name, seed, overrides)
    start = time.perf_counter()
    model.fit(train[features], train.is_correct)
    fit_seconds = time.perf_counter()-start
    start = time.perf_counter()
    probs = model.predict_proba(val[features])[:, 1]
    infer_seconds = time.perf_counter()-start
    metrics, scored = score(val, observations, probs)
    winner = val.iloc[group_winners(val, probs)].copy()
    refs = val[val.is_correct.eq(1)][KEYS+["arc_from_node", "arc_to_node"]].rename(
        columns={"arc_from_node": "ref_from", "arc_to_node": "ref_to"},
    )
    winner = winner.merge(refs, on=KEYS, how="left", validate="one_to_one")
    winner["exact_reverse"] = (~winner.is_correct.astype(bool) & winner.arc_from_node.eq(winner.ref_to)
                               & winner.arc_to_node.eq(winner.ref_from))
    trip = route_metrics(winner, observations, routes)
    assert sum(row["hits"] for row in trip) == metrics["top1_correct"]
    metrics.update(
        candidate_retrieval_ceiling=float(observations.has_positive.mean()),
        macro_trip_top1=float(np.mean([t["coverage_top1"] for t in trip])),
        worst_trip=min(trip, key=lambda t: t["coverage_top1"]),
        mean_trip_arc_set_f1=float(np.mean([t["directed_arc_set_f1"] for t in trip])),
        exact_reverse_errors=int(winner.exact_reverse.sum()),
    )
    continuity = [t["classifier_collapsed_continuity"] for t in trip if t["classifier_collapsed_continuity"] is not None]
    team = dict(
        model="XGBoost" if name == "xgboost" else "LightGBM", owner="omen", seed=seed,
        training_trip_count=train.trip_id.nunique(), cv_macro_f1=None,
        student_test_top1_accuracy=None, student_test_macro_f1=None, student_test_log_loss=None,
        route_continuity=float(np.mean(continuity)) if continuity else None,
        training_seconds=fit_seconds, inference_seconds=infer_seconds, hyperparameters=params, run_completed=True,
    )
    result = dict(
        **team, dataset_version=VERSION, dataset_manifest_sha256=manifest_sha, feature_order=features,
        validation=metrics, per_trip=trip, route_definition="Classifier argmax, consecutive deduplication; no HMM.",
    )
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    np.testing.assert_allclose(joblib.load(buf).predict_proba(val[features].iloc[:128])[:, 1], probs[:128])
    parquet = io.BytesIO()
    scored.to_parquet(parquet, index=False)
    native = (bytes(model.get_booster().save_raw(raw_format="ubj")) if name == "xgboost"
              else model.booster_.model_to_string().encode())
    del model
    return result, team, buf.getvalue(), parquet.getvalue(), native


def put(archive, name, result, team, binary, predictions, native):
    archive.writestr(f"{name}/results.json", json.dumps(result, indent=2, allow_nan=False))
    archive.writestr(f"{name}/team_summary.json", json.dumps(team, indent=2, allow_nan=False))
    archive.writestr(f"{name}/model.joblib", binary)
    archive.writestr(f"{name}/validation_predictions.parquet", predictions)
    archive.writestr(f"{name}/model.{'ubj' if name == 'xgboost' else 'txt'}", native)


def search_params(trial, name):
    params = dict(
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", .03, .18, log=True),
        n_estimators=trial.suggest_int("n_estimators", 90, 270, step=30),
        subsample=trial.suggest_float("subsample", .70, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", .70, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", .15, 15, log=True),
    )
    if name == "xgboost":
        params["min_child_weight"] = trial.suggest_float("min_child_weight", 1, 12, log=True)
    else:
        params["num_leaves"] = trial.suggest_int("num_leaves", 15, 63)
        params["min_child_samples"] = trial.suggest_int("min_child_samples", 12, 80)
    return params


def run(args):
    sys.path.insert(0, str(args.repo.resolve() / "phase3/omen"))
    from train_canonical import score

    manifest_path = args.cache / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == VERSION
    manifest_sha = digest(manifest_path)
    features, seed = manifest["feature_columns"], manifest["seed"]
    train, train_obs = load_split(args.cache, manifest, "train")
    val, val_obs = load_split(args.cache, manifest, "val")
    assert not set(train.trip_id) & set(val.trip_id)
    for table in (train, val):
        table[features] = table[features].astype(np.float32)
    assert digest(args.cache / "reference_routes.json") == manifest["file_sha256"]["reference_routes.json"]
    routes = json.loads((args.cache / "reference_routes.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    folds = fold_sets(train, args.folds)
    results = dict(dataset_version=VERSION, dataset_manifest_sha256=manifest_sha, baselines={},
                   package_versions={p: version(p) for p in ("xgboost", "lightgbm", "optuna", "scikit-learn", "pandas")})
    print(f"TRAIN {len(train):,} rows; VALIDATION {len(val):,} rows; training CV folds {args.folds}", flush=True)
    for name in ("xgboost", "lightgbm"):
        result, team, model, predictions, native = final_fit(
            name, seed, {}, train, val, val_obs, features, routes, manifest_sha, score,
        )
        with ZipFile(args.output / f"{name}_baseline.zip", "x", compression=ZIP_DEFLATED) as archive:
            put(archive, name, result, team, model, predictions, native)
        results["baselines"][name] = result
        print(f"{name}: validation {result['validation']['top1_coverage_aware']:.2%} Top-1; "
              f"macro F1 {result['validation']['macro_f1']:.4f}", flush=True)
    for name in results["baselines"]:
        value, fold_scores = cross_validate(name, seed, {}, folds, train, train_obs, features)
        results["baselines"][name]["training_cv_top1"] = value
        results["baselines"][name]["training_cv_folds"] = fold_scores
        print(f"{name}: training CV {value:.2%} coverage-aware Top-1", flush=True)
    winner = max(results["baselines"], key=lambda name: results["baselines"][name]["training_cv_top1"])
    results["selected_model_from_training_cv"] = winner
    baseline_data = json.dumps(results, indent=2, allow_nan=False) + "\n"
    (args.output / "baseline_comparison.json").write_text(baseline_data, encoding="utf-8")

    if args.trials:
        import optuna

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=2))

        def objective(trial):
            value, detail = cross_validate(winner, seed, search_params(trial, winner), folds, train, train_obs, features)
            trial.set_user_attr("fold_metrics", detail)
            return value

        def save_trial(study, _):
            study.trials_dataframe().to_csv(args.output / "optuna_trials.csv", index=False)
            history = [
                dict(number=t.number, state=t.state.name, value=t.value, params=t.params,
                     fold_metrics=t.user_attrs.get("fold_metrics"),
                     duration_seconds=t.duration.total_seconds() if t.duration else None)
                for t in study.trials
            ]
            (args.output / "optuna_trials.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        study.optimize(objective, n_trials=args.trials, timeout=args.seconds, callbacks=[save_trial])
        trial = study.best_trial
        result, team, model, predictions, native = final_fit(
            winner, seed, trial.params, train, val, val_obs, features, routes, manifest_sha, score,
        )
        result.update(training_cv_top1=trial.value, training_cv_folds=trial.user_attrs["fold_metrics"],
                      optuna_trial=trial.number, completed_trials=len(study.trials))
        with ZipFile(args.output / f"{winner}_tuned.zip", "x", compression=ZIP_DEFLATED) as archive:
            put(archive, winner, result, team, model, predictions, native)
        results["tuned"] = result
        print(f"Tuned {winner}: CV {trial.value:.2%}; validation {result['validation']['top1_coverage_aware']:.2%}",
              flush=True)
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seconds", type=int, default=150)
    with threadpool_limits(limits=4):
        run(parser.parse_args())
