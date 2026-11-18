"""
PathSnap Hackathon — Phase 3: Supervised Model Training & Evaluation
Models Owned:
1. Random Forest Classifier
2. CatBoost Classifier

Trained on 64 training trips (phase2/features/train/*.parquet).
Evaluated on 16 validation trips (phase2/features/val/*.parquet).
Uses the canonical 22 model features defined in Phase 2 FeatureEngineer.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    average_precision_score,
)
from catboost import CatBoostClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Phase3Training")

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE2_DIR = PROJECT_ROOT / "phase2"
PHASE3_DIR = PROJECT_ROOT / "phase3"
TRAIN_DIR = PHASE2_DIR / "features" / "train"
VAL_DIR = PHASE2_DIR / "features" / "val"
MODELS_DIR = PHASE3_DIR / "models"
RESULTS_DIR = PHASE3_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical 22 predictive features from FeatureEngineer.FEATURE_COLUMNS
FEATURE_COLUMNS = [
    # Geometric features
    "distance_to_arc_m",
    "dist_diff_from_best_m",
    "projection_t",
    "candidate_rank",
    "normalized_rank",
    "log_distance",
    # Movement & Heading features
    "gps_speed_mps",
    "gps_bearing_deg",
    "arc_bearing_deg",
    "heading_diff_deg",
    "cos_heading_diff",
    "speed_heading_alignment",
    "is_counter_flow",
    # Road Network features
    "arc_length_m",
    "from_node_out_degree",
    "to_node_in_degree",
    "is_terminal_arc",
    # Sequential & Transition features
    "step_gps_dist_m",
    "transition_dist_m",
    "transition_dist_ratio",
    "is_directly_connected",
    "log_transition_dist",
]

METADATA_COLUMNS = [
    "trip_id",
    "obs_index",
    "candidate_arc_id",
    "is_correct",
]


def load_dataset(feature_dir: Path, max_trips: int = None) -> pd.DataFrame:
    """
    Stream and concatenate feature parquet files, keeping only predictive features
    and essential metadata to strictly control memory usage on M1 Air.
    """
    parquet_files = sorted(list(feature_dir.glob("*.parquet")))
    if max_trips:
        parquet_files = parquet_files[:max_trips]

    logger.info(f"Loading {len(parquet_files)} parquet files from {feature_dir}...")
    dfs = []
    total_rows = 0

    required_cols = FEATURE_COLUMNS + METADATA_COLUMNS

    for f in parquet_files:
        # Load only necessary columns
        df_trip = pd.read_parquet(f, columns=required_cols)
        # Ensure correct dtypes to minimize memory footprint
        for c in FEATURE_COLUMNS:
            df_trip[c] = df_trip[c].astype(np.float32)
        df_trip["is_correct"] = df_trip["is_correct"].astype(np.int8)
        df_trip["candidate_rank"] = df_trip["candidate_rank"].astype(np.int8)
        dfs.append(df_trip)
        total_rows += len(df_trip)

    combined_df = pd.concat(dfs, ignore_index=True)
    mem_mb = combined_df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"Loaded {total_rows} rows from {len(parquet_files)} trips ({mem_mb:.2f} MB in RAM).")
    return combined_df


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    df_eval: pd.DataFrame,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes all candidate-level and observation-level metrics:
    - Candidate binary Macro F1, Precision, Recall, Accuracy
    - Binary Precision, Recall, F1 for the positive class (is_correct=1)
    - Log loss, ROC-AUC, PR-AUC
    - Observation Top-1 Accuracy (argmax predicted probability per obs_index)
    - Coverage-aware Top-1 Accuracy (% of correct predictions among observations having a true candidate)
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Candidate-level classification metrics
    cand_acc = float(accuracy_score(y_true, y_pred))
    cand_macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    cand_macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    cand_macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    cand_bin_f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    cand_bin_precision = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    cand_bin_recall = float(recall_score(y_true, y_pred, average="binary", zero_division=0))

    # Log loss (clipped to avoid inf)
    clipped_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    loss = float(log_loss(y_true, clipped_prob))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    # Observation-level Top-1 Accuracy
    # Group by (trip_id, obs_index) and pick candidate with highest probability
    df_eval_copy = df_eval[["trip_id", "obs_index", "candidate_arc_id", "is_correct", "distance_to_arc_m"]].copy()
    df_eval_copy["predicted_prob"] = y_prob

    # Tie-breaking: sort by predicted_prob desc, then distance_to_arc_m asc
    df_eval_sorted = df_eval_copy.sort_values(
        by=["trip_id", "obs_index", "predicted_prob", "distance_to_arc_m"],
        ascending=[True, True, False, True]
    )

    # Pick top candidate per observation
    top_candidates = df_eval_sorted.groupby(["trip_id", "obs_index"]).first().reset_index()

    total_obs = len(top_candidates)
    correct_obs = int((top_candidates["is_correct"] == 1).sum())

    # Coverage computation: observations where AT LEAST ONE candidate in the candidate set was correct
    obs_with_true_cand = df_eval_copy.groupby(["trip_id", "obs_index"])["is_correct"].max().reset_index()
    covered_obs_count = int((obs_with_true_cand["is_correct"] == 1).sum())
    candidate_coverage_pct = float((covered_obs_count / total_obs * 100.0) if total_obs > 0 else 0.0)

    # Conditional Top-1: % of correct predictions among observations having at least one true candidate
    # (denominator = covered_obs_count = 39,166)
    conditional_top1_pct = float(
        (correct_obs / covered_obs_count * 100.0) if covered_obs_count > 0 else 0.0
    )
    # Coverage-Aware Top-1: % of correct predictions among ALL observations
    # (denominator = total_obs = 39,460, penalizes candidate retrieval misses)
    coverage_aware_top1_pct = float(
        (correct_obs / total_obs * 100.0) if total_obs > 0 else 0.0
    )

    return {
        "candidate_binary_macro_f1": round(cand_macro_f1, 4),
        "candidate_binary_precision": round(cand_macro_precision, 4),
        "candidate_binary_recall": round(cand_macro_recall, 4),
        "candidate_binary_accuracy": round(cand_acc * 100.0, 2),
        "candidate_positive_f1": round(cand_bin_f1, 4),
        "candidate_positive_precision": round(cand_bin_precision, 4),
        "candidate_positive_recall": round(cand_bin_recall, 4),
        "log_loss": round(loss, 5),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "conditional_top1_accuracy": round(conditional_top1_pct, 2),
        "coverage_aware_top1_accuracy": round(coverage_aware_top1_pct, 2),
        "top1_conditional": round(conditional_top1_pct / 100.0, 6),
        "top1_coverage_aware": round(coverage_aware_top1_pct / 100.0, 6),
        "observation_top1_accuracy": round(coverage_aware_top1_pct, 2),
        "candidate_coverage_pct": round(candidate_coverage_pct, 2),
        "total_observations": total_obs,
        "correct_observations": correct_obs,
        "covered_observations": covered_obs_count,
        "observations_without_positive": total_obs - covered_obs_count,
        "total_candidates": len(y_true),
    }


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    df_val: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Trains and benchmarks RandomForestClassifier.
    Hyperparameters strictly aligned with Phase 3 Readiness spec:
    n_estimators=200, max_depth=14, max_features='sqrt', n_jobs=-1, random_state=42.
    """
    hyperparameters = {
        "n_estimators": 200,
        "max_depth": 14,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
        "class_weight": "balanced",
    }
    logger.info(f"--- Training Random Forest Classifier ---")
    logger.info(f"Hyperparameters: {hyperparameters}")

    rf = RandomForestClassifier(**hyperparameters)

    # Measure training time
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - t0
    logger.info(f"Random Forest training completed in {train_time_sec:.2f} seconds.")

    # Measure inference time
    t1 = time.perf_counter()
    y_prob = rf.predict_proba(X_val)[:, 1]
    infer_time_sec = time.perf_counter() - t1
    infer_latency_ms_per_cand = (infer_time_sec / len(X_val)) * 1000.0
    num_obs = len(df_val.groupby(["trip_id", "obs_index"]))
    infer_latency_ms_per_obs = (infer_time_sec / num_obs) * 1000.0 if num_obs > 0 else 0.0

    logger.info(f"Random Forest inference on {len(X_val)} candidates ({num_obs} obs) completed in {infer_time_sec:.3f} s.")
    logger.info(f"Latency: {infer_latency_ms_per_obs:.4f} ms/observation, {infer_latency_ms_per_cand:.5f} ms/candidate.")

    # Compute metrics
    metrics = compute_metrics(y_val, y_prob, df_val)
    metrics["model_name"] = "Random Forest"
    metrics["model_family"] = "Bootstrap Bagging Ensemble"
    metrics["hyperparameters"] = hyperparameters
    metrics["train_time_sec"] = round(train_time_sec, 2)
    metrics["infer_time_sec"] = round(infer_time_sec, 3)
    metrics["latency_ms_per_obs"] = round(infer_latency_ms_per_obs, 4)
    metrics["latency_ms_per_cand"] = round(infer_latency_ms_per_cand, 5)

    # Feature importances
    importances = rf.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
    metrics["feature_importances"] = {k: round(float(v), 5) for k, v in feat_imp}

    # Save model artifact
    model_path = MODELS_DIR / "random_forest_classifier.joblib"
    joblib.dump(rf, model_path, compress=3)
    logger.info(f"Saved Random Forest model artifact to {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB).")
    metrics["saved_artifact"] = str(model_path.relative_to(PROJECT_ROOT))

    # Save metrics JSON
    json_path = RESULTS_DIR / "random_forest_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved Random Forest metrics to {json_path}")

    return metrics


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    df_val: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Trains and benchmarks CatBoostClassifier.
    Hyperparameters strictly aligned with Phase 3 Readiness spec:
    depth=6, learning_rate=0.05, iterations=400, l2_leaf_reg=3.0, random_seed=42.
    """
    hyperparameters = {
        "depth": 6,
        "learning_rate": 0.05,
        "iterations": 400,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "verbose": 50,
        "thread_count": -1,
    }
    logger.info(f"--- Training CatBoost Classifier ---")
    logger.info(f"Hyperparameters: {hyperparameters}")

    cb = CatBoostClassifier(**hyperparameters)

    # Measure training time
    t0 = time.perf_counter()
    cb.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=40,
        verbose=50,
    )
    train_time_sec = time.perf_counter() - t0
    logger.info(f"CatBoost training completed in {train_time_sec:.2f} seconds.")

    # Measure inference time
    t1 = time.perf_counter()
    y_prob = cb.predict_proba(X_val)[:, 1]
    infer_time_sec = time.perf_counter() - t1
    infer_latency_ms_per_cand = (infer_time_sec / len(X_val)) * 1000.0
    num_obs = len(df_val.groupby(["trip_id", "obs_index"]))
    infer_latency_ms_per_obs = (infer_time_sec / num_obs) * 1000.0 if num_obs > 0 else 0.0

    logger.info(f"CatBoost inference on {len(X_val)} candidates ({num_obs} obs) completed in {infer_time_sec:.3f} s.")
    logger.info(f"Latency: {infer_latency_ms_per_obs:.4f} ms/observation, {infer_latency_ms_per_cand:.5f} ms/candidate.")

    # Compute metrics
    metrics = compute_metrics(y_val, y_prob, df_val)
    metrics["model_name"] = "CatBoost"
    metrics["model_family"] = "Symmetric Oblivious Decision Trees"
    metrics["hyperparameters"] = hyperparameters
    metrics["best_iteration"] = cb.get_best_iteration()
    metrics["train_time_sec"] = round(train_time_sec, 2)
    metrics["infer_time_sec"] = round(infer_time_sec, 3)
    metrics["latency_ms_per_obs"] = round(infer_latency_ms_per_obs, 4)
    metrics["latency_ms_per_cand"] = round(infer_latency_ms_per_cand, 5)

    # Feature importances
    importances = cb.get_feature_importance()
    feat_imp = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
    metrics["feature_importances"] = {k: round(float(v), 5) for k, v in feat_imp}

    # Save model artifact
    model_cbm_path = MODELS_DIR / "catboost_classifier.cbm"
    cb.save_model(str(model_cbm_path))
    model_joblib_path = MODELS_DIR / "catboost_classifier.joblib"
    joblib.dump(cb, model_joblib_path, compress=3)
    logger.info(f"Saved CatBoost model artifact to {model_cbm_path} and {model_joblib_path}.")
    metrics["saved_artifact"] = str(model_cbm_path.relative_to(PROJECT_ROOT))

    # Save metrics JSON
    json_path = RESULTS_DIR / "catboost_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved CatBoost metrics to {json_path}")

    return metrics


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    df_val: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Trains and benchmarks Logistic Regression with StandardScaler.
    Hyperparameters strictly aligned with Omen canonical spec:
    C=1.0, solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=42.
    """
    hyperparameters = {
        "C": 1.0,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": 42,
    }
    logger.info(f"--- Training Logistic Regression Classifier ---")
    logger.info(f"Hyperparameters: {hyperparameters}")

    model = make_pipeline(StandardScaler(), LogisticRegression(**hyperparameters))

    # Measure training time
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - t0
    logger.info(f"Logistic Regression training completed in {train_time_sec:.2f} seconds.")

    # Measure inference time
    t1 = time.perf_counter()
    y_prob = model.predict_proba(X_val)[:, 1]
    infer_time_sec = time.perf_counter() - t1
    infer_latency_ms_per_cand = (infer_time_sec / len(X_val)) * 1000.0
    num_obs = len(df_val.groupby(["trip_id", "obs_index"]))
    infer_latency_ms_per_obs = (infer_time_sec / num_obs) * 1000.0 if num_obs > 0 else 0.0

    logger.info(f"Logistic Regression inference on {len(X_val)} candidates ({num_obs} obs) completed in {infer_time_sec:.3f} s.")
    logger.info(f"Latency: {infer_latency_ms_per_obs:.4f} ms/observation, {infer_latency_ms_per_cand:.5f} ms/candidate.")

    # Compute metrics
    metrics = compute_metrics(y_val, y_prob, df_val)
    metrics["model_name"] = "Logistic Regression"
    metrics["model_family"] = "Linear Generalized Linear Model"
    metrics["hyperparameters"] = hyperparameters
    metrics["train_time_sec"] = round(train_time_sec, 2)
    metrics["infer_time_sec"] = round(infer_time_sec, 3)
    metrics["latency_ms_per_obs"] = round(infer_latency_ms_per_obs, 4)
    metrics["latency_ms_per_cand"] = round(infer_latency_ms_per_cand, 5)

    # Feature coefficients
    lr_step = model.named_steps["logisticregression"]
    coefs = lr_step.coef_[0]
    feat_coef = sorted(zip(FEATURE_COLUMNS, coefs), key=lambda x: abs(x[1]), reverse=True)
    metrics["feature_coefficients"] = {k: round(float(v), 5) for k, v in feat_coef}

    # Save model artifact
    model_path = MODELS_DIR / "logistic_regression_classifier.joblib"
    joblib.dump(model, model_path, compress=3)
    logger.info(f"Saved Logistic Regression artifact to {model_path}.")
    metrics["saved_artifact"] = str(model_path.relative_to(PROJECT_ROOT))

    # Save metrics JSON
    json_path = RESULTS_DIR / "logistic_regression_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved Logistic Regression metrics to {json_path}")

    return metrics


def train_extra_trees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    df_val: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Trains and benchmarks ExtraTreesClassifier.
    Hyperparameters strictly aligned with Adarsh canonical spec:
    n_estimators=100, max_depth=16, min_samples_leaf=20, max_features='sqrt', bootstrap=False, n_jobs=-1, random_state=42.
    """
    hyperparameters = {
        "n_estimators": 100,
        "max_depth": 16,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "bootstrap": False,
        "n_jobs": -1,
        "random_state": 42,
    }
    logger.info(f"--- Training Extra Trees Classifier ---")
    logger.info(f"Hyperparameters: {hyperparameters}")

    et = ExtraTreesClassifier(**hyperparameters)

    # Measure training time
    t0 = time.perf_counter()
    et.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - t0
    logger.info(f"Extra Trees training completed in {train_time_sec:.2f} seconds.")

    # Measure inference time
    t1 = time.perf_counter()
    y_prob = et.predict_proba(X_val)[:, 1]
    infer_time_sec = time.perf_counter() - t1
    infer_latency_ms_per_cand = (infer_time_sec / len(X_val)) * 1000.0
    num_obs = len(df_val.groupby(["trip_id", "obs_index"]))
    infer_latency_ms_per_obs = (infer_time_sec / num_obs) * 1000.0 if num_obs > 0 else 0.0

    logger.info(f"Extra Trees inference on {len(X_val)} candidates ({num_obs} obs) completed in {infer_time_sec:.3f} s.")
    logger.info(f"Latency: {infer_latency_ms_per_obs:.4f} ms/observation, {infer_latency_ms_per_cand:.5f} ms/candidate.")

    # Compute metrics
    metrics = compute_metrics(y_val, y_prob, df_val)
    metrics["model_name"] = "Extra Trees"
    metrics["model_family"] = "Extremely Randomized Trees Ensemble"
    metrics["hyperparameters"] = hyperparameters
    metrics["train_time_sec"] = round(train_time_sec, 2)
    metrics["infer_time_sec"] = round(infer_time_sec, 3)
    metrics["latency_ms_per_obs"] = round(infer_latency_ms_per_obs, 4)
    metrics["latency_ms_per_cand"] = round(infer_latency_ms_per_cand, 5)

    # Feature importances
    importances = et.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
    metrics["feature_importances"] = {k: round(float(v), 5) for k, v in feat_imp}

    # Save model artifact
    model_path = MODELS_DIR / "extra_trees_classifier.joblib"
    joblib.dump(et, model_path, compress=3)
    logger.info(f"Saved Extra Trees model artifact to {model_path}.")
    metrics["saved_artifact"] = str(model_path.relative_to(PROJECT_ROOT))

    # Save metrics JSON
    json_path = RESULTS_DIR / "extra_trees_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved Extra Trees metrics to {json_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark PathSnap supervised models on Phase 2 features.")
    parser.add_argument("--model", type=str, choices=["rf", "catboost", "lr", "et", "all", "coordinated"], default="coordinated",
                        help="Which model to train: rf, catboost, lr, et, coordinated (lr+rf+et+cb), or all.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Optional limit on number of trips loaded for fast testing.")
    args = parser.parse_args()

    logger.info("=== Starting PathSnap Phase 3 Model Training (M1 Architecture) ===")
    logger.info(f"Features: 22 canonical columns")
    logger.info(f"Train Dir: {TRAIN_DIR}")
    logger.info(f"Val Dir: {VAL_DIR}")

    # Load datasets
    df_train = load_dataset(TRAIN_DIR, max_trips=args.sample)
    df_val = load_dataset(VAL_DIR, max_trips=args.sample)

    logger.info(f"Train rows: {len(df_train)} (Positive: {(df_train['is_correct']==1).sum()}, Negative: {(df_train['is_correct']==0).sum()})")
    logger.info(f"Val rows: {len(df_val)} (Positive: {(df_val['is_correct']==1).sum()}, Negative: {(df_val['is_correct']==0).sum()})")

    X_train = df_train[FEATURE_COLUMNS].values
    y_train = df_train["is_correct"].values

    X_val = df_val[FEATURE_COLUMNS].values
    y_val = df_val["is_correct"].values

    results = {}

    # Load existing metrics for RF and CatBoost if available
    rf_json = RESULTS_DIR / "random_forest_metrics.json"
    cb_json = RESULTS_DIR / "catboost_metrics.json"
    if rf_json.exists():
        with open(rf_json) as f:
            results["Random Forest"] = json.load(f)
    if cb_json.exists():
        with open(cb_json) as f:
            results["CatBoost"] = json.load(f)

    if args.model in ("lr", "coordinated", "all"):
        lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val, df_val)
        results["Logistic Regression"] = lr_metrics

    if args.model in ("rf", "all"):
        rf_metrics = train_random_forest(X_train, y_train, X_val, y_val, df_val)
        results["Random Forest"] = rf_metrics

    if args.model in ("et", "coordinated", "all"):
        et_metrics = train_extra_trees(X_train, y_train, X_val, y_val, df_val)
        results["Extra Trees"] = et_metrics

    if args.model in ("catboost", "all"):
        cb_metrics = train_catboost(X_train, y_train, X_val, y_val, df_val)
        results["CatBoost"] = cb_metrics

    # Produce comparison summary table for coordinated models
    summary_rows = []
    for model_name, m in results.items():
        correct = m.get("correct_observations", 0)
        total = m.get("total_observations", 39460)
        covered = m.get("covered_observations", 39166)
        cond_top1 = round((correct / covered * 100.0), 2) if covered > 0 else 0.0
        cov_top1 = round((correct / total * 100.0), 2) if total > 0 else 0.0

        summary_rows.append({
            "Model": model_name,
            "Conditional Top-1 (%)": cond_top1,
            "Coverage-Aware Top-1 (%)": cov_top1,
            "Candidate Macro F1": m.get("candidate_binary_macro_f1"),
            "Candidate Accuracy (%)": m.get("candidate_binary_accuracy"),
            "Log Loss": m.get("log_loss"),
            "ROC-AUC": m.get("roc_auc"),
            "Train Time (s)": m.get("train_time_sec"),
            "Obs Latency (ms)": m.get("latency_ms_per_obs"),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "coordinated_models_benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("\n" + summary_df.to_string(index=False))
    logger.info(f"Coordinated benchmark summary saved to {summary_csv}")


if __name__ == "__main__":
    main()
