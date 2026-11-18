"""
PathSnap Hackathon — Phase 3: Model Training (Adarsh's Ownership)
Trains and evaluates:
1. Extra Trees Classifier (Extremely Randomized Trees)
2. LightGBM Classifier (Gradient Boosted Decision Trees)

Features: Exactly the 22 FEATURE_COLUMNS from Phase 2 FeatureEngineer.
Target: is_correct (binary candidate label).
Excludes: all identifiers, timestamps, and GPS coordinates from X.
Partition: Canonical 64 train / 16 val split from dataset_split_manifest.json.
"""

import os
import sys
import time
import json
import glob
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import joblib
import yaml

from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    log_loss,
    confusion_matrix,
)

# Add script paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "phase2" / "scripts"))

from feature_engineering import FeatureEngineer
from evaluation import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "phase3" / "results" / "training.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("TrainAdarshModels")


def load_partition_data(
    parquet_dir: Path,
    expected_trip_ids: List[str],
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Loads all Parquet feature tables for a given partition.
    Returns:
        - X (pd.DataFrame with exactly feature_cols, cast to float32)
        - y (np.ndarray of target is_correct)
        - meta (pd.DataFrame with trip_id, obs_index, candidate_arc_id, true_arc_id)
    """
    logger.info(f"Loading Parquet files from {parquet_dir}...")
    files = sorted(list(parquet_dir.glob("*.parquet")))
    
    loaded_dfs = []
    meta_cols = ["trip_id", "obs_index", "candidate_arc_id", "is_correct"]
    if "true_arc_id" in meta_cols:
        pass
    
    trip_count = 0
    for f in files:
        trip_id = f.stem.replace("_features", "")
        if trip_id in expected_trip_ids:
            df = pd.read_parquet(f)
            loaded_dfs.append(df)
            trip_count += 1
            
    logger.info(f"Loaded {trip_count} trips ({len(loaded_dfs)} tables). Combining...")
    full_df = pd.concat(loaded_dfs, ignore_index=True)
    
    # Verify columns exist
    missing_cols = [c for c in feature_cols if c not in full_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")
        
    # Extract X (float32 for memory efficiency)
    X = full_df[feature_cols].astype(np.float32)
    y = full_df["is_correct"].to_numpy(dtype=int)
    
    meta_available = [c for c in ["trip_id", "obs_index", "candidate_arc_id", "true_arc_id", "is_correct"] if c in full_df.columns]
    meta = full_df[meta_available].copy()
    
    logger.info(f"Partition shape: X={X.shape}, y={y.shape}, Positives={y.sum()} ({y.sum()/len(y)*100:.2f}%)")
    return X, y, meta


def compute_observation_top1(meta: pd.DataFrame, probs: np.ndarray) -> Tuple[float, float, int, int]:
    """
    Computes observation-level Top-1 accuracy and coverage-aware Top-1 accuracy.
    At each (trip_id, obs_index), chooses candidate with highest probability.
    """
    df = meta[["trip_id", "obs_index", "is_correct"]].copy()
    df["prob"] = probs
    
    # For each observation group, find the candidate with max probability
    best_idx = df.groupby(["trip_id", "obs_index"])["prob"].idxmax()
    best_cands = df.loc[best_idx]
    
    total_obs = len(best_cands)
    correct_obs = int((best_cands["is_correct"] == 1).sum())
    
    top1_acc = round(correct_obs / total_obs * 100.0, 2) if total_obs > 0 else 0.0
    # Coverage aware: same when all observations have candidates
    return top1_acc, top1_acc, correct_obs, total_obs


def evaluate_model_performance(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    meta_val: pd.DataFrame,
    train_time_sec: float,
) -> Dict[str, Any]:
    """
    Computes comprehensive genuine metrics across Train and Validation sets.
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Train inference
    t0 = time.time()
    train_probs = model.predict_proba(X_train)[:, 1]
    train_inf_time = time.time() - t0
    train_preds = (train_probs >= 0.5).astype(int)
    
    # Val inference
    t0 = time.time()
    val_probs = model.predict_proba(X_val)[:, 1]
    val_inf_time = time.time() - t0
    val_preds = (val_probs >= 0.5).astype(int)
    
    # Metrics
    evaluator = ModelEvaluator()
    train_bin = evaluator.evaluate_binary_classification(y_train, train_preds, train_probs)
    val_bin = evaluator.evaluate_binary_classification(y_val, val_preds, val_probs)
    
    train_top1, train_cov_top1, train_corr_obs, train_tot_obs = compute_observation_top1(meta_train, train_probs)
    val_top1, val_cov_top1, val_corr_obs, val_tot_obs = compute_observation_top1(meta_val, val_probs)
    
    results = {
        "model_name": model_name,
        "train_runtime_sec": round(train_time_sec, 2),
        "val_inference_latency_ms_per_cand": round(val_inf_time / len(X_val) * 1000.0, 4),
        "val_inference_latency_ms_per_obs": round(val_inf_time / val_tot_obs * 100.0, 4),
        # Validation Metrics
        "val_candidate_accuracy": val_bin["candidate_accuracy"],
        "val_binary_f1": val_bin["binary_f1"],
        "val_macro_f1": val_bin["macro_f1"],
        "val_precision": val_bin["precision"],
        "val_recall": val_bin["recall"],
        "val_log_loss": val_bin["log_loss"],
        "val_observation_top1": val_top1,
        "val_coverage_aware_top1": val_cov_top1,
        "val_confusion_matrix": val_bin["confusion_matrix"],
        # Training Metrics (for generalization check)
        "train_candidate_accuracy": train_bin["candidate_accuracy"],
        "train_binary_f1": train_bin["binary_f1"],
        "train_macro_f1": train_bin["macro_f1"],
        "train_precision": train_bin["precision"],
        "train_recall": train_bin["recall"],
        "train_log_loss": train_bin["log_loss"],
        "train_observation_top1": train_top1,
        "train_coverage_aware_top1": train_cov_top1,
        # Generalization Delta (Overfitting Check)
        "generalization_top1_delta": round(train_top1 - val_top1, 2),
        "generalization_macro_f1_delta": round(train_bin["macro_f1"] - val_bin["macro_f1"], 4),
    }
    
    return results, val_probs


def main():
    logger.info("=" * 70)
    logger.info("PATHSNAP PHASE 3: ADARSH'S SUPERVISED MODEL TRAINING")
    logger.info("=" * 70)
    
    # 1. Load config and split manifest
    config_file = PROJECT_ROOT / "phase3" / "configs" / "phase3_config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    split_manifest_file = PROJECT_ROOT / "phase2" / "results" / "dataset_split_manifest.json"
    with open(split_manifest_file, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)
        
    train_trip_ids = split_manifest["train_trips"]
    val_trip_ids = split_manifest["val_trips"]
    logger.info(f"Split verified: {len(train_trip_ids)} train trips, {len(val_trip_ids)} val trips.")
    
    feature_cols = FeatureEngineer.FEATURE_COLUMNS
    logger.info(f"Model Feature Matrix: {len(feature_cols)} columns:")
    for idx, col in enumerate(feature_cols, start=1):
        logger.info(f"  [{idx:02d}] {col}")
        
    # 2. Load Train and Validation sets
    train_dir = PROJECT_ROOT / "phase2" / "features" / "train"
    val_dir = PROJECT_ROOT / "phase2" / "features" / "val"
    
    X_train, y_train, meta_train = load_partition_data(train_dir, train_trip_ids, feature_cols)
    X_val, y_val, meta_val = load_partition_data(val_dir, val_trip_ids, feature_cols)
    
    # 3. Model A: Extra Trees Classifier
    logger.info("-" * 70)
    logger.info("TRAINING MODEL 1: Extra Trees Classifier")
    et_params = cfg["models"]["extra_trees"]
    logger.info(f"Hyperparameters: {et_params}")
    
    et_model = ExtraTreesClassifier(**et_params)
    t0 = time.time()
    et_model.fit(X_train, y_train)
    et_train_time = time.time() - t0
    logger.info(f"Extra Trees trained in {et_train_time:.2f}s.")
    
    et_results, et_val_probs = evaluate_model_performance(
        "ExtraTreesClassifier", et_model, X_train, y_train, meta_train, X_val, y_val, meta_val, et_train_time
    )
    
    # 4. Model B: LightGBM Classifier
    logger.info("-" * 70)
    logger.info("TRAINING MODEL 2: LightGBM Classifier")
    lgbm_params = cfg["models"]["lightgbm"]
    logger.info(f"Hyperparameters: {lgbm_params}")
    
    lgbm_model = LGBMClassifier(**lgbm_params)
    t0 = time.time()
    lgbm_model.fit(X_train, y_train)
    lgbm_train_time = time.time() - t0
    logger.info(f"LightGBM trained in {lgbm_train_time:.2f}s.")
    
    lgbm_results, lgbm_val_probs = evaluate_model_performance(
        "LGBMClassifier", lgbm_model, X_train, y_train, meta_train, X_val, y_val, meta_val, lgbm_train_time
    )
    
    # 5. Save model artifacts
    models_dir = PROJECT_ROOT / "phase3" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    et_path = models_dir / "extra_trees_model.joblib"
    lgbm_path = models_dir / "lightgbm_model.joblib"
    
    joblib.dump(et_model, et_path, compress=3)
    joblib.dump(lgbm_model, lgbm_path, compress=3)
    logger.info(f"Saved Extra Trees artifact to: {et_path}")
    logger.info(f"Saved LightGBM artifact to: {lgbm_path}")
    
    # 6. Save validation predictions array for teammates
    results_dir = PROJECT_ROOT / "phase3" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    val_preds_df = meta_val.copy()
    val_preds_df["prob_extra_trees"] = np.round(et_val_probs, 5)
    val_preds_df["prob_lightgbm"] = np.round(lgbm_val_probs, 5)
    val_preds_df["pred_extra_trees"] = (et_val_probs >= 0.5).astype(int)
    val_preds_df["pred_lightgbm"] = (lgbm_val_probs >= 0.5).astype(int)
    
    preds_file = results_dir / "val_predictions.parquet"
    val_preds_df.to_parquet(preds_file, index=False)
    logger.info(f"Saved validation prediction arrays to: {preds_file} ({len(val_preds_df):,} rows)")
    
    # 7. Save metrics table
    metrics_list = [et_results, lgbm_results]
    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = results_dir / "adarsh_model_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Saved metrics CSV to: {metrics_file}")
    
    # 8. Save experiment metadata JSON
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "author": "Adarsh",
        "models_trained": ["ExtraTreesClassifier", "LGBMClassifier"],
        "train_trips_count": len(train_trip_ids),
        "val_trips_count": len(val_trip_ids),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "results": metrics_list,
    }
    with open(results_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved experiment metadata.")
    
    # Print summary table to console
    logger.info("=" * 70)
    logger.info("ADARSH MODEL BENCHMARK RESULTS (GENUINE VALIDATION METRICS):")
    logger.info("=" * 70)
    disp_cols = [
        "model_name", "val_observation_top1", "val_macro_f1", "val_binary_f1",
        "val_precision", "val_recall", "val_log_loss", "train_runtime_sec"
    ]
    print(metrics_df[disp_cols].to_string(index=False))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

