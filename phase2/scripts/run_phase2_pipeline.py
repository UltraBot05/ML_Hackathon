"""
PathSnap Hackathon — Phase 2: Complete Pipeline Runner
Executes end-to-end processing across training trips:
- Trajectory kinematic processing
- Road network graph building
- Spatial KD-Tree candidate generation
- Ground-truth label alignment
- Multi-dimensional feature engineering
- Trip-level train/validation partition (80/20 split, seed 42)
- Baseline matcher evaluation (Nearest Road, Direction-Aware, Graph-Constrained HMM)
- Reconstructed route continuity and quality metrics
- Diagnostic plotting and interactive Folium HTML map generation
- Final SHA-256 raw dataset integrity verification
"""

import sys
import os
import time
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import yaml
import logging

# Ensure script dir in path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import DataLoader, TripData
from trajectory_processor import TrajectoryProcessor
from road_graph import RoadNetworkGraph
from spatial_index import RoadSpatialIndex
from candidate_generation import CandidateGenerator
from label_generation import LabelGenerator
from feature_engineering import FeatureEngineer
from baseline_matcher import NearestRoadMatcher, DirectionAwareMatcher, GraphConstrainedHMMMatcher
from route_reconstruction import RouteReconstructor
from evaluation import ModelEvaluator
from visualization import Phase2Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "phase2" / "logs" / "phase2_execution.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("Phase2Pipeline")


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_dataset_integrity(manifest_path: Path) -> bool:
    """Verifies all 360 dataset files match SHA-256 hashes."""
    if not manifest_path.exists():
        logger.error(f"Integrity manifest missing: {manifest_path}")
        return False

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    file_items = manifest["files"].items() if "files" in manifest else manifest.items()
    mismatches = []
    for rel_path, info in file_items:
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            mismatches.append((rel_path, "MISSING"))
            continue
        hasher = hashlib.sha256()
        with open(full_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        if hasher.hexdigest() != info["sha256"]:
            mismatches.append((rel_path, "HASH_MISMATCH"))

    if mismatches:
        logger.critical(f"Dataset integrity compromised! Mismatches: {mismatches}")
        return False
    logger.info("Raw dataset integrity verified: 100% SHA-256 match across all files.")
    return True


def run_pipeline(
    max_trips: Optional[int] = None,
    val_ratio: float = 0.20,
    random_seed: int = 42,
):
    logger.info("=" * 70)
    logger.info("STARTING PATHSNAP PHASE 2 PIPELINE")
    logger.info("=" * 70)

    config_path = PROJECT_ROOT / "phase2" / "configs" / "phase2_config.yaml"
    cfg = load_config(config_path)

    manifest_path = PROJECT_ROOT / "docs" / "phase2" / "source_dataset_integrity_manifest.json"
    logger.info("Pre-execution dataset integrity verification...")
    if not verify_dataset_integrity(manifest_path):
        raise RuntimeError("Pre-execution dataset integrity check failed!")

    # Setup directories
    features_train_dir = PROJECT_ROOT / "phase2" / "features" / "train"
    features_val_dir = PROJECT_ROOT / "phase2" / "features" / "val"
    results_dir = PROJECT_ROOT / "phase2" / "results"
    plots_dir = results_dir / "plots"

    features_train_dir.mkdir(parents=True, exist_ok=True)
    features_val_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover training trips
    loader = DataLoader(PROJECT_ROOT / "Dataset")
    trip_dirs = loader.discover_trip_dirs("train")
    logger.info(f"Discovered {len(trip_dirs)} training trips in Dataset/train/.")

    if max_trips is not None and max_trips > 0:
        logger.info(f"Running pipeline on sample of {max_trips} trips.")
        trip_dirs = trip_dirs[:max_trips]

    # 2. Trip-level Train / Validation Split
    np.random.seed(random_seed)
    all_trip_ids = [td[0] for td in trip_dirs]
    shuffled_ids = np.random.permutation(all_trip_ids)

    n_val = max(1, int(len(shuffled_ids) * val_ratio)) if len(shuffled_ids) > 1 else 0
    val_trip_ids = set(shuffled_ids[:n_val])
    train_trip_ids = set(shuffled_ids[n_val:])

    logger.info(f"Split partition: {len(train_trip_ids)} train trips, {len(val_trip_ids)} validation trips.")

    # Save split manifest
    split_manifest = {
        "random_seed": random_seed,
        "val_ratio": val_ratio,
        "train_trips": sorted(list(train_trip_ids)),
        "val_trips": sorted(list(val_trip_ids)),
    }
    with open(results_dir / "dataset_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    # 3. Instantiate pipeline components
    proc = TrajectoryProcessor(
        max_speed_threshold_mps=cfg["trajectory_processing"]["max_speed_threshold_mps"],
        gap_threshold_seconds=cfg["trajectory_processing"]["gap_threshold_seconds"],
    )
    cand_gen = CandidateGenerator(
        search_radius_m=cfg["candidate_generation"]["search_radius_meters"],
        max_candidates=cfg["candidate_generation"]["max_candidates_per_point"],
        fallback_radius_m=cfg["candidate_generation"]["fallback_search_radius_meters"],
    )
    labeler = LabelGenerator()
    fe = FeatureEngineer(
        max_candidates=cfg["candidate_generation"]["max_candidates_per_point"],
        max_network_cutoff_m=cfg["road_graph"]["max_network_search_meters"],
    )
    reconstructor = RouteReconstructor()
    evaluator = ModelEvaluator()

    base_a = NearestRoadMatcher()
    base_b = DirectionAwareMatcher(
        distance_weight=cfg["baselines"]["direction_aware"]["distance_weight"],
        heading_weight=cfg["baselines"]["direction_aware"]["heading_weight"],
    )
    base_c = GraphConstrainedHMMMatcher(
        emission_sigma_m=cfg["baselines"]["graph_constrained"]["emission_sigma_meters"],
        transition_beta=cfg["baselines"]["graph_constrained"]["transition_beta"],
        disconnection_penalty=cfg["baselines"]["graph_constrained"]["disconnection_penalty"],
    )

    # Accumulators for aggregate evaluation
    trip_candidate_stats = []
    baseline_trip_results = []
    total_pipeline_start = time.time()

    sample_trip_data: Optional[TripData] = None
    sample_obs: Optional[List] = None
    sample_cands: Optional[List] = None
    sample_graph: Optional[RoadNetworkGraph] = None
    sample_routes: Dict[str, List[int]] = {}

    for idx, (trip_id, trip_path) in enumerate(trip_dirs, start=1):
        t_start = time.time()
        logger.info(f"[{idx}/{len(trip_dirs)}] Processing Trip {trip_id}...")

        trip = loader.load_trip(trip_id, "train")
        if trip is None:
            logger.warning(f"Could not load trip {trip_id}. Skipping.")
            continue

        # A. Trajectory processing
        obs = proc.process_trip_track(trip_id, trip.track)

        # B. Road graph and spatial index
        graph = RoadNetworkGraph(trip_id, trip.nodes, trip.arcs)
        sindex = RoadSpatialIndex(graph)

        # C. Candidate generation
        cands, c_stats = cand_gen.generate_candidates_for_trip(trip, obs, graph, sindex)
        trip_candidate_stats.append(c_stats)

        # D. Ground-truth label generation
        obs_map, align_stats = labeler.align_trip_observations(trip, obs, cands)
        labeled_cands = labeler.assign_candidate_labels(cands, obs_map)
        gt_list = [obs_map.get(i) for i in range(len(obs))]

        # E. Feature Engineering
        feat_df = fe.compute_candidate_features(labeled_cands, obs, graph)

        # Save feature table to train or val partition (parquet preferred, fallback to csv.gz)
        target_dir = features_val_dir if trip_id in val_trip_ids else features_train_dir
        try:
            feat_file = target_dir / f"{trip_id}_features.parquet"
            feat_df.to_parquet(feat_file, index=False)
        except Exception:
            feat_file = target_dir / f"{trip_id}_features.csv.gz"
            feat_df.to_csv(feat_file, index=False, compression="gzip")

        # F. Baseline Experiments
        # 1. Nearest Road (A)
        t0_a = time.time()
        pred_a = base_a.match_trip(cands, obs, graph)
        lat_a = (time.time() - t0_a) / len(obs) * 1000.0
        eval_a = evaluator.evaluate_observation_predictions(pred_a, gt_list)
        col_a = reconstructor.collapse_observation_sequence(pred_a)
        rec_a = reconstructor.reconstruct_and_bridge_route(col_a, graph)
        cont_a, _, _ = reconstructor.evaluate_continuity(rec_a, graph)
        sim_a = reconstructor.compare_with_ground_truth(rec_a, trip.route)

        # 2. Direction-Aware (B)
        t0_b = time.time()
        pred_b = base_b.match_trip(cands, obs, graph)
        lat_b = (time.time() - t0_b) / len(obs) * 1000.0
        eval_b = evaluator.evaluate_observation_predictions(pred_b, gt_list)
        col_b = reconstructor.collapse_observation_sequence(pred_b)
        rec_b = reconstructor.reconstruct_and_bridge_route(col_b, graph)
        cont_b, _, _ = reconstructor.evaluate_continuity(rec_b, graph)
        sim_b = reconstructor.compare_with_ground_truth(rec_b, trip.route)

        # 3. Graph-Constrained HMM (C)
        t0_c = time.time()
        pred_c = base_c.match_trip(cands, obs, graph)
        lat_c = (time.time() - t0_c) / len(obs) * 1000.0
        eval_c = evaluator.evaluate_observation_predictions(pred_c, gt_list)
        col_c = reconstructor.collapse_observation_sequence(pred_c)
        rec_c = reconstructor.reconstruct_and_bridge_route(col_c, graph)
        cont_c, _, _ = reconstructor.evaluate_continuity(rec_c, graph)
        sim_c = reconstructor.compare_with_ground_truth(rec_c, trip.route)

        # Record trip baseline metrics
        partition_name = "val" if trip_id in val_trip_ids else "train"
        for m_name, ev, cont, sim, lat in [
            ("nearest_road", eval_a, cont_a, sim_a, lat_a),
            ("direction_aware", eval_b, cont_b, sim_b, lat_b),
            ("graph_constrained_hmm", eval_c, cont_c, sim_c, lat_c),
        ]:
            baseline_trip_results.append({
                "trip_id": trip_id,
                "partition": partition_name,
                "matcher": m_name,
                "top1_accuracy": ev["top1_accuracy"],
                "macro_f1": ev["macro_f1"],
                "precision": ev["precision"],
                "recall": ev["recall"],
                "route_continuity_pct": cont,
                "route_f1": sim["route_f1"],
                "route_jaccard": sim["route_jaccard"],
                "route_precision": sim["route_precision"],
                "route_recall": sim["route_recall"],
                "avg_latency_ms": round(lat, 4),
            })

        # Save first trip for visualization
        if sample_trip_data is None:
            sample_trip_data = trip
            sample_obs = obs
            sample_cands = cands
            sample_graph = graph
            sample_routes = {
                "nearest_road": rec_a,
                "direction_aware": rec_b,
                "graph_constrained_hmm": rec_c,
            }

        elapsed = time.time() - t_start
        logger.info(
            f"Trip {trip_id} finished in {elapsed:.2f}s | "
            f"A: {eval_a['top1_accuracy']}% | B: {eval_b['top1_accuracy']}% | C: {eval_c['top1_accuracy']}%"
        )

    # 4. Save metrics summaries
    cand_df = pd.DataFrame(trip_candidate_stats)
    cand_df.to_csv(results_dir / "candidate_coverage_summary.csv", index=False)

    base_results_df = pd.DataFrame(baseline_trip_results)
    base_results_df.to_csv(results_dir / "baseline_trip_metrics.csv", index=False)

    # Summary table averaged across trips per matcher
    summary_metrics = (
        base_results_df.groupby("matcher")
        .agg({
            "top1_accuracy": "mean",
            "macro_f1": "mean",
            "precision": "mean",
            "recall": "mean",
            "route_continuity_pct": "mean",
            "route_f1": "mean",
            "route_jaccard": "mean",
            "avg_latency_ms": "mean",
        })
        .reset_index()
    )
    summary_metrics = summary_metrics.round(4)
    summary_metrics.to_csv(results_dir / "baseline_metrics_summary.csv", index=False)

    logger.info("=" * 70)
    logger.info("AGGREGATE BASELINE BENCHMARK RESULTS:")
    print(summary_metrics.to_string(index=False))
    logger.info("=" * 70)

    # 5. Generate Visualizations
    logger.info("Generating diagnostic visualizations...")
    viz = Phase2Visualizer(plots_dir)
    viz.plot_baseline_comparison(summary_metrics)

    if sample_trip_data is not None:
        viz.plot_candidate_inspection(
            sample_trip_data, sample_obs, sample_cands, sample_graph, obs_start=15, obs_count=8
        )
        viz.generate_interactive_map(
            sample_trip_data, sample_obs, sample_routes, sample_graph
        )

    # 6. Post-execution dataset integrity verification
    logger.info("Post-execution dataset integrity verification...")
    if not verify_dataset_integrity(manifest_path):
        raise RuntimeError("Post-execution dataset integrity check failed! Raw files were modified.")

    total_time = time.time() - total_pipeline_start
    logger.info(f"Phase 2 pipeline execution completed successfully in {total_time:.2f}s.")


def main():
    parser = argparse.ArgumentParser(description="PathSnap Phase 2 Pipeline Runner")
    parser.add_argument("--sample", type=int, default=None, help="Process only first N trips")
    parser.add_argument("--full", action="store_true", help="Process all training trips")
    args = parser.parse_args()

    max_trips = None
    if args.sample is not None:
        max_trips = args.sample
    elif not args.full:
        # Default to sample 10 if neither is specified for safe development
        max_trips = 10

    run_pipeline(max_trips=max_trips)


if __name__ == "__main__":
    main()
