"""
Test script to benchmark all three baselines on Trip 00000001.
"""

import sys
import time
from pathlib import Path

# Add scripts directory
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from trajectory_processor import TrajectoryProcessor
from road_graph import RoadNetworkGraph
from spatial_index import RoadSpatialIndex
from candidate_generation import CandidateGenerator
from label_generation import LabelGenerator
from baseline_matcher import NearestRoadMatcher, DirectionAwareMatcher, GraphConstrainedHMMMatcher
from route_reconstruction import RouteReconstructor
from evaluation import ModelEvaluator


def main():
    loader = DataLoader("D:/ML_Hackathon/ML_Hackathon/Dataset")
    t0 = loader.load_trip("00000001", "train")
    if t0 is None:
        print("Failed to load trip 00000001")
        return

    proc = TrajectoryProcessor()
    obs = proc.process_trip_track(t0.trip_id, t0.track)
    graph = RoadNetworkGraph(t0.trip_id, t0.nodes, t0.arcs)
    sindex = RoadSpatialIndex(graph)
    gen = CandidateGenerator(search_radius_m=60.0, max_candidates=10)
    cands, c_stats = gen.generate_candidates_for_trip(t0, obs, graph, sindex)

    labeler = LabelGenerator()
    obs_map, align_stats = labeler.align_trip_observations(t0, obs, cands)
    gt_list = [obs_map.get(i) for i in range(len(obs))]

    evaluator = ModelEvaluator()
    reconstructor = RouteReconstructor()

    # Baseline A: Nearest Road
    t_a0 = time.time()
    base_a = NearestRoadMatcher()
    pred_a = base_a.match_trip(cands, obs, graph)
    lat_a = (time.time() - t_a0) / len(obs) * 1000.0
    eval_a = evaluator.evaluate_observation_predictions(pred_a, gt_list)
    col_a = reconstructor.collapse_observation_sequence(pred_a)
    rec_a = reconstructor.reconstruct_and_bridge_route(col_a, graph)
    cont_a, _, _ = reconstructor.evaluate_continuity(rec_a, graph)
    route_sim_a = reconstructor.compare_with_ground_truth(rec_a, t0.route)

    print("=== Baseline A (Nearest Road) ===")
    print(f"Top-1 Accuracy: {eval_a['top1_accuracy']}%")
    print(f"Observation Macro F1: {eval_a['macro_f1']}")
    print(f"Route Continuity: {cont_a}%")
    print(f"Route F1: {route_sim_a['route_f1']}")
    print(f"Latency: {lat_a:.4f} ms/point")

    # Baseline B: Direction-Aware
    t_b0 = time.time()
    base_b = DirectionAwareMatcher()
    pred_b = base_b.match_trip(cands, obs, graph)
    lat_b = (time.time() - t_b0) / len(obs) * 1000.0
    eval_b = evaluator.evaluate_observation_predictions(pred_b, gt_list)
    col_b = reconstructor.collapse_observation_sequence(pred_b)
    rec_b = reconstructor.reconstruct_and_bridge_route(col_b, graph)
    cont_b, _, _ = reconstructor.evaluate_continuity(rec_b, graph)
    route_sim_b = reconstructor.compare_with_ground_truth(rec_b, t0.route)

    print("\n=== Baseline B (Direction-Aware) ===")
    print(f"Top-1 Accuracy: {eval_b['top1_accuracy']}%")
    print(f"Observation Macro F1: {eval_b['macro_f1']}")
    print(f"Route Continuity: {cont_b}%")
    print(f"Route F1: {route_sim_b['route_f1']}")
    print(f"Latency: {lat_b:.4f} ms/point")

    # Baseline C: Graph-Constrained HMM
    t_c0 = time.time()
    base_c = GraphConstrainedHMMMatcher()
    pred_c = base_c.match_trip(cands, obs, graph)
    lat_c = (time.time() - t_c0) / len(obs) * 1000.0
    eval_c = evaluator.evaluate_observation_predictions(pred_c, gt_list)
    col_c = reconstructor.collapse_observation_sequence(pred_c)
    rec_c = reconstructor.reconstruct_and_bridge_route(col_c, graph)
    cont_c, _, _ = reconstructor.evaluate_continuity(rec_c, graph)
    route_sim_c = reconstructor.compare_with_ground_truth(rec_c, t0.route)

    print("\n=== Baseline C (Graph-Constrained HMM) ===")
    print(f"Top-1 Accuracy: {eval_c['top1_accuracy']}%")
    print(f"Observation Macro F1: {eval_c['macro_f1']}")
    print(f"Route Continuity: {cont_c}%")
    print(f"Route F1: {route_sim_c['route_f1']}")
    print(f"Latency: {lat_c:.4f} ms/point")


if __name__ == "__main__":
    main()

