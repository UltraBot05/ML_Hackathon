"""
PathSnap Hackathon — Independent Route-Label Alignment Diagnostic
TASK B: Independent verification of ground-truth route alignment without candidate-set conditioning.

Algorithm:
1. Reconstructs metric road geometry from raw .nodes and .arcs.
2. Traverses the true chronological .route arc sequence.
3. For each GPS point, evaluates true geometric projection against route arcs in a monotonic
   progression window, INDEPENDENT of whether the arc is present in the KD-tree candidate set.
4. Cross-checks the independently established reference arc against Phase 2's candidate-proxy label:
   - True candidate miss rate (reference arc absent from candidate set)
   - Spatial rank of true reference arc in candidates
   - Opposite-direction arc retrieval
   - Agreement rate and disagreement taxonomy
"""

import sys
import math
import collections
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE2_SCRIPTS = PROJECT_ROOT / "phase2" / "scripts"
sys.path.insert(0, str(PHASE2_SCRIPTS))

from data_loader import DataLoader, TripData
from trajectory_processor import TrajectoryProcessor, ProcessedObservation
from road_graph import RoadNetworkGraph
from spatial_index import RoadSpatialIndex
from candidate_generation import CandidateGenerator
from label_generation import LabelGenerator


def point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """Computes perpendicular metric distance and along-segment projection factor t in [0, 1]."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.hypot(px - x1, py - y1), 0.0

    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, t))
    proj_x = x1 + t_clamped * dx
    proj_y = y1 + t_clamped * dy
    dist = math.hypot(px - proj_x, py - proj_y)
    return dist, t_clamped


def calculate_angular_diff(deg1: float, deg2: float) -> float:
    diff = abs(deg1 - deg2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def run_independent_alignment(
    trip_data: TripData,
    processed_obs: List[ProcessedObservation],
    road_graph: RoadNetworkGraph,
    candidates: List[Dict[str, Any]],
    phase2_labels: Dict[int, Optional[int]],
    forward_window: int = 15,
    backward_window: int = 2,
) -> Dict[str, Any]:
    """
    Performs candidate-independent ground-truth route alignment.
    """
    route = trip_data.route
    if not route:
        return {"error": "No ground truth route"}

    cands_by_obs = collections.defaultdict(list)
    for c in candidates:
        cands_by_obs[c["obs_index"]].append(c)

    route_ptr = 0
    num_obs = len(processed_obs)

    indep_aligned_arcs = {}
    indep_aligned_dists = {}
    agreement_count = 0
    candidate_hit_count = 0
    candidate_rank_distribution = collections.defaultdict(int)
    absent_from_candidates_count = 0
    opposite_direction_count = 0
    phase2_none_indep_valid_count = 0
    indep_none_count = 0

    disagreement_details = []

    # Map arc to its reverse arc if exists in graph
    arc_nodes = {aid: (arc.from_node, arc.to_node) for aid, arc in road_graph.arcs_dict.items()}
    node_pair_to_arc = {nodes: aid for aid, nodes in arc_nodes.items()}

    for t, obs in enumerate(processed_obs):
        px, py = road_graph.project_point(obs.lon, obs.lat)
        obs_bearing = obs.bearing_deg if obs.bearing_deg is not None else 0.0
        obs_speed = obs.speed_mps if obs.speed_mps is not None else 0.0

        # Search monotonically along route sequence ONLY (independent of candidates)
        start_j = max(0, route_ptr - backward_window)
        end_j = min(len(route), route_ptr + forward_window + 1)

        best_route_arc = None
        best_dist = float("inf")
        best_j = route_ptr

        for j in range(start_j, end_j):
            arc_id = route[j]
            arc = road_graph.arcs_dict.get(arc_id)
            if arc is None:
                continue

            dist, proj_t = point_to_segment_distance(px, py, arc.x1, arc.y1, arc.x2, arc.y2)
            heading_diff = calculate_angular_diff(obs_bearing, arc.bearing_deg)

            # Penalize wrong-way motion if moving at reasonable speed
            cost = dist
            if obs_speed > 3.0 and heading_diff > 120.0:
                cost += 50.0  # strong penalty for opposite direction

            if cost < best_dist:
                best_dist = cost
                best_route_arc = arc_id
                best_j = j

        # If best match is within 120m, accept it
        if best_route_arc is not None and best_dist < 120.0:
            indep_aligned_arcs[t] = best_route_arc
            indep_aligned_dists[t] = best_dist
            route_ptr = best_j
        else:
            # Fallback: scan whole route
            min_r_arc = None
            min_r_dist = float("inf")
            min_r_j = route_ptr
            for j, arc_id in enumerate(route):
                arc = road_graph.arcs_dict.get(arc_id)
                if arc is None:
                    continue
                d, _ = point_to_segment_distance(px, py, arc.x1, arc.y1, arc.x2, arc.y2)
                if d < min_r_dist:
                    min_r_dist = d
                    min_r_arc = arc_id
                    min_r_j = j

            if min_r_dist < 120.0:
                indep_aligned_arcs[t] = min_r_arc
                indep_aligned_dists[t] = min_r_dist
                route_ptr = min_r_j
            else:
                indep_aligned_arcs[t] = None
                indep_none_count += 1

        true_arc = indep_aligned_arcs[t]
        p2_arc = phase2_labels.get(t)

        # Check agreement
        if true_arc == p2_arc:
            agreement_count += 1
        else:
            if p2_arc is None and true_arc is not None:
                phase2_none_indep_valid_count += 1

        # Check presence in candidate set
        cand_list = cands_by_obs.get(t, [])
        cand_arc_ids = [c["candidate_arc_id"] for c in cand_list]

        if true_arc is not None:
            if true_arc in cand_arc_ids:
                candidate_hit_count += 1
                rank = cand_arc_ids.index(true_arc) + 1
                candidate_rank_distribution[rank] += 1
            else:
                absent_from_candidates_count += 1
                # Check if reverse arc was retrieved
                u, v = arc_nodes.get(true_arc, (-1, -1))
                rev_arc = node_pair_to_arc.get((v, u))
                if rev_arc is not None and rev_arc in cand_arc_ids:
                    opposite_direction_count += 1

                if len(disagreement_details) < 5:
                    disagreement_details.append({
                        "obs_index": t,
                        "indep_arc": true_arc,
                        "phase2_arc": p2_arc,
                        "indep_dist_m": round(indep_aligned_dists.get(t, 0.0), 2),
                        "cand_count": len(cand_list),
                        "cand_arcs": cand_arc_ids[:3],
                    })

    agreement_pct = (agreement_count / num_obs * 100.0) if num_obs > 0 else 0.0
    true_candidate_recall_pct = (candidate_hit_count / (num_obs - indep_none_count) * 100.0) if (num_obs - indep_none_count) > 0 else 0.0

    return {
        "trip_id": trip_data.trip_id,
        "total_obs": num_obs,
        "agreement_count": agreement_count,
        "agreement_pct": round(agreement_pct, 2),
        "true_candidate_recall_pct": round(true_candidate_recall_pct, 2),
        "absent_from_candidates": absent_from_candidates_count,
        "opposite_direction_candidates": opposite_direction_count,
        "phase2_none_but_indep_found": phase2_none_indep_valid_count,
        "rank_distribution": dict(sorted(candidate_rank_distribution.items())),
        "disagreement_samples": disagreement_details,
    }


def main():
    loader = DataLoader(PROJECT_ROOT / "Dataset")
    proc = TrajectoryProcessor()
    c_gen = CandidateGenerator()
    p2_labeler = LabelGenerator()

    # Representative validation trips covering diverse road environments:
    # 00000001 (standard urban), 00000005 (clean short), 00000016 (complex),
    # 00000043 (high noise), 00000061 (arterial), 00000084 (huge 16k-obs trip)
    test_trips = ["00000001", "00000005", "00000013", "00000016", "00000024", "00000028", "00000043", "00000084"]

    print("=== PathSnap Independent Route-Label Alignment Diagnostic ===")
    print(f"Auditing {len(test_trips)} representative validation trips...\n")

    summary_rows = []

    for trip_id in test_trips:
        trip = loader.load_trip(trip_id, "train")
        if trip is None:
            continue

        obs = proc.process_trip_track(trip.trip_id, trip.track)
        graph = RoadNetworkGraph(trip.trip_id, trip.nodes, trip.arcs)
        s_index = RoadSpatialIndex(graph)
        cands, _ = c_gen.generate_candidates_for_trip(trip, obs, graph, s_index)

        # Run Phase 2 proxy labeler
        p2_labels, p2_stats = p2_labeler.align_trip_observations(trip, obs, cands)

        # Run independent route alignment
        diag = run_independent_alignment(trip, obs, graph, cands, p2_labels)

        summary_rows.append({
            "Trip ID": trip_id,
            "Total Obs": diag["total_obs"],
            "Label Agreement (%)": diag["agreement_pct"],
            "Independent True Recall (%)": diag["true_candidate_recall_pct"],
            "Phase2 Proxy Recall (%)": p2_stats["candidate_recall_pct"],
            "Absent from Candidates": diag["absent_from_candidates"],
            "Opposite Direction Arcs": diag["opposite_direction_candidates"],
            "Phase2 None (Found Indep)": diag["phase2_none_but_indep_found"],
            "Rank 1 Hits": diag["rank_distribution"].get(1, 0),
        })

        print(f"Trip {trip_id}: Agreement = {diag['agreement_pct']}%, Indep Recall = {diag['true_candidate_recall_pct']}%, Absent = {diag['absent_from_candidates']}")

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Summary Table ===")
    print(summary_df.to_string(index=False))

    out_csv = PROJECT_ROOT / "phase3" / "results" / "independent_label_audit_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved independent diagnostic summary to: {out_csv}")


if __name__ == "__main__":
    main()
