"""
PathSnap Hackathon — Phase 2: Feature Engineering Pipeline
Computes geometric, kinematic, heading alignment, topological network,
and sequential transition features for road arc candidates.
Prepares clean, leakage-free feature tables ready for Phase 3 ML models.
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import logging

from road_graph import RoadNetworkGraph
from trajectory_processor import ProcessedObservation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FeatureEngineering")


def calculate_angular_diff(deg1: float, deg2: float) -> float:
    """Computes absolute minimal angular difference in degrees [0, 180]."""
    diff = abs(deg1 - deg2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


class FeatureEngineer:
    """
    Computes discriminative features for candidate road arcs.
    """

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

    def __init__(self, max_candidates: int = 10, max_network_cutoff_m: float = 2000.0):
        self.max_candidates = max_candidates
        self.max_network_cutoff_m = max_network_cutoff_m

    def compute_candidate_features(
        self,
        labeled_candidates: List[Dict[str, Any]],
        processed_obs: List[ProcessedObservation],
        road_graph: RoadNetworkGraph,
    ) -> pd.DataFrame:
        """
        Takes candidate dictionaries (with or without labels), computes all feature columns,
        and returns an enriched pandas DataFrame.
        """
        if not labeled_candidates:
            return pd.DataFrame()

        # Build lookup for observations
        obs_dict: Dict[int, ProcessedObservation] = {obs.obs_index: obs for obs in processed_obs}

        # Identify previous top-1 candidate for each observation for transition features
        prev_best_arc_id: Optional[int] = None
        obs_prev_best: Dict[int, Optional[int]] = {}

        # Sort candidates by obs_index then rank
        candidates_sorted = sorted(labeled_candidates, key=lambda c: (c["obs_index"], c["candidate_rank"]))

        # First pass: map obs_index -> rank 1 arc_id
        obs_rank1: Dict[int, int] = {}
        for c in candidates_sorted:
            if c["obs_index"] not in obs_rank1 and c["candidate_rank"] == 1:
                obs_rank1[c["obs_index"]] = c["candidate_arc_id"]

        # Cache node degree lookups
        out_degrees = {nid: len(adj) for nid, adj in road_graph.adj_out.items()}
        in_degrees = {nid: len(adj) for nid, adj in road_graph.adj_in.items()}

        rows = []
        for c in candidates_sorted:
            obs_idx = c["obs_index"]
            obs = obs_dict.get(obs_idx)

            # 1. Geometric features
            dist_m = float(c["distance_to_arc_m"])
            dist_diff = float(c.get("dist_diff_from_best_m", 0.0))
            proj_t = float(c["projection_t"])
            rank = int(c["candidate_rank"])
            norm_rank = rank / float(self.max_candidates)
            log_dist = math.log1p(max(0.0, dist_m))

            # 2. Movement & Heading features
            speed_mps = obs.speed_mps if obs else float(c.get("gps_speed_mps", 0.0))
            gps_bearing = obs.bearing_deg if obs else float(c.get("gps_bearing_deg", 0.0))
            arc_bearing = float(c["arc_bearing_deg"])

            heading_diff = calculate_angular_diff(gps_bearing, arc_bearing)
            rad_diff = math.radians(heading_diff)
            cos_heading = math.cos(rad_diff)

            # Weight heading alignment by speed (heading is unreliable when stationary)
            speed_factor = min(1.0, max(0.0, speed_mps / 5.0))
            speed_heading_alignment = cos_heading * speed_factor

            # Counter-flow flag: heading difference > 120 deg while moving at > 3 m/s
            is_counter_flow = int(heading_diff > 120.0 and speed_mps > 3.0)

            # 3. Road Network features
            arc_length = float(c["arc_length_m"])
            from_node = int(c["arc_from_node"])
            to_node = int(c["arc_to_node"])

            out_deg = out_degrees.get(from_node, 1)
            in_deg = in_degrees.get(to_node, 1)
            is_terminal = int(out_deg == 1 and in_deg == 1)

            # 4. Sequential & Transition features
            step_gps_dist = obs.step_distance_m if obs else 0.0
            prev_rank1_arc = obs_rank1.get(obs_idx - 1)

            if prev_rank1_arc is not None and obs_idx > 0:
                curr_arc_id = int(c["candidate_arc_id"])
                trans_dist = road_graph.candidate_transition_distance(
                    prev_rank1_arc, curr_arc_id, max_cutoff_m=self.max_network_cutoff_m
                )
                if math.isinf(trans_dist):
                    trans_dist = self.max_network_cutoff_m

                # Directly connected check
                prev_arc_obj = road_graph.get_arc(prev_rank1_arc)
                if prev_arc_obj is not None:
                    is_connected = int(prev_arc_obj.to_node == from_node or prev_rank1_arc == curr_arc_id)
                else:
                    is_connected = 0
            else:
                trans_dist = 0.0
                is_connected = 1

            trans_ratio = trans_dist / (step_gps_dist + 1.0)
            log_trans_dist = math.log1p(min(self.max_network_cutoff_m, trans_dist))

            # Assemble complete row
            row = {
                # Metadata / Identifiers
                "trip_id": c["trip_id"],
                "obs_index": obs_idx,
                "timestamp": c.get("timestamp", 0.0),
                "gps_lon": c["gps_lon"],
                "gps_lat": c["gps_lat"],
                "candidate_arc_id": c["candidate_arc_id"],
                "arc_from_node": from_node,
                "arc_to_node": to_node,
                # Geometric features
                "distance_to_arc_m": round(dist_m, 3),
                "dist_diff_from_best_m": round(dist_diff, 3),
                "projection_t": round(proj_t, 4),
                "candidate_rank": rank,
                "normalized_rank": round(norm_rank, 4),
                "log_distance": round(log_dist, 4),
                # Movement & Heading features
                "gps_speed_mps": round(speed_mps, 3),
                "gps_bearing_deg": round(gps_bearing, 2),
                "arc_bearing_deg": round(arc_bearing, 2),
                "heading_diff_deg": round(heading_diff, 2),
                "cos_heading_diff": round(cos_heading, 4),
                "speed_heading_alignment": round(speed_heading_alignment, 4),
                "is_counter_flow": is_counter_flow,
                # Road Network features
                "arc_length_m": round(arc_length, 3),
                "from_node_out_degree": out_deg,
                "to_node_in_degree": in_deg,
                "is_terminal_arc": is_terminal,
                # Sequential & Transition features
                "step_gps_dist_m": round(step_gps_dist, 3),
                "transition_dist_m": round(trans_dist, 3),
                "transition_dist_ratio": round(trans_ratio, 4),
                "is_directly_connected": is_connected,
                "log_transition_dist": round(log_trans_dist, 4),
            }

            # Optional Ground Truth Label (for training partition)
            if "is_correct" in c:
                row["is_correct"] = int(c["is_correct"])
            if "true_arc_id" in c:
                row["true_arc_id"] = int(c["true_arc_id"])

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

