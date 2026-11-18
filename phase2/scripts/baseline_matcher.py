"""
PathSnap Hackathon — Phase 2: Baseline Map Matchers
Implements three baseline matching algorithms:
1. Nearest Road Segment Matcher (Baseline A)
2. Direction-Aware Proximity Matcher (Baseline B)
3. Graph-Constrained Sequence Matcher via HMM Viterbi (Baseline C)
"""

import math
import collections
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import logging

from road_graph import RoadNetworkGraph
from trajectory_processor import ProcessedObservation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BaselineMatcher")


def calculate_angular_diff(deg1: float, deg2: float) -> float:
    """Computes absolute minimal angular difference in degrees [0, 180]."""
    diff = abs(deg1 - deg2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


class NearestRoadMatcher:
    """
    Baseline A: Simple geometric nearest road segment matcher.
    Assigns each observation to the candidate arc with minimal Euclidean distance.
    """

    def __init__(self, name: str = "nearest_road"):
        self.name = name

    def match_trip(
        self,
        candidates: List[Dict[str, Any]],
        processed_obs: List[ProcessedObservation],
        road_graph: Optional[RoadNetworkGraph] = None,
    ) -> List[int]:
        """
        Returns list of predicted arc IDs, one per observation in order of obs_index.
        """
        obs_candidates = collections.defaultdict(list)
        for c in candidates:
            obs_candidates[c["obs_index"]].append(c)

        predictions = []
        for obs in processed_obs:
            cands = obs_candidates.get(obs.obs_index, [])
            if not cands:
                predictions.append(-1)
            else:
                best_cand = min(cands, key=lambda c: c["distance_to_arc_m"])
                predictions.append(int(best_cand["candidate_arc_id"]))

        return predictions


class DirectionAwareMatcher:
    """
    Baseline B: Direction-aware matcher.
    Scores candidates by combining exponential distance decay and cosine heading alignment.
    """

    def __init__(
        self,
        name: str = "direction_aware",
        distance_weight: float = 0.65,
        heading_weight: float = 0.35,
        distance_sigma_m: float = 15.0,
    ):
        self.name = name
        self.distance_weight = distance_weight
        self.heading_weight = heading_weight
        self.distance_sigma_m = distance_sigma_m

    def match_trip(
        self,
        candidates: List[Dict[str, Any]],
        processed_obs: List[ProcessedObservation],
        road_graph: Optional[RoadNetworkGraph] = None,
    ) -> List[int]:
        """
        Returns list of predicted arc IDs based on distance + heading score.
        """
        obs_dict = {obs.obs_index: obs for obs in processed_obs}
        obs_candidates = collections.defaultdict(list)
        for c in candidates:
            obs_candidates[c["obs_index"]].append(c)

        predictions = []
        for obs in processed_obs:
            cands = obs_candidates.get(obs.obs_index, [])
            if not cands:
                predictions.append(-1)
                continue

            best_score = -float("inf")
            best_arc_id = -1

            gps_speed = obs.speed_mps
            gps_bearing = obs.bearing_deg

            for c in cands:
                dist = c["distance_to_arc_m"]
                arc_bearing = c["arc_bearing_deg"]

                # Distance score: exp(-dist / sigma)
                dist_score = math.exp(-dist / self.distance_sigma_m)

                # Heading alignment: cos(angle_diff)
                diff = calculate_angular_diff(gps_bearing, arc_bearing)
                cos_val = math.cos(math.radians(diff))
                # Only give positive weight if aligned, penalty if opposite
                heading_score = max(-0.5, cos_val)

                # If stationary (speed < 1.0 m/s), heading is noisy, rely purely on distance
                if gps_speed < 1.0:
                    score = dist_score
                else:
                    score = self.distance_weight * dist_score + self.heading_weight * heading_score

                if score > best_score:
                    best_score = score
                    best_arc_id = int(c["candidate_arc_id"])

            predictions.append(best_arc_id)

        return predictions


class GraphConstrainedHMMMatcher:
    """
    Baseline C: Graph-constrained Hidden Markov Model with Viterbi decoding.
    Finds the globally optimal sequence of road segments by balancing GPS measurement
    proximity (emission) with road network connectivity and shortest-path distance (transition).
    """

    def __init__(
        self,
        name: str = "graph_constrained_hmm",
        emission_sigma_m: float = 6.0,
        transition_beta: float = 40.0,
        disconnection_penalty: float = -50.0,
        max_network_cutoff_m: float = 1500.0,
    ):
        self.name = name
        self.emission_sigma_m = emission_sigma_m
        self.transition_beta = transition_beta
        self.disconnection_penalty = disconnection_penalty
        self.max_network_cutoff_m = max_network_cutoff_m

    def _emission_log_prob(self, distance_m: float) -> float:
        """Gaussian emission log-probability."""
        return -0.5 * (distance_m / self.emission_sigma_m) ** 2 - math.log(self.emission_sigma_m * 2.5066)

    def _transition_log_prob(self, graph_dist: float, gps_step_dist: float) -> float:
        """Exponential transition log-probability based on routing discrepancy."""
        if math.isinf(graph_dist):
            return self.disconnection_penalty
        diff = abs(graph_dist - gps_step_dist)
        return -diff / self.transition_beta - math.log(self.transition_beta)

    def match_trip(
        self,
        candidates: List[Dict[str, Any]],
        processed_obs: List[ProcessedObservation],
        road_graph: RoadNetworkGraph,
    ) -> List[int]:
        """
        Executes Viterbi trellis decoding over candidates to recover the most likely path.
        """
        if not processed_obs:
            return []

        # Group candidates by obs_index
        obs_cands: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
        for c in candidates:
            obs_cands[c["obs_index"]].append(c)

        T = len(processed_obs)

        # Trellis structures:
        # viterbi[t][cand_idx] = log_prob
        # backpointer[t][cand_idx] = prev_cand_idx
        viterbi: List[Dict[int, float]] = []
        backpointer: List[Dict[int, int]] = []

        # Initialize t = 0
        t0_cands = obs_cands.get(0, [])
        if not t0_cands:
            # Fallback if no candidate at t=0
            return NearestRoadMatcher().match_trip(candidates, processed_obs, road_graph)

        init_v = {}
        for idx, c in enumerate(t0_cands):
            emit = self._emission_log_prob(c["distance_to_arc_m"])
            init_v[idx] = emit

        viterbi.append(init_v)
        backpointer.append({})

        # Forward Viterbi recursion
        for t in range(1, T):
            curr_cands = obs_cands.get(t, [])
            prev_cands = obs_cands.get(t - 1, [])

            if not curr_cands or not prev_cands:
                # Discontinuity fallback: assign equal prior
                curr_v = {}
                curr_bp = {}
                for idx, c in enumerate(curr_cands):
                    curr_v[idx] = self._emission_log_prob(c["distance_to_arc_m"])
                    curr_bp[idx] = 0
                viterbi.append(curr_v)
                backpointer.append(curr_bp)
                continue

            gps_step = processed_obs[t].step_distance_m
            curr_v = {}
            curr_bp = {}

            # Pre-compute transitions for efficiency
            for j_idx, curr_c in enumerate(curr_cands):
                emit = self._emission_log_prob(curr_c["distance_to_arc_m"])
                best_prob = -float("inf")
                best_prev = 0

                for i_idx, prev_c in enumerate(prev_cands):
                    prev_prob = viterbi[t - 1].get(i_idx, -float("inf"))
                    if prev_prob == -float("inf"):
                        continue

                    # Network transition distance
                    graph_dist = road_graph.candidate_transition_distance(
                        int(prev_c["candidate_arc_id"]),
                        int(curr_c["candidate_arc_id"]),
                        max_cutoff_m=self.max_network_cutoff_m,
                    )

                    trans = self._transition_log_prob(graph_dist, gps_step)
                    total_p = prev_prob + trans

                    if total_p > best_prob:
                        best_prob = total_p
                        best_prev = i_idx

                curr_v[j_idx] = best_prob + emit
                curr_bp[j_idx] = best_prev

            viterbi.append(curr_v)
            backpointer.append(curr_bp)

        # Backward termination and path recovery
        predictions = [-1] * T

        # Best candidate at last time step
        last_t = T - 1
        if not viterbi[last_t]:
            return NearestRoadMatcher().match_trip(candidates, processed_obs, road_graph)

        best_last_idx = max(viterbi[last_t].keys(), key=lambda k: viterbi[last_t][k])
        curr_idx = best_last_idx

        for t in range(last_t, -1, -1):
            t_cands = obs_cands.get(t, [])
            if t_cands and curr_idx < len(t_cands):
                predictions[t] = int(t_cands[curr_idx]["candidate_arc_id"])
            elif t_cands:
                predictions[t] = int(t_cands[0]["candidate_arc_id"])
            else:
                predictions[t] = -1

            if t > 0:
                curr_idx = backpointer[t].get(curr_idx, 0)

        return predictions
