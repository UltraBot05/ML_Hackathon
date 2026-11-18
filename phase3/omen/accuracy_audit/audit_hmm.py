"""ML-emission Viterbi decoder from Adarsh's Phase 3 integration.

Extracted unchanged from phase3/scripts/evaluate_ml_hmm.py at
768a80b6100ca4332ce7d27b5736b64ed64cbac4.
"""

from __future__ import annotations

import collections
import math
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import numpy as np

    from road_graph import RoadNetworkGraph
    from trajectory_processor import ProcessedObservation


class MLEmissionHMMMatcher:
    """
    Decodes the optimal sequence of road arcs by combining machine learning emission potentials
    with topological road network transition penalties via Viterbi trellis dynamic programming.
    """

    def __init__(
        self,
        name: str = "ml_hmm_viterbi",
        transition_beta: float = 40.0,
        disconnection_penalty: float = -50.0,
        max_network_cutoff_m: float = 1500.0,
        prob_epsilon: float = 1e-6,
    ):
        self.name = name
        self.transition_beta = transition_beta
        self.disconnection_penalty = disconnection_penalty
        self.max_network_cutoff_m = max_network_cutoff_m
        self.prob_epsilon = prob_epsilon

    def _transition_log_prob(self, graph_dist: float, gps_step_dist: float) -> float:
        """Exponential transition log-probability based on routing discrepancy."""
        if math.isinf(graph_dist):
            return self.disconnection_penalty
        diff = abs(graph_dist - gps_step_dist)
        return -diff / self.transition_beta - math.log(self.transition_beta)

    def match_trip(
        self,
        candidates: List[Dict[str, Any]],
        candidate_probs: np.ndarray,
        processed_obs: List[ProcessedObservation],
        road_graph: RoadNetworkGraph,
    ) -> List[int]:
        """
        Executes Viterbi decoding using ML predicted candidate probabilities as emission scores.
        """
        if not processed_obs:
            return []

        # Attach probs to candidate dicts
        for c, p in zip(candidates, candidate_probs):
            c["ml_prob"] = float(p)

        # Group candidates by obs_index
        obs_cands: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
        for c in candidates:
            obs_cands[c["obs_index"]].append(c)

        T = len(processed_obs)
        viterbi: List[Dict[int, float]] = []
        backpointer: List[Dict[int, int]] = []

        # Initialize t = 0
        t0_cands = obs_cands.get(0, [])
        if not t0_cands:
            return [-1] * T

        init_v = {}
        for idx, c in enumerate(t0_cands):
            emit = math.log(max(self.prob_epsilon, c["ml_prob"]))
            init_v[idx] = emit

        viterbi.append(init_v)
        backpointer.append({})

        # Forward Viterbi recursion
        for t in range(1, T):
            curr_cands = obs_cands.get(t, [])
            prev_cands = obs_cands.get(t - 1, [])

            if not curr_cands or not prev_cands:
                # Discontinuity fallback
                curr_v = {}
                curr_bp = {}
                for idx, c in enumerate(curr_cands):
                    curr_v[idx] = math.log(max(self.prob_epsilon, c["ml_prob"]))
                    curr_bp[idx] = 0
                viterbi.append(curr_v)
                backpointer.append(curr_bp)
                continue

            gps_step = processed_obs[t].step_distance_m
            curr_v = {}
            curr_bp = {}

            for j_idx, curr_c in enumerate(curr_cands):
                emit = math.log(max(self.prob_epsilon, curr_c["ml_prob"]))
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
        last_t = T - 1
        if not viterbi[last_t]:
            return predictions

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
