"""
PathSnap Hackathon — Phase 2: Ground-Truth Label Generation
Aligns raw GPS observations with ground-truth route arc sequences via monotonic window tracking.
Assigns binary classification label 'is_correct' to candidate arcs.
"""

from typing import List, Dict, Any, Tuple, Optional
import collections
import logging

from data_loader import TripData
from trajectory_processor import ProcessedObservation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LabelGenerator")


class LabelGenerator:
    """
    Aligns trajectory GPS observations to ground-truth route arcs and assigns candidate labels.
    """

    def __init__(self, forward_window_size: int = 15, backward_window_size: int = 2):
        self.forward_window_size = forward_window_size
        self.backward_window_size = backward_window_size

    def align_trip_observations(
        self,
        trip_data: TripData,
        processed_obs: List[ProcessedObservation],
        candidates: List[Dict[str, Any]],
    ) -> Tuple[Dict[int, Optional[int]], Dict[str, Any]]:
        """
        Determines the ground-truth active arc for each observation index.
        Returns:
            - mapping: obs_index -> true_arc_id (or None if unaligned)
            - stats: dictionary of alignment summary metrics
        """
        route = trip_data.route
        if not route:
            logger.warning(f"Trip {trip_data.trip_id} has no route data.")
            return {obs.obs_index: None for obs in processed_obs}, {
                "trip_id": trip_data.trip_id,
                "total_obs": len(processed_obs),
                "matched_obs": 0,
                "unmatched_obs": len(processed_obs),
                "match_rate_pct": 0.0,
                "candidate_recall_pct": 0.0,
            }

        # Index candidates by obs_index
        cands_by_obs = collections.defaultdict(list)
        for c in candidates:
            cands_by_obs[c["obs_index"]].append(c)

        route_set = set(route)
        obs_to_true_arc: Dict[int, Optional[int]] = {}
        route_pointer = 0
        num_obs = len(processed_obs)
        matched_obs = 0
        true_in_candidates = 0

        for i in range(num_obs):
            c_list = cands_by_obs[i]
            c_dict = {c["candidate_arc_id"]: c for c in c_list}

            # Search in a monotonic window around current route pointer
            start_j = max(0, route_pointer - self.backward_window_size)
            end_j = min(len(route), route_pointer + self.forward_window_size + 1)

            best_arc = None
            best_dist = float("inf")
            best_j = route_pointer

            for j in range(start_j, end_j):
                r_arc = route[j]
                if r_arc in c_dict:
                    d = c_dict[r_arc]["distance_to_arc_m"]
                    if d < best_dist:
                        best_dist = d
                        best_arc = r_arc
                        best_j = j

            if best_arc is not None:
                obs_to_true_arc[i] = best_arc
                route_pointer = best_j
                matched_obs += 1
                true_in_candidates += 1
            else:
                # Fallback: check if any candidate is in the route
                c_in_route = [c for c in c_list if c["candidate_arc_id"] in route_set]
                if c_in_route:
                    best_c = min(c_in_route, key=lambda c: c["distance_to_arc_m"])
                    obs_to_true_arc[i] = best_c["candidate_arc_id"]
                    matched_obs += 1
                    true_in_candidates += 1
                    # Update pointer if this arc exists in route
                    try:
                        route_pointer = route.index(best_c["candidate_arc_id"], max(0, route_pointer - 2))
                    except ValueError:
                        pass
                else:
                    # Observation has no route arc in its candidate set (missed candidate)
                    obs_to_true_arc[i] = None

        match_rate = round(matched_obs / num_obs * 100.0, 2) if num_obs > 0 else 0.0
        candidate_recall = round(true_in_candidates / num_obs * 100.0, 2) if num_obs > 0 else 0.0

        stats = {
            "trip_id": trip_data.trip_id,
            "total_obs": num_obs,
            "matched_obs": matched_obs,
            "unmatched_obs": num_obs - matched_obs,
            "match_rate_pct": match_rate,
            "candidate_recall_pct": candidate_recall,
        }

        return obs_to_true_arc, stats

    def assign_candidate_labels(
        self,
        candidates: List[Dict[str, Any]],
        obs_to_true_arc: Dict[int, Optional[int]],
    ) -> List[Dict[str, Any]]:
        """
        Enriches candidate dictionaries with 'true_arc_id' and binary target 'is_correct'.
        """
        labeled_candidates = []
        for c in candidates:
            c_copy = dict(c)
            true_arc = obs_to_true_arc.get(c["obs_index"])
            c_copy["true_arc_id"] = true_arc if true_arc is not None else -1
            c_copy["is_correct"] = 1 if (true_arc is not None and c["candidate_arc_id"] == true_arc) else 0
            labeled_candidates.append(c_copy)
        return labeled_candidates

