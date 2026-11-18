"""
PathSnap Hackathon — Phase 2: Candidate Road Generation
Extracts spatial road arc candidates for each GPS observation using metric KD-Trees.
Computes candidate distance ranks, along-arc projection positions, and coverage statistics.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import time
import logging

from data_loader import TripData
from trajectory_processor import ProcessedObservation
from road_graph import RoadNetworkGraph
from spatial_index import RoadSpatialIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CandidateGenerator")


class CandidateGenerator:
    """Generates candidate road segments for trajectory observations."""

    def __init__(
        self,
        search_radius_m: float = 60.0,
        max_candidates: int = 10,
        fallback_radius_m: float = 120.0,
    ):
        self.search_radius_m = search_radius_m
        self.max_candidates = max_candidates
        self.fallback_radius_m = fallback_radius_m

    def generate_candidates_for_trip(
        self,
        trip_data: TripData,
        processed_obs: List[ProcessedObservation],
        road_graph: RoadNetworkGraph,
        spatial_index: RoadSpatialIndex,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generates candidate records for all observations of a single trip.
        Returns: (candidate_records, trip_candidate_stats)
        """
        start_time = time.time()
        candidates_list = []
        candidate_counts = []
        distances = []

        pts_with_candidates = 0
        pts_without_candidates = 0

        for obs in processed_obs:
            px, py = road_graph.project_point(obs.lon, obs.lat)
            candidates = spatial_index.query_candidate_arcs(
                px=px,
                py=py,
                search_radius_m=self.search_radius_m,
                max_candidates=self.max_candidates,
                fallback_radius_m=self.fallback_radius_m,
            )

            count = len(candidates)
            candidate_counts.append(count)

            if count > 0:
                pts_with_candidates += 1
            else:
                pts_without_candidates += 1

            for c in candidates:
                distances.append(c["distance_m"])
                record = {
                    "trip_id": trip_data.trip_id,
                    "obs_index": obs.obs_index,
                    "timestamp": obs.timestamp,
                    "gps_lon": obs.lon,
                    "gps_lat": obs.lat,
                    "gps_speed_mps": obs.speed_mps,
                    "gps_bearing_deg": obs.bearing_deg,
                    "candidate_arc_id": c["arc_id"],
                    "candidate_rank": c["candidate_rank"],
                    "distance_to_arc_m": c["distance_m"],
                    "dist_diff_from_best_m": c["dist_diff_from_best_m"],
                    "projection_t": c["projection_t"],
                    "proj_x": c["proj_x"],
                    "proj_y": c["proj_y"],
                    "arc_length_m": c["arc_length_m"],
                    "arc_bearing_deg": c["arc_bearing_deg"],
                    "arc_from_node": c["from_node"],
                    "arc_to_node": c["to_node"],
                }
                candidates_list.append(record)

        elapsed = time.time() - start_time
        num_pts = len(processed_obs)

        stats = {
            "trip_id": trip_data.trip_id,
            "partition": trip_data.partition,
            "total_observations": num_pts,
            "points_with_candidates": pts_with_candidates,
            "points_without_candidates": pts_without_candidates,
            "coverage_pct": round(pts_with_candidates / num_pts * 100.0, 2) if num_pts > 0 else 0.0,
            "total_candidates_generated": len(candidates_list),
            "mean_candidates_per_point": round(sum(candidate_counts) / num_pts, 2) if num_pts > 0 else 0.0,
            "max_candidates_per_point": max(candidate_counts) if candidate_counts else 0,
            "min_candidate_distance_m": round(min(distances), 2) if distances else 0.0,
            "mean_candidate_distance_m": round(sum(distances) / len(distances), 2) if distances else 0.0,
            "max_candidate_distance_m": round(max(distances), 2) if distances else 0.0,
            "runtime_seconds": round(elapsed, 3),
        }

        return candidates_list, stats

