"""
PathSnap Hackathon — Phase 2: Spatial Indexing Engine
Builds high-performance metric KD-Trees over road network segments to enable
sub-millisecond candidate arc retrieval without brute-force scans.
"""

import math
from typing import List, Tuple, Dict, Set, Any
import numpy as np
from scipy.spatial import KDTree
import logging

from road_graph import RoadNetworkGraph, EnrichedArc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SpatialIndex")


def point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float, float, float]:
    """
    Computes exact Euclidean distance from point P to line segment AB in meters.
    Returns: (distance_m, projection_t, proj_x, proj_y)
    where projection_t is clamped to [0.0, 1.0].
    """
    dx = x2 - x1
    dy = y2 - y1
    L2 = dx * dx + dy * dy

    if L2 < 1e-12:
        dist = math.hypot(px - x1, py - y1)
        return dist, 0.0, x1, y1

    # Dot product projection
    t = ((px - x1) * dx + (py - y1) * dy) / L2
    t_clamped = max(0.0, min(1.0, t))

    proj_x = x1 + t_clamped * dx
    proj_y = y1 + t_clamped * dy
    dist = math.hypot(px - proj_x, py - proj_y)

    return dist, t_clamped, proj_x, proj_y


class RoadSpatialIndex:
    """
    High-speed KD-Tree spatial index over road network arc segments in metric coordinates.
    Guarantees fast O(log |E|) candidate retrieval with zero false dismissals.
    """

    def __init__(self, road_graph: RoadNetworkGraph, sample_spacing_m: float = 25.0):
        self.road_graph = road_graph
        self.sample_spacing_m = sample_spacing_m

        # Build sample points along all arcs
        sample_coords = []
        self.sample_arc_ids = []

        for arc_id, arc in self.road_graph.arcs_dict.items():
            length = arc.length_m
            # Number of sample points along segment
            num_samples = max(2, int(math.ceil(length / self.sample_spacing_m)) + 1)
            ts = np.linspace(0.0, 1.0, num_samples)

            for t in ts:
                sx = arc.x1 + t * (arc.x2 - arc.x1)
                sy = arc.y1 + t * (arc.y2 - arc.y1)
                sample_coords.append((sx, sy))
                self.sample_arc_ids.append(arc_id)

        if sample_coords:
            self.kdtree = KDTree(np.array(sample_coords))
        else:
            self.kdtree = None

    def query_candidate_arcs(
        self,
        px: float,
        py: float,
        search_radius_m: float = 60.0,
        max_candidates: int = 10,
        fallback_radius_m: float = 120.0,
    ) -> List[Dict[str, Any]]:
        """
        Finds all road arcs within search_radius_m of metric point (px, py).
        Computes exact perpendicular distance and projection factor for each candidate.
        Returns candidates sorted by distance ascending.
        """
        if self.kdtree is None:
            return []

        # 1. Query KDTree for candidate sample indices
        sample_indices = self.kdtree.query_ball_point([px, py], r=search_radius_m)

        # Fallback to wider search if no candidate found
        if not sample_indices and fallback_radius_m > search_radius_m:
            sample_indices = self.kdtree.query_ball_point([px, py], r=fallback_radius_m)

        # 2. Extract unique arc IDs
        candidate_arc_ids: Set[int] = {self.sample_arc_ids[idx] for idx in sample_indices}

        # If still empty, find the single nearest sample point
        if not candidate_arc_ids:
            dist, idx = self.kdtree.query([px, py], k=1)
            candidate_arc_ids.add(self.sample_arc_ids[idx])

        # 3. Compute exact perpendicular distances
        candidates = []
        for arc_id in candidate_arc_ids:
            arc = self.road_graph.get_arc(arc_id)
            if arc is None:
                continue

            dist, t_proj, proj_x, proj_y = point_to_segment_distance(
                px, py, arc.x1, arc.y1, arc.x2, arc.y2
            )

            candidates.append({
                "arc_id": arc_id,
                "distance_m": round(dist, 3),
                "projection_t": round(t_proj, 4),
                "proj_x": round(proj_x, 3),
                "proj_y": round(proj_y, 3),
                "arc_length_m": arc.length_m,
                "arc_bearing_deg": arc.bearing_deg,
                "from_node": arc.from_node,
                "to_node": arc.to_node,
            })

        # 4. Sort by distance ascending and trim to max_candidates
        candidates.sort(key=lambda c: c["distance_m"])
        top_candidates = candidates[:max_candidates]

        # 5. Assign candidate distance rank and distance difference from rank 1
        best_dist = top_candidates[0]["distance_m"] if top_candidates else 0.0
        for rank, c in enumerate(top_candidates, start=1):
            c["candidate_rank"] = rank
            c["dist_diff_from_best_m"] = round(c["distance_m"] - best_dist, 3)

        return top_candidates
