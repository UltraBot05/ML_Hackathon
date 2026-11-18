"""
PathSnap Hackathon — Phase 2: Road Network Graph Engine
Represents directed road networks, node geometries, arc headings, lengths,
and provides efficient shortest-path operations in metric projected coordinates.
"""

import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import networkx as nx
import numpy as np
import logging

from data_loader import RoadNode, RoadArc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RoadGraph")

EARTH_RADIUS = 6_371_000.0


def calculate_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Computes compass bearing from (lon1, lat1) to (lon2, lat2) in degrees [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


@dataclass
class ProjectedNode:
    node_id: int
    lon: float
    lat: float
    x: float  # meters
    y: float  # meters


@dataclass
class EnrichedArc:
    arc_id: int
    from_node: int
    to_node: int
    length_m: float
    bearing_deg: float
    x1: float
    y1: float
    x2: float
    y2: float
    mid_x: float
    mid_y: float


class RoadNetworkGraph:
    """
    Engine representing directed road network subgraphs with local metric projection.
    Provides fast spatial lookup, adjacency lists, and network shortest-path queries.
    """

    def __init__(self, trip_id: str, nodes: List[RoadNode], arcs: List[RoadArc]):
        self.trip_id = trip_id
        self.raw_nodes = nodes
        self.raw_arcs = arcs

        # 1. Determine local projection origin (center of bounding box)
        lons = [n.lon for n in nodes]
        lats = [n.lat for n in nodes]
        self.min_lon, self.max_lon = min(lons), max(lons)
        self.min_lat, self.max_lat = min(lats), max(lats)
        self.center_lon = (self.min_lon + self.max_lon) / 2.0
        self.center_lat = (self.min_lat + self.max_lat) / 2.0
        self.cos_lat0 = math.cos(math.radians(self.center_lat))

        # 2. Project nodes to metric coordinates (meters)
        self.nodes_dict: Dict[int, ProjectedNode] = {}
        for n in nodes:
            x = EARTH_RADIUS * math.radians(n.lon - self.center_lon) * self.cos_lat0
            y = EARTH_RADIUS * math.radians(n.lat - self.center_lat)
            self.nodes_dict[n.node_id] = ProjectedNode(node_id=n.node_id, lon=n.lon, lat=n.lat, x=x, y=y)

        # 3. Enrich arcs with lengths, bearings, coordinates, and midpoints
        self.arcs_dict: Dict[int, EnrichedArc] = {}
        self.adj_out: Dict[int, List[Tuple[int, int]]] = {n.node_id: [] for n in nodes}
        self.adj_in: Dict[int, List[Tuple[int, int]]] = {n.node_id: [] for n in nodes}

        self.nx_graph = nx.DiGraph()
        self.nx_graph.add_nodes_from(range(len(nodes)))

        for a in arcs:
            u_node = self.nodes_dict.get(a.from_node)
            v_node = self.nodes_dict.get(a.to_node)
            if u_node is None or v_node is None:
                continue

            dx = v_node.x - u_node.x
            dy = v_node.y - u_node.y
            length = math.hypot(dx, dy)
            bearing = calculate_bearing(u_node.lon, u_node.lat, v_node.lon, v_node.lat)
            mid_x = (u_node.x + v_node.x) / 2.0
            mid_y = (u_node.y + v_node.y) / 2.0

            enriched = EnrichedArc(
                arc_id=a.arc_id,
                from_node=a.from_node,
                to_node=a.to_node,
                length_m=round(length, 3),
                bearing_deg=round(bearing, 2),
                x1=u_node.x,
                y1=u_node.y,
                x2=v_node.x,
                y2=v_node.y,
                mid_x=mid_x,
                mid_y=mid_y,
            )
            self.arcs_dict[a.arc_id] = enriched
            self.adj_out[a.from_node].append((a.to_node, a.arc_id))
            self.adj_in[a.to_node].append((a.from_node, a.arc_id))

            # Add to NetworkX graph with length weight
            self.nx_graph.add_edge(a.from_node, a.to_node, weight=length, arc_id=a.arc_id)

        # Cache for single source shortest path distance dictionaries
        self._source_dist_cache: Dict[int, Dict[int, float]] = {}

    def project_point(self, lon: float, lat: float) -> Tuple[float, float]:
        """Converts arbitrary (lon, lat) to the trip's local metric (x, y) in meters."""
        x = EARTH_RADIUS * math.radians(lon - self.center_lon) * self.cos_lat0
        y = EARTH_RADIUS * math.radians(lat - self.center_lat)
        return x, y

    def get_arc(self, arc_id: int) -> Optional[EnrichedArc]:
        return self.arcs_dict.get(arc_id)

    def get_distances_from_node(self, from_node: int, max_cutoff_m: float = 2000.0) -> Dict[int, float]:
        """
        Computes shortest path distances from from_node to all reachable nodes within max_cutoff_m.
        Caches results per from_node.
        """
        if from_node in self._source_dist_cache:
            return self._source_dist_cache[from_node]

        try:
            dists = nx.single_source_dijkstra_path_length(
                self.nx_graph, source=from_node, cutoff=max_cutoff_m, weight="weight"
            )
        except (nx.NodeNotFound, KeyError):
            dists = {from_node: 0.0}

        self._source_dist_cache[from_node] = dists
        return dists

    def shortest_path_distance(self, from_node: int, to_node: int, max_cutoff_m: float = 2000.0) -> float:
        """
        Computes directed shortest path distance in meters between two network nodes.
        Returns 0.0 if same node; inf if disconnected or exceeds cutoff.
        """
        if from_node == to_node:
            return 0.0

        dists = self.get_distances_from_node(from_node, max_cutoff_m)
        return dists.get(to_node, float("inf"))

    def candidate_transition_distance(self, prev_arc_id: int, curr_arc_id: int, max_cutoff_m: float = 2000.0) -> float:
        """
        Computes directed network distance to transition from prev_arc to curr_arc.
        If prev_arc == curr_arc: transition distance is 0.0.
        Otherwise: shortest path from prev_arc.to_node to curr_arc.from_node + curr_arc.length.
        """
        if prev_arc_id == curr_arc_id:
            return 0.0

        prev_arc = self.get_arc(prev_arc_id)
        curr_arc = self.get_arc(curr_arc_id)
        if prev_arc is None or curr_arc is None:
            return float("inf")

        # Direct topological connection: to_node == from_node
        if prev_arc.to_node == curr_arc.from_node:
            return curr_arc.length_m

        path_dist = self.shortest_path_distance(prev_arc.to_node, curr_arc.from_node, max_cutoff_m)
        if math.isinf(path_dist):
            return float("inf")
        return path_dist + curr_arc.length_m
