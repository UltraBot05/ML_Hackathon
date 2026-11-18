"""
PathSnap Hackathon — Phase 2: Route Reconstruction Engine
Converts observation-level matched arc sequences into topologically contiguous road routes.
Performs consecutive deduplication, connectivity auditing, gap bridging via shortest path,
and computes route-level quality metrics against ground-truth routes.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import networkx as nx
import logging

from road_graph import RoadNetworkGraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RouteReconstruction")


class RouteReconstructor:
    """
    Reconstructs contiguous vehicle routes from matched observation arc sequences.
    """

    def __init__(self, max_gap_bridge_m: float = 600.0, max_gap_arcs: int = 5):
        self.max_gap_bridge_m = max_gap_bridge_m
        self.max_gap_arcs = max_gap_arcs

    def collapse_observation_sequence(self, matched_arcs: List[int]) -> List[int]:
        """
        Deduplicates consecutive identical arc IDs, ignoring invalid (-1) predictions.
        [10, 10, 10, 12, 12, 15] -> [10, 12, 15]
        """
        collapsed = []
        for arc_id in matched_arcs:
            if arc_id < 0:
                continue
            if not collapsed or collapsed[-1] != arc_id:
                collapsed.append(arc_id)
        return collapsed

    def evaluate_continuity(
        self, collapsed_arcs: List[int], road_graph: RoadNetworkGraph
    ) -> Tuple[float, int, int]:
        """
        Evaluates topological continuity of a sequence of arcs.
        Returns: (continuity_pct, connected_transitions, total_transitions)
        """
        if len(collapsed_arcs) <= 1:
            return 100.0, 0, 0

        total_transitions = len(collapsed_arcs) - 1
        connected_transitions = 0

        for i in range(total_transitions):
            curr_arc = road_graph.get_arc(collapsed_arcs[i])
            next_arc = road_graph.get_arc(collapsed_arcs[i + 1])

            if curr_arc is not None and next_arc is not None:
                if curr_arc.to_node == next_arc.from_node:
                    connected_transitions += 1

        continuity_pct = round(connected_transitions / total_transitions * 100.0, 2)
        return continuity_pct, connected_transitions, total_transitions

    def reconstruct_and_bridge_route(
        self,
        collapsed_arcs: List[int],
        road_graph: RoadNetworkGraph,
    ) -> List[int]:
        """
        Bridges topological gaps between disconnected consecutive arcs using Dijkstra shortest paths.
        """
        if len(collapsed_arcs) <= 1:
            return list(collapsed_arcs)

        reconstructed = [collapsed_arcs[0]]

        for i in range(len(collapsed_arcs) - 1):
            curr_id = collapsed_arcs[i]
            next_id = collapsed_arcs[i + 1]

            curr_arc = road_graph.get_arc(curr_id)
            next_arc = road_graph.get_arc(next_id)

            if curr_arc is None or next_arc is None:
                reconstructed.append(next_id)
                continue

            # Directly connected
            if curr_arc.to_node == next_arc.from_node:
                reconstructed.append(next_id)
                continue

            # Check if bridging is possible via shortest path
            try:
                path_nodes = nx.shortest_path(
                    road_graph.nx_graph,
                    source=curr_arc.to_node,
                    target=next_arc.from_node,
                    weight="weight",
                )
                # Check path length
                path_len = nx.shortest_path_length(
                    road_graph.nx_graph,
                    source=curr_arc.to_node,
                    target=next_arc.from_node,
                    weight="weight",
                )

                if path_len <= self.max_gap_bridge_m and len(path_nodes) - 1 <= self.max_gap_arcs:
                    # Extract intermediate arc IDs
                    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                        edge_data = road_graph.nx_graph.get_edge_data(u, v)
                        if edge_data and "arc_id" in edge_data:
                            reconstructed.append(edge_data["arc_id"])
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

            reconstructed.append(next_id)

        # Deduplicate consecutive duplicates that may have arisen
        final_route = []
        for a in reconstructed:
            if not final_route or final_route[-1] != a:
                final_route.append(a)

        return final_route

    def compare_with_ground_truth(
        self, reconstructed_arcs: List[int], true_route: List[int]
    ) -> Dict[str, float]:
        """
        Computes set overlap and sequential precision/recall against ground-truth route.
        """
        rec_set = set(reconstructed_arcs)
        true_set = set(true_route)

        intersection = rec_set.intersection(true_set)
        union = rec_set.union(true_set)

        jaccard = (len(intersection) / len(union)) if union else 1.0
        precision = (len(intersection) / len(rec_set)) if rec_set else 0.0
        recall = (len(intersection) / len(true_set)) if true_set else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "route_precision": round(precision, 4),
            "route_recall": round(recall, 4),
            "route_f1": round(f1, 4),
            "route_jaccard": round(jaccard, 4),
            "pred_route_len": len(reconstructed_arcs),
            "true_route_len": len(true_route),
        }

