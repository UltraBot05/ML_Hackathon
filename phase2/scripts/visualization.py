"""
PathSnap Hackathon — Phase 2: Visualization Suite
Generates high-resolution diagnostic plots and interactive Folium maps:
1. Spatial candidate generation diagram (GPS points, search radii, candidate arcs)
2. Trajectory map-matching comparison (Ground truth vs Baseline routes)
3. Baseline benchmark metric comparison charts
4. Interactive multi-layer Folium HTML map for deep visual inspection
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
import logging

from data_loader import TripData
from trajectory_processor import ProcessedObservation
from road_graph import RoadNetworkGraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Visualizer")


class Phase2Visualizer:
    """Generates Phase 2 diagnostic visual artifacts."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_baseline_comparison(
        self,
        metrics_df,
        output_filename: str = "baseline_performance_comparison.png",
    ) -> Path:
        """
        Creates a 4-panel comparison bar chart across baseline matchers:
        - Top-1 Accuracy (%)
        - Macro F1 Score
        - Route Continuity (%)
        - Latency per point (ms)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("PathSnap Phase 2 — Baseline Map-Matching Benchmark", fontsize=16, fontweight="bold")

        colors = ["#4C72B0", "#55A868", "#C44E52"]
        matchers = metrics_df["matcher"].tolist()

        # Panel 1: Top-1 Accuracy
        ax1 = axes[0, 0]
        bars1 = ax1.bar(matchers, metrics_df["top1_accuracy"], color=colors, alpha=0.85, edgecolor="black")
        ax1.set_title("Observation Top-1 Accuracy (%)", fontweight="bold")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0, 105)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")

        # Panel 2: Route Continuity
        ax2 = axes[0, 1]
        bars2 = ax2.bar(matchers, metrics_df["route_continuity_pct"], color=colors, alpha=0.85, edgecolor="black")
        ax2.set_title("Reconstructed Route Continuity (%)", fontweight="bold")
        ax2.set_ylabel("Continuity (%)")
        ax2.set_ylim(0, 105)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")

        # Panel 3: Route F1 Score
        ax3 = axes[1, 0]
        bars3 = ax3.bar(matchers, metrics_df["route_f1"], color=colors, alpha=0.85, edgecolor="black")
        ax3.set_title("Ground-Truth Route F1 Score", fontweight="bold")
        ax3.set_ylabel("F1 Score")
        ax3.set_ylim(0, 1.1)
        ax3.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars3:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")

        # Panel 4: Latency per Point (Log Scale)
        ax4 = axes[1, 1]
        bars4 = ax4.bar(matchers, metrics_df["avg_latency_ms"], color=colors, alpha=0.85, edgecolor="black")
        ax4.set_title("Inference Latency per Observation (ms)", fontweight="bold")
        ax4.set_ylabel("Milliseconds / Point")
        ax4.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f"{yval:.4f} ms", ha="center", va="bottom", fontweight="bold")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = self.output_dir / output_filename
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info(f"Saved baseline comparison plot to {out_path}")
        return out_path

    def plot_candidate_inspection(
        self,
        trip_data: TripData,
        processed_obs: List[ProcessedObservation],
        candidates: List[Dict[str, Any]],
        road_graph: RoadNetworkGraph,
        obs_start: int = 10,
        obs_count: int = 8,
        output_filename: str = "candidate_generation_inspection.png",
    ) -> Path:
        """
        Plots a high-detail spatial diagram showing GPS points, search radii,
        and nearby candidate road segments with perpendicular projections.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_title(
            f"Trip {trip_data.trip_id} — Candidate Road Segment Retrieval & Projections",
            fontsize=14,
            fontweight="bold",
        )

        import collections
        cands_by_obs = collections.defaultdict(list)
        for c in candidates:
            cands_by_obs[c["obs_index"]].append(c)

        sub_obs = processed_obs[obs_start : obs_start + obs_count]
        colors = plt.cm.tab10(range(obs_count))

        for idx, obs in enumerate(sub_obs):
            px, py = road_graph.project_point(obs.lon, obs.lat)
            col = colors[idx]

            # Plot GPS point
            ax.scatter(px, py, color=col, s=80, zorder=5, label=f"GPS #{obs.obs_index}")

            # Plot search circle (60m)
            circle = plt.Circle((px, py), 60.0, color=col, fill=False, linestyle=":", alpha=0.4)
            ax.add_patch(circle)

            # Plot candidate arcs and projection lines
            obs_c = cands_by_obs.get(obs.obs_index, [])
            for c in obs_c:
                arc = road_graph.get_arc(c["candidate_arc_id"])
                if arc is not None:
                    # Draw arc segment
                    is_best = (c["candidate_rank"] == 1)
                    arc_color = "red" if is_best else "gray"
                    arc_alpha = 0.8 if is_best else 0.4
                    arc_width = 2.0 if is_best else 1.0
                    ax.plot([arc.x1, arc.x2], [arc.y1, arc.y2], color=arc_color, alpha=arc_alpha, linewidth=arc_width, zorder=3)

                    # Draw perpendicular projection line
                    ax.plot([px, c["proj_x"]], [py, c["proj_y"]], color=col, linestyle="--", alpha=0.6, linewidth=1.0, zorder=4)

        ax.set_xlabel("Local Projected X (meters)", fontweight="bold")
        ax.set_ylabel("Local Projected Y (meters)", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        out_path = self.output_dir / output_filename
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info(f"Saved candidate inspection plot to {out_path}")
        return out_path

    def generate_interactive_map(
        self,
        trip_data: TripData,
        processed_obs: List[ProcessedObservation],
        reconstructed_routes: Dict[str, List[int]],
        road_graph: RoadNetworkGraph,
        output_filename: str = "interactive_map_matching.html",
        tiles: Optional[str] = "CartoDB positron",
    ) -> Path:
        """
        Creates a high-performance interactive Folium map with togglable layers:
        - Raw GPS trajectory observations (blue markers + line)
        - Ground truth route (green line)
        - Baseline A reconstructed route (orange line)
        - Baseline B reconstructed route (purple line)
        - Baseline C reconstructed route (red line)
        """
        center_lat = road_graph.center_lat
        center_lon = road_graph.center_lon

        if tiles and str(tiles).lower() in ("cartodb positron", "cartodbpositron", "default"):
            fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)
            folium.TileLayer(
                tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png?key=cb1_5bhz_1_52b498fdc9aac4015cc35f5a",
                attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                name="CartoDB Positron (Clean Default)",
                subdomains="abcd",
                max_zoom=19,
                control=True,
                overlay=False,
            ).add_to(fmap)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                attr="Tiles &copy; Esri",
                name="Esri WorldStreetMap",
                max_zoom=19,
                control=True,
                overlay=False,
            ).add_to(fmap)
            folium.TileLayer(
                tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                name="OpenStreetMap",
                max_zoom=19,
                control=True,
                overlay=False,
            ).add_to(fmap)
        elif tiles and str(tiles).lower() != "none":
            fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=tiles)
        else:
            fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)

        # Layer 1: Raw GPS observations
        gps_fg = folium.FeatureGroup(name="Raw GPS Observations (Blue)", show=True)
        gps_coords = [(o.lat, o.lon) for o in processed_obs]
        folium.PolyLine(gps_coords, color="#1f77b4", weight=3, opacity=0.7, tooltip="Raw GPS Track").add_to(gps_fg)

        # Add sample markers every 10 points
        for i in range(0, len(processed_obs), max(1, len(processed_obs) // 50)):
            o = processed_obs[i]
            folium.CircleMarker(
                location=[o.lat, o.lon],
                radius=4,
                color="#1f77b4",
                fill=True,
                fill_color="#1f77b4",
                tooltip=f"Obs #{o.obs_index} | Speed: {o.speed_kmh} km/h | Time: {o.timestamp}s",
            ).add_to(gps_fg)
        gps_fg.add_to(fmap)

        # Helper to extract arc line coordinates
        def get_route_coords(arc_ids: List[int]) -> List[Tuple[float, float]]:
            coords = []
            for aid in arc_ids:
                arc = road_graph.get_arc(aid)
                if arc is not None:
                    u = road_graph.nodes_dict.get(arc.from_node)
                    v = road_graph.nodes_dict.get(arc.to_node)
                    if u and v:
                        coords.append((u.lat, u.lon))
                        coords.append((v.lat, v.lon))
            return coords

        # Layer 2: Ground Truth Route
        if trip_data.route:
            gt_fg = folium.FeatureGroup(name="Ground-Truth Route (Green)", show=True)
            gt_coords = get_route_coords(trip_data.route)
            folium.PolyLine(gt_coords, color="#2ca02c", weight=5, opacity=0.85, tooltip="Ground-Truth Route").add_to(gt_fg)
            gt_fg.add_to(fmap)

        # Baseline Routes
        layer_styles = {
            "nearest_road": ("Baseline A: Nearest (Orange)", "#ff7f0e", 3, False),
            "direction_aware": ("Baseline B: Direction-Aware (Purple)", "#9467bd", 3.5, False),
            "graph_constrained_hmm": ("Baseline C: Graph HMM (Red)", "#d62728", 4, True),
        }

        for matcher_name, arc_ids in reconstructed_routes.items():
            if matcher_name in layer_styles:
                label, color, weight, show = layer_styles[matcher_name]
                fg = folium.FeatureGroup(name=label, show=show)
                coords = get_route_coords(arc_ids)
                folium.PolyLine(coords, color=color, weight=weight, opacity=0.8, tooltip=label).add_to(fg)
                fg.add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)

        out_path = self.output_dir / output_filename
        fmap.save(str(out_path))
        logger.info(f"Saved interactive Folium map to {out_path}")
        return out_path
