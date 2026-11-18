"""
PathSnap Hackathon - Phase 1: Road Network and Route Exploration
Analyzes node/arc topology, graph connectivity, bounding boxes, and ground-truth route continuity.
Computationally efficient and memory safe (O(V+E) linear graph algorithms).
Produces network_summary.csv and route_integrity.csv.
"""

from pathlib import Path
import csv
import math
import statistics
import networkx as nx

# Dynamic root detection
CURRENT_FILE = Path(__file__).resolve()
if (CURRENT_FILE.parents[1] / "Dataset").exists():
    PROJECT_ROOT = CURRENT_FILE.parents[1]
elif (CURRENT_FILE.parents[2] / "Dataset").exists():
    PROJECT_ROOT = CURRENT_FILE.parents[2]
else:
    PROJECT_ROOT = Path(r"D:\ML_Hackathon\ML_Hackathon")

DATASET = PROJECT_ROOT / "Dataset"
CSV_OUT = PROJECT_ROOT / "phase1" / "results" / "csv"
CSV_OUT.mkdir(parents=True, exist_ok=True)
LEGACY_OUT = PROJECT_ROOT / "phase1_results"
LEGACY_OUT.mkdir(parents=True, exist_ok=True)

PARTITIONS = ["train", "student_test"]


def load_nodes(path):
    nodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                nodes.append((float(parts[0]), float(parts[1])))
    return nodes


def load_arcs(path):
    arcs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                arcs.append((int(float(parts[0])), int(float(parts[1]))))
    return arcs


def load_route(path):
    route = []
    if not path.exists():
        return route
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                route.append(int(float(parts[0])))
    return route


def load_track_bbox(path):
    if not path.exists():
        return None
    lons, lats = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                lons.append(float(parts[0]))
                lats.append(float(parts[1]))
    if not lons:
        return None
    return min(lons), max(lons), min(lats), max(lats)


def analyze_network_for_trip(trip_dir, trip_id, partition):
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"
    track_path = trip_dir / f"{trip_id}.track"

    nodes = load_nodes(nodes_path)
    arcs = load_arcs(arcs_path)
    route = load_route(route_path)
    track_bbox = load_track_bbox(track_path)

    node_count = len(nodes)
    arc_count = len(arcs)

    if node_count == 0 or arc_count == 0:
        empty_res = {
            "partition": partition,
            "trip_id": trip_id,
            "nodes": node_count,
            "arcs": arc_count,
            "route_arcs": len(route),
            "status": "EMPTY_GRAPH"
        }
        empty_route = {
            "partition": partition,
            "trip_id": trip_id,
            "route_arcs_count": len(route),
            "unique_arcs_count": 0,
            "route_discontinuities": 0,
            "route_continuity_pct": 0.0,
            "repeated_arcs_count": 0,
            "all_arc_ids_valid": False,
            "min_arc_id": None,
            "max_arc_id": None,
            "total_arcs_in_graph": 0,
            "route_status": "EMPTY"
        }
        return empty_res, empty_route

    # Node bounding box
    node_lons = [n[0] for n in nodes]
    node_lats = [n[1] for n in nodes]
    min_node_lon, max_node_lon = min(node_lons), max(node_lons)
    min_node_lat, max_node_lat = min(node_lats), max(node_lats)

    # GPS inside network bounding box?
    gps_in_bbox = None
    if track_bbox:
        t_min_lon, t_max_lon, t_min_lat, t_max_lat = track_bbox
        gps_in_bbox = (
            min_node_lon <= t_min_lon and t_max_lon <= max_node_lon and
            min_node_lat <= t_min_lat and t_max_lat <= max_node_lat
        )

    # Build directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(node_count))
    G.add_edges_from(arcs)

    # Degree stats
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    isolated_nodes = sum(1 for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0)

    # Connected components
    num_scc = nx.number_strongly_connected_components(G)
    num_wcc = nx.number_weakly_connected_components(G)

    # Route continuity check
    route_discontinuities = 0
    route_repeated_arcs = 0
    all_valid = True

    if route:
        seen_arcs = set()
        for idx in range(len(route)):
            arc_id = route[idx]
            if not (0 <= arc_id < arc_count):
                all_valid = False
            if arc_id in seen_arcs:
                route_repeated_arcs += 1
            seen_arcs.add(arc_id)

            if idx > 0:
                prev_arc_id = route[idx - 1]
                if 0 <= prev_arc_id < arc_count and 0 <= arc_id < arc_count:
                    prev_to_node = arcs[prev_arc_id][1]
                    curr_from_node = arcs[arc_id][0]
                    if prev_to_node != curr_from_node:
                        route_discontinuities += 1
                else:
                    route_discontinuities += 1

    route_continuity_pct = (
        ((len(route) - 1 - route_discontinuities) / (len(route) - 1) * 100.0)
        if len(route) > 1 else 100.0
    )

    net_summary = {
        "partition": partition,
        "trip_id": trip_id,
        "nodes": node_count,
        "arcs": arc_count,
        "route_arcs": len(route),
        "min_lon": round(min_node_lon, 5),
        "max_lon": round(max_node_lon, 5),
        "min_lat": round(min_node_lat, 5),
        "max_lat": round(max_node_lat, 5),
        "gps_strictly_inside_bbox": gps_in_bbox,
        "isolated_nodes": isolated_nodes,
        "isolated_node_pct": round(isolated_nodes / node_count * 100.0, 2),
        "avg_in_degree": round(statistics.mean(in_degrees), 2),
        "max_in_degree": max(in_degrees),
        "avg_out_degree": round(statistics.mean(out_degrees), 2),
        "max_out_degree": max(out_degrees),
        "strongly_connected_components": num_scc,
        "weakly_connected_components": num_wcc,
        "route_discontinuities": route_discontinuities,
        "route_continuity_pct": round(route_continuity_pct, 2),
        "route_repeated_arcs": route_repeated_arcs
    }

    route_integrity = {
        "partition": partition,
        "trip_id": trip_id,
        "route_arcs_count": len(route),
        "unique_arcs_count": len(set(route)) if route else 0,
        "route_discontinuities": route_discontinuities,
        "route_continuity_pct": round(route_continuity_pct, 2),
        "repeated_arcs_count": route_repeated_arcs,
        "all_arc_ids_valid": all_valid,
        "min_arc_id": min(route) if route else None,
        "max_arc_id": max(route) if route else None,
        "total_arcs_in_graph": arc_count,
        "route_status": "PERFECT_CONTINUITY" if route_discontinuities == 0 else "MINOR_DISCONTINUITY"
    }

    return net_summary, route_integrity


def run_network_analysis():
    records = []
    route_records = []
    print("Running road network analysis across partitions...\n")

    for partition in PARTITIONS:
        folder = DATASET / partition
        if not folder.exists():
            continue

        for trip_dir in sorted(folder.iterdir()):
            if not trip_dir.is_dir():
                continue
            trip_id = trip_dir.name
            stats, route_stats = analyze_network_for_trip(trip_dir, trip_id, partition)
            records.append(stats)
            route_records.append(route_stats)

    # Save to CSV
    for target_dir in [CSV_OUT, LEGACY_OUT]:
        out_csv = target_dir / "network_summary.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"Saved: {out_csv.resolve()}")

        route_csv = target_dir / "route_integrity.csv"
        with open(route_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=route_records[0].keys())
            writer.writeheader()
            writer.writerows(route_records)
        print(f"Saved: {route_csv.resolve()}")

    # Print summary
    train_records = [r for r in records if r["partition"] == "train"]
    test_records = [r for r in records if r["partition"] == "student_test"]

    print("\n========== ROAD NETWORK AUDIT SUMMARY ==========")
    for name, subset in [("TRAIN", train_records), ("STUDENT_TEST", test_records)]:
        nodes = [r["nodes"] for r in subset]
        arcs = [r["arcs"] for r in subset]
        routes = [r["route_arcs"] for r in subset]
        discont = [r["route_discontinuities"] for r in subset]
        cont_pct = [r["route_continuity_pct"] for r in subset]
        iso = [r["isolated_nodes"] for r in subset]

        print(f"\n{name} ({len(subset)} trips):")
        print(f"  Nodes per trip   : mean={statistics.mean(nodes):.1f}, min={min(nodes)}, max={max(nodes)}")
        print(f"  Arcs per trip    : mean={statistics.mean(arcs):.1f}, min={min(arcs)}, max={max(arcs)}")
        print(f"  Route arcs       : mean={statistics.mean(routes):.1f}, min={min(routes)}, max={max(routes)}")
        print(f"  Isolated nodes   : mean={statistics.mean(iso):.1f} (avg {statistics.mean([r['isolated_node_pct'] for r in subset]):.2f}%)")
        print(f"  Route continuity : mean={statistics.mean(cont_pct):.2f}% (trips with perfect 100% continuity: {sum(d == 0 for d in discont)}/{len(subset)})")


if __name__ == "__main__":
    run_network_analysis()
