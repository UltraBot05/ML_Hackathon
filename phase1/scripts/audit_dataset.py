"""
PathSnap Hackathon - Phase 1: Dataset Audit
Audits file presence, row counts, coordinates, timestamps, and network integrity.
Supports running from project root or script directory.
"""

from pathlib import Path
import csv
import math
import sys

# Locate project root dynamically
CURRENT_FILE = Path(__file__).resolve()
# Candidate roots: parent of scripts (phase1), or parent of phase1 (project root)
if (CURRENT_FILE.parents[1] / "Dataset").exists():
    PROJECT_ROOT = CURRENT_FILE.parents[1]
elif (CURRENT_FILE.parents[2] / "Dataset").exists():
    PROJECT_ROOT = CURRENT_FILE.parents[2]
else:
    PROJECT_ROOT = Path(r"D:\ML_Hackathon\ML_Hackathon")

DATASET = PROJECT_ROOT / "Dataset"
OUTPUT_DIR = PROJECT_ROOT / "phase1" / "results" / "csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Also preserve phase1_results for backward compatibility
LEGACY_OUTPUT_DIR = PROJECT_ROOT / "phase1_results"
LEGACY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTITIONS = ["train", "student_test"]
EXTENSIONS = [".track", ".nodes", ".arcs", ".route"]


def read_numeric_rows(path):
    rows = []
    malformed = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                values = [float(x) for x in line.split()]
                rows.append(values)
            except ValueError:
                malformed += 1
    return rows, malformed


def valid_lon_lat(lon, lat):
    return (
        math.isfinite(lon)
        and math.isfinite(lat)
        and -180 <= lon <= 180
        and -90 <= lat <= 90
    )


def run_audit():
    audit_rows = []
    gps_rows = []
    route_rows = []
    file_rows = []

    for partition in PARTITIONS:
        partition_dir = DATASET / partition
        if not partition_dir.exists():
            print(f"Partition missing: {partition}")
            continue

        trip_dirs = sorted(p for p in partition_dir.iterdir() if p.is_dir())
        print(f"\nAuditing {partition}: {len(trip_dirs)} trips")

        for trip_dir in trip_dirs:
            trip_id = trip_dir.name
            files = {ext: trip_dir / f"{trip_id}{ext}" for ext in EXTENSIONS}

            missing = [ext for ext, path in files.items() if not path.exists()]
            empty = [
                ext for ext, path in files.items()
                if path.exists() and path.stat().st_size == 0
            ]

            # File metadata
            for ext, path in files.items():
                file_rows.append({
                    "partition": partition,
                    "trip_id": trip_id,
                    "extension": ext,
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else None
                })

            # Initialize counters
            gps_count = nodes_count = arcs_count = route_count = 0
            invalid_coords = malformed_track = 0
            non_monotonic = duplicate_timestamps = 0
            time_gaps = []
            timestamps = []

            invalid_arc_nodes = 0
            invalid_route_arc_ids = 0
            malformed_nodes = malformed_arcs = malformed_route = 0

            # TRACK
            if files[".track"].exists():
                track, malformed_track = read_numeric_rows(files[".track"])
                gps_count = len(track)

                previous_time = None
                seen_timestamps = set()

                for row in track:
                    if len(row) < 3:
                        malformed_track += 1
                        continue

                    lon, lat, timestamp = row[:3]

                    if not valid_lon_lat(lon, lat):
                        invalid_coords += 1

                    if not math.isfinite(timestamp):
                        malformed_track += 1
                        continue

                    timestamps.append(timestamp)

                    if timestamp in seen_timestamps:
                        duplicate_timestamps += 1
                    seen_timestamps.add(timestamp)

                    if previous_time is not None:
                        gap = timestamp - previous_time
                        time_gaps.append(gap)
                        if gap <= 0:
                            non_monotonic += 1

                    previous_time = timestamp

            # NODES
            nodes = []
            if files[".nodes"].exists():
                nodes, malformed_nodes = read_numeric_rows(files[".nodes"])
                nodes_count = len(nodes)
                for row in nodes:
                    if len(row) < 2 or not valid_lon_lat(row[0], row[1]):
                        malformed_nodes += 1

            # ARCS
            arcs = []
            if files[".arcs"].exists():
                arcs, malformed_arcs = read_numeric_rows(files[".arcs"])
                arcs_count = len(arcs)
                for row in arcs:
                    if len(row) < 2:
                        malformed_arcs += 1
                        continue

                    from_node, to_node = row[:2]
                    if (
                        not from_node.is_integer()
                        or not to_node.is_integer()
                        or from_node < 0
                        or to_node < 0
                        or from_node >= nodes_count
                        or to_node >= nodes_count
                    ):
                        invalid_arc_nodes += 1

            # ROUTE
            route = []
            if files[".route"].exists():
                route, malformed_route = read_numeric_rows(files[".route"])
                route_count = len(route)
                for row in route:
                    if len(row) < 1:
                        malformed_route += 1
                        continue

                    arc_id = row[0]
                    if (
                        not arc_id.is_integer()
                        or arc_id < 0
                        or arc_id >= arcs_count
                    ):
                        invalid_route_arc_ids += 1

            # GPS statistics
            if time_gaps:
                avg_gap = sum(time_gaps) / len(time_gaps)
                min_gap = min(time_gaps)
                max_gap = max(time_gaps)
                one_sec_pct = (
                    sum(abs(g - 1.0) < 1e-6 for g in time_gaps)
                    / len(time_gaps) * 100
                )
            else:
                avg_gap = min_gap = max_gap = one_sec_pct = None

            gps_rows.append({
                "partition": partition,
                "trip_id": trip_id,
                "gps_points": gps_count,
                "invalid_coordinates": invalid_coords,
                "malformed_track_rows": malformed_track,
                "duplicate_timestamps": duplicate_timestamps,
                "non_monotonic_intervals": non_monotonic,
                "avg_time_gap_sec": avg_gap,
                "min_time_gap_sec": min_gap,
                "max_time_gap_sec": max_gap,
                "one_second_interval_pct": one_sec_pct
            })

            route_rows.append({
                "partition": partition,
                "trip_id": trip_id,
                "nodes": nodes_count,
                "arcs": arcs_count,
                "route_entries": route_count,
                "invalid_arc_node_references": invalid_arc_nodes,
                "invalid_route_arc_references": invalid_route_arc_ids,
                "malformed_nodes": malformed_nodes,
                "malformed_arcs": malformed_arcs,
                "malformed_route": malformed_route
            })

            audit_rows.append({
                "partition": partition,
                "trip_id": trip_id,
                "missing_files": ",".join(missing),
                "empty_files": ",".join(empty),
                "gps_points": gps_count,
                "nodes": nodes_count,
                "arcs": arcs_count,
                "route_entries": route_count,
                "status": "CHECK" if missing or empty else "FILES_PRESENT"
            })

    def save_csv(filename, rows):
        if not rows:
            return
        for target_dir in [OUTPUT_DIR, LEGACY_OUTPUT_DIR]:
            path = target_dir / filename
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved: {path}")

    save_csv("trip_audit.csv", audit_rows)
    save_csv("file_inventory.csv", file_rows)
    save_csv("gps_audit.csv", gps_rows)
    save_csv("network_route_audit.csv", route_rows)

    print("\n===== PHASE 1 AUDIT COMPLETE =====")
    print(f"Trips audited: {len(audit_rows)}")
    print(f"Reports saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_audit()

