"""
PathSnap Hackathon - Phase 1: Trajectory Statistics
Computes distance, duration, speeds, sampling intervals, and anomaly counts per trip.
Strictly separates train and student_test partitions.
Supports running from project root or script directory.
"""

from pathlib import Path
import csv
import math
import statistics

# Dynamic project root detection
CURRENT_FILE = Path(__file__).resolve()
if (CURRENT_FILE.parents[1] / "Dataset").exists():
    PROJECT_ROOT = CURRENT_FILE.parents[1]
elif (CURRENT_FILE.parents[2] / "Dataset").exists():
    PROJECT_ROOT = CURRENT_FILE.parents[2]
else:
    PROJECT_ROOT = Path(r"D:\ML_Hackathon\ML_Hackathon")

DATASET = PROJECT_ROOT / "Dataset"
OUTPUT_DIR = PROJECT_ROOT / "phase1" / "results" / "csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_OUTPUT_DIR = PROJECT_ROOT / "phase1_results"
LEGACY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTITIONS = ["train", "student_test"]
EARTH_RADIUS = 6_371_000  # meters


def haversine(lon1, lat1, lon2, lat2):
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS * math.asin(math.sqrt(min(1.0, a)))


def analyze_track(path):
    points = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            if len(values) < 3:
                continue
            lon, lat, timestamp = map(float, values[:3])
            points.append((lon, lat, timestamp))

    if not points:
        return {
            "gps_points": 0,
            "total_distance_km": 0.0,
            "duration_seconds": 0.0,
            "avg_time_gap_seconds": 0.0,
            "median_time_gap_seconds": 0.0,
            "min_time_gap_seconds": 0.0,
            "max_time_gap_seconds": 0.0,
            "avg_speed_kmh": 0.0,
            "max_segment_speed_kmh": 0.0,
            "segments_over_55m_s": 0,
            "zero_time_gaps": 0,
            "negative_time_gaps": 0
        }

    distances = []
    time_gaps = []
    speeds = []
    zero_gaps = 0
    negative_gaps = 0

    for i in range(1, len(points)):
        lon1, lat1, t1 = points[i - 1]
        lon2, lat2, t2 = points[i]
        dt = t2 - t1

        if dt < 0:
            negative_gaps += 1
            continue
        elif dt == 0:
            zero_gaps += 1
            continue

        dist = haversine(lon1, lat1, lon2, lat2)
        spd = dist / dt

        distances.append(dist)
        time_gaps.append(dt)
        speeds.append(spd)

    total_dist = sum(distances)
    duration = points[-1][2] - points[0][2] if len(points) >= 2 else 0.0

    return {
        "gps_points": len(points),
        "total_distance_km": total_dist / 1000.0,
        "duration_seconds": duration,
        "avg_time_gap_seconds": statistics.mean(time_gaps) if time_gaps else 0.0,
        "median_time_gap_seconds": statistics.median(time_gaps) if time_gaps else 0.0,
        "min_time_gap_seconds": min(time_gaps) if time_gaps else 0.0,
        "max_time_gap_seconds": max(time_gaps) if time_gaps else 0.0,
        "avg_speed_kmh": (total_dist / duration) * 3.6 if duration > 0 else 0.0,
        "max_segment_speed_kmh": max(speeds) * 3.6 if speeds else 0.0,
        "segments_over_55m_s": sum(s > 55.0 for s in speeds),
        "zero_time_gaps": zero_gaps,
        "negative_time_gaps": negative_gaps
    }


def run_trajectory_stats():
    results = []

    for partition in PARTITIONS:
        folder = DATASET / partition
        if not folder.exists():
            continue

        for trip_dir in sorted(folder.iterdir()):
            if not trip_dir.is_dir():
                continue

            trip_id = trip_dir.name
            track_file = trip_dir / f"{trip_id}.track"
            if not track_file.exists():
                continue

            stats = analyze_track(track_file)
            results.append({
                "partition": partition,
                "trip_id": trip_id,
                **stats
            })

    # Save to both locations
    for target_dir in [OUTPUT_DIR, LEGACY_OUTPUT_DIR]:
        csv_path = target_dir / "trajectory_stats.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved: {csv_path.resolve()}")

    # Summary table
    summary_rows = []
    print("\n========== PARTITION SUMMARY ==========")
    for partition in PARTITIONS:
        subset = [r for r in results if r["partition"] == partition]
        if not subset:
            continue

        point_counts = [r["gps_points"] for r in subset]
        distances = [r["total_distance_km"] for r in subset]
        durations = [r["duration_seconds"] for r in subset]
        speeds = [r["avg_speed_kmh"] for r in subset]
        max_speeds = [r["max_segment_speed_kmh"] for r in subset]
        anom_segs = [r["segments_over_55m_s"] for r in subset]

        summary = {
            "partition": partition,
            "trip_count": len(subset),
            "total_gps_observations": sum(point_counts),
            "mean_gps_points": round(statistics.mean(point_counts), 2),
            "median_gps_points": round(statistics.median(point_counts), 2),
            "min_gps_points": min(point_counts),
            "max_gps_points": max(point_counts),
            "mean_distance_km": round(statistics.mean(distances), 2),
            "median_distance_km": round(statistics.median(distances), 2),
            "min_distance_km": round(min(distances), 2),
            "max_distance_km": round(max(distances), 2),
            "mean_duration_min": round(statistics.mean(durations) / 60.0, 2),
            "median_duration_min": round(statistics.median(durations) / 60.0, 2),
            "mean_avg_speed_kmh": round(statistics.mean(speeds), 2),
            "median_avg_speed_kmh": round(statistics.median(speeds), 2),
            "max_segment_speed_kmh": round(max(max_speeds), 2),
            "total_segments_over_55m_s": sum(anom_segs),
            "trips_with_segments_over_55m_s": sum(s > 0 for s in anom_segs)
        }
        summary_rows.append(summary)

        print(f"\n{partition.upper()}:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    # Save summary table
    for target_dir in [OUTPUT_DIR, LEGACY_OUTPUT_DIR]:
        sum_path = target_dir / "partition_summary.csv"
        with open(sum_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved: {sum_path.resolve()}")


if __name__ == "__main__":
    run_trajectory_stats()

