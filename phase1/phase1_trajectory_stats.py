
from pathlib import Path
import csv
import math
import statistics

DATASET = Path(r"D:\ML_Hackathon\ML_Hackathon\Dataset")
OUT = Path("phase1_results")
OUT.mkdir(exist_ok=True)

PARTITIONS = ["train", "student_test"]
EARTH_RADIUS = 6_371_000  # metres


def haversine(lon1, lat1, lon2, lat2):
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS * math.asin(math.sqrt(min(1, a)))


def analyze_track(path):
    points = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            values = line.split()

            if len(values) < 3:
                continue

            lon, lat, timestamp = map(float, values[:3])
            points.append((lon, lat, timestamp))

    distances = []
    time_gaps = []
    speeds = []

    for i in range(1, len(points)):
        lon1, lat1, t1 = points[i - 1]
        lon2, lat2, t2 = points[i]

        dt = t2 - t1

        if dt <= 0:
            continue

        distance = haversine(lon1, lat1, lon2, lat2)
        speed = distance / dt

        distances.append(distance)
        time_gaps.append(dt)
        speeds.append(speed)

    total_distance = sum(distances)
    duration = (
        points[-1][2] - points[0][2]
        if len(points) >= 2 else 0
    )

    return {
        "gps_points": len(points),
        "total_distance_km": total_distance / 1000,
        "duration_seconds": duration,
        "avg_time_gap_seconds": (
            statistics.mean(time_gaps) if time_gaps else 0
        ),
        "median_time_gap_seconds": (
            statistics.median(time_gaps) if time_gaps else 0
        ),
        "max_time_gap_seconds": (
            max(time_gaps) if time_gaps else 0
        ),
        "avg_speed_kmh": (
            (total_distance / duration) * 3.6
            if duration > 0 else 0
        ),
        "max_segment_speed_kmh": (
            max(speeds) * 3.6 if speeds else 0
        ),
        "segments_over_55m_s": sum(s > 55 for s in speeds),
    }


results = []

for partition in PARTITIONS:
    folder = DATASET / partition

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

        print(
            f"{partition:12} {trip_id} | "
            f"{stats['gps_points']:6} points | "
            f"{stats['total_distance_km']:8.2f} km | "
            f"{stats['avg_speed_kmh']:6.2f} km/h"
        )


# Export individual trip statistics
csv_path = OUT / "trajectory_stats.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved: {csv_path.resolve()}")


# Correct aggregate statistics
print("\n========== PARTITION SUMMARY ==========")

for partition in PARTITIONS:
    subset = [r for r in results if r["partition"] == partition]

    if not subset:
        continue

    point_counts = [r["gps_points"] for r in subset]
    distances = [r["total_distance_km"] for r in subset]
    durations = [r["duration_seconds"] for r in subset]
    speeds = [r["avg_speed_kmh"] for r in subset]

    print(f"\n{partition.upper()}")
    print(f"Trips: {len(subset)}")
    print(f"Total GPS points: {sum(point_counts)}")
    print(f"Mean GPS points/trip: {statistics.mean(point_counts):.2f}")
    print(f"Median GPS points/trip: {statistics.median(point_counts):.2f}")
    print(f"Min GPS points/trip: {min(point_counts)}")
    print(f"Max GPS points/trip: {max(point_counts)}")
    print(f"Mean distance: {statistics.mean(distances):.2f} km")
    print(f"Median distance: {statistics.median(distances):.2f} km")
    print(f"Mean duration: {statistics.mean(durations)/60:.2f} min")
    print(f"Mean trip average speed: {statistics.mean(speeds):.2f} km/h")
