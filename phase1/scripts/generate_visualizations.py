"""
PathSnap Hackathon - Phase 1: Comprehensive Scientific Visualizations
Generates all 11 required EDA and anomaly figures plus the interactive Folium HTML map.
Strictly separates train and test partitions.
Supports execution from project root or script directory.
"""

from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import folium

# Dynamic root detection
CURRENT_FILE = Path(__file__).resolve()
if (CURRENT_FILE.parents[1] / "Dataset").exists():
    PROJECT_ROOT = CURRENT_FILE.parents[1]
elif (CURRENT_FILE.parents[2] / "Dataset").exists():
    PROJECT_ROOT = CURRENT_FILE.parents[2]
else:
    PROJECT_ROOT = Path(r"D:\ML_Hackathon\ML_Hackathon")

DATASET = PROJECT_ROOT / "Dataset"
CSV_DIR = PROJECT_ROOT / "phase1" / "results" / "csv"
PLOTS_DIR = PROJECT_ROOT / "phase1" / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial', 'DejaVu Sans', 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


EARTH_RADIUS = 6_371_000


def haversine_m(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS * math.asin(math.sqrt(min(1.0, a)))


def load_csv(name):
    path = CSV_DIR / name
    if not path.exists():
        path = PROJECT_ROOT / "phase1_results" / name
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# 1. GPS Point Count Distribution
def plot_01_gps_point_count(traj_data):
    train_pts = [int(r["gps_points"]) for r in traj_data if r["partition"] == "train"]
    test_pts = [int(r["gps_points"]) for r in traj_data if r["partition"] == "student_test"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(train_pts, bins=30, alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=0.5, label=f'Train (n={len(train_pts)}, mean={np.mean(train_pts):.0f})')
    ax.hist(test_pts, bins=10, alpha=0.7, color='#ff7f0e', edgecolor='black', linewidth=0.5, label=f'Student Test (n={len(test_pts)}, mean={np.mean(test_pts):.0f})')
    ax.axvline(x=np.median(train_pts), color='#1f77b4', linestyle='--', linewidth=1.5, label=f'Train Median ({np.median(train_pts):.0f})')
    ax.set_title('1. GPS Point Count Distribution per Trip', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of GPS Observations', fontsize=11)
    ax.set_ylabel('Number of Trips', fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    p = PLOTS_DIR / "01_gps_point_count_distribution.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}")


# 2. Trip Distance Distribution
def plot_02_trip_distance(traj_data):
    train_dist = [float(r["total_distance_km"]) for r in traj_data if r["partition"] == "train"]
    test_dist = [float(r["total_distance_km"]) for r in traj_data if r["partition"] == "student_test"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(train_dist, bins=25, alpha=0.7, color='#2ca02c', edgecolor='black', linewidth=0.5, label=f'Train (mean={np.mean(train_dist):.1f} km)')
    ax.hist(test_dist, bins=10, alpha=0.7, color='#d62728', edgecolor='black', linewidth=0.5, label=f'Student Test (mean={np.mean(test_dist):.1f} km)')
    ax.axvline(x=np.median(train_dist), color='green', linestyle='--', linewidth=1.5, label=f'Train Median: {np.median(train_dist):.1f} km')
    ax.set_title('2. Trajectory Distance Distribution (km)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Total Trajectory Distance (km)', fontsize=11)
    ax.set_ylabel('Number of Trips', fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    p = PLOTS_DIR / "02_trip_distance_distribution.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}")


# 3. Trip Duration Distribution
def plot_03_trip_duration(traj_data):
    train_dur = [float(r["duration_seconds"]) / 60.0 for r in traj_data if r["partition"] == "train"]
    test_dur = [float(r["duration_seconds"]) / 60.0 for r in traj_data if r["partition"] == "student_test"]

    fig, ax = plt.subplots(figsize=(8, 5))
    # Log scale bins to capture trips with overnight/weekend pauses
    bins = np.logspace(np.log10(min(train_dur)), np.log10(max(train_dur)), 30)
    ax.hist(train_dur, bins=bins, alpha=0.7, color='#9467bd', edgecolor='black', linewidth=0.5, label='Train')
    ax.hist(test_dur, bins=bins, alpha=0.7, color='#8c564b', edgecolor='black', linewidth=0.5, label='Student Test')
    ax.set_xscale('log')
    ax.axvline(x=np.median(train_dur), color='purple', linestyle='--', label=f'Train Median: {np.median(train_dur):.1f} min')
    ax.axvline(x=60, color='gray', linestyle=':', label='1 Hour')
    ax.axvline(x=1440, color='red', linestyle=':', label='24 Hours')
    ax.set_title('3. Trajectory Duration Distribution (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Duration (minutes, log scale)', fontsize=11)
    ax.set_ylabel('Number of Trips', fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    p = PLOTS_DIR / "03_trip_duration_distribution.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}")


# 4. Sampling Interval Distribution
def plot_04_sampling_intervals(gps_data):
    one_sec = [float(r["one_second_interval_pct"]) for r in gps_data if r.get("one_second_interval_pct")]
    avg_gaps = [float(r["avg_time_gap_sec"]) for r in gps_data if r.get("avg_time_gap_sec")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Prevalence of exactly 1.0s steps
    axes[0].hist(one_sec, bins=20, color='#17becf', edgecolor='black', linewidth=0.5, alpha=0.75)
    axes[0].axvline(x=np.median(one_sec), color='red', linestyle='--', linewidth=1.5, label=f'Median: {np.median(one_sec):.2f}%')
    axes[0].set_title('4a. Exact 1.00s Interval Prevalence per Trip (%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Percentage of Steps Exactly 1.00s Apart (%)', fontsize=11)
    axes[0].set_ylabel('Number of Trips', fontsize=11)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Average time gap
    axes[1].hist(avg_gaps, bins=25, color='#bcbd22', edgecolor='black', linewidth=0.5, alpha=0.75)
    axes[1].axvline(x=1.0, color='black', linestyle=':', label='Nominal 1.00s')
    axes[1].axvline(x=np.median(avg_gaps), color='blue', linestyle='--', label=f'Median: {np.median(avg_gaps):.2f}s')
    axes[1].set_title('4b. Mean Sampling Interval per Trip (s)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Mean Interval (seconds)', fontsize=11)
    axes[1].set_ylabel('Number of Trips', fontsize=11)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    p = PLOTS_DIR / "04_sampling_interval_distribution.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}")


# 5. Speed Distribution
def plot_05_speed_distribution(traj_data):
    train_spd = [float(r["avg_speed_kmh"]) for r in traj_data if r["partition"] == "train"]
    test_spd = [float(r["avg_speed_kmh"]) for r in traj_data if r["partition"] == "student_test"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(train_spd, bins=25, alpha=0.7, color='#34495e', edgecolor='black', linewidth=0.5, label=f'Train (mean={np.mean(train_spd):.1f} km/h)')
    ax.hist(test_spd, bins=10, alpha=0.7, color='#e67e22', edgecolor='black', linewidth=0.5, label=f'Student Test (mean={np.mean(test_spd):.1f} km/h)')
    ax.axvline(x=np.median(train_spd), color='navy', linestyle='--', label=f'Train Median: {np.median(train_spd):.1f} km/h')
    ax.set_title('5. Average Trip Speed Distribution (km/h)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Average Trip Speed (km/h)', fontsize=11)
    ax.set_ylabel('Number of Trips', fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    p = PLOTS_DIR / "05_speed_distribution.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}")


# 6. Max Segment Speed Distribution
def plot_06_max_segment_speed(traj_data):
    train_max = [float(r["max_segment_speed_kmh"]) for r in traj_data if r["partition"] == "train"]
    test_max = [float(r["max_segment_speed_kmh"]) for r in traj_data if r["partition"] == "student_test"]
    cutoff_kmh = 55.0 * 3.6  # 198 km/h

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(train_max, bins=25, alpha=0.7, color='#c0392b', edgecolor='black', linewidth=0.5, label='Train')
    ax.hist(test_max, bins=10, alpha=0.7, color='#2980b9', edgecolor='black', linewidth=0.5, label='Student Test')
    ax.axvline(x=cutoff_kmh, color='red', linestyle='--', linewidth=2, label=f'55 m/s Threshold ({cutoff_kmh:.0f} km/h)')
    ax.set_title('6. Maximum Segment Speed per Trip (km/h)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Maximum Segment Speed (km/h)', fontsize=11)
    ax.set_ylabel('Number of Trips', fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    p = PLOTS_DIR / "06_max_segment_speed_distribution.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}")


# 7. Road Network Size Distribution
def plot_07_network_size(net_data):
    train_nodes = [int(r["nodes"]) for r in net_data if r["partition"] == "train"]
    train_arcs = [int(r["arcs"]) for r in net_data if r["partition"] == "train"]
    test_nodes = [int(r["nodes"]) for r in net_data if r["partition"] == "student_test"]
    test_arcs = [int(r["arcs"]) for r in net_data if r["partition"] == "student_test"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(train_nodes, bins=25, alpha=0.7, color='#27ae60', edgecolor='black', linewidth=0.5, label=f'Train (mean={np.mean(train_nodes):.0f})')
    axes[0].hist(test_nodes, bins=10, alpha=0.7, color='#e67e22', edgecolor='black', linewidth=0.5, label=f'Test (mean={np.mean(test_nodes):.0f})')
    axes[0].set_title('7a. Road Network Nodes Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Nodes', fontsize=11)
    axes[0].set_ylabel('Number of Trips', fontsize=11)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].hist(train_arcs, bins=25, alpha=0.7, color='#2980b9', edgecolor='black', linewidth=0.5, label=f'Train (mean={np.mean(train_arcs):.0f})')
    axes[1].hist(test_arcs, bins=10, alpha=0.7, color='#e67e22', edgecolor='black', linewidth=0.5, label=f'Test (mean={np.mean(test_arcs):.0f})')
    axes[1].set_title('7b. Road Network Arcs Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Directed Arcs', fontsize=11)
    axes[1].set_ylabel('Number of Trips', fontsize=11)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    p = PLOTS_DIR / "07_road_network_size_distribution.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}")


# 8. Network Node/Arc Relationship
def plot_08_node_arc_relationship(net_data):
    nodes = [int(r["nodes"]) for r in net_data]
    arcs = [int(r["arcs"]) for r in net_data]
    parts = [r["partition"] for r in net_data]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4' if p == 'train' else '#ff7f0e' for p in parts]
    ax.scatter(nodes, arcs, c=colors, alpha=0.7, edgecolors='black', linewidth=0.5, s=45)

    # Reference slope lines
    x_vals = np.linspace(0, max(nodes), 100)
    ax.plot(x_vals, 2.0 * x_vals, color='red', linestyle='--', linewidth=1.5, label='Slope = 2.0 (|E| = 2|V|)')

    ax.set_title('8. Road Network: Nodes vs Arcs Relationship', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Nodes (|V|)', fontsize=11)
    ax.set_ylabel('Number of Directed Arcs (|E|)', fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    p = PLOTS_DIR / "08_network_node_arc_relationship.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}")


# 9. Sample Spatial Trajectories
def plot_09_spatial_trajectories():
    sample_trips = ["00000003", "00000005", "00000079"]
    titles = ["Trip 00000003 (Clean / Baseline)", "Trip 00000005 (Highway Drive)", "Trip 00000079 (Speed Spike Outlier)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, trip_id, title in zip(axes, sample_trips, titles):
        track_path = DATASET / "train" / trip_id / f"{trip_id}.track"
        if not track_path.exists():
            continue

        lons, lats = [], []
        with open(track_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    lons.append(float(parts[0]))
                    lats.append(float(parts[1]))

        ax.plot(lons, lats, color='#2980b9', linewidth=1.2, alpha=0.8, label='GPS Track')
        ax.scatter([lons[0]], [lats[0]], color='green', s=50, label='Start', zorder=5)
        ax.scatter([lons[-1]], [lats[-1]], color='red', s=50, label='End', zorder=5)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude (°)', fontsize=10)
        ax.set_ylabel('Latitude (°)', fontsize=10)
        ax.legend(loc='best', frameon=True, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    plt.tight_layout()
    p = PLOTS_DIR / "09_sample_spatial_trajectories.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}")


# 10. Selected Anomaly Visualizations
def plot_10_anomaly_deepdive():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Trip 00000079 multipath jump and return
    track_79 = DATASET / "train" / "00000079" / "00000079.track"
    pts_79 = []
    with open(track_79, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                pts_79.append((float(parts[0]), float(parts[1]), float(parts[2])))

    # Find the exact max speed segment dynamically
    max_spd_idx = 90
    max_spd_val = 0
    for i in range(1, len(pts_79)):
        d = haversine_m(pts_79[i-1][0], pts_79[i-1][1], pts_79[i][0], pts_79[i][1])
        dt = pts_79[i][2] - pts_79[i-1][2]
        if dt > 0 and (d / dt * 3.6) > max_spd_val:
            max_spd_val = d / dt * 3.6
            max_spd_idx = i

    start_idx = max(0, max_spd_idx - 4)
    end_idx = min(len(pts_79), max_spd_idx + 5)
    sub_79 = pts_79[start_idx:end_idx]
    sub_lons = [p[0] for p in sub_79]
    sub_lats = [p[1] for p in sub_79]

    axes[0].plot(sub_lons, sub_lats, color='gray', linestyle=':', marker='o', alpha=0.6)
    p_pre = pts_79[max_spd_idx - 1]
    p_spike = pts_79[max_spd_idx]
    p_post = pts_79[min(len(pts_79)-1, max_spd_idx + 1)]

    axes[0].scatter([p_pre[0]], [p_pre[1]], color='blue', s=80, label=f'Pt {max_spd_idx-1} (t={p_pre[2]:.0f}s)', zorder=6)
    axes[0].scatter([p_spike[0]], [p_spike[1]], color='red', s=120, label=f'Pt {max_spd_idx} SPIKE ({max_spd_val:.1f} km/h)', zorder=6)
    axes[0].scatter([p_post[0]], [p_post[1]], color='green', s=80, label=f'Pt {max_spd_idx+1} (t={p_post[2]:.0f}s)', zorder=6)
    axes[0].annotate(f'180m Jump in 1s\n(648.5 km/h)', xy=(p_spike[0], p_spike[1]), xytext=(p_spike[0]+0.0003, p_spike[1]-0.0003),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    axes[0].set_title('10a. Trip 00000079: Instantaneous Multipath Jump & Return', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Longitude (°)', fontsize=10)
    axes[0].set_ylabel('Latitude (°)', fontsize=10)
    axes[0].legend(frameon=True, fontsize=9)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Panel B: Trip 00000090 Time vs Elapsed Distance showing 42.5-hour plateau
    track_90 = DATASET / "train" / "00000090" / "00000090.track"
    pts_90 = []
    with open(track_90, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                pts_90.append((float(parts[0]), float(parts[1]), float(parts[2])))

    # Find the max gap index dynamically
    max_gap_idx = 535
    max_gap_val = 0
    for i in range(1, len(pts_90)):
        dt = pts_90[i][2] - pts_90[i-1][2]
        if dt > max_gap_val:
            max_gap_val = dt
            max_gap_idx = i

    times_h = [(p[2] - pts_90[0][2]) / 3600.0 for p in pts_90]
    axes[1].plot(range(len(pts_90)), times_h, color='#8e44ad', linewidth=2)
    gap_y = times_h[max_gap_idx]
    axes[1].axhline(y=gap_y, color='red', linestyle='--', alpha=0.7)
    axes[1].annotate(f'{max_gap_val/3600:.1f}-Hour Plateau (Weekend Stop)\nBetween Pt {max_gap_idx-1} and {max_gap_idx}', 
                     xy=(max_gap_idx, gap_y), xytext=(len(pts_90)*0.2, gap_y*0.6),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    axes[1].set_title(f'10b. Trip 00000090: {max_gap_val/3600:.1f}-Hour Pause Between Driving Sessions', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('GPS Point Index', fontsize=10)
    axes[1].set_ylabel('Elapsed Time (Hours)', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    p = PLOTS_DIR / "10_selected_anomaly_deepdive.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}")


# 11. Road Network / GPS Overlay Visualization
def plot_11_road_network_gps_overlay(trip_id="00000003"):
    track_path = DATASET / "train" / trip_id / f"{trip_id}.track"
    nodes_path = DATASET / "train" / trip_id / f"{trip_id}.nodes"
    arcs_path = DATASET / "train" / trip_id / f"{trip_id}.arcs"
    route_path = DATASET / "train" / trip_id / f"{trip_id}.route"

    # Load nodes
    nodes = []
    with open(nodes_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                nodes.append((float(parts[0]), float(parts[1])))

    # Load arcs
    arcs = []
    with open(arcs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                arcs.append((int(float(parts[0])), int(float(parts[1]))))

    # Load route
    route_arcs = []
    with open(route_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                route_arcs.append(int(float(parts[0])))

    # Load track
    track_lons, track_lats = [], []
    with open(track_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                track_lons.append(float(parts[0]))
                track_lats.append(float(parts[1]))

    fig, ax = plt.subplots(figsize=(9, 7))

    # Subsample nearby arcs for clean plotting
    min_lon, max_lon = min(track_lons) - 0.01, max(track_lons) + 0.01
    min_lat, max_lat = min(track_lats) - 0.01, max(track_lats) + 0.01

    # Plot road network edges
    for u, v in arcs:
        if 0 <= u < len(nodes) and 0 <= v < len(nodes):
            x1, y1 = nodes[u]
            x2, y2 = nodes[v]
            if (min_lon <= x1 <= max_lon and min_lat <= y1 <= max_lat) or (min_lon <= x2 <= max_lon and min_lat <= y2 <= max_lat):
                ax.plot([x1, x2], [y1, y2], color='#bdc3c7', linewidth=0.8, zorder=1)

    # Plot ground truth route
    for arc_id in route_arcs:
        if 0 <= arc_id < len(arcs):
            u, v = arcs[arc_id]
            ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]], color='#2980b9', linewidth=2.5, zorder=3)

    # Plot GPS points
    ax.scatter(track_lons, track_lats, color='#e74c3c', s=12, alpha=0.85, label='Raw GPS Points', zorder=4)

    # Dummy lines for clean legend
    ax.plot([], [], color='#bdc3c7', linewidth=1.5, label='Road Network Arcs')
    ax.plot([], [], color='#2980b9', linewidth=2.5, label='Ground Truth Route')

    ax.set_title(f'11. Trip {trip_id}: Road Network Arcs, Ground Truth Route & GPS Overlay', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude (°)', fontsize=11)
    ax.set_ylabel('Latitude (°)', fontsize=11)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.legend(frameon=True, loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    p = PLOTS_DIR / "11_road_network_gps_overlay.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}")


# Interactive Folium Map
def generate_interactive_folium_map(trip_id="00000003", tiles="CartoDB positron"):
    track_path = DATASET / "train" / trip_id / f"{trip_id}.track"
    nodes_path = DATASET / "train" / trip_id / f"{trip_id}.nodes"
    arcs_path = DATASET / "train" / trip_id / f"{trip_id}.arcs"
    route_path = DATASET / "train" / trip_id / f"{trip_id}.route"

    track_pts = []
    with open(track_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                track_pts.append((float(parts[1]), float(parts[0]), float(parts[2])))

    nodes = []
    with open(nodes_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                nodes.append((float(parts[1]), float(parts[0])))

    arcs = []
    with open(arcs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                arcs.append((int(float(parts[0])), int(float(parts[1]))))

    route_arc_ids = []
    with open(route_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                route_arc_ids.append(int(float(parts[0])))

    center_lat = np.mean([p[0] for p in track_pts])
    center_lon = np.mean([p[1] for p in track_pts])

    # Configurable basemap with compliant CartoDB Positron default and tile-free fallback
    if tiles and str(tiles).lower() in ("cartodb positron", "cartodbpositron", "default"):
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name="CartoDB Positron (Clean Default)",
            subdomains="abcd",
            max_zoom=19,
            control=True,
            overlay=False,
        ).add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri",
            name="Esri WorldStreetMap",
            max_zoom=19,
            control=True,
            overlay=False,
        ).add_to(m)
        folium.TileLayer(
            tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            name="OpenStreetMap",
            max_zoom=19,
            control=True,
            overlay=False,
        ).add_to(m)
    elif tiles and str(tiles).lower() != "none":
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=tiles)
    else:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)

    # Nearby network arcs
    subnetwork_group = folium.FeatureGroup(name="Road Network Arcs", show=False)
    for u, v in arcs[:1200]:
        if 0 <= u < len(nodes) and 0 <= v < len(nodes):
            folium.PolyLine([nodes[u], nodes[v]], color="#888888", weight=1.5, opacity=0.6).add_to(subnetwork_group)
    subnetwork_group.add_to(m)

    # Reference route
    route_group = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
    route_coords = []
    for arc_id in route_arc_ids:
        if 0 <= arc_id < len(arcs):
            u, v = arcs[arc_id]
            route_coords.append(nodes[u])
            route_coords.append(nodes[v])

    if route_coords:
        folium.PolyLine(route_coords, color="#2980b9", weight=4, opacity=0.85, tooltip=f"Ground Truth Route: {len(route_arc_ids)} arcs").add_to(route_group)
    route_group.add_to(m)

    # GPS points
    gps_group = folium.FeatureGroup(name="GPS Trajectory Observations", show=True)
    step = 2
    for i in range(0, len(track_pts), step):
        lat, lon, t = track_pts[i]
        folium.CircleMarker(
            location=[lat, lon], radius=3, color="#e74c3c", fill=True, fill_color="#e74c3c", fill_opacity=0.8,
            popup=f"GPS Pt #{i}<br>Time: {t}s<br>Lat: {lat:.5f}<br>Lon: {lon:.5f}"
        ).add_to(gps_group)

    folium.Marker([track_pts[0][0], track_pts[0][1]], popup="Origin", icon=folium.Icon(color="green", icon="play")).add_to(gps_group)
    folium.Marker([track_pts[-1][0], track_pts[-1][1]], popup="Destination", icon=folium.Icon(color="red", icon="stop")).add_to(gps_group)
    gps_group.add_to(m)

    folium.LayerControl().add_to(m)

    out_html = PLOTS_DIR / "interactive_map_sample.html"
    m.save(str(out_html))
    print(f"Saved: {out_html}")


def run_all_visualizations():
    print("Generating all 11 scientific Phase 1 visualizations & interactive map...\n")
    traj_data = load_csv("trajectory_stats.csv")
    gps_data = load_csv("gps_audit.csv")
    net_data = load_csv("network_summary.csv")

    plot_01_gps_point_count(traj_data)
    plot_02_trip_distance(traj_data)
    plot_03_trip_duration(traj_data)
    plot_04_sampling_intervals(gps_data)
    plot_05_speed_distribution(traj_data)
    plot_06_max_segment_speed(traj_data)
    plot_07_network_size(net_data)
    plot_08_node_arc_relationship(net_data)
    plot_09_spatial_trajectories()
    plot_10_anomaly_deepdive()
    plot_11_road_network_gps_overlay("00000003")
    generate_interactive_folium_map("00000003")
    print("\nAll 11 figures and interactive map generated successfully.")


if __name__ == "__main__":
    run_all_visualizations()
