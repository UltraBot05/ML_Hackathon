"""
PathSnap Hackathon - Master Phase 1 Pipeline Runner
Executes the complete Phase 1 audit, trajectory statistics, anomaly investigation,
road network analysis, visualization pipeline, and artifact validation end-to-end.
"""

import sys
import time
from pathlib import Path

# Add script folder to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from audit_dataset import run_audit
from trajectory_statistics import run_trajectory_stats
from investigate_anomalies import run_investigation
from network_analysis import run_network_analysis
from generate_visualizations import run_all_visualizations
from validate_and_manifest import validate_all_artifacts


def run_pipeline():
    start_total = time.time()
    print("=" * 80)
    print("PATHSNAP HACKATHON — PHASE 1 COMPLETE REPRODUCIBLE PIPELINE")
    print("=" * 80)

    stages = [
        ("Step 1: Dataset Structural Audit", run_audit),
        ("Step 2: Trajectory Statistics & Summaries", run_trajectory_stats),
        ("Step 3: Anomaly Deep-Dive Investigation & Classification", run_investigation),
        ("Step 4: Road Network Topology & Route Integrity Analysis", run_network_analysis),
        ("Step 5: High-Quality Scientific Visualizations (11 figures + Folium map)", run_all_visualizations),
        ("Step 6: Comprehensive Artifact Validation & Manifest Generation", validate_all_artifacts),
    ]

    for step_name, func in stages:
        t0 = time.time()
        print(f"\n>>> Running {step_name}...")
        func()
        print(f">>> Completed {step_name} in {time.time() - t0:.2f}s\n")

    print("=" * 80)
    print(f"ALL PHASE 1 TASKS COMPLETED & VALIDATED SUCCESSFULLY IN {time.time() - start_total:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
