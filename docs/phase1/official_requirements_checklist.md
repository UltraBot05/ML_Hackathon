# PathSnap Hackathon — Phase 1 Official Requirements Checklist

This document maps all Phase 1 requirements from the official hackathon documents to their implementation evidence, compliance status, and actions.

---

## Official Documents Reference

- **[DE]**: *Dataset Exploration.pdf* (Supporting Document: Dataset Description and Exploration)
- **[DEL]**: *Deliverables.pdf* (Technical Report & Submission Guidelines)
- **[MG]**: *Main Guidelines.pdf* (Challenge Overview & Distributed Deployment)
- **[MTG]**: *Models and Training Guidelines.pdf* (Modeling Expectations & Algorithms)
- **[APP]**: *Assessment and Plagiarism Policy.pdf* (Evaluation Criteria & Academic Integrity)

---

## Phase 1 Requirements Compliance Matrix

| Requirement | Source Document | Mandatory / Optional | Status | Evidence in Repository | Remaining Action |
|:---|:---:|:---:|:---:|:---|:---|
| **Dataset Inventory & Partition Audit**<br>Count trips (80 train, 10 student test), file counts (360 files), and byte sizes | [DE] §4, §7, §8;<br>[DEL] §2.2 | **Mandatory** | **COMPLETED** | `phase1/scripts/audit_dataset.py`, `phase1/results/csv/trip_audit.csv`, `file_inventory.csv` | None. 100% verified against disk. |
| **Verify File Format & Column Semantics**<br>`.track` (lon, lat, timestamp in s), `.nodes` (lon, lat), `.arcs` (from, to), `.route` (arc sequence) | [DE] §3 | **Mandatory** | **COMPLETED** | Verified in `audit_dataset.py` line 14–40; column semantics confirmed via [DE] §3. | None. Documented in reports. |
| **Coordinate Bounds & Geodesic Distance**<br>Check angular coordinates ($[-180, 180]$, $[-90, 90]$); compute geodesic distance (Haversine) | [DE] §5 | **Mandatory** | **COMPLETED** | `haversine()` implemented in `trajectory_statistics.py`; 0 invalid coords logged in `gps_audit.csv`. | None. Projected metrics noted for Phase 2. |
| **Trajectory Statistics**<br>Trajectory length, duration, sampling interval distributions, and average/segment speeds | [DE] §4, §8 | **Mandatory** | **COMPLETED** | `phase1/results/csv/trajectory_stats.csv`, `partition_summary.csv`. | None. Recalculated from actual files. |
| **Nominal 1-Hz Sampling Verification**<br>Verify nominal 1-Hz interval (~97% at 1.0s) and document irregular gaps | [DE] §3, §4 | **Mandatory** | **COMPLETED** | `gps_audit.csv` confirms ~97.0% exact 1.00s intervals; plotted in `sampling_intervals_analysis.png`. | None. |
| **Data Quality & Monotonicity Audit**<br>Audit malformed rows, duplicate timestamps, negative $\Delta t$, and non-monotonic sequences | [DE] §4 | **Mandatory** | **COMPLETED** | `gps_audit.csv` confirms 0 malformed rows, 0 duplicate timestamps, 0 non-monotonic intervals. | None. |
| **Road Network Size & Topology Audit**<br>Node counts, arc counts, degree distribution, isolated nodes, and connected components | [DE] §4, §8 | **Mandatory** | **COMPLETED** | `network_analysis.py`, `phase1/results/csv/network_summary.csv`. 0 isolated nodes confirmed. | None. |
| **Reference Route Integrity & Continuity**<br>Verify arc reference validity $[0, \|E\|-1]$ and consecutive node-to-node route continuity | [DE] §3, §6 | **Mandatory** | **COMPLETED** | `network_route_audit.csv` (0 invalid arcs); `network_summary.csv` (99.93% average continuity). | None. |
| **Investigate Anomalies in Training Data**<br>Deep-dive on velocity spikes (>55 m/s) and long temporal gaps (>300 s) on TRAIN trips only | [DE] §4, §8 | **Mandatory** | **COMPLETED** | `investigate_anomalies.py`, `anomaly_segments.csv`, `anomaly_investigation_report.md`. | Mitigations deferred to Phase 2. |
| **Strict Test Data Protection & Anti-Leakage**<br>Zero fitting, zero hyperparameter tuning, zero heuristic derivations using `student_test` | [DE] §7 | **Mandatory** | **COMPLETED** | `student_test` isolated; only descriptive structural audit performed. No data leakage. | Strictly maintain boundary in Phase 2–4. |
| **Non-Destructive Exploration**<br>Raw files must not be modified, filtered, or deleted | [DE] §1; [APP] §1 | **Mandatory** | **COMPLETED** | 100% of raw files in `Dataset/train/` and `Dataset/student_test/` untouched. | Preserve raw dataset intact. |
| **Interactive Folium Map**<br>Visualise GPS trajectory observations and road network on an interactive map | [DE] §8;<br>[DEL] §2.2 | **Mandatory** | **COMPLETED** | Generated `phase1/results/plots/interactive_map_sample.html` with GPS markers, arcs, and route overlay. | None. |
| **High-Quality EDA Visualizations**<br>Plots of point count, distance, duration, speeds, sampling intervals, and topology | [DE] §8;<br>[DEL] §2.2 | **Mandatory** | **COMPLETED** | 5 high-resolution PNG plots generated in `phase1/results/plots/`. | None. |
| **Consolidated Technical Documentation**<br>Formal technical report detailing structure, stats, network, anomalies, and limitations | [DEL] §2.2 | **Mandatory** | **COMPLETED** | `phase1_dataset_exploration_report.md`, `phase1_executive_summary.md`. | None. Ready for team review. |
| **Full Pipeline Reproducibility**<br>Single-command execution from project root | [DEL] §1 | **Mandatory** | **COMPLETED** | `python run_phase1.py` completes all 5 stages in ~42 seconds. | None. |

---

## Detailed Evaluation of Hackathon Restrictions

### 1. Permitted vs Prohibited Use of `student_test`
- **Permitted by Official Rules ([DE] §7):** "Development-stage evaluation and model comparison." Descriptive file-existence audits and observation counts are necessary to verify evaluation readiness.
- **Strictly Prohibited by Official Rules ([DE] §7):**
  - Observations from the same trip must not span across partitions.
  - Test data must NOT be used for model training.
  - Test labels must NOT be used for feature selection or threshold calibration.
  - Learned preprocessing transformations (e.g. scalers, spatial coordinate bounds) must be fitted using training data only.
- **Repository Audit Finding:** All anomaly investigations, velocity cutoff choices (55 m/s), moving speed calculations, and preprocessing plans were derived exclusively from the 80 `train` trips. No data leakage occurred.

### 2. Preprocessing Constraints for Phase 2
- Official documentation mandates that teams must document and justify all data-cleaning decisions.
- Automatic or blind deletion of GPS points is discouraged because ground truth routes may correspond to segments where GPS was noisy.
- All cleaning (Kalman filtering, session splitting) must be implemented transparently in Phase 2 source code.

