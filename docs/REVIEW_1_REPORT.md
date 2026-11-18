# PathSnap Hackathon — Review 1 Technical Report
**Project:** PathSnap — Distributed Map-Matching & Route Reconstruction  
**Institution:** PES University | Hackathon Date: November 18, 2026  
**Track:** Machine Learning & Spatial Intelligence  
**Author / Teammate:** M1 Air Lead (Supervised Tree Ensembles, Independent Scoring Audit & Evaluator UI)  
**Branch:** `team/mac-ui-models` | Commit: Canonical Phase 2/3 (`59f573e`)  

---

## 1. Executive Summary

Map-matching noisy GPS trajectories to dense, directed urban road networks is a foundational challenge in modern intelligent transportation systems. Raw GPS fixes suffer from multipath reflection, satellite clock drift, urban canyon signal shadowing, and sampling frequency jitter. Furthermore, complex road topologies (parallel service roads, multi-lane divided carriageways, overpasses, and high-density intersections) render naive Euclidean proximity matching severely error-prone.

This report documents the end-to-end engineering and scientific contributions delivered for **Review 1**:
1. **Dataset Ingestion, Integrity & Anomaly Audit:** 100% SHA-256 raw dataset verification across 90 trips (80 train, 10 student_test), isolating 981 GPS velocity spikes ($> 55\text{ m/s}$) with zero data leakage.
2. **Spatial Indexing & Candidate Retrieval:** Metric 2D KD-Tree over sampled road arcs achieving **100% candidate coverage** ($R_1=60\text{m}, R_2=120\text{m}$) across all 208,417 training GPS points in 22.5 seconds ($> 9{,}200\text{ points/sec}$).
3. **Discriminative 22-Feature Engineering Pipeline:** Extraction of geometric proximity, velocity-attenuated heading alignment, topological node degrees, and network transition ratios stored in Apache Parquet format.
4. **Independent Route-Label Alignment Audit:** Built an independent geometric alignment diagnostic reconciling GPS points against the ordered `.route` path, proving **95.6% label agreement** and **>98.9% true ground truth candidate retrieval recall**.
5. **Phase 3 Supervised Machine Learning Execution:** Training and validation of **Random Forest**, **CatBoost**, **Logistic Regression**, and **Extra Trees** locally on Apple Silicon M1, cross-benchmarking with **XGBoost** and **LightGBM** (via Omen and Adarsh) on an identical 64/16 trip split.
6. **Resolving Metric Discrepancies & Demystifying "96%–97%" Claims:** Reconciled mathematical definitions of Conditional vs. Coverage-Aware Top-1 and demonstrated why pointwise Top-1 across all validation trips genuinely sits at **70%–75%**, whereas route continuity reaches **98%–100%**.
7. **Distributed Evaluator UI Architecture:** Streamlit and Folium interactive frontend with robust tile fallbacks (`CartoDB positron`, `Esri`) interfacing with the Omen ZeroTier API (`http://10.92.151.151:8000`) and local model backends.

---

## 2. Dataset Overview & Forensic Data Audit

### 2.1 File System Structure & Entity Semantics
The competition dataset comprises 90 distinct trip folders split into two partitions:
- **`train/`:** 80 trips (permitted for training, feature extraction, parameter tuning, and validation).
- **`student_test/`:** 10 trips (strictly held-out and protected from training or tuning).

Each trip directory contains four synchronized raw text files:
1. **`.track`:** Time-series GPS observations `[longitude, latitude, timestamp_seconds]`.
2. **`.nodes`:** Spatial road vertices `[longitude, latitude]`, 0-indexed implicitly by line number.
3. **`.arcs`:** Directed road segments `[from_node_id, to_node_id]`, 0-indexed implicitly by line number.
4. **`.route`:** Chronological sequence of true traversed `arc_id` values (ground truth available only in `train/`).

### 2.2 Forensic Audit & Dataset Statistics

| Dimension | Training Partition (`train/`) | Test Partition (`student_test/`) | Combined Dataset |
|---|---|---|---|
| **Total Trips** | 80 trips | 10 trips | 90 trips |
| **Total GPS Observations** | 208,417 fixes | 19,160 fixes | 227,577 fixes |
| **Mean Points per Trip** | 2,605.2 points | 1,916.0 points | 2,528.6 points |
| **Median Points per Trip** | 1,529.0 points | 1,420.0 points | 1,518.0 points |
| **Min / Max Points** | 361 / 19,313 points | 262 / 3,958 points | 262 / 19,313 points |
| **Mean Trajectory Distance**| 37.23 km | 29.70 km | 36.39 km |
| **Min / Max Distance** | 6.33 km / 169.25 km | 5.55 km / 71.72 km | 5.55 km / 169.25 km |
| **Mean Duration** | 94.67 minutes | 33.75 minutes | 87.90 minutes |
| **Mean Average Speed** | 48.60 km/h | 53.53 km/h | 49.15 km/h |
| **SHA-256 Integrity Audit**| **100% Unaltered** | **100% Unaltered** | **360 Files Verified** |

### 2.3 Anomaly Taxonomy & Preprocessing Safeguards
Our exploratory data analysis identified several physical anomalies inherent in real-world telematics:
- **GPS Speed Spikes:** 981 trajectory segments exhibited instantaneous velocity $> 55\text{ m/s}$ ($198\text{ km/h}$), peaking at unphysical extremes caused by multi-path reflection or satellite reacquisition delays. These were flagged rather than deleted to avoid time-series discontinuity.
- **Low-Speed Heading Drift:** When vehicle speed drops below $1.5\text{ m/s}$ (traffic stops, signals), GPS receiver bearing drifts randomly. We introduced `speed_heading_alignment` to mathematically attenuate heading influence during stops.
- **Topological Discontinuities in Route Ground Truth:** 11 trips exhibited minor topological gaps where derived routes skipped intermediate micro-segments ($< 10\text{m}$). Our Dijkstra gap-bridging module resolves these with $> 98.9\%$ continuity.

---

## 3. Spatial Trajectory & Road Graph Visualization

### 3.1 Visual Evidence (Phase 1 & Phase 2 Artifacts)
- **Interactive Multi-Layer Map (Phase 1):** `phase1/results/plots/interactive_map_sample.html` (1.1 MB)  
  Renders raw GPS observation markers, directed road network arcs, and reference ground-truth route sequences using Leaflet/Folium.
- **Map-Matching Ground Truth Overlay (Phase 2):** `phase2/results/plots/interactive_map_matching.html` (196 KB)  
  Displays synchronized multi-layer comparison between raw GPS (Blue), Ground Truth (Green), Nearest Road (Orange), Direction-Aware (Purple), and Graph HMM (Red).
- **Tile Access Issue Resolution:** Configured tiles to use compliant HTTPS tile providers (`CartoDB positron`, `Esri WorldStreetMap`, and tile-free fallback) to eliminate HTTP 403 tile blocks. High-resolution screenshots generated:
  - `phase1/results/plots/map_sample_folium_screenshot.png` (1.1 MB)
  - `phase2/results/plots/map_matching_folium_screenshot.png` (971 KB)
  - `phase2/results/plots/map_matching_static_fallback.png` (517 KB)

---

## 4. 22-Feature Engineering Pipeline

To provide maximum discriminative power to supervised classifiers, Phase 2 constructed 22 features across four mathematical domains:

```
+---------------------------------------------------------------------------------------------------+
|                            22-FEATURE TABULAR INPUT REPRESENTATION                                |
+---------------------------------------------------------------------------------------------------+
|  [Domain 1: Geometric Proximity] (6 Features)                                                     |
|  • distance_to_arc_m       : Perpendicular metric point-to-segment Euclidean distance             |
|  • dist_diff_from_best_m   : Distance margin relative to nearest candidate (d_i - d_1)            |
|  • projection_t            : Along-arc linear projection fraction in [0.0, 1.0]                   |
|  • candidate_rank          : Distance-based ordinal position {1, ..., 10}                         |
|  • normalized_rank         : candidate_rank / 10.0                                                |
|  • log_distance            : ln(1.0 + distance_to_arc_m)                                          |
|                                                                                                   |
|  [Domain 2: Movement & Heading Alignment] (7 Features)                                            |
|  • gps_speed_mps           : Instantaneous vehicle speed (m/s)                                    |
|  • gps_bearing_deg         : Trajectory compass heading [0, 360)                                  |
|  • arc_bearing_deg         : Road segment orientation [0, 360)                                    |
|  • heading_diff_deg        : Minimal angular difference |theta_gps - theta_arc| in [0, 180]        |
|  • cos_heading_diff        : cos(Delta theta) in [-1.0, 1.0]                                      |
|  • speed_heading_alignment : cos(Delta theta) * min(1.0, v / 5.0)                                 |
|  • is_counter_flow         : Binary indicator (Delta theta > 120 deg and v > 3 m/s)               |
|                                                                                                   |
|  [Domain 3: Road Graph Topology] (4 Features)                                                     |
|  • arc_length_m            : Arc length in meters                                                 |
|  • from_node_out_degree    : Outgoing arcs from start vertex                                      |
|  • to_node_in_degree       : Incoming arcs to end vertex                                         |
|  • is_terminal_arc         : Single-predecessor single-successor continuation flag                 |
|                                                                                                   |
|  [Domain 4: Sequential Network Transitions] (5 Features)                                          |
|  • step_gps_dist_m         : Geodesic displacement over preceding time interval                  |
|  • transition_dist_m       : Shortest network path distance from candidate at t-1                 |
|  • transition_dist_ratio   : d_network / max(1.0, d_gps)                                          |
|  • is_directly_connected   : Direct topological adjacency flag                                    |
|  • log_transition_dist     : ln(1.0 + transition_dist_m)                                          |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Independent Route-Label Alignment Audit (Task B)

### 5.1 Motivation & Diagnostic Design
The Phase 2 label generator selected its ground-truth reference arc from among the generated candidates (proxy labeling). To independently verify whether this proxy introduced artificial inflation or missed ground-truth arcs, we authored an independent route alignment diagnostic (`phase2/scripts/independent_route_alignment.py`) that strictly projects GPS coordinates against the ground truth `.route` arc sequence *independently of candidate retrieval*.

### 5.2 Verification Findings across 8 Validation Trips (28,964 Observations)
Evaluating representative trips spanning clean urban corridors, complex roundabouts, and noisy long-haul journeys:

| Trip ID | Total Obs | Label Agreement | Indep. True Recall | Phase 2 Proxy Recall | Absent from Candidates | Opposite Dir Arcs | Rank 1 Hits |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000001` | 1,315 | 99.77% | 99.85% | 99.85% | 2 | 0 | 1,068 |
| `00000005` | 1,045 | 99.71% | 100.00% | 100.00% | 0 | 0 | 1,000 |
| `00000013` | 2,580 | 97.17% | 98.10% | 100.00% | 49 | 0 | 2,144 |
| `00000016` | 1,305 | 87.97% | 98.70% | 99.16% | 17 | 1 | 714 |
| `00000024` | 801 | 100.00% | 100.00% | 100.00% | 0 | 0 | 699 |
| `00000028` | 3,565 | 99.27% | 99.63% | 99.41% | 13 | 0 | 1,831 |
| `00000043` | 2,322 | 93.88% | 95.56% | 93.93% | 101 | 0 | 1,200 |
| `00000084` | 16,031 | 94.19% | 99.55% | 99.76% | 72 | 7 | 7,938 |
| **Combined**| **28,964** | **95.60%** | **99.10%** | **99.25%** | **254** | **8** | **16,594** |

### 5.3 Diagnostic Conclusions
1. **High Proxy Agreement (95.6%):** The canonical training labels match the true geometric `.route` alignment in 95.6% of observations. Disagreements occur almost exclusively at complex junctions/slip roads (e.g., Trip `00000016`).
2. **True Candidate Retrieval Recall is >99.1%:** Ground truth arcs are captured within the $R=120\text{m}$ candidate set $>99.1\%$ of the time. The 294 validation observations lacking a positive candidate across the entire 16 trips ($0.75\%$) represent true physical signal loss / off-network driving.
3. **Opposite-Direction Candidates are Rare (<0.03%):** Only 8 instances out of 28,964 observations involved reversed arcs, confirming the road graph's directional integrity.

---

## 6. Demystifying the "96%–97%" Claim (Task C)

In discussions, other teams have reported "96%–97% accuracy" using XGBoost or LightGBM. Our rigorous audit proves that **pointwise directed Top-1 accuracy across the entire 16-trip validation set is genuinely 70%–75%**, and explains the three reasons why other teams quote 96%–97%:

```
+---------------------------------------------------------------------------------------------------+
|                           FOUR DISTINCT EVALUATION METRICS IN MAP MATCHING                        |
+---------------------------------------------------------------------------------------------------+
| 1. Candidate Binary Accuracy (86% – 93%):                                                          |
|    Measures binary 0/1 correctness across all 266,367 candidate rows. Due to 85.3% negative       |
|    imbalance, a trivial "all zero" baseline achieves 85.3%. A 92.5% score is candidate accuracy, |
|    NOT route-matching Top-1 accuracy.                                                             |
|                                                                                                   |
| 2. Pointwise Directed Top-1 Accuracy (70% – 75% Full Split):                                      |
|    Strict argmax prediction per observation. Must predict the exact directed arc ID.             |
|    - Conditional Top-1: Correct / Covered Obs (N = 39,166) = 73.48% – 75.41%                      |
|    - Coverage-Aware Top-1: Correct / Total Obs (N = 39,460) = 72.93% – 74.85%                     |
|                                                                                                   |
| 3. Clean Trip Selection Bias (95% – 100% on specific trips):                                     |
|    On clean suburban/arterial trips, raw ML Top-1 is naturally 95%–100%:                          |
|    - Trip 00000024: 97.75% – 100.0%                                                              |
|    - Trip 00000005: 97.70%                                                                        |
|    - Trip 00000001: 95.58% – 97.26%                                                               |
|    However, massive noisy trips like Trip 00000084 (16,031 observations, 40.6% of the val set)   |
|    achieve ~47%–48% raw Top-1, pulling the weighted aggregate down to ~70%–75%.                   |
|                                                                                                   |
| 4. Post-Processed Route Continuity (98% – 100%):                                                  |
|    After HMM Viterbi trellis decoding and Dijkstra shortest-path bridging, the connected route  |
|    continuity is 98%–100%. Teams presenting route continuity as "accuracy" report 96%–99%.        |
+---------------------------------------------------------------------------------------------------+
```

---

## 7. Authoritative Six-Model Benchmark Leaderboard (Task D)

All models evaluated on the **exact same 16 validation trips** (`phase2/features/val/`, 266,367 candidate rows, 39,460 total observations, 39,166 covered observations) using the canonical 22 features (commit `59f573e`):

| Model | Model Family | Candidate Accuracy | Candidate Macro F1 | Log Loss | ROC-AUC | Conditional Top-1 | Coverage-Aware Top-1 | Train Time (s) | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear Model (L2 Regularized) | 86.27% | 0.7859 | 0.2924 | 0.9460 | **74.94%** | **74.38%** | 1.75s | 0.0003 ms/obs |
| **Random Forest** | Bootstrap Bagging Ensemble | 89.82% | 0.8264 | 0.2393 | 0.9580 | **73.48%** | **72.93%** | 146.57s | 0.0228 ms/obs |
| **Extra Trees** | Extremely Randomized Trees | 92.53% | 0.8440 | 0.1744 | 0.9594 | **75.41%** | **74.85%** | 27.65s | 0.0123 ms/obs |
| **XGBoost** | Gradient Boosted Decision Trees | 91.11% | 0.8189 | 0.2024 | 0.9542 | **70.11%** | **69.59%** | 8.40s | 0.0039 ms/obs |
| **LightGBM** | Histogram Gradient Boosting | 91.01% | 0.8166 | 0.2209 | 0.9518 | **69.14%** | **68.62%** | 4.72s | 0.0009 ms/obs |
| **CatBoost** | Symmetric Oblivious Decision Trees | 92.17% | 0.8353 | 0.1780 | 0.9604 | **74.20%** | **73.64%** | 9.00s | 0.0005 ms/obs |

### 7.1 Baseline Matchers vs. Supervised Classifiers

| Paradigm | Model / Matcher | Conditional Top-1 (%) | Coverage-Aware Top-1 (%) | Macro F1 | Obs Latency (ms) |
|---|---|---|---|---|---|
| **Baseline A** | Nearest Road Segment | 59.76% | 59.32% | 0.5881 | **0.0053 ms** |
| **Baseline B** | Direction-Aware Matcher | 76.89% | 76.32% | 0.7600 | 0.0113 ms |
| **Baseline C** | Graph-Constrained HMM | 83.22% | 82.60% | 0.8220 | 0.9890 ms |
| **Supervised ML (Best Single)**| **Extra Trees Classifier** | **75.41%** | **74.85%** | **0.8440** | 0.0123 ms |
| **Supervised ML (Best Boosting)**| **CatBoost Classifier** | **74.20%** | **73.64%** | **0.8353** | **0.0005 ms** |
| **Post-Processing (Hybrid)**| **LightGBM + HMM Viterbi** | **86.26% (Macro)** | — | **0.8911 (Route)** | 0.6500 ms |

### 7.2 Key Scientific Insights & Diagnostic Analysis

1. **Resolution of Top-1 Metric Swapping:**
   - Previous internal tables had swapped the labels for Conditional Top-1 ($N=39,166$) and Coverage-Aware Top-1 ($N=39,460$).
   - Mathematically, Conditional Top-1 is strictly greater than or equal to Coverage-Aware Top-1:
     $$\text{Conditional Top-1} = \frac{N_{\text{correct}}}{N_{\text{covered}}} \ge \frac{N_{\text{correct}}}{N_{\text{total}}} = \text{Coverage-Aware Top-1}$$
   - All historical metrics, numerators ($N_{\text{correct}}$), and denominators ($N_{\text{covered}}=39,166$, $N_{\text{total}}=39,460$) have been reconciled and verified.

2. **Extra Trees and CatBoost Lead Among Supervised Models:**
   - **Extra Trees** achieves the highest pointwise Top-1 accuracy (**75.41% conditional, 74.85% coverage-aware**) and highest Macro F1 (**0.8440**). The extreme randomization of split thresholds across features acts as an effective regularizer against correlated spatial features.
   - **CatBoost** delivers the best boosting performance: **74.20% conditional Top-1**, lowest log loss (**0.1780**), highest ROC-AUC (**0.9604**), and ultrafast training (**9.00s**) and inference (**0.0005 ms/obs**) on M1.

3. **Sequential Trellis Integration (The HMM Leap):**
   - Pointwise classifiers evaluate observations independently, meaning spatial noise at single fixes can cause prediction jitter.
   - When ML candidate probabilities are passed as emission potentials into an **HMM Viterbi Decoder with Dijkstra network bridging**, performance leaps to **86.26% trip Top-1**, **99.15% route continuity**, and **0.8911 Route F1**.

---

## 8. Distributed Evaluator UI Architecture (Task F)

The evaluation frontend is implemented in Streamlit (`evaluator_ui/app.py`):
- **Server Health Check:** Pings Omen ZeroTier API at `http://10.92.151.151:8000/health`.
- **Local Fallback:** If the remote API is unreachable, seamlessly switches to local saved models (`CatBoost`, `Random Forest`, `Extra Trees`, `Logistic Regression`).
- **Interactive Folium Map:** Displays raw GPS observations, top-k candidate arcs, predicted match, and ground truth route with custom color coding and popup metadata.
- **Trip Inspection:** Allows selecting any of the 16 validation trips (or training trips) for deep forensic inspection of edge cases.

---

## 9. Review 1 Checklist & Readiness

- [x] Canonical 22 features confirmed and frozen across all teammates (commit `59f573e`).
- [x] All 6 models trained and verified on identical split.
- [x] Metric swapping resolved with exact numerators/denominators.
- [x] Independent route-alignment diagnostic confirmed >99.1% true candidate recall.
- [x] Folium tile HTTP 403 access fixed with compliant HTTPS providers.
- [x] Streamlit Evaluator UI running and tested locally.
- [x] Review 1 Viva defense documentation complete (`docs/MAC_VIVA.md`).
