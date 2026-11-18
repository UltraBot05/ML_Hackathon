# PathSnap Hackathon — Review 1 Technical Report
**Project:** PathSnap — Distributed Map-Matching & Route Reconstruction  
**Institution:** PES University | Hackathon Date: November 18, 2026  
**Track:** Machine Learning & Spatial Intelligence  
**Author / Teammate:** M1 Air Lead (Supervised Tree Ensembles & Evaluator UI)  

---

## 1. Executive Summary

Map-matching noisy GPS trajectories to dense, directed urban road networks is a foundational challenge in modern intelligent transportation systems. Raw GPS fixes suffer from multipath reflection, satellite clock drift, urban canyon signal shadowing, and sampling frequency jitter. Furthermore, complex road topologies (parallel service roads, multi-lane divided carriageways, overpasses, and high-density intersections) render naive Euclidean proximity matching severely error-prone.

This report documents the end-to-end engineering and scientific contributions delivered for **Review 1**:
1. **Dataset Ingestion, Integrity & Anomaly Audit:** 100% SHA-256 raw dataset verification across 90 trips (80 train, 10 student_test), isolating 981 GPS velocity spikes ($> 55\text{ m/s}$) with zero data leakage.
2. **Spatial Indexing & Candidate Retrieval:** Metric 2D KD-Tree over sampled road arcs achieving **100% candidate coverage** ($R_1=60\text{m}, R_2=120\text{m}$) across all 208,417 training GPS points in 22.5 seconds ($> 9{,}200\text{ points/sec}$).
3. **Discriminative 22-Feature Engineering Pipeline:** Extraction of geometric proximity, velocity-attenuated heading alignment, topological node degrees, and network transition ratios stored in Apache Parquet format.
4. **Empirical Baseline Benchmarking:** Implementation and evaluation of three distinct matching paradigms (Nearest Road, Direction-Aware, and Graph-Constrained HMM Viterbi).
5. **Phase 3 Supervised Machine Learning Execution:** Training and validation of **Random Forest Classifier** and **CatBoost Classifier** on the canonical 22-feature tables using strict trip-level partitioning (64 train / 16 val trips).
6. **Distributed Evaluator UI Architecture:** Streamlit and Folium interactive frontend designed to interface with Omen server via ZeroTier mesh networking (`http://10.92.151.151:8000`) with robust local fallback.

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
- **Spatial Distribution Plots:**
  - `phase1/results/plots/01_gps_point_count_distribution.png`: Log-normal distribution of trajectory lengths.
  - `phase1/results/plots/05_speed_distribution.png`: Bimodal velocity distribution (urban stop-and-go vs arterial cruising).
  - `phase1/results/plots/08_network_node_arc_relationship.png`: Strong linear correlation between vertex count and directed edge density.
  - `phase2/results/plots/candidate_generation_inspection.png`: Spatial KD-Tree candidate search radii ($60\text{m}$ / $120\text{m}$) and candidate selection.

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

## 5. Model Selection Rationale

Why Random Forest and CatBoost represent the optimal pairing for urban trajectory map matching:

1. **Random Forest (Bagging Ensemble):**
   - **Variance Reduction:** GPS noise is stochastic and high-variance. Bagging averages over 200 deep decision trees (`max_depth=14`), dampening random multipath coordinate errors.
   - **Feature Decorrelation:** Random subspace selection ($m = \sqrt{22} \approx 4$ features per split) prevents the dominant `distance_to_arc_m` from monopolizing tree roots, forcing trees to evaluate heading agreement and network transition feasibility.
   - **Parallel M1 CPU Scaling:** Embarrassingly parallel training utilizing all 8 ARM cores efficiently.

2. **CatBoost (Ordered Gradient Boosting):**
   - **Ordered Boosting:** Eliminates prediction shift and target leakage by computing gradients on permutations where the target instance was excluded from prior trees.
   - **Symmetric (Oblivious) Trees:** Constant depth across all leaves acts as structural regularization against overfitting on complex intersection topologies. Oblivious trees map to bitwise index operations, achieving microsecond inference latency on Apple Silicon.
   - **Superior Regularization:** Explicit $L_2$ leaf regularization (`l2_leaf_reg=3.0`) prevents extreme log-odds predictions on ambiguous parallel roads.

---

## 6. Empirical Results & Comprehensive Benchmark

### 6.1 Baseline Matchers vs. Supervised Classifiers
All models and baselines were evaluated on the **exact same 16 validation trips** (`phase2/features/val/`, 266,367 candidate rows, 39,460 observations, 39,166 covered observations) using identical ground-truth alignment targets.

| Paradigm | Model / Matcher | Top-1 Accuracy (%) | Coverage-Aware Top-1 (%) | Macro F1 | Binary Accuracy (%) | Log Loss | ROC-AUC | Obs Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **Baseline A** | Nearest Road Segment | 59.76% | 59.76% | 0.5881 | — | — | — | **0.0053 ms** |
| **Baseline B** | Direction-Aware Matcher | 76.89% | 76.89% | 0.7600 | — | — | — | 0.0113 ms |
| **Baseline C** | Graph-Constrained HMM | 83.22% | 83.22% | 0.8220 | — | — | — | 0.9890 ms |
| **Supervised ML** | **Random Forest Classifier** | **72.93%** | **73.48%** | **0.8264** | **89.82%** | **0.23928** | **0.9580** | **0.0228 ms** |
| **Supervised ML** | **CatBoost Classifier** | **73.64%** | **74.20%** | **0.8353** | **92.17%** | **0.17804** | **0.9604** | **0.0005 ms** |

### 6.2 Key Scientific Insights & Diagnostic Analysis

1. **CatBoost Outperforms Random Forest across All Metrics:**
   - CatBoost achieves higher Top-1 Accuracy (**73.64%** vs 72.93%), higher Candidate Macro F1 (**0.8353** vs 0.8264), higher ROC-AUC (**0.9604** vs 0.9580), and significantly lower Log Loss (**0.17804** vs 0.23928).
   - CatBoost trains in **9.00 seconds** (vs 146.57s for RF) and delivers **0.0005 ms/observation** inference latency (45x faster than RF), proving the power of symmetric oblivious trees compiled on Apple Silicon M1.

2. **Feature Importance Hierarchy:**
   - Both models identify `projection_t`, `dist_diff_from_best_m`, and `log_distance` as the top three most influential predictive features.
   - For CatBoost, `projection_t` contributes **25.14%** and `dist_diff_from_best_m` contributes **21.51%** of total split gain. This demonstrates that road position relative to intersection junctions and distance margin against the nearest candidate are far more discriminative than raw absolute distance.
   - Kinematic alignment (`speed_heading_alignment`, `arc_bearing_deg`) accounts for $> 15\%$ of feature importance, successfully filtering wrong-way candidates on divided roads.

3. **Classification vs. Sequence Trellis Trade-off:**
   - Supervised classifiers evaluate each observation independently, achieving high candidate-level discriminative power (Macro F1 $> 0.835$).
   - Baseline C utilizes full trajectory Viterbi sequence decoding across the graph, which smooths out point-level ambiguities over time.
   - **Recommended Next Step for Phase 3 Final Submission:** Hybrid ML-HMM emission model, feeding CatBoost candidate probabilities $\hat{P}(c \mid P_t)$ directly into the HMM Viterbi trellis!

---

## 7. Limitations of Derived Reference Labels

1. **Discrete Route vs. Continuous Dynamics:** Ground truth `.route` files specify discrete arc sequences. GPS points recorded during lane transitions or complex intersections may fall equidistant between two valid arcs.
2. **Monotonic Window Alignment Boundary Jitter:** Sequential trajectory-to-route alignment uses dynamic sliding windows. While achieving $> 99.8\%$ candidate recall, extreme traffic jams (velocity $\approx 0\text{ m/s}$) can introduce localized temporal boundary blur.
3. **Noisy Ground Truth Segments:** A small fraction of ground-truth routes contain geometric micro-arcs ($< 5\text{m}$) that GPS points physically bypass. Models penalize these as false negatives even when the macroscopic route is correct.

---

## 8. Summary of Review 1 Deliverables
- **Codebase Integrity:** 100% clean commit history on branch `team/mac-ui-models`.
- **Pre-existing Code Respected:** Zero recreation of Phase 1/2 preprocessing or candidate generators.
- **Model Artifacts Saved:** `phase3/models/random_forest_classifier.joblib` and `phase3/models/catboost_classifier.cbm`.
- **Streamlit Evaluator UI:** Fully built at `evaluator_ui/app.py` with Folium map integration and ZeroTier API connectivity.
- **Viva Preparation:** Complete theoretical guide authored at `docs/MAC_VIVA.md`.
