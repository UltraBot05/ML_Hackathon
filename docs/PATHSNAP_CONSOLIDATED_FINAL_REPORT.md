# PathSnap: Distributed Map-Matching & Route Reconstruction
**PES University PathSnap Hackathon** | November 18, 2026  
**Final Consolidated Engineering & Scientific Technical Report**

---

### Team Attribution & Contributions
- **Adarsh:** Phase 1/2 Data Engineering, Spatial KD-Tree Indexing, 22-Feature Engineering Pipeline, Extra Trees, LightGBM, Graph HMM Viterbi Decoder.
- **Teammate (Omen Lead):** Independent Pipeline Audit, Logistic Regression, XGBoost, Optuna Bayesian Tuning, Decoder Parameter Search, FastAPI Microservice, Final 24-Feature Integration.
- **M1 Air Lead (Mac Lead):** Random Forest, CatBoost, Six-Model Benchmark Consolidation, Independent Geometric Route Audit, Folium Visual Maps, Streamlit Evaluator UI, Viva Defense Guide.

---

## 1. Executive Summary & Problem Formulation
Map-matching telematics trajectories to directed urban road network graphs is a foundational requirement for modern intelligent transportation systems. Raw GPS fixes suffer from satellite clock drift, multipath reflections, and low-speed positional jitter.

**Milestone Achievement:** Our collaborative team architecture successfully reproduced **97.7036% directed Top-1 accuracy on student_test** and **97.5145% directed Top-1 on the Lecturer evaluation set** using our integrated 24-feature LightGBM + HMM Viterbi + route-boundary realignment pipeline.

---

## 2. Phase 1: Data Ingestion, Integrity & Forensic Anomaly Audit
- **Total Dataset:** 90 trips total (80 train, 10 student-test held-out), 360 text files verified 100% SHA-256 unaltered.
- **Training Scale:** 208,417 observations across 80 trips.
- **Speed Spikes Isolated:** 981 trajectory segments exceeding 55 m/s (198 km/h), peaking at 648.5 km/h on Trip 00000079. Flagged rather than deleted to avoid time-series discontinuity.
- **Low-Speed Heading Drift:** Attenuated via `speed_heading_alignment`.

---

## 3. Phase 2: Spatial Indexing & 22-Feature Engineering
Dual-radius 2D KD-Tree ($R_1 = 60m, R_2 = 120m$) achieving 100% candidate coverage in 22.5s (>9,200 points/sec).
- **Geometric Proximity (6 features):** `distance_to_arc_m`, `dist_diff_from_best_m`, `projection_t`, `candidate_rank`, `normalized_rank`, `log_distance`.
- **Kinematics & Heading (7 features):** `gps_speed_mps`, `gps_bearing_deg`, `arc_bearing_deg`, `heading_diff_deg`, `cos_heading_diff`, `speed_heading_alignment`, `is_counter_flow`.
- **Road Graph Topology (4 features):** `arc_length_m`, `from_node_out_degree`, `to_node_in_degree`, `is_terminal_arc`.
- **Sequential Transitions (5 features):** `step_gps_dist_m`, `transition_dist_m`, `transition_dist_ratio`, `is_directly_connected`, `log_transition_dist`.

---

## 4. Independent Route-Label Alignment Audit
Evaluating 28,964 validation observations across 8 diverse trips using `phase2/scripts/independent_route_alignment.py`:
- **Label Agreement:** **95.60%** overall agreement with independent ground truth geometry.
- **True Candidate Recall:** **99.10%** of ground-truth arcs captured in KD-Tree candidate sets.
- **Reversed Arcs:** Only 8 points out of 28,964 (<0.03%).

---

## 5. Authoritative Six-Model Benchmark (Canonical 22 Features)
Evaluated on the exact 16 validation trips (266,367 candidate rows, 39,460 observations, 39,166 covered observations):

| Model | Model Family | Candidate Acc | Macro F1 | Log Loss | ROC-AUC | Conditional Top-1 | Coverage-Aware Top-1 | Fit Time | Obs Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear ($L_2$ Regularized) | 86.27% | 0.7859 | 0.2924 | 0.9460 | 74.94% | 74.38% | 1.75s | 0.0003 ms |
| **Random Forest** | Bootstrap Bagging Ensemble | 89.82% | 0.8264 | 0.2393 | 0.9580 | 73.48% | 72.93% | 146.57s | 0.0228 ms |
| **Extra Trees** | Extremely Randomized Trees | 92.53% | **0.8440** | **0.1744** | 0.9594 | **75.41%** | **74.85%** | 27.65s | 0.0123 ms |
| **XGBoost** | Gradient Boosted Decision Trees | 91.11% | 0.8189 | 0.2024 | 0.9542 | 70.11% | 69.59% | 8.40s | 0.0039 ms |
| **LightGBM** | Histogram Gradient Boosting | 91.01% | 0.8166 | 0.2209 | 0.9518 | 69.14% | 68.62% | **4.72s** | 0.0009 ms |
| **CatBoost** | Symmetric Oblivious Trees | 92.17% | 0.8353 | 0.1780 | **0.9604** | 74.20% | 73.64% | 9.00s | **0.0005 ms** |

---

## 6. Optuna Tuning & Decoder Investigation (Omen Lead)
Bayesian optimization over gradient boosting hyperparameters and HMM sequence transition parameters:
- `beta = 40.0`, disconnection penalty `-50.0`, network cutoff `1500m`.
- Temperature scaling on classifier probabilities for calibrated emission potentials.

---

## 7. ML + HMM Viterbi Sequential Decoding (Adarsh & Omen)
Integrating LightGBM candidate emission probabilities with HMM Viterbi decoding and Dijkstra shortest-path bridging:
- Macro average trip Top-1 increases from **82.35%** to **86.26%** (+3.91%).
- Reconstructed route continuity reaches **99.15%**.
- Route F1 reaches **0.8911**.

---

## 8. Final 24-Feature Integrated Architecture (Target Results)
- **Configuration:** 24 features (including boundary realignment and transition delta features) + LightGBM + HMM Viterbi + Dijkstra network bridging + Route-Boundary Realignment.
- **Reproduced Scores:**
  - `student_test` Directed Top-1: **97.7036%**
  - Lecturer Evaluation Set Directed Top-1: **97.5145%**
  - Route Continuity: **>99.5%**
  *(Detailed final accuracy table reserved for coordinator sign-off).*

---

## 9. Distributed System Architecture & Evaluator UI (Mac Lead)
- **ZeroTier Mesh:** Private link to Omen server at `http://10.92.151.151:8000 / 8002` (~38ms latency).
- **Folium Rendering:** Converts GeoJSON `[lon, lat]` to `[lat, lon]`, auto-zooms to trajectory bounding box, displays raw GPS fixes (Blue), ground truth (Green), and model prediction (Red).
- **Zero Simulated Fallback:** Real HTTP communication during demo mode.
