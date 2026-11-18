# PathSnap: Complete Project Pipeline & Six-Model Benchmark Guide
**Demo Presentation & Technical Defense Reference**  
**Institution:** PES University | PathSnap Hackathon (November 18, 2026)  
**Authors:** Adarsh, Omen Lead, Mac Lead (team/mac-ui-models)  

---

## 1. High-Level Problem & Project Objective

- **Problem:** Raw GPS telemetry recorded from vehicles suffers from severe physical noise:
  - Satellite clock drift & ionospheric refraction
  - Multipath reflections in urban canyons (signals bouncing off glass/concrete buildings)
  - Low-speed drift and jitter when stationary at traffic signals
  - Tunnel blackouts and sampling gaps
- **Objective 1 (Map-Matching):** Given an ordered sequence of noisy GPS fixes $P = (p_1, p_2, \dots, p_T)$, identify the exact directed road segment arc $e_t$ in the road network graph $G = (V, E)$ that the vehicle was physically traversing.
- **Objective 2 (Route Reconstruction):** Reconstruct the complete, unbroken, topologically contiguous physical driving path $R = [e_1, e_2, \dots, e_M]$ such that consecutive arcs connect continuously without non-physical graph disconnects ($v(e_i) == u(e_{i+1})$).

---

## 2. End-to-End System Pipeline Architecture

```
+-------------------------------------------------------------------------------------------------+
|                                 PATHSNAP END-TO-END PIPELINE                                    |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|  [STAGE 1: RAW INGESTION & FORENSIC AUDIT]                                                      |
|  • 90 Trips (80 Train / 10 Student Test held-out), 360 text files verified 100% SHA-256.        |
|  • 208,417 Train GPS points. Flagged 981 velocity spikes (> 55 m/s) without dropping points.   |
|  • Identified low-speed heading noise (< 1.5 m/s) and formulated speed-attenuated alignment.    |
|                                     │                                                           |
|                                     ▼                                                           |
|  [STAGE 2: SPATIAL INDEXING & CANDIDATE RETRIEVAL]                                              |
|  • 2D Metric KD-Tree over densified road arc coordinates (dual radius: R1=60m, R2=120m).        |
|  • 100% Candidate Coverage across all 208,417 training points in 22.5s (> 9,200 points/sec).   |
|  • Generates 7–10 candidate road segments per GPS fix (266k candidates across 16 val trips).    |
|                                     │                                                           |
|                                     ▼                                                           |
|  [STAGE 3: 22-FEATURE TABULAR EXTRACTION]                                                       |
|  • Geometric Proximity (6): distance_to_arc, projection_t, candidate_rank, dist_diff_best, etc. |
|  • Kinematics & Heading (7): gps_speed, heading_diff, cos_heading_diff, speed_heading_align, ... |
|  • Road Topology (4): arc_length_m, from_node_out_degree, to_node_in_degree, is_terminal_arc.   |
|  • Sequential Transitions (5): step_gps_dist, transition_dist, transition_dist_ratio, etc.     |
|  • Stored in Parquet (float32). Zero leakage: NO coordinates, NO IDs, NO timestamps in X.      |
|                                     │                                                           |
|                                     ▼                                                           |
|  [STAGE 4: SUPERVISED ML CANDIDATE CLASSIFICATION]                                              |
|  • Strict Trip Split: 64 Train Trips (1,143,501 rows) | 16 Val Trips (266,367 rows, 39,460 obs).|
|  • Evaluated 6 Classifiers: Logistic Reg, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost|
|  • Pointwise directed Top-1: 70% – 75.4% | Candidate Macro F1: 0.78 – 0.844.                     |
|                                     │                                                           |
|                                     ▼                                                           |
|  [STAGE 5: SEQUENTIAL TRELLIS DECODING (HMM VITERBI)]                                           |
|  • ML candidate probabilities feed into an HMM Viterbi trellis as emission potentials.         |
|  • Road network topology provides transition probabilities (exponential distance prior beta=40).|
|  • Dijkstra shortest-path bridging resolves physical gaps: Trip Top-1 leaps to 86.26%, 99.15% cont.|
|                                     │                                                           |
|                                     ▼                                                           |
|  [STAGE 6: FINAL 24-FEATURE INTEGRATED PIPELINE]                                                |
|  • 24 Features + LightGBM + HMM Viterbi + Route-Boundary Realignment.                          |
|  • Reproduces 97.7036% Top-1 on student_test and 97.5145% on Lecturer evaluation set!          |
|                                     │                                                           |
|                                     ▼                                                           |
|  [STAGE 7: DISTRIBUTED EVALUATOR UI & INFERENCE API]                                            |
|  • Omen FastAPI backend on ZeroTier private mesh (10.92.151.151:8000/8002).                     |
|  • Streamlit frontend on Mac M1: dynamically renders GeoJSON polylines in Folium ([lon, lat]    |
|    -> [lat, lon] conversion, auto-bounding, real-time Top-1, Route F1, continuity & latency).  |
+-------------------------------------------------------------------------------------------------+
```

---

## 3. How We Ran the 6 Models: Setup, Data Contract & Parameters

### The Evaluation Protocol (Zero Leakage)
- **Data Partitioning:** Strictly grouped by `trip_id`. 64 trips for training ($1{,}143{,}501$ candidate rows) and 16 trips for validation ($266{,}367$ candidate rows, $39{,}460$ GPS observations).
- **Why Trip-Grouped?** Random row splitting allows adjacent GPS seconds ($t$ and $t+1$) to leak between train and test, producing fake $>98\%$ point accuracy that fails on unseen trips. Trip splitting tests true geographical generalization.
- **Features Used ($X$):** Exactly the canonical 22 features. Identifiers (`trip_id`, `obs_index`, `candidate_arc_id`, `true_arc_id`), coordinates (`lat`, `lon`), and timestamps are strictly excluded.
- **Target ($y$):** `is_correct` (binary indicator, $\approx 12.5\%\text{--}14.7\%$ positive rows due to candidate set size).

### Model Configurations & Hyperparameters

1. **Logistic Regression ($L_2$ Regularized):**
   - *Pipeline:* `StandardScaler()` + `LogisticRegression(C=1.0, solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=42)`
   - *Fit Time:* **1.75s** | *Inference Latency:* **0.0003 ms/obs**
   - *Behavior:* Learns linear log-odds boundary. Highest weights assigned to `speed_heading_alignment` (+1.48) and `dist_diff_from_best_m` (-1.19). Well-calibrated argmax margins.

2. **Random Forest (Bootstrap Bagging Ensemble):**
   - *Parameters:* `n_estimators=200`, `max_depth=14`, `max_features='sqrt'` ($m=\lfloor\sqrt{22}\rfloor=4$), `class_weight='balanced'`, `n_jobs=-1`
   - *Fit Time:* **146.57s** | *Inference Latency:* **0.0228 ms/obs**
   - *Behavior:* Fits 200 deep CART trees on bootstrap samples. Subspace randomization prevents distance features from monopolizing root splits, forcing trees to evaluate heading alignment and topology.

3. **Extra Trees (Extremely Randomized Trees):**
   - *Parameters:* `n_estimators=100`, `max_depth=16`, `min_samples_leaf=20`, `max_features='sqrt'`, `bootstrap=False`, `n_jobs=-1`, `random_state=42`
   - *Fit Time:* **27.65s** | *Inference Latency:* **0.0123 ms/obs**
   - *Behavior:* Randomizes both feature subsets and split cut-points uniformly at random. Outperforms standard Random Forest on collinear spatial data by reducing variance and feature correlation.

4. **XGBoost (Histogram Gradient Boosting):**
   - *Parameters:* `objective='binary:logistic'`, `tree_method='hist'`, `n_estimators=150`, `max_depth=5`, `learning_rate=0.08`, `subsample=0.85`, `colsample_bytree=0.85`, `random_state=42`
   - *Fit Time:* **8.40s** | *Inference Latency:* **0.0039 ms/obs**
   - *Behavior:* Gradient-boosted shallow decision trees using histogram binning for fast split finding.

5. **LightGBM (Leaf-Wise Histogram Boosting):**
   - *Parameters:* `n_estimators=150`, `max_depth=8`, `num_leaves=31`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_samples=20`, `random_state=42`
   - *Fit Time:* **4.72s** | *Inference Latency:* **0.0009 ms/obs**
   - *Behavior:* Fast leaf-wise tree growth with gradient-based one-side sampling (GOSS). Fastest training speed among tree models.

6. **CatBoost (Symmetric Oblivious Decision Trees):**
   - *Parameters:* `depth=6` ($2^6=64$ leaves), `learning_rate=0.05`, `iterations=400`, `l2_leaf_reg=3.0`, `loss_function='Logloss'`, `random_seed=42`
   - *Fit Time:* **9.00s** | *Inference Latency:* **0.0005 ms/obs**
   - *Behavior:* Oblivious symmetric trees evaluate identical split conditions across all nodes at each depth level. Compiles to bitwise operations on CPU, yielding the lowest log loss (0.1780) and highest ROC-AUC (0.9604).

---

## 4. Specific Model Accuracies & Authoritative Leaderboard

Evaluated on the exact 16 validation trips ($39{,}460$ total observations, $39{,}166$ covered observations, $266{,}367$ candidate rows):

| Model | Candidate Accuracy (%) | Candidate Macro F1 | Log Loss | ROC-AUC | Conditional Top-1 (%) | Coverage-Aware Top-1 (%) | Correct Observations | Training Time (s) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees** | **92.53%** | **0.8440** | **0.1744** | 0.9594 | **75.41%** | **74.85%** | **29,537** | 27.65s | 0.0123 ms/obs |
| **Logistic Regression** | 86.27% | 0.7859 | 0.2924 | 0.9460 | **74.94%** | **74.38%** | **29,352** | **1.75s** | **0.0003 ms/obs** |
| **CatBoost** | 92.17% | 0.8353 | 0.1780 | **0.9604** | **74.20%** | **73.64%** | **29,060** | 9.00s | **0.0005 ms/obs** |
| **Random Forest** | 89.82% | 0.8264 | 0.2393 | 0.9580 | **73.48%** | **72.93%** | **28,778** | 146.57s | 0.0228 ms/obs |
| **XGBoost** | 91.11% | 0.8189 | 0.2024 | 0.9542 | **70.11%** | **69.59%** | **27,459** | 8.40s | 0.0039 ms/obs |
| **LightGBM** | 91.01% | 0.8166 | 0.2209 | 0.9518 | **69.14%** | **68.62%** | **27,079** | 4.72s | 0.0009 ms/obs |

### Baseline Comparison (Heuristic Matchers)

| Paradigm | Model / Matcher | Conditional Top-1 (%) | Coverage-Aware Top-1 (%) | Candidate Macro F1 | Obs Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline A** | Nearest Road Segment | 59.76% | 59.32% | 0.5881 | 0.0053 ms |
| **Baseline B** | Direction-Aware Matcher | 76.89% | 76.32% | 0.7600 | 0.0113 ms |
| **Baseline C** | Graph-Constrained HMM | 83.22% | 82.60% | 0.8220 | 0.9890 ms |
| **Supervised Point ML** | **Extra Trees Classifier** | **75.41%** | **74.85%** | **0.8440** | 0.0123 ms |
| **Sequential Hybrid** | **LightGBM + HMM Viterbi** | **86.26% (Macro)** | — | **0.8911 (Route)** | 0.6500 ms |
| **Final Integration** | **24-feat LightGBM+HMM+Realignment** | **97.70% (student_test)** | **97.51% (Lecturer)** | **>0.95** | 0.8200 ms |

---

## 5. Critical Technical Nuances for Demo & Viva Defense

### A. Conditional Top-1 vs. Coverage-Aware Top-1
- **Total Validation Observations ($N_{\text{total}}$):** $39,460$
- **Covered Observations ($N_{\text{covered}}$):** $39,166$ (observations having at least 1 positive candidate retrieved by KD-Tree)
- **Candidate Misses ($N_{\text{uncovered}}$):** $294$ ($0.75\%$)
$$\text{Conditional Top-1} = \frac{N_{\text{correct}}}{39,166} \quad\ge\quad \text{Coverage-Aware Top-1} = \frac{N_{\text{correct}}}{39,460}$$
- **Resolution:** Our earlier report accidentally swapped the metric headers. The higher score ($73.48\%\text{--}75.41\%$) is the Conditional Top-1; the lower score ($72.93\%\text{--}74.85\%$) is Coverage-Aware Top-1.

### B. Demystifying the "96%–97%" Claim (Why Pointwise ML is 70%–75%)
When other teams claim 96%–97%, they are conflating three different metrics:
1. **Candidate Binary Accuracy (86%–93%):** Because candidate generation yields 7–10 arcs per point, ~85.3% of candidate rows are negative. A model predicting all zeros achieves 85.3% accuracy! A score of 92.5% is candidate row accuracy, not route Top-1.
2. **Clean Trip Selection Bias (95%–100%):** On clean suburban/highway trips, raw ML Top-1 is indeed 95%–100% (Trip `00000024` is 97.75%–100%, Trip `00000005` is 97.70%). However, massive noisy trips like Trip `00000084` (16,031 observations, 40.6% of the validation set) have raw pointwise accuracy of ~47% due to urban canyon multipath reflections, pulling down the full-split average to 70%–75%.
3. **Reconstructed Route Continuity (98%–100%):** After HMM Viterbi decoding and Dijkstra shortest-path bridging, the connected route continuity is 98%–100%. Teams presenting route continuity as "accuracy" report 96%–99%.

### C. The Leap from Pointwise ML to Sequential HMM Decoding
- **Pointwise ML (70%–75%):** Evaluates each observation independently. Jitter in a single noisy GPS fix can cause the argmax to jump to an adjacent parallel road.
- **HMM Viterbi Trellis (86%–97%+):** Combines ML emission probabilities with graph transition probabilities:
  $$\delta_t(j) = \max_i \left[\delta_{t-1}(i) + \ln P_{\text{trans}}(i \to j)\right] + \ln P_{\text{emission}}(p_t \mid j)$$
  If a candidate requires an illegal U-turn or highway backtrack, the transition penalty ($d_{\text{network}} \gg d_{\text{gps}}$) overrules local noise, guaranteeing a continuous, physically valid driving path.

### D. Independent Route Alignment Audit
- Built `phase2/scripts/independent_route_alignment.py` to test whether proxy candidate labels biased evaluation.
- Projecting GPS fixes directly onto the ground-truth `.route` arc geometry across 28,964 observations confirmed:
  - **95.6% label agreement** between canonical labels and independent geometry.
  - **>99.1% true candidate recall**, proving candidate generation captures the true arc $>99\%$ of the time.
  - Reversed arcs occur in less than **0.03%** of observations.

---

## 6. How to Demo the Live Streamlit UI

1. **Start the UI:**
   ```bash
   .venv/bin/streamlit run evaluator_ui/app.py --server.headless true --server.port 8501
   ```
2. **Key UI Controls to Showcase:**
   - **Host & Port:** Shows ZeroTier server address (`10.92.151.151:8000 / 8002`) with live latency probe (~38ms).
   - **Trip Selector:** Switch between clean trips (e.g., `00000024`, `00000005`) and challenging urban trips (`00000084`, `00000016`).
   - **Interactive Folium Map:**
     - Raw GPS Observations (Blue dots with index & velocity popups).
     - Ground Truth Route (Green polyline).
     - Model Predicted Route (Red/Orange polyline converted from GeoJSON `[lon, lat]` to Folium `[lat, lon]`).
     - Bounding box auto-fit and Folium LayerControls.
   - **Metrics Panel:** Displays live Directed Top-1, Route F1, Route Continuity, and Latency.
   - **6-Model Benchmark Tab:** Presents the authoritative 22-feature comparison table.
   - **API Debugger Tab:** Demonstrates live `POST /predict` queries and raw JSON responses.
