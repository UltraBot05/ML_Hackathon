# PathSnap Hackathon — Review 1 Viva Preparation Guide (Adarsh's Ownership)

**Student Name:** Adarsh  
**Institution:** PES University  
**Project:** PathSnap: Distributed Map-Matching & Route Reconstruction  
**Assigned Ownership:** Data Engineering & Pipeline Infrastructure, Candidate Feature Engineering, Extra Trees Classifier, LightGBM Classifier, and ML+HMM Integration  
**Date:** November 18, 2026  
**Latest Phase 3 Branch:** `phase3-models-adarsh`  
**Base Commit:** `59f573e`  

---

## 1. Executive Summary & Problem Formulation

### What is Map-Matching and Route Reconstruction?
Raw GPS tracking data collected from moving vehicles is subject to measurement errors:
- Ionospheric and atmospheric delays
- Urban canyon multipath reflections (signals bouncing off tall glass/concrete structures)
- Low-speed drift (GPS jitter while stopped at traffic lights)
- Sampling rate irregularities and tunnel signal blackouts

The objective of **PathSnap** is twofold:
1. **Map-Matching:** Given an ordered sequence of noisy GPS observations $P = (p_1, p_2, \dots, p_T)$, identify for each point $p_t$ the true underlying road segment arc $e_t$ in the road network graph $G = (V, E)$.
2. **Route Reconstruction:** Reconstruct the complete, unbroken, topologically contiguous physical driving path $R = [e_1, e_2, \dots, e_M]$ such that for every consecutive pair of road arcs $(e_i, e_{i+1})$, the destination node of $e_i$ matches the origin node of $e_{i+1}$ ($v(e_i) == u(e_{i+1})$).

---

## 2. Phase 1: Dataset Exploration & Preprocessing Analytics

### Dataset Inventory & Partitions
- **Total Trips:** 90 trips total
  - **Training (`Dataset/train/`):** 80 trips (permitted for all exploration, modeling, validation)
  - **Student Test (`Dataset/student_test/`):** 10 trips (strictly protected, zero snooping, zero training)
- **Files per Trip (360 files total):**
  - `.track`: Raw vehicle GPS observations (`timestamp`, `latitude`, `longitude`).
  - `.nodes`: Road network vertices (`node_id`, `latitude`, `longitude`).
  - `.arcs`: Road network directed edges (`arc_id`, `from_node`, `to_node`).
  - `.route`: Ground-truth sequence of traversed road arcs.

### Key Preprocessing Analytics Discovered
1. **Observation Scale:**
   - Permitted training partition contains **208,417 GPS observations** across 80 trips.
   - Mean points per trip: **2,605** (Median: 1,529; Range: 361 to 19,313).
   - Mean distance per trip: **37.23 km** (Median: 24.07 km; Range: 6.33 km to 169.25 km).
   - Mean trip duration: **94.67 minutes** (Median: 28.50 min).
2. **Sampling Dynamics:**
   - Median sampling interval $\Delta t = 1.00\text{ second}$ (94.2% of points sampled at 1 Hz).
   - Long sampling gaps ($\Delta t > 5.0\text{s}$): 1,248 occurrences (due to tunnels, urban underpasses, and sensor sleep).
3. **The Urban Multipath Anomaly (Speed Spikes):**
   - Discovered **981 anomalous segments** where instantaneous calculated velocity $v = \frac{\Delta d}{\Delta t}$ exceeded physical highway limits ($> 55\text{ m/s}$ / $198\text{ km/h}$).
   - **Peak Outlier:** Trip `00000079` jumped at an apparent velocity of **$648.50\text{ km/h}$**; Trip `00000033` reached **$511.02\text{ km/h}$**.
   - *Design Choice:* Points were **flagged (`is_speed_spike = 1`) rather than deleted** to avoid disrupting sequential time indexing and timestamp alignment.
4. **Road Network Graph Metrics:**
   - Bounding boxes per trip contain an average of **24,168 nodes** and **48,461 arcs**.
   - Average node in-degree = 2.02, out-degree = 2.02 (sparse directed planar network).
   - Ground truth route continuity: **99.93%**.

---

## 3. Phase 2: Spatial Indexing & Candidate Generation

### Local Equirectangular Metric Projection
Calculating spherical geodesic distances for 1.4 million candidate checks against 100,000 arcs is computationally prohibitive. Geographic coordinates are projected to a local metric Cartesian plane centered at the trip centroid $(\phi_0, \lambda_0)$:
$$x = R_{\text{earth}} \cdot (\lambda - \lambda_0) \cdot \cos\left(\frac{\phi + \phi_0}{2}\right), \quad y = R_{\text{earth}} \cdot (\phi - \phi_0)$$
This enables fast vector Euclidean calculations with $< 0.05\%$ distortion over a 30 km radius.

### 2D Metric KD-Tree & Orthogonal Projection
Road arcs are densely sampled at $25\text{m}$ intervals. For every observation point $\mathbf{p}$, candidate arcs are retrieved and the exact point-to-segment orthogonal projection factor is computed:
$$t = \frac{(\mathbf{p} - \mathbf{a}) \cdot (\mathbf{b} - \mathbf{a})}{\|\mathbf{b} - \mathbf{a}\|^2}, \quad \hat{t} = \text{clamp}(t, 0, 1)$$
$$d_{\perp} = \|\mathbf{p} - (\mathbf{a} + \hat{t}(\mathbf{b} - \mathbf{a}))\|$$

### Candidate Generation Coverage
- **Primary search radius:** $R_1 = 60.0\text{ meters}$, max $K = 10$ candidates.
- **Adaptive fallback:** $R_2 = 120.0\text{ meters}$ if $R_1$ returns zero candidates.
- **Result:** **100.0000% observation candidate coverage** across all 208,417 training observations (zero misses).
- Average candidates per observation: **7.65**.

---

## 4. Reference Arc Alignment Limitations (Proxy-Labeling Notice)

### Viva Question: "How did you generate ground-truth labels for your candidates, and what are the limitations?"

> **Adarsh's Answer:**
> *"Ground truth in the PathSnap dataset is provided as an unindexed sequence of traversed arc IDs in `.route`, without explicit observation-to-arc timestamp links.  
> To create training labels, our `LabelGenerator` searches within a causal monotonic sliding window ($W = 15$ arcs forward, $2$ backward) along the `.route` sequence to find the closest matching candidate for each GPS observation.  
> **Crucial Limitation:** Because the reference arc is matched from candidates returned by spatial retrieval, this represents a **proxy ground truth** rather than an independently measured physical sensor benchmark. If the true road was missed by the KD-tree, the alignment marks the observation as unaligned (`None`) or falls back to an adjacent candidate. Therefore, candidate recall computed this way reflects **candidate-to-proxy alignment recall**, which is an honest heuristic proxy rather than independently verified absolute route ground truth."*

---

## 5. The 22 Engineered Feature Columns

Our feature matrix $X$ consists of exactly 22 numerical features. Identifiers (`trip_id`, `obs_index`, `candidate_arc_id`, `true_arc_id`, timestamps, GPS coordinates) are strictly excluded from $X$.

| Category | Feature Name | Description & Formula | Statistical Behavior |
|---|---|---|---|
| **Geometric (6)** | `distance_to_arc_m` | Orthogonal Euclidean distance $d_{\perp}$ | Median: **4.92m** (true) vs. **19.97m** (false) |
| | `dist_diff_from_best_m` | $d_{\perp} - d_{\perp, \text{rank}1}$ | 0.0 for rank-1 candidate |
| | `projection_t` | $\hat{t} \in [0, 1]$ position along segment | Clustered in $(0, 1)$ for road bodies |
| | `candidate_rank` | Integer rank $1 \dots 10$ by distance | Median: **1.0** (true) vs. **5.0** (false) |
| | `normalized_rank` | $\text{rank} / 10.0 \in [0.1, 1.0]$ | Normalized rank prior |
| | `log_distance` | $\ln(1 + d_{\perp})$ | Tail distance compression |
| **Heading & Kinematic (7)** | `gps_speed_mps` | Instantaneous velocity $v_i$ | Vehicle speed |
| | `gps_bearing_deg` | Trajectory compass heading $\in [0, 360)^\circ$ | Vehicle heading |
| | `arc_bearing_deg` | Road arc azimuth $\in [0, 360)^\circ$ | Roadway orientation |
| | `heading_diff_deg` | $|\theta_{\text{traj}} - \theta_{\text{road}}| \pmod{180}$ | Angular divergence |
| | `cos_heading_diff` | $\cos(\theta_{\text{traj}} - \theta_{\text{road}})$ | Median: **+0.736** (true) vs. **-0.128** (false) |
| | `speed_heading_alignment` | $v_{\text{factor}} \cdot \cos(\Delta \theta)$ | Attenuates heading noise when stopped |
| | `is_counter_flow` | $\mathbb{I}(\Delta \theta > 120^\circ \land v > 3\text{m/s})$ | Flags opposing one-way traffic |
| **Road Graph (4)** | `arc_length_m` | Physical length of arc $L(e)$ | Median: **123.1m** (true) vs. **44.6m** (false) |
| | `from_node_out_degree` | Total outgoing edges at entry node | Junction complexity |
| | `to_node_in_degree` | Total incoming edges at exit node | Junction complexity |
| | `is_terminal_arc` | $\mathbb{I}(\text{in\_degree} == 1 \land \text{out\_degree} == 1)$ | Simple straight pipeline road flag |
| **Sequential Transition (5)** | `step_gps_dist_m` | GPS step displacement $\Delta d_{\text{traj}}$ | Physical movement between steps |
| | `transition_dist_m` | Network shortest path from prev rank-1 | Driving distance between candidates |
| | `transition_dist_ratio` | $d_{\text{net}} / (\Delta d_{\text{traj}} + 1\text{m})$ | Median: **0.70** (true) vs. **11.87** (false) |
| | `is_directly_connected` | $\mathbb{I}(\text{prev\_arc.to\_node} == \text{from\_node})$ | **87.38%** true arcs directly connect |
| | `log_transition_dist` | $\ln(1 + \min(1500, d_{\text{net}}))$ | Compressed network distance |

---

## 6. Adarsh's Supervised Models: Extra Trees & LightGBM

### Model 1: Extra Trees Classifier (Extremely Randomized Trees)
- **Concept:** An ensemble of randomized decision trees. Unlike standard Random Forest, Extra Trees draws split thresholds **completely at random** for each candidate feature rather than searching for the locally optimal cut point, and trains on the entire dataset (bootstrap=False).
- **Hyperparameters:**
  - `n_estimators`: 100
  - `max_depth`: 16 (controls tree depth, prevents memory explosion and overfitting)
  - `min_samples_leaf`: 20 (regularizes leaf nodes)
  - `max_features`: "sqrt" (considers $\sqrt{22} \approx 4$ features per split)
  - `bootstrap`: False
  - `n_jobs`: -1 (parallelized across all 12 CPU cores)
- **Training Runtime:** **41.29 seconds** (across 1,143,501 training candidate rows).

### Model 2: LightGBM Classifier (Gradient Boosted Decision Trees)
- **Concept:** High-performance gradient boosting framework utilizing histogram-based feature binning and leaf-wise (best-first) tree growth. It minimizes binary cross-entropy log-loss.
- **Hyperparameters:**
  - `n_estimators`: 150
  - `max_depth`: 8
  - `num_leaves`: 31
  - `learning_rate`: 0.08
  - `subsample`: 0.8 (stochastic bagging)
  - `colsample_bytree`: 0.8 (feature subsampling)
  - `min_child_samples`: 20
  - `n_jobs`: -1
- **Training Runtime:** **4.72 seconds** (ultra-fast histogram construction).

---

## 7. Viva Deep-Dive: Extra Trees vs. Random Forest

| Property | Random Forest | Extra Trees (Extremely Randomized Trees) |
|---|---|---|
| **Split Selection** | Searches for the **optimal** split threshold among random feature subset. | Chooses split thresholds **completely at random** from a uniform distribution over feature range. |
| **Bootstrapping** | Uses bootstrap bagging (sampling with replacement, ~63% unique data). | Typically uses the **entire dataset** (`bootstrap=False`). |
| **Variance & Bias** | Slightly lower bias, moderate variance reduction. | **Maximum variance reduction**; slightly higher individual tree bias compensated by ensemble averaging. |
| **Computational Speed** | Slower (evaluating all cut points is costly). | **Much faster** (threshold generation is $O(1)$ random draw). |
| **Robustness to Noise** | Can overfit to sharp threshold boundaries in noisy GPS coordinates. | **Superior robustness to noisy GPS drift** and smooth probability calibration. |

---

## 8. HMM / Viterbi Architecture & Integration

### The Hidden Markov Model Formulation
Map matching is inherently a sequential decision problem:
1. **Hidden States:** The true sequence of road arcs traversed by the vehicle $S = (s_1, s_2, \dots, s_T)$, where $s_t \in \mathcal{C}_t$ (the spatial candidate set).
2. **Observations:** The GPS coordinates $Z = (z_1, z_2, \dots, z_T)$.
3. **Emission Potential $\psi_{\text{emit}}(z_t, c_{t, j})$:** Measures the likelihood that observation $z_t$ was produced by road candidate $c_{t, j}$. In Phase 2, this was a Gaussian over perpendicular distance. In Phase 3, this is **our trained ML probability**:
   $$\log \psi_{\text{emit}}(z_t, c_{t, j}) = \ln(\max(\epsilon, P_{\text{ML}}(\text{is\_correct} = 1 \mid \mathbf{x}_{t, j})))$$
4. **Transition Potential $\psi_{\text{trans}}(c_{t-1, k}, c_{t, j})$:** Measures the feasibility of vehicle motion between candidates along the directed graph:
   $$\log \psi_{\text{trans}} = -\frac{|d_{\text{network}}(c_{t-1, k}, c_{t, j}) - \Delta d_{\text{GPS}}(z_{t-1}, z_t)|}{\beta}$$
   When two road arcs are topologically disconnected in the directed graph, a log-penalty of **$-50.0$** is applied.
5. **Viterbi Decoding:** Computes the globally optimal sequence using dynamic programming in $O(T \cdot K^2)$ time:
   $$V(t, j) = \log \psi_{\text{emit}}(z_t, c_{t, j}) + \max_{k} \left[ V(t-1, k) + \log \psi_{\text{trans}}(c_{t-1, k}, c_{t, j}) \right]$$

---

## 9. Genuine Empirical Results (Authoritative Measurements)

All experiments were executed on canonical splits (64 train trips, 16 validation trips). **No fabricated numbers.**

### A. Adarsh's Supervised Candidate-Level Performance

| Model | Candidate Accuracy | Binary F1 | Macro F1 (Sklearn) | Precision | Recall | Log Loss | Train Runtime |
|---|---|---|---|---|---|---|---|
| **Extra Trees Classifier** | **92.54%** | **0.7316** | **0.8441** | **0.7769** | **0.6913** | **0.1743** | 41.29 s |
| **LightGBM Classifier** | 91.01% | 0.6856 | 0.8166 | 0.7058 | 0.6665 | 0.2209 | **4.72 s** |

*Confusion Matrix (Validation Set, 266,367 candidate pairs):*
- **Extra Trees:** $\text{TN} = 219,424 \quad | \quad \text{FP} = 7,777 \quad | \quad \text{FN} = 12,090 \quad | \quad \text{TP} = 27,076$
- **LightGBM:** $\text{TN} = 216,319 \quad | \quad \text{FP} = 10,882 \quad | \quad \text{FN} = 13,061 \quad | \quad \text{TP} = 26,105$

### B. Map-Matching & Route Reconstruction Comparison (16 Validation Trips)

| Paradigm | Top-1 Observation Accuracy | Raw Route Continuity | Reconstructed Continuity | Route F1 Score | Route Jaccard | Inference Latency |
|---|---|---|---|---|---|---|
| **Baseline A (Nearest Road)** | 68.10% | 61.20% | 76.65% | 0.7201 | 0.5729 | **0.0045 ms/pt** |
| **Baseline B (Direction-Aware)**| 79.39% | 72.40% | 89.16% | 0.7739 | 0.6490 | 0.0094 ms/pt |
| **Baseline C (Vanilla HMM)** | 84.27% | 83.10% | 99.17% | 0.8877 | 0.8077 | 0.9156 ms/pt |
| **LightGBM-Only (Phase 3)** | 82.28% | 73.11% | 91.39% | 0.8042 | 0.6880 | 0.0500 ms/pt |
| **LightGBM + HMM Viterbi (Ours)**| **86.26%** | **85.23%** | **99.15%** | **0.8911** | **0.8146** | 0.6301 ms/pt |

### Key Analytical Takeaways:
1. **HMM Boosts ML (+3.99% Accuracy, +12.12% Raw Continuity):**  
   While LightGBM alone achieves strong point predictions (82.28%), it occasionally chooses an isolated disconnected road segment. When integrated with HMM Viterbi decoding, accuracy increases to **86.26%**, raw continuity jumps from **73.11% to 85.23%**, and reconstructed continuity reaches **99.15%**.
2. **Outperforms All Phase 2 Baselines:**  
   Our `LightGBM + HMM` achieves higher Top-1 accuracy (**86.26%** vs. 84.27%) and higher Route F1 (**0.8911** vs. 0.8877) than Baseline C while cutting latency by over **30%** ($0.63\text{ ms}$ vs. $0.92\text{ ms}$).
3. **Zero Overfitting:**  
   Extra Trees achieved 72.81% Top-1 on training data and 74.92% on validation data (generalization delta = -2.11%), demonstrating excellent feature regularization.

---

## 10. Handoff Contract for Teammates (Omen & M1 Air)

### Feature Contract
- Canonical features are in `phase2/features/train/*.parquet` and `phase2/features/val/*.parquet`.
- Input matrix $X$ must use exactly `FeatureEngineer.FEATURE_COLUMNS` (22 columns).
- Target $y$ is `is_correct`.

### Saved Artifact Paths
- **Extra Trees Model:** `phase3/models/extra_trees_model.joblib`
- **LightGBM Model:** `phase3/models/lightgbm_model.joblib`
- **Validation Predictions:** `phase3/results/val_predictions.parquet` (contains candidate-level predicted probabilities for blending and ensembling)
- **Metrics Summary:** `phase3/results/adarsh_model_metrics.csv`
- **ML + HMM Comparison:** `phase3/results/ml_hmm_comparison.csv`
- **Branch:** `phase3-models-adarsh` (all Phase 1 and 2 tests 100% passing).
