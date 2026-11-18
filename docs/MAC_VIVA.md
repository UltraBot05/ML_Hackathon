# PathSnap Hackathon — Comprehensive Viva Preparation Guide
**Author:** M1 Air Teammate (Random Forest, CatBoost & Evaluation UI Lead)  
**Hackathon:** PES University PathSnap Hackathon (November 18, 2026)  
**Document:** `docs/MAC_VIVA.md`  

---

## 1. Random Forest Classifier

### 1.1 Bootstrap Aggregation (Bagging)
- **Concept:** Ensemble meta-estimator that fits multiple decision tree classifiers in parallel on bootstrap samples of the training set.
- **Bootstrap Sampling:** Given a training set $D$ of $N$ candidate rows, each tree is trained on a sample $D_b$ of size $N$ drawn uniformly with replacement ($D_b \sim \text{Bootstrap}(D)$).
- **Statistical Property:** Approximately $1 - 1/e \approx 63.2\%$ of unique observations are sampled per tree; the remaining $\approx 36.8\%$ form the Out-Of-Bag (OOB) sample.
- **Variance Reduction:** For $B$ identically distributed (but independent) trees each with variance $\sigma^2$ and pairwise correlation $\rho$:
  $$\text{Var}\left(\frac{1}{B}\sum_{b=1}^B T_b(x)\right) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2$$
  As $B \to \infty$, the second term approaches 0, and total ensemble variance is bounded by $\rho \sigma^2$. Decorrelating individual trees reduces $\rho$.

### 1.2 Feature Randomness (Random Subspace Method)
- At each split node in every tree, instead of searching across all $p=22$ features, a random subset of $m = \sqrt{p} = \lfloor\sqrt{22}\rfloor = 4$ features is evaluated.
- **Why it matters:** If one feature is overwhelmingly dominant (e.g., `distance_to_arc_m` or `dist_diff_from_best_m`), standard decision trees would split on it at the root of almost every tree, making the trees strongly correlated ($\rho \to 1$). Feature subsampling forces diverse subtrees that capture non-distance signals like heading alignment, speed, and topological transition continuity.

### 1.3 Tree Depth and Hyperparameter Configuration
- `n_estimators = 200`: Sufficient trees to stabilize vote averaging across the candidate space without excessive memory overhead.
- `max_depth = 14`: Controls model capacity and prevents individual trees from memorizing anomalous GPS noise or specific road geometries.
- `max_features = 'sqrt'`: Enforces $m \approx 4$ random features per split node.
- `class_weight = 'balanced'`: Compensates for candidate class imbalance (typically $\approx 1$ positive candidate out of $7\text{--}10$ generated candidates per GPS point, $\approx 12.5\%$ positive class).

---

## 2. CatBoost Classifier

### 2.1 Gradient Boosting Principles
- **Additive Formulation:** Unlike Bagging (which averages independent trees), Gradient Boosting constructs an ensemble sequentially:
  $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
  where $\eta$ is the learning rate, and each new weak learner $h_m(x)$ is fit to the negative gradient (pseudo-residuals) of the loss function $\mathcal{L}(y, F_{m-1}(x))$:
  $$r_{i,m} = -\left[\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$
- For binary cross-entropy (Logloss), the negative gradient equals $y_i - \sigma(F(x_i))$, pointing in the exact direction of probability correction.

### 2.2 Ordered Boosting
- **Problem with Classic GBDT:** In standard gradient boosting (XGBoost/LightGBM), the gradients used to train $h_m(x)$ are calculated on the same instances $x_i$ that were used to train earlier iterations $F_{m-1}$, inducing target leakage and prediction shift (overfitting).
- **CatBoost Solution (Ordered Boosting):** Simulates an online learning protocol by generating random permutations of the training dataset. When calculating the gradient for instance $x_i$, CatBoost uses a model trained strictly on instances preceding $x_i$ in the permutation:
  $$\hat{y}_i \text{ evaluated without using } (x_i, y_i)$$
  This eliminates statistical bias in gradient estimates and improves generalization on unseen test trips.

### 2.3 Symmetric (Oblivious) Decision Trees
- **Structure:** At each depth level of the tree, the exact same split criterion (feature and threshold) is applied across all nodes at that depth.
- **Properties:**
  - Depth $d$ produces exactly $2^d$ leaves.
  - Evaluation index is computed via binary bitwise arithmetic: $\text{leaf\_idx} = \sum_{k=0}^{d-1} 2^k \cdot \mathbb{I}(x_{j_k} > \theta_k)$.
  - Highly cache-friendly, enabling rapid vector execution on CPU (critical for M1 ARM architecture).
  - Acts as structural regularization, preventing deep asymmetric leaf overfitting.
- **Hyperparameters:** `depth = 6` ($2^6 = 64$ leaves), `learning_rate = 0.05`, `iterations = 400`, `l2_leaf_reg = 3.0`.

---

## 3. Bagging versus Boosting: Deep Comparison

| Dimension | Random Forest (Bagging) | CatBoost (Boosting) |
|---|---|---|
| **Ensemble Paradigm** | Independent parallel trees | Sequential residual correction |
| **Error Reduction** | Primarily reduces **variance** | Reduces **bias** and **variance** |
| **Weak Learner Capacity** | Deep, unpruned, high-variance trees (`max_depth=14`) | Shallow, constrained, high-bias trees (`depth=6`) |
| **Tree Architecture** | Asymmetric CART binary trees | Symmetric oblivious decision tables |
| **Sensitivity to Outliers** | High robustness (averaging dampens outliers) | Moderately sensitive (can over-focus on hard residuals if unregularized) |
| **Parallelization** | Embarrassingly parallel across CPU cores | Parallel across histogram binning & permutation splits |
| **Output Interpretation** | Direct empirical fraction of tree votes | Additive log-odds mapped via sigmoid function $\sigma(z)$ |

---

## 4. Why Our 22 Features Help

Phase 2 created 22 discriminative features grouped into four physical domains:

1. **Local Geometric Proximity (6 Features):**
   - `distance_to_arc_m`, `log_distance`: Logarithmic and raw Euclidean distance from GPS fix to segment.
   - `dist_diff_from_best_m`: Distance delta relative to nearest candidate. Crucial for resolving parallel roads (e.g. elevated expressway vs. ground service road).
   - `projection_t`: Normalized position $[0, 1]$ along road arc. Detects if vehicle is near an intersection or mid-block.
   - `candidate_rank`, `normalized_rank`: Relative spatial ordinality.

2. **Kinematic & Heading Alignment (7 Features):**
   - `gps_speed_mps`, `gps_bearing_deg`, `arc_bearing_deg`: Raw dynamic states.
   - `heading_diff_deg`, `cos_heading_diff`: Angular agreement $\Delta \theta = |\theta_{\text{gps}} - \theta_{\text{arc}}|$.
   - `speed_heading_alignment`: Modulates cosine heading by velocity: $\cos(\Delta\theta) \cdot \min(1.0, v/5.0)$. When a vehicle is stopped or moving $< 1\text{ m/s}$, GPS heading drifts randomly; this feature attenuates misleading heading signals at low speeds.
   - `is_counter_flow`: Binary flag triggering when motion opposes directed road orientation ($\Delta\theta > 120^\circ \land v > 3\text{ m/s}$). Immediately discards one-way wrong-direction candidates.

3. **Road Graph Topology (4 Features):**
   - `arc_length_m`: Metric segment length.
   - `from_node_out_degree`, `to_node_in_degree`: Graph connectivity metrics. Higher degree indicates complex junctions (roundabouts, highway interchanges).
   - `is_terminal_arc`: Segments with single predecessor and successor (unbroken corridors).

4. **Sequential Transition Dynamics (5 Features):**
   - `step_gps_dist_m`: Geodesic distance moved by vehicle over $\Delta t$.
   - `transition_dist_m`, `log_transition_dist`: Shortest network path length on directed graph between candidate at $t-1$ and candidate at $t$.
   - `transition_dist_ratio`: $\frac{d_{\text{network}}(c_{t-1}, c_t)}{d_{\text{gps}}(P_{t-1}, P_t)}$. For physically valid vehicle transitions, this ratio is $\approx 1.0$. If a candidate requires a non-physical U-turn or highway backtrack, this ratio spikes to $\gg 5.0$, enabling tree splits to reject it.
   - `is_directly_connected`: Binary topological continuity check ($v_{t-1}^* == u_t$ or $e_{t-1}^* == e_t$).

---

## 5. Why Validation Is Grouped by Trip (Zero Leakage)

- **The Leakage Threat:** GPS points along a single trip are autocorrelated in time, space, driver habits, and road geometry. If points were partitioned using random row-level k-fold cross-validation:
  - An observation at $t=10\text{s}$ could be in `train` while $t=11\text{s}$ is in `val`.
  - The model would simply memorize specific segment IDs and spatial coordinates, yielding artificially inflated $> 98\%$ accuracy that completely collapses on novel trips.
- **Trip-Level Splitting:** Partitioning strictly by `trip_id` (64 train trips, 16 validation trips, fixed seed 42) ensures that entire routes, novel geographical subregions, and complete driver trajectories are held out. Models are forced to learn transferable physical relationships (distance margins, velocity-attenuated heading alignment, network transition ratios) rather than memorizing local coordinates.

---

## 6. Supervised Classification versus HMM Sequence Decoding

| Concept | Supervised Point Classifier (RF / CatBoost) | Hidden Markov Model (HMM) Viterbi |
|---|---|---|
| **Unit of Decision** | Candidate scoring per observation point | Joint path optimization over full sequence |
| **Emission Model** | Learned non-linear interactions across 22 geometric/kinematic features | Parametric Gaussian distance + heading distribution |
| **Transition Model** | Pairwise transition feature (`transition_dist_ratio`) | First-order Markov chain with exponential decay prior $e^{-|\Delta d|/\beta}$ |
| **Global Continuity** | Independent per-point argmax can suffer momentary topological discontinuities | Guarantees mathematically optimal path across trellis |
| **Inference Latency** | Ultra-fast vector inference ($< 0.05\text{ ms/candidate}$) | Trellis dynamic programming $\mathcal{O}(T \cdot K^2)$ ($0.8\text{--}1.0\text{ ms/point}$) |
| **Best-of-Both Hybrid** | Use supervised classifier probabilities $\hat{P}(c \mid P_t)$ as refined emission probabilities inside the Viterbi trellis! | |

---

## 7. Evaluator UI & ZeroTier Distributed Architecture

```
+-----------------------------------------------------------------------------+
|                          DISTRIBUTED SYSTEM TOPOLOGY                         |
+-----------------------------------------------------------------------------+
|                                                                             |
|   [MacBook Air M1 Frontend]                [Omen Server / Workstation]      |
|   • Streamlit Web Application               • FastAPI / Uvicorn Server       |
|   • Interactive Folium Maps                 • 6 Supervised Models Loaded     |
|   • Model Selector & Metrics Dashboard      • Road Graph & HMM Pipeline      |
|   • Port: 8501                              • ZeroTier IP: 10.92.151.151:8000|
|            │                                              │                 |
|            │──────────── HTTP GET /health ───────────────>│                 |
|            │<─────────── Status: OK, Models ──────────────│                 |
|            │                                              │                 |
|            │──────────── POST /predict ──────────────────>│                 |
|            │    Payload: {"trip_id": "...", "model": ".."}│                 |
|            │<─────────── JSON Result: ────────────────────│                 |
|            │    {"matched_arcs": [...], "metrics": {...}} │                 |
|                                                                             |
+-----------------------------------------------------------------------------+
```

- **Communication Protocol:** RESTful JSON over ZeroTier software-defined private mesh network.
- **Fail-Safe UI Design:**
  - Health probe checks connectivity before sending heavy payloads.
  - Graceful fallback: If ZeroTier node is offline, UI provides local evaluation mode using local model artifacts and validation Parquet features.
  - Zero hardcoding: All model names, trip options, evaluation metrics, and GeoJSON road coordinates are populated dynamically from API responses or local evaluation manifests.

---

## 8. Authoritative Six-Model Leaderboard on Canonical 22 Features

Benchmark evaluated across all 16 validation trips (266,367 candidate rows, 39,460 GPS observations, 39,166 covered observations):

| Model | Model Family | Candidate Accuracy | Candidate Macro F1 | Log Loss | ROC-AUC | Conditional Top-1 | Coverage-Aware Top-1 | Train Time (s) | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear Model (L2 Regularized) | 86.27% | 0.7859 | 0.2924 | 0.9460 | **74.94%** | **74.38%** | 1.75s | 0.0003 ms/obs |
| **Random Forest** | Bootstrap Bagging Ensemble | 89.82% | 0.8264 | 0.2393 | 0.9580 | **73.48%** | **72.93%** | 146.57s | 0.0228 ms/obs |
| **Extra Trees** | Extremely Randomized Trees | 92.53% | 0.8440 | 0.1744 | 0.9594 | **75.41%** | **74.85%** | 27.65s | 0.0123 ms/obs |
| **XGBoost** | Gradient Boosted Decision Trees | 91.11% | 0.8189 | 0.2024 | 0.9542 | **70.11%** | **69.59%** | 8.40s | 0.0039 ms/obs |
| **LightGBM** | Histogram Gradient Boosting | 91.01% | 0.8166 | 0.2209 | 0.9518 | **69.14%** | **68.62%** | 4.72s | 0.0009 ms/obs |
| **CatBoost** | Symmetric Oblivious Decision Trees | 92.17% | 0.8353 | 0.1780 | 0.9604 | **74.20%** | **73.64%** | 9.00s | 0.0005 ms/obs |

### Exact Numerators and Denominators for Top-1 Metrics:
- **Total Validation Observations ($N_{\text{total}}$):** $39,460$
- **Covered Observations ($N_{\text{covered}}$):** $39,166$ (observations having at least one true positive candidate in the retrieved set)
- **Candidate Misses ($N_{\text{uncovered}}$):** $294$ ($0.75\%$)
- **Conditional Top-1:** $\frac{N_{\text{correct}}}{N_{\text{covered}}}$ (conditioned on candidate presence)
- **Coverage-Aware Top-1:** $\frac{N_{\text{correct}}}{N_{\text{total}}}$ (unconditional, penalizes the 294 candidate misses)

---

## 9. Answering the Toughest Viva Questions

### Q1: "Other teams claim 96%–97% accuracy with XGBoost or LightGBM. Why are your Top-1 scores 70%–75%?"
**Viva Answer:**
> "That discrepancy comes down to three distinct factors that must not be conflated:
> 1. **Candidate Binary Accuracy vs. Pointwise Top-1:** Because candidate generation produces 7–10 candidates per point, roughly 85.3% of candidate rows are negative. A classifier predicting zero everywhere achieves 85.3% candidate accuracy. Our models achieve 86%–93% candidate accuracy and >0.83 Macro F1, but pointwise Top-1 requires identifying the exact single correct directed arc out of all candidates.
> 2. **Trip Selection Bias:** On clean suburban trips, our models also achieve **95%–100% Top-1** (e.g., Trip `00000024` is 97.75%–100%, Trip `00000005` is 97.70%). However, massive trips with severe urban canyon noise and stop-and-go multipath jitter—specifically Trip `00000084` with 16,031 points (over 40% of the entire validation set)—have raw pointwise accuracy of ~47%–48%, which pulls the observation-weighted validation average down to 70%–75%.
> 3. **Route Continuity vs. Pointwise Accuracy:** When other teams report 96%–99%, they are often reporting **reconstructed route continuity** after HMM Viterbi decoding and shortest-path gap bridging. Our LightGBM+HMM pipeline also achieves **99.15% route continuity** and **86.26% trip-level Top-1**!"

### Q2: "Phase 2 assigned ground truth labels using candidates. How do you know candidate retrieval didn't miss ground truth arcs?"
**Viva Answer:**
> "We explicitly suspected this proxy-label circularity and built an **independent route-alignment diagnostic** (`phase2/scripts/independent_route_alignment.py`). This script projected raw GPS points directly onto the ordered `.route` arc geometry *without using the candidate pool*.
> Evaluating 28,964 validation observations across 8 diverse trips proved:
> 1. **95.6% label agreement** with the canonical Phase 2 labels.
> 2. **True candidate retrieval recall is >99.1%**, meaning the KD-tree candidate search ($R_1=60\text{m}, R_2=120\text{m}$) successfully captures the true arc in over 99% of observations.
> 3. True candidate misses ($0.75\%$ across the 16 validation trips) occur almost exclusively during off-network driving or severe GPS dropouts, not KD-Tree failure."

### Q3: "Why did Extra Trees and CatBoost outperform XGBoost and LightGBM point-wise?"
**Viva Answer:**
> "1. **Extra Trees (75.41% Top-1):** In tabular spatial data, features like `distance_to_arc_m`, `dist_diff_from_best_m`, and `log_distance` are highly collinear. Extra Trees randomizes both the feature subset AND the split threshold uniformly at random. This extreme threshold randomization prevents the dominant distance features from masking subtle heading and topology signals, producing the highest candidate Macro F1 (0.8440) and highest Top-1.
> 2. **CatBoost (74.20% Top-1, 0.1780 Log Loss):** CatBoost uses symmetric oblivious decision trees with ordered boosting and explicit $L_2$ regularization. Oblivious trees enforce uniform split structure across all leaves at each depth, acting as strong structural regularization that prevents overfitting to intricate junction shapes. Furthermore, CatBoost's ordered boosting avoids target shift on trip partitions."

### Q4: "Why does Logistic Regression perform so well (74.94% Top-1) compared to complex GBDTs?"
**Viva Answer:**
> "Logistic Regression fits a linear boundary in the log-odds space. For Top-1 ranking, the absolute probability values do not matter as much as the relative ordering (argmax) among the candidates of a single observation. The standardized logistic regression weights heavily on `speed_heading_alignment` (+1.48) and `dist_diff_from_best_m` (-1.19), producing monotonic, well-calibrated score margins that cleanly separate the best arc without the risk of leaf-node threshold jitter that can affect unregularized tree splits."
