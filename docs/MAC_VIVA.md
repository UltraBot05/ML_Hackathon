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

## 8. Actual Experimental Findings on Canonical 22 Features

Benchmark evaluated across all 16 validation trips (266,367 candidate rows, 39,460 GPS observations):

| Model / Matcher | Top-1 Accuracy (%) | Coverage-Aware Top-1 (%) | Macro F1 | Binary Accuracy (%) | Log Loss | ROC-AUC | Train Time (s) | Inference Latency |
|---|---|---|---|---|---|---|---|---|
| **CatBoost Classifier** | **73.64%** | **74.20%** | **0.8353** | **92.17%** | **0.17804** | **0.9604** | **9.00 s** | **0.0005 ms/obs** |
| **Random Forest Classifier** | **72.93%** | **73.48%** | **0.8264** | **89.82%** | **0.23928** | **0.9580** | **146.57 s** | **0.0228 ms/obs** |
| **Baseline C: Graph HMM** | 83.22% | 83.22% | 0.8220 | — | — | — | — | 0.9890 ms/obs |
| **Baseline B: Direction-Aware** | 76.89% | 76.89% | 0.7600 | — | — | — | — | 0.0113 ms/obs |
| **Baseline A: Nearest Road** | 59.76% | 59.76% | 0.5881 | — | — | — | — | 0.0053 ms/obs |

### Key Viva Takeaways:
1. **CatBoost superior efficiency on Apple Silicon:** CatBoost trained in 9.0s (16x faster than Random Forest) and ran inference in 0.0005 ms/obs (45x faster than Random Forest, ~2,000x faster than Baseline C HMM).
2. **Top Features by Gini / Split Gain:**
   - `projection_t` (25.14% in CatBoost): Along-arc projection identifies whether the fix is near an intersection.
   - `dist_diff_from_best_m` (21.51% in CatBoost, 16.25% in RF): Margin relative to closest road segment discriminates parallel carriageways.
   - `log_distance` & `distance_to_arc_m` (combined ~18%): Proximity scale.
   - Kinematic alignment (`speed_heading_alignment`, `arc_bearing_deg`): Eliminates counter-flow one-way candidates.
3. **Pointwise vs Sequential:** Independent candidate classifiers achieve higher candidate Macro F1 (0.8353) than HMM (0.8220), but HMM benefits from sequence continuity. The winning architecture is a **Hybrid CatBoost-HMM Trellis**.

