# PathSnap Hackathon — Phase 3 Readiness & Model Architecture Report
**Project:** Distributed Map-Matching & Route Reconstruction  
**Phase:** 2 -> 3 Transition Specification  
**Status:** Ready for Phase 3 Execution  

---

## 1. Executive Summary

Phase 2 has successfully established a clean, standardized, leak-free feature engineering and spatial candidate generation pipeline. This document defines the exact data contracts, tabular feature schemas, partition splits, and model architectures for the **six planned Phase 3 machine learning models**:

1. **LightGBM** (Light Gradient Boosting Machine)
2. **Extra Trees Classifier** (Extremely Randomized Trees)
3. **Logistic Regression** (L2-Regularized Linear Baseline)
4. **XGBoost** (Extreme Gradient Boosting)
5. **CatBoost** (Categorical & Ordered Boosting)
6. **Random Forest** (Bootstrap Aggregated Forest)

> [!IMPORTANT]
> **Strict Phase Separation:** As mandated by the hackathon instructions, none of these six models have been trained during Phase 2. Phase 2 delivers the fully processed, validated data tables, ground-truth alignment targets, baseline benchmarks, and feature contracts ready for Phase 3 execution.

---

## 2. Standardized Data Contract & Feature Schema

All candidate tables generated in Phase 2 adhere to a unified schema stored in Apache Parquet format under:
- Training Features: `phase2/features/train/<trip_id>_features.parquet` (64 trips)
- Validation Features: `phase2/features/val/<trip_id>_features.parquet` (16 trips)

### 2.1 Feature Dictionary (22 Input Features)

| Feature Name | Dtype | Range | Description |
|---|---|---|---|
| `distance_to_arc_m` | `float64` | $[0.0, 120.0]$ | Perpendicular metric distance from GPS point to candidate arc |
| `dist_diff_from_best_m` | `float64` | $[0.0, 120.0]$ | Distance margin relative to the 1-nearest candidate ($d_i - d_1$) |
| `projection_t` | `float64` | $[0.0, 1.0]$ | Normalized along-arc projection factor (0=start node, 1=end node) |
| `candidate_rank` | `int64` | $\{1, \dots, 10\}$ | Spatial distance rank among candidate set for observation |
| `normalized_rank` | `float64` | $[0.1, 1.0]$ | Normalized distance rank (`candidate_rank / 10.0`) |
| `log_distance` | `float64` | $[0.0, 4.8]$ | Log-transformed distance $\ln(1.0 + d)$ |
| `gps_speed_mps` | `float64` | $[0.0, 55.0]$ | Instantaneous GPS vehicle speed in meters per second |
| `gps_bearing_deg` | `float64` | $[0.0, 360.0)$ | GPS trajectory compass heading |
| `arc_bearing_deg` | `float64` | $[0.0, 360.0)$ | Road segment compass orientation |
| `heading_diff_deg` | `float64` | $[0.0, 180.0]$ | Absolute angular discrepancy $|\theta_{\text{gps}} - \theta_{\text{arc}}|$ |
| `cos_heading_diff` | `float64` | $[-1.0, 1.0]$ | Cosine of angular discrepancy |
| `speed_heading_alignment`| `float64` | $[-1.0, 1.0]$ | Heading alignment attenuated by speed $\cos(\Delta\theta) \cdot \min(1.0, v/5.0)$ |
| `is_counter_flow` | `int64` | $\{0, 1\}$ | Flag for motion opposing one-way arc flow ($\Delta\theta > 120^\circ \land v > 3\text{ m/s}$) |
| `arc_length_m` | `float64` | $[1.0, 5000.0]$ | Metric road segment length in meters |
| `from_node_out_degree` | `int64` | $[1, 20]$ | Outgoing road segment count from arc start node |
| `to_node_in_degree` | `int64` | $[1, 20]$ | Incoming road segment count to arc end node |
| `is_terminal_arc` | `int64` | $\{0, 1\}$ | Flag indicating single-in / single-out continuation segment |
| `step_gps_dist_m` | `float64` | $[0.0, 500.0]$ | Geodesic distance moved by GPS over preceding time step |
| `transition_dist_m` | `float64` | $[0.0, 2000.0]$ | Shortest network path distance from previous top candidate |
| `transition_dist_ratio` | `float64` | $[0.0, 50.0]$ | Ratio of network transition distance to GPS step distance |
| `is_directly_connected` | `int64` | $\{0, 1\}$ | Flag indicating topological adjacency ($v_{t-1}^* == u_t \lor e_{t-1}^* == e_t$) |
| `log_transition_dist` | `float64` | $[0.0, 7.6]$ | Log-transformed network transition distance |

### 2.2 Target Variables & Identifiers
- **Classification Target:** `is_correct` $\in \{0, 1\}$ (1 if candidate arc == true traversed arc, else 0).
- **Evaluation Grouping:** `trip_id`, `obs_index` (for grouping candidates per GPS point).
- **Ground Truth Arc ID:** `true_arc_id` $\in \mathbb{Z}$ (reference road segment).

---

## 3. Trip-Level Train / Validation Partitioning

To avoid data leakage across time-series observations, splitting is conducted strictly at the **trip level** rather than random point sampling.

- **Total Training Partition Trips:** 80
- **Train Split (80%):** 64 trips (`phase2/features/train/`)
- **Validation Split (20%):** 16 trips (`phase2/features/val/`)
- **Random Seed:** 42 (fixed for reproducibility)
- **Split Manifest:** Saved in `phase2/results/dataset_split_manifest.json`

Because each trip occurs in a distinct geographic area and time window, trip-level evaluation ensures that models generalize to unseen road networks and novel driver trajectories.

---

## 4. Planned Phase 3 Model Architectures & Optimization Plan

```
+---------------------------------------------------------------------------------------------------+
|                               PHASE 3 PLANNED MODEL ARCHITECTURES                                 |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Linear Baseline]               [Randomized Forest]             [Bootstrap Bagging]              |
|  • Model 3: Logistic Regression  • Model 2: Extra Trees          • Model 6: Random Forest         |
|  - Fast linear baseline          - Extreme random thresholds     - 200 trees, sqrt features       |
|  - L2 penalty, balanced weights  - Fast parallel training        - Robust against overfitting     |
|                                                                                                   |
|  [Histogram GBDT]                [Exact Gradient Boosting]       [Oblivious Trees]                |
|  • Model 1: LightGBM             • Model 4: XGBoost              • Model 5: CatBoost              |
|  - Leaf-wise tree growth         - Depth-wise tree growth        - Symmetric oblivious splits     |
|  - High-speed histogram binning  - L1/L2 regularization          - Resists target leakage         |
|                                                                                                   |
|                                    ▼                                                              |
|                     [Candidate Softmax Scoring & Viterbi]                                         |
|                     • Softmax candidate probability P(c | P_t)                                    |
|                     • Hybrid ML emission + HMM transition trellis                                 |
+---------------------------------------------------------------------------------------------------+
```

### 4.1 Model 1: LightGBM (Primary Candidate Classifier)
- **Objective:** Binary cross-entropy `binary_logloss` or `lambdarank` (pairwise ranking per observation).
- **Hyperparameters:** `num_leaves=31`, `learning_rate=0.05`, `n_estimators=300`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Expected Advantage:** Fast training on $> 500{,}000$ candidate rows; excellent capture of non-linear interactions between distance, heading alignment, and transition distance ratios.

### 4.2 Model 2: Extra Trees (Extremely Randomized Trees)
- **Objective:** Variance-reduced ensemble classifier.
- **Hyperparameters:** `n_estimators=200`, `max_depth=16`, `min_samples_split=5`, `n_jobs=-1`.
- **Expected Advantage:** Randomized split thresholds smooth decision boundaries across noisy GPS measurements.

### 4.3 Model 3: Logistic Regression (Interpretable Linear Baseline)
- **Objective:** L2-regularized logistic regression with standard feature scaling.
- **Hyperparameters:** `C=1.0`, `solver='lbfgs'`, `class_weight='balanced'`.
- **Expected Advantage:** Microsecond inference latency ($< 0.001\text{ ms/candidate}$); establishes linear separability baseline.

### 4.4 Model 4: XGBoost (Gradient Boosted Decision Trees)
- **Objective:** `binary:logistic` with exact greedy split search.
- **Hyperparameters:** `max_depth=6`, `learning_rate=0.05`, `n_estimators=300`, `gamma=1.0`, `reg_lambda=2.0`.
- **Expected Advantage:** Robust regularization against high-density intersection noise.

### 4.5 Model 5: CatBoost (Ordered Boosting)
- **Objective:** Logloss on symmetric oblivious trees.
- **Hyperparameters:** `depth=6`, `learning_rate=0.05`, `iterations=400`, `l2_leaf_reg=3.0`.
- **Expected Advantage:** Oblivious trees prevent overfitting on dominant features (e.g. `distance_to_arc_m`).

### 4.6 Model 6: Random Forest Classifier
- **Objective:** Bootstrap aggregation over decorrelated decision trees.
- **Hyperparameters:** `n_estimators=200`, `max_depth=14`, `max_features='sqrt'`, `n_jobs=-1`.
- **Expected Advantage:** Benchmark ensemble performance without boosting complexity.

---

## 5. Phase 3 Evaluation Protocol

In Phase 3, all six models will be evaluated using the identical protocol and validation split established in Phase 2:
1. **Candidate-Level Metrics:** Precision, Recall, Macro F1, ROC-AUC, PR-AUC.
2. **Observation-Level Metrics:** Top-1 Accuracy (% of observations where top-predicted candidate is the true road), Top-3 Accuracy.
3. **Route-Level Metrics:** Reconstructed Route Continuity (%), Ground-Truth Route F1, Route Jaccard Similarity.
4. **Efficiency Metrics:** Training duration, inference latency (ms/point), and memory consumption.

