# PathSnap Review 1 — Authoritative Six-Model Benchmark Leaderboard

**Data Contract**: Canonical 22 Phase 2 features (commit `59f573e`), strictly leak-free (no identifiers, coordinates, or reference route info).  
**Split**: 64 Training Trips (1,143,501 candidate rows), 16 Validation Trips (266,367 candidate rows).  
**Validation Observations**: Total $N_{\text{total}} = 39,460$; Covered with $\ge 1$ positive candidate $N_{\text{covered}} = 39,166$; Uncovered candidate misses = 294 ($0.75\%$).  

### Supervised Classification Leaderboard

| Model | Model Family | Candidate Accuracy | Candidate Macro F1 | Log Loss | ROC-AUC | Conditional Top-1 | Coverage-Aware Top-1 | Train Time (s) | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear Model (L2 Regularized) | 86.27% | 0.7859 | 0.2924 | 0.9460 | **74.94%** | **74.38%** | 1.75s | 0.0003 ms/obs |
| **Random Forest** | Bootstrap Bagging Ensemble | 89.82% | 0.8264 | 0.2393 | 0.9580 | **73.48%** | **72.93%** | 146.57s | 0.0228 ms/obs |
| **Extra Trees** | Extremely Randomized Trees | 92.53% | 0.8440 | 0.1744 | 0.9594 | **75.41%** | **74.85%** | 27.65s | 0.0123 ms/obs |
| **XGBoost** | Gradient Boosted Decision Trees | 91.11% | 0.8189 | 0.2024 | 0.9542 | **70.11%** | **69.59%** | 8.40s | 0.0039 ms/obs |
| **LightGBM** | Histogram Gradient Boosting | 91.01% | 0.8166 | 0.2209 | 0.9518 | **69.14%** | **68.62%** | 4.72s | 0.0009 ms/obs |
| **CatBoost** | Symmetric Oblivious Decision Trees | 92.17% | 0.8353 | 0.1780 | 0.9604 | **74.20%** | **73.64%** | 9.00s | 0.0005 ms/obs |

### Key Architectural & Performance Insights

1. **Top-1 Observation Accuracy Range (70% – 75%)**:
   - Across all 16 unseen validation trips (39,460 observations), pointwise directed Top-1 accuracy sits consistently between **69.59%** (XGBoost) and **74.85%** (Extra Trees).
   - **Extra Trees** achieves the highest Top-1 accuracy (**75.41% conditional, 74.85% coverage-aware**) and highest Macro F1 (**0.8440**), benefiting from extreme random thresholding that mitigates correlation among dense spatial features.
   - **Logistic Regression** proves surprisingly resilient at **74.94% conditional Top-1**, outperforming gradient boosters point-wise because its linear log-odds probabilities produce well-calibrated argmax rankings.
   - **CatBoost** delivers the best boosting balance: **74.20% conditional Top-1**, lowest log loss (**0.1780**), highest ROC-AUC (**0.9604**), and blazingly fast M1 training (**9.00s**) and inference (**0.0005 ms/obs**).

2. **Sequential Decoding Leap (ML + HMM Viterbi)**:
   - While pure pointwise ML achieves ~70%–75% Top-1, integrating ML emission probabilities into an **HMM Viterbi Decoder with Dijkstra network bridging** boosts average trip Top-1 to **86.26%**, route continuity to **99.15%**, and Route F1 to **0.8911** (demonstrated on LightGBM+HMM by Adarsh).
   - Pointwise classification treats each observation independently; HMM Viterbi enforces road network topology, eliminating physically impossible jumps.

3. **Demystifying the "96%–97%" Claim (Why Our 70%–75% Is Genuine)**:
   - **Candidate Binary Accuracy**: Predictors predicting zero achieve ~85.3% accuracy due to class imbalance (~14.7% positive rows). Models achieve **86%–92.5%** candidate binary accuracy.
   - **Trip Selection Bias**: On clean suburban/highway trips, raw pointwise Top-1 is indeed **95%–100%** (e.g., Trip `00000024` is 97.75%–100%, Trip `00000005` is 97.7%). However, challenging trips with severe GPS noise, urban canyons, or tunnels (e.g., Trip `00000084` with 16,031 observations) pull down the aggregate validation average to ~70%–75%.
   - **Route Continuity vs Pointwise Accuracy**: Reconstructed route continuity after network bridging is **98%–100%**, which non-rigorous presentations often conflate with pointwise Top-1.
