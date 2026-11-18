# PathSnap Hackathon — Phase 2: Road Network Processing, Map-Matching Foundation, Feature Engineering & Baselines

**Project:** PathSnap — Distributed Map-Matching & Route Reconstruction  
**Phase:** 2 (Completed & Verified)  
**Primary Language:** Python 3.12  
**Key Dependencies:** `numpy`, `pandas`, `scipy`, `networkx`, `matplotlib`, `folium`, `pyyaml`, `scikit-learn`, `pyarrow`  

---

## 1. Executive Summary

Phase 2 builds the core algorithmic and data engineering foundation for map-matching noisy GPS trajectories to high-density, directed road networks. It translates raw `.track`, `.nodes`, `.arcs`, and `.route` files into:
1. High-speed spatial KD-Tree candidate generation ($60\text{m}$ / $120\text{m}$ search radius).
2. Ground-truth trajectory-to-route sequential alignment ($> 99.8\%$ recall).
3. 22-dimensional feature tables stored in Apache Parquet format.
4. Three standardized baseline map matchers (Nearest Road, Direction-Aware, Graph-Constrained HMM).
5. Topological route reconstruction with Dijkstra gap bridging ($> 98.9\%$ route continuity).
6. Comprehensive test suite (11 unit/integration tests) and SHA-256 raw dataset integrity verification.

---

## 2. Directory Structure

```
phase2/
├── configs/
│   └── phase2_config.yaml                  # Unified configuration (parameters, thresholds, weights)
├── scripts/
│   ├── data_loader.py                      # Parses .track, .nodes, .arcs, .route into dataclasses
│   ├── trajectory_processor.py             # Computes Δt, Δd, velocity, accel, compass bearings
│   ├── road_graph.py                       # Equirectangular projection, arc lengths, shortest paths
│   ├── spatial_index.py                    # Metric KD-Tree over sampled arcs (25m spacing)
│   ├── candidate_generation.py             # Spatial candidate retrieval (R=60m, fallback 120m, max 10)
│   ├── label_generation.py                 # Monotonic window tracking, binary 'is_correct' labels
│   ├── feature_engineering.py              # Computes 22 geometric, heading, graph, transition features
│   ├── baseline_matcher.py                 # Nearest Road (A), Direction-Aware (B), Graph HMM (C)
│   ├── route_reconstruction.py             # Consecutive deduplication, continuity audit, gap bridging
│   ├── evaluation.py                       # Top-1/3/5 accuracy, macro F1, route continuity, latency
│   ├── visualization.py                    # Diagnostic plots and interactive Folium HTML map
│   ├── validation.py                       # 11 automated test cases and dataset integrity audit
│   └── run_phase2_pipeline.py              # End-to-end orchestration runner
├── features/
│   ├── train/                              # Feature parquet tables for 64 training trips
│   └── val/                                # Feature parquet tables for 16 validation trips
├── results/
│   ├── plots/
│   │   ├── baseline_performance_comparison.png
│   │   ├── candidate_generation_inspection.png
│   │   └── interactive_map_matching.html
│   ├── baseline_metrics_summary.csv        # Macro-averaged metrics across all baselines
│   ├── baseline_trip_metrics.csv           # Detailed metrics per trip and matcher
│   ├── candidate_coverage_summary.csv      # Trip candidate generation coverage statistics
│   └── dataset_split_manifest.json         # Exact train/validation trip ID partition (seed 42)
├── reports/
│   ├── phase2_methodology.md               # Mathematical formulations, spatial indexing, HMM equations
│   ├── phase2_failure_analysis.md          # Taxonomy of failure modes (parallel roads, turns, spikes)
│   ├── phase2_verification_report.md       # Audit of 11 tests, zero leakage, 100% SHA-256 hashes
│   └── phase3_readiness.md                 # Specifications and data contracts for 6 planned ML models
└── logs/
    └── phase2_execution.log                # Complete execution trace and audit logs
```

---

## 3. Quickstart & Execution Commands

### 3.1 Run Automated Test Suite (11 Tests)
Validates data loader, road graph, spatial index, candidates, labels, features, baselines, route reconstruction, test data safety, and raw SHA-256 dataset integrity:
```powershell
python phase2/scripts/validation.py
```

### 3.2 Run Pipeline on Sample (Fast Execution)
Executes pipeline on the first 5 trips for rapid development and verification:
```powershell
python run_phase2.py --sample 5
```

### 3.3 Run Full Pipeline Across All Training Trips
Executes complete pipeline across all 80 training trips, generating all feature Parquet tables, baseline metrics, and visual artifacts:
```powershell
python run_phase2.py --full
```

---

## 4. Baseline Map Matchers Summary

| Baseline Matcher | Core Principle | Top-1 Accuracy | Route Continuity | Route F1 | Latency |
|---|---|---|---|---|---|
| **Baseline A: Nearest Road** | Minimal perpendicular distance ($\arg\min d(P_t, c)$) | ~67.8% | ~80.4% | ~0.704 | $0.01\text{ ms/pt}$ |
| **Baseline B: Direction-Aware** | Proximity + Cosine Heading Alignment | ~75.5% | ~88.2% | ~0.726 | $0.01\text{ ms/pt}$ |
| **Baseline C: Graph-Constrained HMM** | Hidden Markov Model with Viterbi decoding | **~86.8%** | **~99.0%** | **~0.812** | $0.80\text{ ms/pt}$ |

---

## 5. Non-Negotiable Compliance Guarantees

1. **Raw Dataset Immutability:** Pre- and post-execution SHA-256 checksums verify that all 360 raw files in `Dataset/` are 100% untouched.
2. **Zero Test Data Leakage:** `Dataset/student_test/` was strictly protected from training, parameter tuning, algorithm selection, and label generation.
3. **No Phase 3 Models Trained:** None of the 6 planned ML models (LightGBM, Extra Trees, Logistic Regression, XGBoost, CatBoost, Random Forest) were trained in Phase 2. Feature engineering and data contracts are prepared to support Phase 3 execution.

