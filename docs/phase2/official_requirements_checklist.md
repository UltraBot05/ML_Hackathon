# PathSnap Hackathon — Phase 2 Official Requirements & Compliance Checklist

---

## 1. Official Documentation Sources

1. **[DE]**: *Dataset Exploration.pdf* (Supporting Document: Dataset Description and Exploration)
2. **[DEL]**: *Deliverables.pdf* (Technical Report & Submission Guidelines)
3. **[MG]**: *Main Guidelines.pdf* (Challenge Overview & Distributed Deployment)
4. **[MTG]**: *Models and Training Guidelines.pdf* (Modeling Expectations & Algorithms)
5. **[APP]**: *Assessment and Plagiarism Policy.pdf* (Evaluation Criteria & Academic Integrity)

---

## 2. Requirements & Implementation Matrix

| Requirement | Official Source | Relevant Section | How Phase 2 Implements It | Verification Status |
|:---|:---:|:---:|:---|:---:|
| **Zero Modification of Raw Dataset** | [DE] §1; [APP] §1 | [DE] p. 1 | All derived outputs written into `phase2/` subdirectories. SHA-256 integrity manifest recorded pre- and post-run. | **VERIFIED** |
| **Strict Protection of `student_test`** | [DE] §7; [MTG] §4, §7 | [DE] p. 3; [MTG] p. 2 | Zero training, zero tuning, zero feature selection, and zero label derivation performed on `student_test`. Phase 2 pipelines train/val splits strictly from 80 `train` trips. | **VERIFIED** |
| **Entity Semantics: `.track`** | [DE] §3 | [DE] p. 1 | Parsed as `longitude, latitude, timestamp_seconds`. Nominal 1-Hz sampling preserved without destructive deletion. | **VERIFIED** |
| **Entity Semantics: `.nodes`** | [DE] §3 | [DE] p. 1 | Parsed as `longitude, latitude`. 0-indexed row position defines `node_id`. | **VERIFIED** |
| **Entity Semantics: `.arcs`** | [DE] §3 | [DE] p. 1 | Parsed as directed connection `[from_node_id, to_node_id]`. 0-indexed row position defines `arc_id`. Directionality strictly enforced. | **VERIFIED** |
| **Entity Semantics: `.route` as Ground Truth** | [DE] §3, §6 | [DE] p. 1, 2 | Parsed as chronological sequence of ground-truth `arc_id` values traversed by the vehicle. Verified continuous path along directed network. | **VERIFIED** |
| **Local Geodesic / Metric Projection** | [DE] §5 | [DE] p. 2 | Point-to-segment distance and along-arc projections calculated using local metric tangent plane (Equirectangular / UTM metric projection) to prevent angular degree distortion. | **VERIFIED** |
| **Spatial Indexing for Candidate Generation** | [DE] §4; [MTG] §5 | [MTG] p. 4 | Implemented efficient KD-Tree spatial index over arc segment bounding boxes / midpoints, bounding candidate search within a configurable radius ($R=50\text{m}$, $\max=10$ candidates). | **VERIFIED** |
| **Candidate Geometric Feature Engineering** | [MTG] §5; [DEL] §2.2 | [MTG] p. 2 | Perpendicular distance, projection factor along segment, distance rank, and heading difference between trajectory displacement and directed road arc. | **VERIFIED** |
| **Movement & Temporal Feature Engineering** | [DE] §4; [MTG] §5 | [MTG] p. 2 | GPS step speed, acceleration, bearing, time delta, and anomaly flags (speed spikes, long pauses). | **VERIFIED** |
| **Road Network Transition Features** | [MTG] §8 | [MTG] p. 4 | Shortest network path distance between consecutive candidates $c_t \to c_{t+1}$, network-to-GPS distance ratio, transition feasibility, and graph connectivity. | **VERIFIED** |
| **Ground-Truth Label Generation** | [MTG] §3, §5, §8 | [MTG] p. 2, 4 | For each GPS point $P_t$, candidate $c_{t,j}$ is labeled `is_correct = 1` if $c_{t,j}$ matches the vehicle's traversed route arc at time $t$, else `0`. | **VERIFIED** |
| **Trip-Level Train / Validation Split** | [DE] §7; [MTG] §4 | [DE] p. 3 | Deterministic split (80/20 train/val, 64 train trips, 16 val trips, seed=42) partitioned strictly by `trip_id`. Zero trip overlap. | **VERIFIED** |
| **Baseline Map Matchers** | [MTG] §8; [DEL] §2.2 | [MTG] p. 4 | Baseline A: Nearest Road; Baseline B: Direction-Aware Nearest Road; Baseline C: Graph-Constrained / HMM-Viterbi heuristic matcher. | **VERIFIED** |
| **Sequential Route Reconstruction** | [MTG] §8, §9 | [MTG] p. 4, 5 | Reconstructs ordered `arc_id` sequences from matched observations, checking topological connectivity and reporting continuity. | **VERIFIED** |
| **Standardized Evaluation Metrics** | [MTG] §5; [DEL] §3.1 | [MTG] p. 2; [DEL] p. 3 | Evaluated Observation-level Top-1 Accuracy, Candidate Miss Rate, Macro F1-score, Route Continuity (%), and runtime latency. | **VERIFIED** |
| **Compatibility with Phase 3 ML Models** | [MTG] §2, §3 | [MTG] p. 1 | Feature tables exported with explicit schema, standardized numeric/categorical types, and metadata ready for LightGBM, Extra Trees, Logistic Regression, XGBoost, CatBoost, and Random Forest. | **VERIFIED** |
| **Evaluator-Testing Data Availability** | [DE] §7 | [DE] p. 3 | Final 10 evaluator test trips are held back by the competition organizers. Not present in local package. | **UNVERIFIED / HELD-OUT** |
| **Explicit Speed Limits / Road Class in Arcs** | [DE] §3 | [DE] p. 1 | Raw `.arcs` files contain only `from_node, to_node`. No explicit lane count, speed limit, or OSM highway tag provided in raw data. | **VERIFIED ABSENT (Derived Geometrically)** |

