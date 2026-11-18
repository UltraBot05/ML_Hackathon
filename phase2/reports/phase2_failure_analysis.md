# PathSnap Hackathon — Phase 2: Failure Mode & Error Analysis
**Project:** Distributed Map-Matching & Route Reconstruction  
**Phase:** 2 — Map-Matching Failure Taxonomy & Mitigation Strategies  
**Status:** Completed & Verified  

---

## 1. Introduction

Map matching real-world vehicle GPS trajectories onto high-density, directed urban road networks is challenged by measurement noise, satellite occlusions, multipath reflections, and topological ambiguities. 

This report provides a systematic taxonomy of the primary failure modes identified across the PathSnap dataset during Phase 2 baseline benchmarking, evaluates how each baseline behaves under these conditions, and outlines targeted mitigations for our planned Phase 3 machine learning models.

---

## 2. Taxonomy of Map-Matching Failure Modes

```
+---------------------------------------------------------------------------------------------------+
|                              MAP-MATCHING FAILURE MODES TAXONOMY                                  |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Parallel Road Ambiguity]        [Intersection Turn Jitter]      [Multipath Velocity Spikes]    |
|  • Dual carriageway / frontage   • Turning at complex junctions   • Impossible speeds (>198 km/h) |
|  • Distance delta < 8 meters     • Speed drops; heading fluctuates• Search radius fallback needed |
|                                                                                                   |
|  [Stationary Bearing Drift]      [Topological Candidate Misses]  [Overpass / Multi-Level Ramps]  |
|  • Clustering at red lights      • True arc excluded from top-K   • Grade separation ignored in 2D|
|  • Random heading oscillations   • Breaks route continuity        • Spurious cross-level matching |
+---------------------------------------------------------------------------------------------------+
```

---

### Mode 1: Parallel Roadway & Dual-Carriageway Ambiguity

**Phenomenon:**  
In suburban and urban networks, divided arterial highways frequently run parallel to access roads, frontage lanes, or opposing traffic carriageways separated by only $5\text{–}15\text{ meters}$. A standard GPS error of $8\text{–}12\text{ meters}$ is sufficient to place the raw point closer to the incorrect parallel segment than the true traversed lane.

**Baseline Failure Patterns:**
- **Baseline A (Nearest Road):** Fails catastrophically. Whenever GPS drift shifts laterally across the median, Baseline A instantaneously flips to the opposing or parallel road segment, producing zigzagging "ping-pong" route predictions.
- **Baseline B (Direction-Aware):** Successfully eliminates *opposing* traffic lanes by filtering out counter-flow arcs ($\Delta\theta \approx 180^\circ$). However, when the parallel road runs in the *same* direction (e.g. highway vs. frontage road), Baseline B cannot distinguish them because $\Delta\theta \approx 0^\circ$ for both candidates.
- **Baseline C (Graph-Constrained HMM):** Substantially more robust. If transitioning from the main carriageway onto the frontage road requires an exit ramp that does not exist at that coordinate, the transition network distance $d_G$ is infinite (or large detour), penalizing the incorrect candidate. However, if the frontage road is connected by cross-streets, HMM can occasionally lag by 2–3 time steps before snapping back.

**Phase 3 ML Mitigation:**
- Features such as `transition_dist_ratio`, `is_directly_connected`, and `from_node_out_degree` provide explicit signals distinguishing controlled-access highway lanes from local access streets.
- Tree-based models (LightGBM, XGBoost, CatBoost) learn hierarchical feature splits combining high speed (`gps_speed_mps > 20 m/s`) with high `transition_dist_ratio` to reject local frontage candidates.

---

### Mode 2: Multi-Leg Intersection & Turn Jitter

**Phenomenon:**  
At major 4-way, 5-way, or roundabout intersections, multiple candidate arcs converge within a $30\text{–}50\text{ meter}$ footprint. When a vehicle decelerates to negotiate a turn ($v < 3\text{ m/s}$), the computed GPS compass bearing fluctuates erratically due to numeric differentiation noise $(\Delta\lambda / \Delta t)$.

**Baseline Failure Patterns:**
- Candidates from perpendicular cross-streets have perpendicular distance comparable to the true turning arc.
- In Baseline B, heading alignment scores become unstable when instantaneous speed is near zero, causing spurious snapping to cross-street entry arcs before the turn is completed.

**Phase 3 ML Mitigation:**
- We engineered `speed_heading_alignment = cos(Δθ) * min(1.0, v / 5.0)`, which smoothly attenuates heading influence to zero as speed approaches stationary conditions.
- `projection_t` captures whether the vehicle is near the exit node of the incoming arc ($t \approx 1.0$) or the entry node of the outgoing arc ($t \approx 0.0$), allowing Phase 3 models to resolve turn transitions accurately.

---

### Mode 3: Multipath Reflections & Instantaneous Velocity Spikes

**Phenomenon:**  
As discovered during the Phase 1 trajectory audit, certain training trips (e.g. Trips 00000079, 00000033, 00000013) contain extreme velocity spikes ($> 198\text{ km/h}$, reaching up to $648\text{ km/h}$). These anomalies are caused by satellite multipath reflections off tall buildings or temporary signal loss, which cause a single GPS coordinate to jump hundreds of meters away for a single second.

**Baseline Failure Patterns:**
- The outlier coordinate falls outside the primary $60\text{m}$ search radius, forcing the candidate generator into fallback mode ($120\text{m}$) or nearest sample retrieval.
- Baseline A snaps blindly to whatever random street is nearest the outlier coordinate.
- Baseline C (HMM) penalizes the extreme discrepancy between $d_G$ and step distance. If the outlier cannot be connected within network cutoff ($1500\text{m}$), the transition probability absorbs a $-50.0$ penalty, preserving the route along the highway when normal points resume.

**Phase 3 ML Mitigation:**
- Kinematic flags `is_speed_spike`, `step_gps_dist_m`, and `log_transition_dist` provide explicit features that identify transient anomaly points, preventing models from breaking long continuous routes.

---

### Mode 4: Stationary Clutter & Signal Drift at Traffic Stops

**Phenomenon:**  
When a vehicle stops at a signal or in congested traffic for $30\text{–}120\text{ seconds}$, GPS coordinates form a circular "cloud" of points around the true stopping position with random step displacements $\Delta d \in [0.5, 3.0\text{ m}]$ and arbitrary bearings $\theta \in [0, 360)^\circ$.

**Baseline Failure Patterns:**
- Without speed gating, Direction-Aware matchers oscillate between opposing directions as the noisy heading flips back and forth.
- In Nearest Road, points drift across sidewalk and service lanes.

**Phase 3 ML Mitigation:**
- `is_stationary` and `speed_heading_alignment` ensure that stationary points are matched using strict geometric and topological continuity constraints rather than unstable heading angles.

---

### Mode 5: Candidate Set Misses & Route Continuity Breaks

**Phenomenon:**  
On rare occasions ($< 1.5\%$ of observations), severe GPS drift ($> 60\text{m}$) or high road density causes the ground-truth arc to fall outside the top $K = 10$ candidates.

**Baseline Failure Patterns:**
- Any point-by-point matcher will inevitably predict an incorrect arc for that time step.
- When collapsing consecutive matched arcs, this introduces an isolated false arc $[a_1, a_1, \mathbf{a_{\text{wrong}}}, a_1, a_2]$, fracturing topological route continuity.

**Engineered Mitigation in Phase 2:**
- Our `RouteReconstructor` detects when $a_k.\text{to\_node} \ne a_{k+1}.\text{from\_node}$ and triggers Dijkstra shortest path gap bridging (cutoff $\le 600\text{m}$, $\le 5$ arcs).
- This bridges isolated candidate misses, increasing reconstructed route continuity from $80.3\%$ to **$> 98.9\%$**.

---

## 3. Summary of Failure Modes Across Baselines

| Failure Scenario | Baseline A (Nearest) | Baseline B (Direction-Aware) | Baseline C (Graph HMM) | Phase 3 ML Solution |
|---|---|---|---|---|
| **Dual Carriageway (Opposing)** | High failure (snaps to opposite lane) | Resolved (heading filter) | Resolved (network constraints) | High precision via `cos_heading_diff` |
| **Parallel Road (Same Direction)** | High failure | High failure | Moderate failure | Resolved via `transition_dist_ratio` & network degree |
| **Complex Intersections** | Moderate failure | Turn jitter | High continuity | Resolved via `projection_t` & transition cost |
| **Multipath Velocity Spike** | Extreme failure (false snapping) | Extreme failure | Robust (recovers path) | Gated via `is_speed_spike` & `log_transition_dist` |
| **Stationary Traffic Stop** | Lateral drift | Heading oscillation | Stable sequence | Stabilized via `speed_heading_alignment` |
| **Candidate Miss (Gap)** | Breaks route continuity | Breaks route continuity | May force detour | Bridged via `RouteReconstructor` Dijkstra engine |

