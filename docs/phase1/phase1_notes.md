# PathSnap Hackathon — Phase 1 Engineering & Design Notes

---

## 1. Coordinate System & Geodesic Distance Formulations

In `Dataset Exploration.pdf` Section 5 (*Geographical Data Considerations*), the organizers state:
> "Longitude and latitude are angular coordinates. Ordinary Euclidean distance in degrees is generally not an appropriate measure of physical distance. Choose and justify an appropriate projected coordinate system or geodesic distance calculation."

### Geodesic Calculation (Phase 1 Implemented)
For Phase 1 trajectory auditing, the **Haversine formula** on a spherical Earth of radius $R = 6,371,000\text{ meters}$ was selected:
$$\Delta\sigma = 2 \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1\cos\phi_2\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$
$$d = R \cdot \Delta\sigma$$

- **Accuracy:** For point-to-point steps at 1-Hz sampling ($\sim 10\text{–}30\text{ meters}$), spherical curvature errors are $< 0.1\%$, vastly smaller than raw GPS measurement error ($\pm 5\text{–}15\text{ meters}$).
- **Performance:** Computes in $O(1)$ per segment, enabling the 208,417-observation dataset to be processed in $< 1\text{ second}$.

### Projection Recommendations for Phase 2 (Preprocessing)
When projecting GPS observations onto road arcs in Phase 2:
1. **Local Equirectangular / UTM Projection:**
   $$x = R \cdot (\lambda - \lambda_0) \cdot \cos(\phi_0), \quad y = R \cdot (\phi - \phi_0)$$
   where $(\phi_0, \lambda_0)$ is the bounding box center of the trip's subnetwork.
2. **Computational Benefit:** Allows standard 2D vector dot-products for point-to-line segment projections and KD-Tree bounding box queries.

---

## 2. File Schema and Representation

| File Ext | Primary Entity | Raw Columns | Inferred Semantics | Line Number Indexing |
|:---:|:---|:---:|:---|:---:|
| `.track` | Observation | `lon lat timestamp` | Angular coordinates in degrees; timestamp in seconds | 1-indexed in file |
| `.nodes` | Intersection / Waypoint | `lon lat` | Angular road network junction coordinates | 0-indexed: line $k \implies \text{node\_id} = k$ |
| `.arcs` | Directed Road Edge | `from_node to_node` | Directed connection from origin node to destination node | 0-indexed: line $k \implies \text{arc\_id} = k$ |
| `.route` | Traversed Path | `arc_id` | Chronological ground truth sequence of traversed arcs | 0-indexed line sequence |

---

## 3. Anomaly Analysis Summary & Classification

Raw records inspected:
1. **Multipath Velocity Spikes (e.g. Trip `00000079`):**
   - $t_1 = 132.0, \text{lon}_1 = 7.61803, \text{lat}_1 = 45.05927$
   - $t_2 = 133.0, \text{lon}_2 = 7.61623, \text{lat}_2 = 45.06038$
   - Distance: $180.14\text{ m}$; $\Delta t = 1.0\text{ s} \implies \text{speed} = 180.14\text{ m/s} = 648.50\text{ km/h}$.
   - At $t_3 = 134.0$, coordinates return to $(7.61809, 45.05932)$.
   - Proves a single reflected coordinate jump.
2. **Multi-Session Pauses (e.g. Trip `00000090`):**
   - Point 1,048: $t = 165,302\text{ s}$
   - Point 1,049: $t = 318,366\text{ s}$
   - $\Delta t = 153,064\text{ s}$ (42.5 hours); distance $= 38.6\text{ m}$.
   - Vehicle stopped over weekend while logger paused. Moving average speed is **32.00 km/h**, whereas gross speed was artificially distorted to **0.20 km/h**.

