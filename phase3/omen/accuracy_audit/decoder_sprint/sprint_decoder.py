"""Isolated GPS-only transition and emission experiments for PathSnap."""

from dataclasses import dataclass
import math

import networkx as nx
import numpy as np
from scipy.special import logsumexp


class DirectedDistances:
    """Cache one explicit superset per source and filter each requested cutoff."""

    def __init__(self, graph, max_cache_cutoff=1500.0):
        self.graph = graph
        self.max_cache_cutoff = max_cache_cutoff
        self.distances = {}

    def distance(self, source, target, cutoff):
        if cutoff < 0:
            return math.inf
        if source == target:
            return 0.0 if source in self.graph else math.inf
        if source not in self.graph or target not in self.graph:
            return math.inf
        saved_cutoff, distances = self.distances.get(source, (-1, None))
        if saved_cutoff < cutoff:
            saved_cutoff = max(cutoff, self.max_cache_cutoff)
            distances = nx.single_source_dijkstra_path_length(
                self.graph, source, cutoff=saved_cutoff, weight="weight",
            )
            self.distances[source] = saved_cutoff, distances
        distance = distances.get(target, math.inf)
        return distance if distance <= cutoff else math.inf


def routing_graph(graph):
    """Preserve parallel candidate IDs without changing original metric weights."""
    directed = graph.nx_graph
    if len(graph.arcs_dict) == directed.number_of_edges():
        return directed
    parallel = nx.MultiDiGraph()
    parallel.add_nodes_from(directed)
    for arc in graph.arcs_dict.values():
        u, v = arc.from_node, arc.to_node
        # All arcs with the same directed endpoints share the same projected length.
        parallel.add_edge(u, v, key=arc.arc_id, weight=directed[u][v]["weight"])
    return parallel


def projected_motion(network, prev, t0, current, t1, *, cutoff=1500.0, jitter=0.0):
    """Directed path from one clamped arc projection to the next."""
    a = min(1.0, max(0.0, t0)) * prev.length_m
    b = min(1.0, max(0.0, t1)) * current.length_m
    if prev.arc_id == current.arc_id and b - a >= -jitter:
        return max(0.0, b-a)
    arc_distance = prev.length_m-a+b
    return arc_distance + network.distance(prev.to_node, current.from_node, cutoff-arc_distance)


def viterbi(arcs, emission, transition, breaks=None):
    size, width = arcs.shape
    if not size:
        return np.empty(0, dtype=int)
    resets = np.zeros(size, dtype=bool)
    resets[0] = True
    if breaks is not None:
        resets[1:] = breaks
    has = (arcs >= 0).any(axis=1)
    resets[1:] |= ~has[:-1] | ~has[1:]
    previous = emission[0].copy()
    back = np.zeros((size, width), dtype=np.int16)
    best_ending = np.zeros(size, dtype=np.int16)
    best_ending[0] = previous.argmax()
    steps = np.arange(width)
    for i in range(1, size):
        if resets[i]:
            previous = emission[i].copy()
        else:
            candidates = previous[:, None] + transition[i-1]
            back[i] = candidates.argmax(axis=0)
            previous = candidates[back[i], steps] + emission[i]
        best_ending[i] = previous.argmax()
    output = np.full(size, -1, dtype=int)
    index = best_ending[-1]
    for i in range(size-1, -1, -1):
        output[i] = arcs[i, index]
        if i:
            index = best_ending[i-1] if resets[i] else back[i, index]
    return output


@dataclass(frozen=True)
class Config:
    transition: str = "legacy"
    emission: str = "gaussian"
    stationary: bool = False
    heading: float = 0.0
    beta: float = 40.0
    geometry_weight: float = .65
    temperature: float = 1.0
    sigma: float = 6.0


def gps_context(points, seconds):
    if len(points) != len(seconds):
        raise ValueError("GPS coordinates and timestamps must be aligned")
    count = len(points)
    idx = np.arange(count)
    gaps = np.diff(seconds) > 30
    blocks = np.r_[0, np.cumsum(gaps)]
    lo, hi = np.maximum(0, idx-4), np.minimum(count-1, idx+4)
    lo = np.where(blocks[lo] == blocks, lo, idx)
    hi = np.where(blocks[hi] == blocks, hi, idx)
    elapsed = seconds[hi] - seconds[lo]
    local = points[hi] - points[lo]
    speed = np.divide(np.linalg.norm(local, axis=1), elapsed, out=np.zeros(count), where=elapsed > 0)
    # Bound scatter in GPS space without consulting a road or a reference label.
    scatter = np.zeros(count)
    for offset in (-4, -3, -2, -1, 1, 2, 3, 4):
        neighbor = np.clip(idx+offset, 0, count-1)
        distance = np.linalg.norm(points[neighbor] - points, axis=1)
        scatter = np.maximum(scatter, np.where(blocks[neighbor] == blocks, distance, 0))
    stationary = (elapsed >= 4) & (speed < 1.0) & (scatter <= 8.0)
    bearing = np.rad2deg(np.arctan2(local[:, 0], local[:, 1])) % 360
    return stationary, gaps, speed, bearing


class TripDecoder:
    def __init__(self, graph, observation_count, step_distance, points, seconds, candidate_rows):
        self.graph = graph
        self.rows = candidate_rows
        self.steps = step_distance
        self.stationary, self.gaps, self.speed, self.bearing = gps_context(points, seconds)
        self.network = DirectedDistances(routing_graph(graph))
        obs = candidate_rows.obs_index.to_numpy(dtype=int)
        rank = candidate_rows.candidate_rank.to_numpy(dtype=int)-1
        self.index = obs, rank
        self.arcs = np.full((observation_count, 1+rank.max()), -1, dtype=int)
        self.arcs[self.index] = candidate_rows.candidate_arc_id
        self.position = np.zeros_like(self.arcs, dtype=float)
        self.position[self.index] = candidate_rows.projection_t
        self.distance = np.zeros_like(self.arcs, dtype=float)
        self.distance[self.index] = candidate_rows.distance_to_arc_m
        self.arc_heading = np.zeros_like(self.arcs, dtype=float)
        self.arc_heading[self.index] = candidate_rows.arc_bearing_deg
        self.length = np.zeros_like(self.arcs, dtype=float)
        self.length[self.index] = candidate_rows.arc_length_m
        self.transitions = {}

    def prime_legacy(self):
        by_obs = self.arcs
        for prev, current in zip(by_obs[:-1, 0], by_obs[1:]):
            if prev >= 0:
                for arc in current[current >= 0]:
                    self.graph.candidate_transition_distance(int(prev), int(arc), max_cutoff_m=2000.0)

    def _endpoint_distances(self, kind):
        graph = self.graph
        ids = self.arcs
        starts = np.full(self.arcs.shape, -1, dtype=int)
        ends = np.full(self.arcs.shape, -1, dtype=int)
        starts[self.index] = self.rows.arc_from_node
        ends[self.index] = self.rows.arc_to_node
        to = np.broadcast_to(starts[1:, None, :], (len(ids)-1, ids.shape[1], ids.shape[1]))
        origin = np.broadcast_to(ends[:-1, :, None], to.shape)
        valid = (origin >= 0) & (to >= 0)
        same = ids[:-1, :, None] == ids[1:, None, :]
        arc_a = self.position[:-1] * self.length[:-1]
        arc_b = self.position[1:] * self.length[1:]
        query = valid & ~same
        projected_kind = kind in {"projection", "projected_jitter"}
        if projected_kind:
            delta = arc_b[:, None, :] - arc_a[:, :, None]
            # A negative measured projection can represent a stationary latent position plus GPS jitter.
            tolerance = 8.0 if kind == "projected_jitter" else 1.0
            direct = same & (delta >= -tolerance)
            query |= valid & same & ~direct
        if kind == "legacy":
            lookup = graph.shortest_path_distance
        else:
            lookup = self.network.distance
        pair_codes = np.column_stack((origin[query], to[query]))
        unique, inverse = np.unique(pair_codes, axis=0, return_inverse=True)
        lengths = np.array([lookup(int(a), int(b), 1500.0) for a, b in unique])
        network = np.full(to.shape, np.inf)
        network[query] = lengths[inverse]
        if not projected_kind:
            return np.where(valid, np.where(same, 0.0, network+self.length[1:, None, :]), np.inf)
        projected = self.length[:-1, :, None] - arc_a[:, :, None] + network + arc_b[:, None, :]
        projected = np.where(projected <= 1500, projected, np.inf)
        return np.where(valid, np.where(direct, np.maximum(delta, 0), projected), np.inf)

    def potentials(self, config):
        key = config.transition, config.beta, config.stationary
        if key not in self.transitions:
            if config.transition not in self.transitions:
                self.transitions[config.transition] = self._endpoint_distances(config.transition)
            distance = self.transitions[config.transition]
            stationary = self.stationary[:-1] & self.stationary[1:]
            same = self.arcs[:-1, :, None] == self.arcs[1:, None, :]
            expected = self.steps[1:, None, None]
            if config.stationary:
                expected = np.where(stationary[:, None, None], 0, expected)
                hold = stationary[:, None, None] & same & (self.arcs[1:, None, :] >= 0)
                distance = np.where(hold, 0.0, distance)
            values = np.where(np.isfinite(distance),
                              -np.abs(distance-expected)/config.beta-math.log(config.beta), -50.0)
            if config.stationary:
                values -= 2.0 * (stationary[:, None, None] & ~same)
            self.transitions[key] = values
        return self.transitions[key]

    def decode(self, config, probabilities=None):
        valid = self.arcs >= 0
        geo = -.5*(self.distance/config.sigma)**2
        if config.emission in {"robust", "hybrid"}:
            geo = np.logaddexp(math.log(.97)+geo,
                               math.log(.03*config.sigma/30)-.5*(self.distance/30)**2)
        if config.stationary:
            geo = np.where(self.stationary[:, None], geo*.35, geo)
        if config.heading:
            reliable = np.clip((self.speed-.8)/3, 0, 1)
            delta = np.deg2rad(self.bearing[:, None]-self.arc_heading)
            geo += config.heading * reliable[:, None] * np.cos(delta)
        score = geo
        if config.emission in {"ml", "hybrid"}:
            ml = np.full(self.arcs.shape, -np.inf)
            p = np.clip(probabilities, 1e-6, 1-1e-6)
            if config.emission == "ml":
                ml[self.index] = np.log(np.maximum(probabilities, 1e-6))
                score = ml
            else:
                ml[self.index] = (np.log(p)-np.log1p(-p))/config.temperature
                geo = np.where(valid, geo, -np.inf)
                # Normalize within each observation before mixing the two log potentials.
                ml -= logsumexp(ml, axis=1, keepdims=True)
                geo -= logsumexp(geo, axis=1, keepdims=True)
                score = config.geometry_weight*geo + (1-config.geometry_weight)*ml
        return viterbi(self.arcs, np.where(valid, score, -np.inf), self.potentials(config),
                       self.gaps if config.stationary else None)
