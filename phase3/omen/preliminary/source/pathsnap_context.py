"""GPS and directed-network context for the provisional candidate classifier."""

import numpy as np


CONTEXT_FEATURES = [
    "projection_fraction", "start_distance_m", "end_distance_m",
    "heading_window3_rad", "heading_window8_rad", "speed_window3_mps",
    "segment_distance_before3_m", "segment_distance_after3_m",
    "path_cost_delta_m", "incoming_cost_delta_m", "outgoing_cost_delta_m",
]
BRIDGE_FEATURES = [
    "bridge_path_cost_delta_m", "bridge_incoming_cost_delta_m", "bridge_outgoing_cost_delta_m",
]
MAX_GAP_S = 30


def motion_window(points, seconds, width):
    observations = np.arange(len(points))
    elapsed = np.diff(seconds, prepend=seconds[0])
    blocks = np.cumsum(elapsed > MAX_GAP_S)
    before = np.maximum(0, observations - width)
    after = np.minimum(len(points) - 1, observations + width)
    before = np.where(blocks[before] == blocks, before, observations)
    after = np.where(blocks[after] == blocks, after, observations)
    motion = points[after] - points[before]
    return before, after, motion, seconds[after] - seconds[before]


def heading_error(motion, vectors, lengths):
    heading = np.arctan2(motion[:, 1], motion[:, 0])
    direction = np.arctan2(vectors[:, 1], vectors[:, 0])
    error = np.abs((heading - direction + np.pi) % (2 * np.pi) - np.pi)
    error[(np.linalg.norm(motion, axis=1) < 1) | (lengths == 0)] = np.pi / 2
    return error


def distance_to_segment(points, starts, vectors, lengths):
    projection = np.sum((points - starts) * vectors, axis=1)
    fraction = np.divide(projection, lengths**2, out=np.zeros_like(projection), where=lengths > 0)
    offsets = points - starts - np.clip(fraction, 0, 1)[:, None] * vectors
    return np.linalg.norm(offsets, axis=1), fraction


def cost_messages(valid, emissions, transitions, breaks):
    """Min-cost evidence from each side of the trip, with resets at GPS gaps."""
    forward = np.zeros(valid.shape)
    backward = np.zeros(valid.shape)
    for i in range(1, len(valid)):
        if not breaks[i - 1]:
            previous = forward[i - 1] + emissions[i - 1]
            costs = np.min(previous[:, None] + transitions[i - 1], axis=0)
            forward[i, valid[i]] = costs[valid[i]] - costs.min()
    for i in range(len(valid) - 2, -1, -1):
        if not breaks[i]:
            following = backward[i + 1] + emissions[i + 1]
            costs = np.min(transitions[i] + following[None, :], axis=1)
            backward[i, valid[i]] = costs[valid[i]] - costs.min()
    combined = forward + backward + emissions
    best = np.where(valid.any(axis=1), combined.min(axis=1), 0)[:, None]
    return combined - best, forward, backward


def bridge_lengths(arcs, lengths, end_nodes, start_nodes):
    """Look up the shortest directed connector using up to two intervening arcs."""
    node_count = int(arcs.max()) + 1
    source, target = arcs.T
    by_source = np.argsort(source, kind="stable")
    boundaries = np.searchsorted(source[by_source], np.arange(node_count + 1))
    outgoing = np.diff(boundaries)[target]
    first = np.repeat(np.arange(len(arcs)), outgoing)
    offsets = np.repeat(np.cumsum(outgoing) - outgoing, outgoing)
    second = by_source[boundaries[target[first]] + np.arange(len(first)) - offsets]
    edge_keys = np.concatenate((source * node_count + target, source[first] * node_count + target[second]))
    edge_lengths = np.concatenate((lengths, lengths[first] + lengths[second]))
    order = np.argsort(edge_keys, kind="stable")
    keys, first_occur = np.unique(edge_keys[order], return_index=True)
    values = np.minimum.reduceat(edge_lengths[order], first_occur)
    queries = end_nodes * node_count + start_nodes
    idx = np.minimum(np.searchsorted(keys, queries), len(keys) - 1)
    return np.where(end_nodes == start_nodes, 0, np.where(keys[idx] == queries, values[idx], np.inf))


def add_context(rows, points, seconds, arcs, starts, vectors, lengths, width, *, bridges=False):
    obs = rows["observation_idx"].to_numpy()
    arc = rows["arc_id"].to_numpy()
    rank = rows["candidate_rank"].to_numpy() - 1
    names = [*CONTEXT_FEATURES, *BRIDGE_FEATURES] if bridges else CONTEXT_FEATURES
    if not len(obs):
        return rows.assign(**{name: np.array([], dtype=np.float32) for name in names})
    road_start, road_vector, length = starts[arc], vectors[arc], lengths[arc]
    road_end = road_start + road_vector
    distance, fraction = distance_to_segment(points[obs], road_start, road_vector, length)
    projected_distance = np.clip(fraction, 0, 1) * length
    before3, after3, motion3, elapsed3 = motion_window(points, seconds, 3)
    _, _, motion8, _ = motion_window(points, seconds, 8)
    heading3 = heading_error(motion3[obs], road_vector, length)
    heading8 = heading_error(motion8[obs], road_vector, length)
    displacement = np.linalg.norm(motion3, axis=1)
    speed = np.divide(displacement, elapsed3, out=np.zeros_like(displacement), where=elapsed3 > 0)
    before_distance, _ = distance_to_segment(points[before3[obs]], road_start, road_vector, length)
    after_distance, _ = distance_to_segment(points[after3[obs]], road_start, road_vector, length)

    dims = (len(points), width)
    ids = np.full(dims, -1, dtype=int)
    ids[obs, rank] = arc
    valid = ids >= 0
    offset = np.zeros(dims)
    offset[obs, rank] = projected_distance
    emissions = np.full(dims, np.inf)
    emissions[obs, rank] = distance + 3 * heading3
    valid_pair = valid[:-1, :, None] & valid[1:, None, :]
    same_arc = ids[:-1, :, None] == ids[1:, None, :]
    exit_node = arcs[ids[:-1], 1]
    entry_node = arcs[ids[1:], 0]
    end_positions = starts[ids[:-1]] + vectors[ids[:-1]]
    start_positions = starts[ids[1:]]
    connector = np.linalg.norm(end_positions[:, :, None] - start_positions[:, None, :], axis=-1)
    connected = exit_node[:, :, None] == entry_node[:, None, :]
    along = offset[1:, None, :] - offset[:-1, :, None]
    routed = lengths[ids[:-1], None] + along + connector
    travel = np.where(same_arc, np.abs(along), routed)
    observed_step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    transition_cost = 0.75 * np.abs(travel - observed_step[:, None, None])
    transition_cost += np.where(same_arc, np.maximum(0, -along), np.where(connected, 0, 18))
    transition_cost = np.where(valid_pair, np.minimum(transition_cost, 250), np.inf)
    breaks = (np.diff(seconds) > MAX_GAP_S) | ~valid_pair.any(axis=(1, 2))
    total, incoming, outgoing = cost_messages(valid, emissions, transition_cost, breaks)
    values = [
        fraction, np.linalg.norm(points[obs] - road_start, axis=1),
        np.linalg.norm(points[obs] - road_end, axis=1), heading3, heading8, speed[obs],
        before_distance, after_distance, total[obs, rank], incoming[obs, rank], outgoing[obs, rank],
    ]
    if bridges:
        exact = bridge_lengths(arcs, lengths, exit_node[:, :, None], entry_node[:, None, :])
        bridge_found = np.isfinite(exact)
        route_length = lengths[ids[:-1], None] + along + np.where(bridge_found, exact, connector)
        travel = np.where(same_arc, np.abs(along), route_length)
        cost = 0.75 * np.abs(travel - observed_step[:, None, None])
        cost += np.where(same_arc, np.maximum(0, -along), np.where(bridge_found, 0, 40))
        cost = np.where(valid_pair, np.minimum(cost, 250), np.inf)
        total, incoming, outgoing = cost_messages(valid, emissions, cost, breaks)
        values.extend([total[obs, rank], incoming[obs, rank], outgoing[obs, rank]])
    return rows.assign(**{name: value.astype(np.float32) for name, value in zip(names, values)})
