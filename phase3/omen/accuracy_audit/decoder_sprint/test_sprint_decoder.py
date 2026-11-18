from types import SimpleNamespace
import math
import os
from pathlib import Path
import tempfile
import unittest

import networkx as nx
import numpy as np

from sprint_decoder import Config, DirectedDistances, projected_motion, routing_graph, viterbi


class RoutingTests(unittest.TestCase):
    def test_every_cutoff_in_both_query_orders(self):
        graph = nx.DiGraph()
        graph.add_weighted_edges_from([(0, 1, 5), (1, 2, 8), (3, 0, 2)], weight="weight")
        for cutoffs in ((7, 20), (20, 7)):
            distances = DirectedDistances(graph, max_cache_cutoff=20)
            for cutoff in cutoffs:
                self.assertEqual(distances.distance(0, 1, cutoff), 5)
                self.assertEqual(distances.distance(0, 2, cutoff), 13 if cutoff >= 13 else math.inf)
                self.assertEqual(distances.distance(0, 3, cutoff), math.inf)

    def test_partial_arcs_and_unreachable_reverse(self):
        graph = nx.MultiDiGraph()
        graph.add_weighted_edges_from([(0, 1, 10), (1, 2, 7), (1, 2, 5), (2, 3, 20)])
        network = DirectedDistances(graph)
        a = SimpleNamespace(arc_id=7, from_node=0, to_node=1, length_m=10)
        b = SimpleNamespace(arc_id=8, from_node=2, to_node=3, length_m=20)
        self.assertAlmostEqual(projected_motion(network, a, .25, a, .75), 5)
        self.assertAlmostEqual(projected_motion(network, a, .8, b, .25), 12)
        self.assertEqual(projected_motion(network, a, .8, a, .2), math.inf)
        graph.add_weighted_edges_from([(1, 0, 10)])
        self.assertEqual(projected_motion(DirectedDistances(graph), a, .8, a, .2), 14)

    def test_parallel_candidate_ids_survive_corrected_graph_construction(self):
        base = nx.DiGraph()
        base.add_weighted_edges_from([(0, 1, 10), (1, 2, 5)])
        arcs = [SimpleNamespace(arc_id=i, from_node=u, to_node=v)
                for i, (u, v) in enumerate(((0, 1), (0, 1), (1, 2)))]
        graph = routing_graph(SimpleNamespace(nx_graph=base, arcs_dict={a.arc_id: a for a in arcs}))
        self.assertEqual(set(graph[0][1]), {0, 1})
        self.assertEqual(DirectedDistances(graph).distance(0, 2, 15), 15)

    def test_gap_restarts_without_changing_the_number_of_observations(self):
        arcs = np.array([[10, 11], [10, 11], [20, 21]])
        emissions = np.array([[3., 0.], [2., 0.], [0., 9.]])
        transitions = np.zeros((2, 2, 2))
        transitions[1, :, 1] = -50
        self.assertEqual(viterbi(arcs, emissions, transitions).tolist(), [10, 10, 20])
        self.assertEqual(viterbi(arcs, emissions, transitions, np.array([False, True])).tolist(), [10, 10, 21])

    @unittest.skipUnless(os.environ.get("PATHSNAP_REPO"), "Set PATHSNAP_REPO for raw-trip verification.")
    def test_raw_inference_does_not_need_a_reference_file(self):
        from run_decoder_sprint import dependencies, prepare

        components = dependencies(Path(os.environ["PATHSNAP_REPO"]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trip = root / "00000001"
            trip.mkdir()
            (trip / "00000001.track").write_text("1.0001 2.0 0\n1.0009 2.0 5\n")
            (trip / "00000001.nodes").write_text("1.0 2.0\n1.001 2.0\n")
            (trip / "00000001.arcs").write_text("0 1\n")
            decoder, *_ = prepare(root, trip.name, components)
            result = decoder.decode(Config(transition="projection"))
            np.testing.assert_array_equal(result, [0, 0])


if __name__ == "__main__":
    unittest.main()
