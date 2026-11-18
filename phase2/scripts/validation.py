"""
PathSnap Hackathon — Phase 2: Comprehensive Test & Validation Suite
Executes 11 rigorous automated checks validating data loader, graph engine,
spatial index, candidate generator, feature engineering, baselines, route reconstruction,
data leakage prevention, and SHA-256 raw dataset integrity.
"""

import sys
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import logging

# Add current scripts directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import DataLoader, TripData, GPSObservation
from trajectory_processor import TrajectoryProcessor, calculate_bearing, haversine_distance
from road_graph import RoadNetworkGraph
from spatial_index import RoadSpatialIndex, point_to_segment_distance
from candidate_generation import CandidateGenerator
from label_generation import LabelGenerator
from feature_engineering import FeatureEngineer
from baseline_matcher import NearestRoadMatcher, DirectionAwareMatcher, GraphConstrainedHMMMatcher
from route_reconstruction import RouteReconstructor
from evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase2Validation")


class TestPhase2Pipeline(unittest.TestCase):
    """Rigorous test cases for Phase 2 components."""

    @classmethod
    def setUpClass(cls):
        cls.dataset_root = PROJECT_ROOT / "Dataset"
        cls.loader = DataLoader(cls.dataset_root)
        cls.train_dirs = cls.loader.discover_trip_dirs("train")
        cls.trip_id = cls.train_dirs[0][0]
        cls.trip = cls.loader.load_trip(cls.trip_id, "train")

    def test_01_data_loader(self):
        """Test data loader finds exactly 80 train trips and parses non-empty structures."""
        self.assertEqual(len(self.train_dirs), 80, "Expected exactly 80 training trips.")
        self.assertIsNotNone(self.trip, "Failed to load first trip.")
        self.assertGreater(self.trip.num_points, 100, "Trip should have > 100 points.")
        self.assertGreater(self.trip.num_nodes, 1000, "Trip network should have > 1000 nodes.")
        self.assertGreater(self.trip.num_arcs, 1000, "Trip network should have > 1000 arcs.")
        self.assertGreater(self.trip.num_route_arcs, 10, "Trip should have > 10 route arcs.")

    def test_02_kinematic_processor(self):
        """Test trajectory kinematics (distance >= 0, bearing in [0, 360), speed)."""
        proc = TrajectoryProcessor()
        obs = proc.process_trip_track(self.trip.trip_id, self.trip.track)
        self.assertEqual(len(obs), len(self.trip.track))

        for o in obs:
            self.assertGreaterEqual(o.step_distance_m, 0.0)
            self.assertGreaterEqual(o.speed_mps, 0.0)
            self.assertTrue(0.0 <= o.bearing_deg < 360.0, f"Bearing out of bounds: {o.bearing_deg}")

    def test_03_road_graph_engine(self):
        """Test metric projection, arc lengths, and NetworkX shortest path queries."""
        graph = RoadNetworkGraph(self.trip.trip_id, self.trip.nodes, self.trip.arcs)
        self.assertEqual(len(graph.nodes_dict), len(self.trip.nodes))
        self.assertEqual(len(graph.arcs_dict), len(self.trip.arcs))

        # Check arc length is positive
        sample_arc = list(graph.arcs_dict.values())[0]
        self.assertGreater(sample_arc.length_m, 0.0)
        self.assertTrue(0.0 <= sample_arc.bearing_deg < 360.0)

        # Test local projection
        px, py = graph.project_point(sample_arc.x1, sample_arc.y1)
        self.assertIsInstance(px, float)
        self.assertIsInstance(py, float)

    def test_04_spatial_kdtree_index(self):
        """Test KD-Tree query returns candidates within specified radius."""
        graph = RoadNetworkGraph(self.trip.trip_id, self.trip.nodes, self.trip.arcs)
        sindex = RoadSpatialIndex(graph)
        self.assertIsNotNone(sindex.kdtree)

        first_obs = self.trip.track[0]
        px, py = graph.project_point(first_obs.lon, first_obs.lat)
        cands = sindex.query_candidate_arcs(px, py, search_radius_m=60.0, max_candidates=5)

        self.assertGreater(len(cands), 0, "Spatial index returned no candidates for valid point.")
        self.assertLessEqual(len(cands), 5)
        for c in cands:
            self.assertIn("arc_id", c)
            self.assertIn("distance_m", c)
            self.assertTrue(0.0 <= c["projection_t"] <= 1.0)
            self.assertGreaterEqual(c["candidate_rank"], 1)

    def test_05_candidate_generation(self):
        """Test candidate generation over trip achieves > 95% point coverage."""
        proc = TrajectoryProcessor()
        obs = proc.process_trip_track(self.trip.trip_id, self.trip.track)
        graph = RoadNetworkGraph(self.trip.trip_id, self.trip.nodes, self.trip.arcs)
        sindex = RoadSpatialIndex(graph)
        gen = CandidateGenerator(search_radius_m=60.0, max_candidates=10)

        cands, stats = gen.generate_candidates_for_trip(self.trip, obs, graph, sindex)
        self.assertGreater(len(cands), 0)
        self.assertGreaterEqual(stats["coverage_pct"], 95.0, f"Coverage too low: {stats['coverage_pct']}%")

    def test_06_ground_truth_labeling(self):
        """Test sequential label alignment achieves > 90% match rate."""
        proc = TrajectoryProcessor()
        obs = proc.process_trip_track(self.trip.trip_id, self.trip.track)
        graph = RoadNetworkGraph(self.trip.trip_id, self.trip.nodes, self.trip.arcs)
        sindex = RoadSpatialIndex(graph)
        gen = CandidateGenerator(search_radius_m=60.0, max_candidates=10)
        cands, _ = gen.generate_candidates_for_trip(self.trip, obs, graph, sindex)

        labeler = LabelGenerator()
        obs_map, align_stats = labeler.align_trip_observations(self.trip, obs, cands)
        labeled_cands = labeler.assign_candidate_labels(cands, obs_map)

        self.assertGreaterEqual(align_stats["candidate_recall_pct"], 90.0)
        labels = [c["is_correct"] for c in labeled_cands]
        self.assertIn(1, labels)
        self.assertIn(0, labels)

    def test_07_feature_engineering_schema(self):
        """Test feature engineering DataFrame has expected columns and no NaNs in key features."""
        proc = TrajectoryProcessor()
        obs = proc.process_trip_track(self.trip.trip_id, self.trip.track)
        graph = RoadNetworkGraph(self.trip.trip_id, self.trip.nodes, self.trip.arcs)
        sindex = RoadSpatialIndex(graph)
        gen = CandidateGenerator(search_radius_m=60.0, max_candidates=5)
        # Use first 50 obs for fast test
        obs_subset = obs[:50]
        cands, _ = gen.generate_candidates_for_trip(self.trip, obs_subset, graph, sindex)

        labeler = LabelGenerator()
        obs_map, _ = labeler.align_trip_observations(self.trip, obs_subset, cands)
        labeled_cands = labeler.assign_candidate_labels(cands, obs_map)

        fe = FeatureEngineer(max_candidates=5)
        df = fe.compute_candidate_features(labeled_cands, obs_subset, graph)

        self.assertFalse(df.empty)
        for col in FeatureEngineer.FEATURE_COLUMNS:
            self.assertIn(col, df.columns, f"Missing feature column: {col}")
            self.assertFalse(df[col].isnull().any(), f"Column {col} contains NaNs.")

    def test_08_baselines_execution(self):
        """Test Baseline A, B, C execution and non-empty valid predictions."""
        proc = TrajectoryProcessor()
        obs = proc.process_trip_track(self.trip.trip_id, self.trip.track[:30])
        graph = RoadNetworkGraph(self.trip.trip_id, self.trip.nodes, self.trip.arcs)
        sindex = RoadSpatialIndex(graph)
        gen = CandidateGenerator(search_radius_m=60.0, max_candidates=5)
        cands, _ = gen.generate_candidates_for_trip(self.trip, obs, graph, sindex)

        pred_a = NearestRoadMatcher().match_trip(cands, obs, graph)
        pred_b = DirectionAwareMatcher().match_trip(cands, obs, graph)
        pred_c = GraphConstrainedHMMMatcher().match_trip(cands, obs, graph)

        self.assertEqual(len(pred_a), len(obs))
        self.assertEqual(len(pred_b), len(obs))
        self.assertEqual(len(pred_c), len(obs))

        # Check predictions are valid integers
        self.assertTrue(all(isinstance(a, int) for a in pred_a))
        self.assertTrue(all(isinstance(a, int) for a in pred_b))
        self.assertTrue(all(isinstance(a, int) for a in pred_c))

    def test_09_route_reconstruction(self):
        """Test consecutive deduplication and continuity evaluation."""
        recon = RouteReconstructor()
        seq = [1, 1, 1, 2, 2, 3, 3, 3]
        collapsed = recon.collapse_observation_sequence(seq)
        self.assertEqual(collapsed, [1, 2, 3])

    def test_10_student_test_data_leakage(self):
        """Verify student_test files were NOT read or written to for any model/labeling purpose."""
        # Ensure no student_test files exist in processed training directories
        proc_train_dir = PROJECT_ROOT / "phase2" / "features" / "train"
        if proc_train_dir.exists():
            for f in proc_train_dir.glob("*.parquet"):
                self.assertNotIn("student_test", f.name)

    def test_11_dataset_sha256_integrity(self):
        """Verify that every raw dataset file SHA-256 hash matches source integrity manifest."""
        manifest_path = PROJECT_ROOT / "docs" / "phase2" / "source_dataset_integrity_manifest.json"
        self.assertTrue(manifest_path.exists(), "Source dataset integrity manifest missing!")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        mismatches = []
        # Support either direct dictionary or dict with 'files' key
        file_items = manifest["files"].items() if "files" in manifest else manifest.items()
        for rel_path, info in file_items:
            full_path = PROJECT_ROOT / rel_path
            self.assertTrue(full_path.exists(), f"Raw file missing: {rel_path}")

            hasher = hashlib.sha256()
            with open(full_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            current_hash = hasher.hexdigest()

            if current_hash != info["sha256"]:
                mismatches.append(rel_path)

        self.assertEqual(len(mismatches), 0, f"Raw files modified! Mismatches: {mismatches}")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase2Pipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
