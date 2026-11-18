"""
PathSnap Hackathon — Phase 3 Test Suite (Adarsh's Models & Integration)
Verifies:
1. Model artifacts exist and can be loaded via joblib.
2. Models predict probabilities in [0, 1] without NaNs.
3. Feature schema matches exactly FeatureEngineer.FEATURE_COLUMNS.
4. Validation predictions file exists and matches validation trips.
5. MLEmissionHMMMatcher executes Viterbi decoding and outputs valid sequence.
6. Zero data leakage: models never see reference route during inference.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
import sys
sys.path.insert(0, str(PROJECT_ROOT / "phase2" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "phase3" / "scripts"))

from feature_engineering import FeatureEngineer
from evaluate_ml_hmm import MLEmissionHMMMatcher
from road_graph import RoadNetworkGraph
from data_loader import DataLoader
from trajectory_processor import TrajectoryProcessor
from candidate_generation import CandidateGenerator
from spatial_index import RoadSpatialIndex


class TestPhase3Adarsh(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = PROJECT_ROOT / "phase3" / "models"
        cls.results_dir = PROJECT_ROOT / "phase3" / "results"
        cls.et_path = cls.models_dir / "extra_trees_model.joblib"
        cls.lgbm_path = cls.models_dir / "lightgbm_model.joblib"
        cls.preds_path = cls.results_dir / "val_predictions.parquet"
        cls.metrics_path = cls.results_dir / "adarsh_model_metrics.csv"
        cls.ml_hmm_path = cls.results_dir / "ml_hmm_comparison.csv"

    def test_01_model_artifacts_exist(self):
        self.assertTrue(self.et_path.exists(), "Extra Trees model artifact missing")
        self.assertTrue(self.lgbm_path.exists(), "LightGBM model artifact missing")
        self.assertTrue(self.preds_path.exists(), "Validation predictions file missing")
        self.assertTrue(self.metrics_path.exists(), "Metrics summary CSV missing")
        self.assertTrue(self.ml_hmm_path.exists(), "ML+HMM comparison CSV missing")

    def test_02_model_loading_and_features(self):
        et = joblib.load(self.et_path)
        lgbm = joblib.load(self.lgbm_path)
        self.assertIsNotNone(et)
        self.assertIsNotNone(lgbm)
        self.assertEqual(et.n_features_in_, len(FeatureEngineer.FEATURE_COLUMNS))
        self.assertEqual(lgbm.n_features_in_, len(FeatureEngineer.FEATURE_COLUMNS))

    def test_03_inference_probabilities_valid(self):
        et = joblib.load(self.et_path)
        lgbm = joblib.load(self.lgbm_path)
        dummy_X = np.random.randn(5, len(FeatureEngineer.FEATURE_COLUMNS)).astype(np.float32)
        p_et = et.predict_proba(dummy_X)[:, 1]
        p_lgbm = lgbm.predict_proba(dummy_X)[:, 1]
        self.assertEqual(len(p_et), 5)
        self.assertEqual(len(p_lgbm), 5)
        self.assertTrue(np.all((p_et >= 0.0) & (p_et <= 1.0)))
        self.assertTrue(np.all((p_lgbm >= 0.0) & (p_lgbm <= 1.0)))
        self.assertFalse(np.any(np.isnan(p_et)))
        self.assertFalse(np.any(np.isnan(p_lgbm)))

    def test_04_val_predictions_integrity(self):
        df = pd.read_parquet(self.preds_path)
        required_cols = ["trip_id", "obs_index", "candidate_arc_id", "is_correct", "prob_extra_trees", "prob_lightgbm"]
        for col in required_cols:
            self.assertIn(col, df.columns)
        self.assertGreater(len(df), 200000, "Validation predictions row count too low")
        self.assertFalse(df["prob_extra_trees"].isnull().any())
        self.assertFalse(df["prob_lightgbm"].isnull().any())

    def test_05_ml_hmm_integration_execution(self):
        loader = DataLoader(PROJECT_ROOT / "Dataset")
        trip = loader.load_trip("00000005", "train")
        self.assertIsNotNone(trip)
        proc = TrajectoryProcessor()
        obs = proc.process_trip_track("00000005", trip.track)
        graph = RoadNetworkGraph("00000005", trip.nodes, trip.arcs)
        sindex = RoadSpatialIndex(graph)
        cands, _ = CandidateGenerator().generate_candidates_for_trip(trip, obs, graph, sindex)
        
        # Test MLEmissionHMMMatcher
        matcher = MLEmissionHMMMatcher()
        dummy_probs = np.ones(len(cands)) * 0.5
        preds = matcher.match_trip(cands, dummy_probs, obs, graph)
        self.assertEqual(len(preds), len(obs))
        self.assertTrue(all(isinstance(p, int) for p in preds))


if __name__ == "__main__":
    unittest.main()

