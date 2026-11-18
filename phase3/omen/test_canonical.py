"""Review 1 tests for observation metrics, artifact reload, and route-free features."""

import io
import json
import os
from pathlib import Path
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd

from model_registry import load_classifier
from train_canonical import feature_contract, score


ROOT = Path(os.environ.get("PATHSNAP_CANONICAL_REPO", Path(__file__).resolve().parents[2]))
ARTIFACT = Path(os.environ.get("PATHSNAP_REVIEW1_ARCHIVE", Path(__file__).parent / "review1" / "experiment.zip"))


class Review1Tests(unittest.TestCase):
    def test_top1_counts_absent_and_no_positive_observations(self):
        rows = pd.DataFrame({
            "trip_id": ["x"] * 4, "obs_index": [0, 0, 1, 2], "candidate_arc_id": [8, 3, 2, 7],
            "true_arc_id": [3, 3, 2, -1], "is_correct": [0, 1, 1, 0], "candidate_rank": [2, 1, 1, 1],
        })
        observations = pd.DataFrame({
            "has_candidates": [True, True, True, False], "has_positive": [True, True, False, False],
        })
        metrics, _ = score(rows, observations, np.array([.3, .3, .8, .1]))
        self.assertEqual(metrics["top1_conditional"], 1.0)
        self.assertEqual(metrics["top1_coverage_aware"], .5)
        self.assertEqual(metrics["observations_without_positive"], 2)
        self.assertEqual(metrics["observations_without_candidates"], 1)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + .8) / 2)

    def test_saved_probabilities_reproduce(self):
        features = feature_contract(ROOT)
        names = ("logistic", "xgboost")
        with ZipFile(ARTIFACT) as archive:
            audit = json.loads(archive.read("audit.json"))
            self.assertEqual(audit["features"], features)
            self.assertFalse(set(audit["train"]["trip_ids"]) & set(audit["validation"]["trip_ids"]))
            self.assertEqual((audit["train"]["parquet_files"], audit["validation"]["parquet_files"]), (64, 16))
            trip = audit["validation"]["trip_ids"][0]
            source = pd.read_parquet(ROOT / "phase2" / "features" / "val" / f"{trip}_features.parquet")
            for name in names:
                with self.subTest(model=name):
                    model, columns = load_classifier(name, dict.fromkeys(names, ARTIFACT), features)
                    scores = pd.read_parquet(io.BytesIO(archive.read(f"{name}/validation_predictions.parquet")))
                    scores = scores[scores.trip_id == trip].head(256)
                    joined = scores.merge(source, on=["trip_id", "obs_index", "candidate_arc_id"],
                                          suffixes=("_saved", ""), validate="one_to_one")
                    self.assertEqual(len(joined), 256)
                    predicted = model.predict_proba(joined[columns].astype(np.float32))[:, 1]
                    np.testing.assert_allclose(predicted, joined["probability"], rtol=1e-6)

    def test_phase2_features_can_be_generated_with_no_route(self):
        features = feature_contract(ROOT)
        from candidate_generation import CandidateGenerator
        from data_loader import DataLoader, TripData
        from feature_engineering import FeatureEngineer
        from road_graph import RoadNetworkGraph
        from spatial_index import RoadSpatialIndex
        from trajectory_processor import TrajectoryProcessor

        trip_id = "00000005"
        dataset = ROOT / "Dataset"
        base = dataset / "train" / trip_id / trip_id
        loader = DataLoader(dataset)
        raw = TripData(trip_id, "train", loader.load_track(base.with_suffix(".track")),
                       loader.load_nodes(base.with_suffix(".nodes")), loader.load_arcs(base.with_suffix(".arcs")), [])
        obs = TrajectoryProcessor(max_speed_threshold_mps=55.0, gap_threshold_seconds=300.0).process_trip_track(
            trip_id, raw.track,
        )
        graph = RoadNetworkGraph(trip_id, raw.nodes, raw.arcs)
        candidates, _ = CandidateGenerator().generate_candidates_for_trip(raw, obs, graph, RoadSpatialIndex(graph))
        prediction_rows = FeatureEngineer().compute_candidate_features(candidates, obs, graph)
        archived_rows = pd.read_parquet(ROOT / "phase2/features/val" / f"{trip_id}_features.parquet")
        columns = ["obs_index", "candidate_arc_id", *features]
        self.assertNotIn("true_arc_id", prediction_rows)
        self.assertNotIn("is_correct", prediction_rows)
        pd.testing.assert_frame_equal(prediction_rows[columns].reset_index(drop=True),
                                      archived_rows[columns].reset_index(drop=True), check_dtype=False)


if __name__ == "__main__":
    unittest.main()
