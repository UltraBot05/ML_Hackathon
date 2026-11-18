import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import httpx
import numpy as np
import pandas as pd

from pathsnap_api import create_app, load_classifier
from pathsnap_context import BRIDGE_FEATURES, CONTEXT_FEATURES, bridge_lengths
from pathsnap_quickstart import candidate_pairs, load_trip_features, ordered_proxy


class CandidateTests(unittest.TestCase):
    def test_long_segments_and_geometric_ties(self):
        points = np.array([[2., 1.], [8., 1.], [200., 50.]])
        starts = np.array([[0., 0.], [5., 10.], [0., 2.]])
        vectors = np.array([[1000., 0.], [1., 0.], [10., 0.]])
        obs, arc, distance, rank = candidate_pairs(points, starts, vectors, radius=3, top_k=2)
        self.assertEqual(list(zip(obs, arc, rank)), [(0, 0, 1), (0, 2, 2), (1, 0, 1), (1, 2, 2)])
        np.testing.assert_allclose(distance, 1)
        proxy, ambiguous = ordered_proxy(points[:2], starts, vectors, np.array([2, 0]))
        self.assertEqual(proxy.tolist(), [2, 2])
        self.assertEqual(ambiguous.tolist(), [True, True])

    def test_context_preserves_baseline_candidates(self):
        root = Path(os.environ.get("PATHSNAP_DATA_ROOT", "Student folder/Student folder"))
        trip = root / "train" / "00000003"
        plain = load_trip_features(trip)[3]
        context = load_trip_features(trip, context=True)[3]
        pd.testing.assert_frame_equal(context[plain.columns], plain)
        self.assertTrue(np.isfinite(context[CONTEXT_FEATURES].to_numpy()).all())

    def test_bridge_uses_directed_paths_of_at_most_two_arcs(self):
        arcs = np.array([[0, 1], [1, 2], [0, 2], [2, 3], [4, 0]])
        distances = bridge_lengths(
            arcs, np.array([2., 3., 9., 4., 1.]), np.array([0, 2, 0, 1, 3, 4]),
            np.array([2, 0, 3, 3, 3, 3]),
        )
        np.testing.assert_allclose(distances, [5., np.inf, 13., 7., 0., np.inf])


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_trips_and_unavailable_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trips = root / "evaluator_test"
            trips.mkdir()
            (trips / "00000001").mkdir()
            (trips / "00000001" / "00000001.track").touch()
            (trips / "00000002").mkdir()
            with patch.dict(os.environ, {"PATHSNAP_DATA_ROOT": str(root), "PATHSNAP_SPLIT": "evaluator_test"}):
                transport = httpx.ASGITransport(app=create_app())
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    health = await client.get("/health")
                    self.assertEqual(health.status_code, 200)
                    self.assertFalse(health.json()["prediction_ready"])
                    self.assertEqual(len(health.json()["models"]), 6)
                    trips_response = await client.get("/trips")
                    self.assertEqual(trips_response.json(), {"trip_ids": ["00000001"]})
                    for trip_id, status in [("00000001", 503), ("00000002", 404), ("../train", 422)]:
                        response = await client.post("/predict", json={"trip_id": trip_id})
                        self.assertEqual(response.status_code, status)


class ModelArtifactTests(unittest.TestCase):
    def test_reload_and_predict(self):
        root = Path(os.environ.get("PATHSNAP_DATA_ROOT", "Student folder/Student folder"))
        directories = os.environ.get(
            "PATHSNAP_ARTIFACT_DIRS", "pathsnap_preliminary;pathsnap_full_proxy;pathsnap_context_proxy",
        ).split(";")
        for directory in map(Path, directories):
            for name in ("logistic", "xgboost"):
                with self.subTest(archive=directory / f"{name}.zip"):
                    path = directory / f"{name}.zip"
                    model, features = load_classifier(path)
                    with ZipFile(path) as archive:
                        report = json.loads(archive.read("results.json"))
                        with archive.open("validation_probabilities.csv") as data:
                            expected = pd.read_csv(data, dtype={"trip_id": str}, nrows=25)
                    partition = "student_test" if report.get("student_test_path") else "train"
                    keys = ["trip_id", "observation_idx", "arc_id"]
                    rows = pd.concat([
                        load_trip_features(
                            root / partition / trip, context=bool(set(features) & set(CONTEXT_FEATURES)),
                            bridges=bool(set(features) & set(BRIDGE_FEATURES)),
                            radius=report["radius_m"], top_k=report["top_k"],
                        )[3]
                        for trip in expected["trip_id"].unique()
                    ])
                    self.assertEqual(features, report["features"])
                    candidates = expected.merge(rows, on=keys, validate="one_to_one")
                    self.assertEqual(len(expected), len(candidates))
                    predicted = model.predict_proba(candidates[features])[:, 1]
                    np.testing.assert_allclose(predicted, expected["probability"], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
