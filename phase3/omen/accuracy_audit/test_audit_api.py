import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from audit_api import REPO_ROOT, create_app


class AuditApiTests(unittest.TestCase):
    def setUp(self):
        partition = Path(self.enterContext(TemporaryDirectory()))
        trip = partition / "00000001"
        trip.mkdir()
        (trip / "00000001.track").write_text("1.0001 2.0 0\n\n1.0009 2.0 5\n", encoding="utf-8")
        (trip / "00000001.nodes").write_text("1.0 2.0\n1.001 2.0\n", encoding="utf-8")
        (trip / "00000001.arcs").write_text("0 1\n", encoding="utf-8")
        incomplete = partition / "00000002"
        incomplete.mkdir()
        (incomplete / "00000002.track").touch()

        self.enterContext(patch.dict(os.environ, {
            "PATHSNAP_TRIP_ROOT": str(partition),
            "PATHSNAP_MODEL_ARCHIVE": str(REPO_ROOT / "phase3/omen/review1/experiment.zip"),
            "PATHSNAP_MODEL": "logistic",
        }))
        self.client = self.enterContext(TestClient(create_app()))

    def test_api_lists_complete_raw_trips_and_archive_models(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["available_models"], ["logistic", "xgboost"])
        self.assertEqual(health.json()["default_model"], "logistic")
        self.assertEqual(self.client.get("/trips").json(), {"trip_ids": ["00000001"]})

    def test_predicts_raw_trip_without_a_reference_route(self):
        response = self.client.post("/predict", json={"trip_id": "00000001"})
        self.assertEqual(response.status_code, 200, response.text)
        prediction = response.json()
        self.assertEqual(set(prediction), {
            "trip_id", "model", "observation_arc_ids", "route_arc_ids", "route_segments",
            "gps_coordinates", "continuity_pct", "timings_ms",
        })
        self.assertEqual(prediction["model"], "logistic")
        self.assertEqual(prediction["observation_arc_ids"], [0, 0])
        self.assertEqual(prediction["route_arc_ids"], [0])
        self.assertEqual(prediction["route_segments"], [{
            "arc_id": 0, "coordinates": [[1.0, 2.0], [1.001, 2.0]],
        }])
        self.assertEqual(prediction["gps_coordinates"], [[1.0001, 2.0], [1.0009, 2.0]])
        self.assertEqual(prediction["continuity_pct"], 100.0)
        timings = prediction["timings_ms"]
        self.assertGreater(timings["total"], 0)
        self.assertGreaterEqual(timings["total"], timings["hmm"])

    def test_predict_rejects_unavailable_input(self):
        for body, status_code in (
            ({"trip_id": "../000001"}, 422),
            ({"trip_id": "00000002"}, 404),
            ({"trip_id": "00000001", "model": "missing"}, 404),
        ):
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/predict", json=body).status_code, status_code)


if __name__ == "__main__":
    unittest.main()
