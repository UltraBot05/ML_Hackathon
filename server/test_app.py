import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from . import app as adapter


class AdapterTests(unittest.TestCase):
    def setUp(self):
        root = self.enterContext(tempfile.TemporaryDirectory())
        trip_dir = Path(root) / "00000007"
        trip_dir.mkdir()
        for ext, contents in {
            "nodes": "10 20\n11 21\n12 22\n",
            "arcs": "1 0\n0 2\n",
            "track": "11 21 1\n12 22 2\n",
        }.items():
            (trip_dir / f"00000007.{ext}").write_text(contents, encoding="utf-8")
        (trip_dir / "00000007.route").mkdir()
        self.enterContext(patch.dict(os.environ, {
            "PATHSNAP_TRIP_ROOT": root,
            "PATHSNAP_PARTITION": "student_test",
            "PATHSNAP_REFERENCE_URL": "http://127.0.0.1:8101",
            "PATHSNAP_REFERENCE_TIMEOUT_SECONDS": "120",
        }))
        self.upstream = self.enterContext(patch.object(adapter, "HTTPConnection"))
        self.connection = self.upstream.return_value
        self.response = self.connection.getresponse.return_value
        self.client = self.enterContext(TestClient(adapter.app))
        self.prediction = {
            "trip_id": "00000007",
            "partition": "student_test",
            "reconstructed_route": [0, 1, 0],
            "n_obs": 2,
            "inference_time_s": 0.125,
            "metrics": {"route_arc_f1_pct": 72.5},
        }

    def reply(self, payload, status=200):
        self.response.status = status
        self.response.read.return_value = json.dumps(payload).encode()

    def predict(self, **overrides):
        payload = {"trip_id": "00000007", "model": "best", "use_hmm": True}
        return self.client.post("/predict", json=payload | overrides)

    def test_prediction_uses_directed_geometry_and_real_reference_output(self):
        self.reply(self.prediction)

        response = self.predict()

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["trip_id"], "00000007")
        self.assertEqual(result["engine"], "ta-reference-lightgbm-24f")
        self.assertEqual(result["feature_schema"], "ta-reference-24f")
        self.assertEqual(result["n_obs"], 2)
        self.assertEqual(result["predicted_arc_ids"], [0, 1, 0])
        self.assertEqual(result["route_segments"], [
            [[11, 21], [10, 20]], [[10, 20], [12, 22]], [[11, 21], [10, 20]],
        ])
        self.assertEqual(result["gps_coordinates"], [[11, 21], [12, 22]])
        self.assertEqual(result["inference_ms"], 125)
        self.assertEqual(result["metrics"], {"route_arc_f1_pct": 72.5})
        self.assertNotIn("observation_arc_ids", result)
        self.assertTrue(result["warnings"])
        args, options = self.connection.request.call_args
        self.assertEqual(args, ("POST", "/api/predict"))
        self.assertEqual(json.loads(options["body"]), {
            "partition": "student_test", "trip_id": "00000007", "generate_map": False,
        })

    def test_predict_returns_observation_ids_only_if_supplied(self):
        self.reply(self.prediction | {"observation_arc_ids": [0, -1]})

        response = self.predict()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["observation_arc_ids"], [0, -1])

    def test_prediction_rejects_invalid_reference_geometry_index(self):
        self.reply(self.prediction | {"reconstructed_route": [-1]})
        self.assertEqual(self.predict().status_code, 502)

    def test_prediction_rejects_a_mismatched_reference_observation_count(self):
        self.reply(self.prediction | {"n_obs": 3})
        self.assertEqual(self.predict().status_code, 502)

    def test_predict_rejects_requests_the_engine_cannot_run(self):
        self.assertEqual(self.predict(use_hmm=False).status_code, 422)
        self.assertEqual(self.predict(model="other").status_code, 422)
        self.assertEqual(self.predict(trip_id="../outside").status_code, 422)
        self.connection.request.assert_not_called()

    def test_reference_timeout_is_reported_as_gateway_timeout(self):
        self.connection.getresponse.side_effect = TimeoutError
        self.assertEqual(self.predict().status_code, 504)

    def test_reference_must_be_loopback(self):
        with patch.dict(os.environ, {"PATHSNAP_REFERENCE_URL": "http://example.com"}):
            self.assertEqual(self.client.get("/health").status_code, 503)
        self.upstream.assert_not_called()

    def test_reference_accepts_an_ipv6_loopback_origin_without_a_port(self):
        self.reply({"status": "ok", "partitions": ["student_test"]})
        with patch.dict(os.environ, {"PATHSNAP_REFERENCE_URL": "http://[::1]"}):
            self.assertEqual(self.client.get("/health").status_code, 200)
        self.upstream.assert_called_once_with("::1", 80, timeout=120)

    def test_health_models_and_trips_report_the_active_engine(self):
        self.reply({"status": "ok", "partitions": ["student_test"]})
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["engine"], "ta-reference-lightgbm-24f")
        self.assertEqual(health.json()["partition"], "student_test")

        models = self.client.get("/models").json()
        self.assertEqual(models["default_model"], "best")
        self.assertEqual(models["models"][0]["feature_count"], 24)
        self.assertEqual(models["models"][0]["feature_schema"], "ta-reference-24f")
        self.assertFalse(models["models"][0]["supports_no_hmm"])

        self.reply({"partition": "student_test", "trips": [{
            "trip_id": "00000007", "n_obs": 2, "n_nodes": 3, "n_arcs": 2,
            "duration_s": 1, "has_server_route": True,
        }]})
        self.assertEqual(self.client.get("/trips").json(), {
            "partition": "student_test", "trips": [{
                "trip_id": "00000007", "n_obs": 2, "n_nodes": 3, "n_arcs": 2,
                "duration_s": 1,
            }],
        })
        self.assertEqual(self.connection.request.call_args.args, (
            "GET", "/api/trips?partition=student_test",
        ))


if __name__ == "__main__":
    unittest.main()
