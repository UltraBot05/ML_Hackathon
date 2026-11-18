"""Integrity tests for the opt-in route-first PathSnap experiment."""

import io
import json
import os
from pathlib import Path
import unittest
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd

from build_audit_dataset import backwards_count, compare_pool, digest
from train_audit_boosters import group_winners


ROOT = Path(os.environ.get("PATHSNAP_REPO", Path(__file__).resolve().parents[3]))
CACHE = os.environ.get("PATHSNAP_AUDIT_CACHE")
RESULTS = os.environ.get("PATHSNAP_AUDIT_RESULTS")


class EvaluationTests(unittest.TestCase):
    def test_distinguishes_candidate_absence_from_missing_proxy(self):
        answer = compare_pool(np.array([0, 0, 1]), np.array([5, 7, 9]), np.array([1, 2, 1]),
                              np.array([7, 3, 4]))
        self.assertEqual(answer["with_candidates"], 2)
        self.assertEqual(answer["proxy_present"], 1)
        self.assertEqual(answer["median_candidates"], 1)
        self.assertEqual(answer["nearest_hits"], 0)

    def test_repeated_route_ids_keep_distinct_route_occurrences(self):
        route = [1, 2, 1, 3]
        self.assertEqual(backwards_count(route, np.array([1, 2, 1, 3])), 0)
        self.assertEqual(backwards_count(route, np.array([1, 3, 2])), 1)

    def test_same_score_uses_rank_and_arc_id_within_trip(self):
        frame = pd.DataFrame({
            "trip_id": ["x", "x", "x", "y", "y"], "obs_index": [0, 0, 0, 0, 0],
            "candidate_arc_id": [4, 3, 10, 7, 9], "candidate_rank": [2, 1, 1, 1, 2],
        })
        probabilities = np.array([.4, .4, .4, .3, .6])
        np.testing.assert_array_equal(group_winners(frame, probabilities), [1, 4])


@unittest.skipUnless(CACHE and RESULTS, "Set PATHSNAP_AUDIT_CACHE and PATHSNAP_AUDIT_RESULTS to verify a local run.")
class ArtifactTests(unittest.TestCase):
    def test_same_features_and_split_and_verified_artifacts(self):
        cache = Path(CACHE)
        results_dir = Path(RESULTS)
        data = cache / "dataset_manifest.json"
        manifest = json.loads(data.read_text(encoding="utf-8"))
        summary = json.loads((results_dir / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["dataset_manifest_sha256"], digest(data))
        self.assertEqual(len(manifest["file_sha256"]), 162)
        train_ids = set(manifest["split"]["train_trips"])
        val_ids = set(manifest["split"]["val_trips"])
        self.assertEqual((len(train_ids), len(val_ids)), (64, 16))
        self.assertTrue(train_ids.isdisjoint(val_ids))
        features = manifest["feature_columns"]
        keys = ["trip_id", "obs_index", "candidate_arc_id"]
        trip = sorted(val_ids)[0]
        old = pd.read_parquet(ROOT / "phase2/features/val" / f"{trip}_features.parquet")
        new = pd.read_parquet(cache / "features/val" / f"{trip}.parquet")
        pd.testing.assert_frame_equal(new[keys+features], old[keys+features])
        for path in sorted(results_dir.glob("*.zip")):
            with self.subTest(archive=path.name), ZipFile(path) as archive:
                self.assertIsNone(archive.testzip())
                name = path.stem.split("_")[0]
                report = json.loads(archive.read(f"{name}/results.json"))
                self.assertEqual(report["feature_order"], features)
                self.assertEqual(report["dataset_manifest_sha256"], digest(data))
                model = joblib.load(io.BytesIO(archive.read(f"{name}/model.joblib")))
                scores = pd.read_parquet(io.BytesIO(archive.read(f"{name}/validation_predictions.parquet")))
                rows = scores.loc[scores.trip_id.eq(trip), keys+["probability"]].head(100).merge(
                    new, on=keys, validate="one_to_one",
                )
                self.assertEqual(len(rows), 100)
                res = model.predict_proba(rows[features].astype(np.float32))[:, 1]
                np.testing.assert_allclose(res, rows.probability, rtol=1e-6)
        for entry in [*summary["baselines"].values(), *([summary["tuned"]] if "tuned" in summary else [])]:
            for f in entry["training_cv_folds"]:
                fit = set(f["train_trips"])
                check = set(f["evaluation_trips"])
                self.assertTrue(fit.isdisjoint(check) and fit | check == train_ids)
            self.assertEqual({r["trip_id"] for r in entry["per_trip"]}, val_ids)


if __name__ == "__main__":
    unittest.main()
