"""
PathSnap Hackathon — Phase 2: Evaluation Framework
Computes empirical metrics across matchers and trips:
- Observation Top-1 / Top-3 / Top-5 Accuracy
- Candidate Miss Rate
- Binary Classification Precision, Recall, Macro-F1
- Route Continuity Percentage
- Route Set Similarity (Jaccard, F1)
- Inference Latency per Observation
"""

from typing import List, Dict, Any, Optional
import time
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Evaluator")


class ModelEvaluator:
    """
    Evaluates map-matching predictions against ground-truth observation targets and route sequences.
    """

    def evaluate_observation_predictions(
        self,
        predictions: List[int],
        ground_truth: List[Optional[int]],
    ) -> Dict[str, float]:
        """
        Computes observation-level Top-1 accuracy, Precision, Recall, and F1.
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Length mismatch: {len(predictions)} preds vs {len(ground_truth)} gt")

        total = len(predictions)
        if total == 0:
            return {"top1_accuracy": 0.0, "total_obs": 0, "correct_obs": 0}

        correct = 0
        valid_gt = 0
        tp = 0
        fp = 0
        fn = 0

        for pred, gt in zip(predictions, ground_truth):
            if gt is not None and gt >= 0:
                valid_gt += 1
                if pred == gt:
                    correct += 1
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            else:
                if pred >= 0:
                    fp += 1

        accuracy = (correct / valid_gt * 100.0) if valid_gt > 0 else 0.0
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "top1_accuracy": round(accuracy, 2),
            "macro_f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "total_obs": total,
            "valid_gt_obs": valid_gt,
            "correct_obs": correct,
        }

    def evaluate_candidate_ranking(
        self,
        candidates: List[Dict[str, Any]],
        ground_truth: Dict[int, Optional[int]],
    ) -> Dict[str, float]:
        """
        Evaluates candidate retrieval quality: Top-1, Top-3, Top-5 recall and candidate miss rate.
        """
        import collections

        cands_by_obs = collections.defaultdict(list)
        for c in candidates:
            cands_by_obs[c["obs_index"]].append(c)

        total_obs = len(ground_truth)
        top1_hit = 0
        top3_hit = 0
        top5_hit = 0
        in_candidates = 0

        for obs_idx, true_arc in ground_truth.items():
            if true_arc is None or true_arc < 0:
                continue

            c_list = cands_by_obs.get(obs_idx, [])
            c_sorted = sorted(c_list, key=lambda c: c["distance_to_arc_m"])
            c_arcs = [c["candidate_arc_id"] for c in c_sorted]

            if c_arcs and c_arcs[0] == true_arc:
                top1_hit += 1
            if true_arc in c_arcs[:3]:
                top3_hit += 1
            if true_arc in c_arcs[:5]:
                top5_hit += 1
            if true_arc in c_arcs:
                in_candidates += 1

        top1_rec = round(top1_hit / total_obs * 100.0, 2) if total_obs > 0 else 0.0
        top3_rec = round(top3_hit / total_obs * 100.0, 2) if total_obs > 0 else 0.0
        top5_rec = round(top5_hit / total_obs * 100.0, 2) if total_obs > 0 else 0.0
        miss_rate = round((1.0 - in_candidates / total_obs) * 100.0, 2) if total_obs > 0 else 0.0

        return {
            "candidate_top1_pct": top1_rec,
            "candidate_top3_pct": top3_rec,
            "candidate_top5_pct": top5_rec,
            "candidate_miss_rate_pct": miss_rate,
        }

