"""
PathSnap Hackathon — Evaluation Framework
Computes empirical metrics across matchers and trips:
- Observation Top-1 / Top-3 / Top-5 Accuracy
- Coverage-Aware Top-1 Accuracy (penalizes candidate misses)
- Standard Sklearn Binary and Macro-F1 across classes
- Log Loss and Candidate Confusion Matrix
- Route Continuity Percentage
- Route Set Similarity (Jaccard, F1)
- Inference Latency per Observation
"""

from typing import List, Dict, Any, Optional
import time
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    log_loss,
    confusion_matrix,
)

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
        Preserves backward compatibility with Phase 2 baseline scripts while documenting
        distinction between positive-class binary F1 and standard two-class Macro F1.
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
        coverage_aware_accuracy = (correct / total * 100.0) if total > 0 else 0.0
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        binary_f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "top1_accuracy": round(accuracy, 2),
            "coverage_aware_top1": round(coverage_aware_accuracy, 2),
            # 'macro_f1' retained as alias to binary_f1 for Phase 2 backward compatibility
            "macro_f1": round(binary_f1, 4),
            "binary_f1": round(binary_f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "total_obs": total,
            "valid_gt_obs": valid_gt,
            "correct_obs": correct,
        }

    def evaluate_binary_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Computes standard scikit-learn binary and two-class Macro-F1 metrics for supervised ML models.
        Macro F1 is defined strictly as the unweighted mean of class 0 F1 and class 1 F1.
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
        bin_f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        mac_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        metrics = {
            "candidate_accuracy": round(float(acc), 4),
            "binary_f1": round(float(bin_f1), 4),
            "macro_f1": round(float(mac_f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "total_samples": int(len(y_true)),
        }

        if y_prob is not None:
            # Clip probabilities to avoid infinite log_loss
            y_prob_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
            try:
                loss = log_loss(y_true, y_prob_clipped)
                metrics["log_loss"] = round(float(loss), 4)
            except Exception as e:
                logger.warning(f"Failed to compute log_loss: {e}")
                metrics["log_loss"] = None

        return metrics

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
