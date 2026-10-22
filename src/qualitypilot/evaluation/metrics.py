"""Explicitly represent undefined metrics as JSON null plus a reason."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from qualitypilot.decision.policy import Thresholds


def metric(value: float | int | None, reason: str | None = None) -> dict:
    return {"value": value, "reason": reason}


def ratio(numerator: int, denominator: int, reason: str) -> dict:
    return metric(numerator / denominator) if denominator else metric(None, reason)


def evaluate(labels: np.ndarray, scores: np.ndarray, thresholds: Thresholds) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape or not labels.size:
        raise ValueError("Labels and scores must be nonempty vectors of equal shape")
    if not set(np.unique(labels)) <= {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores")
    predicted = scores > thresholds.accept
    tn, fp, fn, tp = map(int, confusion_matrix(labels, predicted, labels=[0, 1]).ravel())
    decisions = np.array([thresholds.decide(float(score)) for score in scores])
    review = decisions == "REVIEW"
    result = {
        "image_auroc": metric(float(roc_auc_score(labels, scores)))
        if len(np.unique(labels)) == 2
        else metric(None, "AUROC requires both classes"),
        "image_average_precision": metric(float(average_precision_score(labels, scores)))
        if labels.sum()
        else metric(None, "Average precision requires positive labels"),
        "single_threshold": {
            "threshold": thresholds.accept,
            "comparison": "score > threshold",
            "precision": ratio(tp, tp + fp, "No predicted positives"),
            "recall": ratio(tp, tp + fn, "No true positives in subset"),
            "f1": ratio(2 * tp, 2 * tp + fp + fn, "No true or predicted positives"),
            "false_positive_rate": ratio(fp, fp + tn, "No normal images in subset"),
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "false_accepts": fn,
            "false_rejects": fp,
            "total_errors": fn + fp,
        },
        "two_thresholds": {
            "accept": thresholds.accept,
            "reject": thresholds.reject,
            "review_count": int(review.sum()),
            "review_fraction": float(review.mean()),
            "review_normal": int(((labels == 0) & review).sum()),
            "review_defect": int(((labels == 1) & review).sum()),
            "false_accepts": int(((labels == 1) & (decisions == "ACCEPT")).sum()),
            "false_rejects": int(((labels == 0) & (decisions == "REJECT")).sum()),
            "note": "Review outcomes are unknown and are excluded from automatic errors.",
        },
        "pixel_auroc": metric(None, "Global baseline has no anomaly map"),
        "dice": metric(None, "Global baseline has no anomaly map"),
        "iou": metric(None, "Global baseline has no anomaly map"),
        "gpu_latency_ms": metric(None, "This sklearn baseline runs on CPU only"),
    }
    return result
