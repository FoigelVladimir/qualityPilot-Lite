"""Empirical normal-score quantiles are operating points, not defect probabilities."""

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    accept: float
    reject: float
    calibration_count: int

    def __post_init__(self) -> None:
        if not isfinite(self.accept) or not isfinite(self.reject) or self.accept > self.reject:
            raise ValueError("Require finite, ordered thresholds")
        if self.calibration_count < 1:
            raise ValueError("Calibration is empty")

    def decide(self, score: float) -> str:
        if not isfinite(score):
            raise ValueError("Anomaly score must be finite")
        if score > self.reject:
            return "REJECT"
        if score > self.accept:
            return "REVIEW"
        return "ACCEPT"


def calibrate(scores: np.ndarray, accept_quantile: float, reject_quantile: float) -> Thresholds:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Expected a nonempty finite vector of normal calibration scores")
    if not 0 < accept_quantile < reject_quantile < 1:
        raise ValueError("Invalid calibration quantiles")
    low, high = np.quantile(scores, [accept_quantile, reject_quantile], method="higher")
    return Thresholds(float(low), float(high), len(scores))
