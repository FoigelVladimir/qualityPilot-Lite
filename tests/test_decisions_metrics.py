import numpy as np
import pytest

from qualitypilot.decision.policy import Thresholds, calibrate
from qualitypilot.evaluation.metrics import evaluate


def test_decision_boundaries():
    thresholds = Thresholds(1, 2, 100)
    assert thresholds.decide(0) == "ACCEPT"
    assert thresholds.decide(1) == "ACCEPT"
    assert thresholds.decide(1.5) == "REVIEW"
    assert thresholds.decide(2) == "REVIEW"
    assert thresholds.decide(3) == "REJECT"
    with pytest.raises(ValueError):
        thresholds.decide(float("nan"))


def test_collapsed_review_region():
    thresholds = calibrate(np.ones(5), 0.95, 0.99)
    assert thresholds.accept == thresholds.reject
    assert thresholds.decide(1) == "ACCEPT"
    assert thresholds.decide(1.1) == "REJECT"


def test_metrics_known_confusion():
    result = evaluate(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.6, 0.9]), Thresholds(0.5, 0.8, 100)
    )
    single = result["single_threshold"]
    assert single["confusion_matrix"] == [[1, 1], [0, 2]]
    assert single["precision"]["value"] == pytest.approx(2 / 3)
    assert single["recall"]["value"] == 1
    assert single["f1"]["value"] == pytest.approx(0.8)
    assert single["false_positive_rate"]["value"] == 0.5
    assert result["two_thresholds"]["review_fraction"] == 0.5
    assert result["two_thresholds"]["false_rejects"] == 0
    assert result["two_thresholds"]["false_accepts"] == 0
    assert result["image_auroc"]["value"] == 0.75


def test_undefined_metrics():
    result = evaluate(np.array([0, 0]), np.array([0.1, 0.2]), Thresholds(1, 2, 10))
    for key in ("image_auroc", "image_average_precision", "pixel_auroc"):
        assert result[key]["value"] is None
        assert result[key]["reason"]
    assert result["single_threshold"]["precision"]["value"] is None
    assert result["single_threshold"]["recall"]["value"] is None


@pytest.mark.parametrize("scores", [[], [np.nan], [[1, 2]]])
def test_invalid_calibration(scores):
    with pytest.raises(ValueError):
        calibrate(np.array(scores), 0.95, 0.99)


@pytest.mark.parametrize(
    "labels,scores", [([], []), ([2], [0.1]), ([0], [np.inf]), ([0, 1], [0.1])]
)
def test_invalid_metric_inputs(labels, scores):
    with pytest.raises(ValueError):
        evaluate(np.array(labels), np.array(scores), Thresholds(1, 2, 10))
