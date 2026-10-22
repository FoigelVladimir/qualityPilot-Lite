import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qualitypilot.cli import run_baseline
from qualitypilot.config import Config
from qualitypilot.data.pipeline import prepare
from qualitypilot.inference.service import Inspector
from qualitypilot.models.baseline import Baseline, features


def test_features_and_reproducible_scores(config: Config):
    samples, _ = prepare(config)
    paths = [config.data_root / s.path for s in samples if s.split == "train"]
    assert features(paths[0], 32).shape == (60,)
    first, second = Baseline(32, 42, 10), Baseline(32, 42, 10)
    first.fit(paths)
    second.fit(paths)
    assert np.array_equal(first.scores(paths), second.scores(paths))
    assert np.isfinite(first.scores(paths)).all()


def test_missing_model(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="qualitypilot baseline"):
        Inspector.load(tmp_path / "absent.joblib")


def test_end_to_end_outputs(config: Config):
    metrics = run_baseline(config)
    expected = [
        "manifest.csv",
        "dataset_summary.json",
        "config.yaml",
        "experiment.json",
        "calibration.csv",
        "metrics.json",
        "results.csv",
        "confusion_matrix.png",
    ]
    assert all((config.output_dir / file).is_file() for file in expected)
    assert json.loads((config.output_dir / "metrics.json").read_text()) == metrics
    rows = pd.read_csv(config.output_dir / "results.csv")
    assert len(rows) == 8
    assert len(rows.path.unique()) == 8
    assert rows.latency_ms.gt(0).all()
    inspector = Inspector.load(config.model_path)
    prediction = inspector.predict(config.data_root / rows.iloc[0].path)
    assert prediction.score == pytest.approx(rows.iloc[0].score)
    assert prediction.decision == rows.iloc[0].decision
    assert set(prediction.to_dict()) == {
        "score",
        "decision",
        "latency_ms",
        "anomaly_map",
        "model",
        "category",
        "device",
        "timings_ms",
        "original_size",
        "pixel_threshold",
    }
    assert prediction.anomaly_map is None
    assert metrics["pixel_auroc"]["value"] is None


def test_test_labels_do_not_change_calibration(config: Config):
    run_baseline(config)
    _, metadata = Baseline.load(config.model_path)
    # Change only the test inventory (normal label/type); train/calibration stay identical.
    root = config.data_root / "bottle"
    (root / "test/scratch").rename(root / "test/other_defect")
    (root / "ground_truth/scratch").rename(root / "ground_truth/other_defect")
    repeated_config = replace(
        config,
        output_dir=config.output_dir / "repeat",
        model_path=config.model_path.with_name("repeated.joblib"),
    )
    run_baseline(repeated_config)
    _, repeated = Baseline.load(repeated_config.model_path)
    assert metadata["thresholds"] == repeated["thresholds"]
