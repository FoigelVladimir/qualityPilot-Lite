import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from qualitypilot.cli import run_baseline, save_inventory
from qualitypilot.config import Config
from qualitypilot.data.pipeline import Sample, prepare
from qualitypilot.evaluation.latency import summarize_latency
from qualitypilot.evaluation.pixels import evaluate_pixels, read_mask
from qualitypilot.evaluation.runner import compare_results, evaluate_model
from qualitypilot.experiments import create_run, normal_paths, verify_reference, verify_source_files
from qualitypilot.inference.images import load_image, resize_map
from qualitypilot.inference.service import Inspector
from qualitypilot.models.patchcore_config import PatchcoreConfig, load_patchcore_config


def test_pixel_metrics_known_values():
    mask = np.array([[0, 1], [0, 1]], dtype=bool)
    values = np.array([[0.1, 0.9], [0.8, 0.7]])
    result = evaluate_pixels([mask], [values], 0.75)
    assert result["pixel_auroc"]["value"] == 0.75
    assert result["dice"]["value"] == 0.5
    assert result["iou"]["value"] == pytest.approx(1 / 3)
    assert result["pixel_tp"] == result["pixel_fp"] == result["pixel_fn"] == 1


def test_undefined_pixel_metrics():
    mask = np.zeros((2, 2), dtype=bool)
    values = np.zeros((2, 2))
    result = evaluate_pixels([mask], [values], None)
    assert all(result[key]["value"] is None for key in ("pixel_auroc", "dice", "iou"))
    assert evaluate_pixels([mask], [values], 1)["dice"]["value"] is None


def test_invalid_pixel_inputs():
    with pytest.raises(ValueError, match="shapes"):
        evaluate_pixels([np.zeros((2, 2))], [np.zeros((3, 3))], 1)
    with pytest.raises(ValueError, match="finite"):
        evaluate_pixels([np.zeros((2, 2))], [np.full((2, 2), np.nan)], 1)


def test_mask_and_map_geometry(tmp_path: Path):
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[:, 10:] = 255
    path = tmp_path / "mask.png"
    Image.fromarray(mask).save(path)
    labels = read_mask(path, (8, 4))
    values = resize_map((mask > 0).astype(np.float32), (8, 4))
    assert labels.shape == values.shape == (4, 8)
    assert not labels[:, :4].any()
    assert labels[:, 4:].all()
    assert values[:, -1].min() == 1
    assert not read_mask(None, (8, 4)).any()


def test_latency_summary():
    summary = summarize_latency([1, 2, 3, 4, 5], "cuda", 3)
    assert summary["count"] == 5
    assert summary["median"] == 3
    assert summary["p95"] == pytest.approx(4.8)
    assert summary["cuda_synchronized"]


@pytest.mark.parametrize("values", [[], [-1], [np.nan], [[1, 2]]])
def test_invalid_latency(values):
    with pytest.raises(ValueError):
        summarize_latency(values, "cpu", 3)


def test_run_directories_are_unique(tmp_path: Path):
    first = create_run(tmp_path, "bottle")
    (first / "sentinel").write_text("preserve")
    second = create_run(tmp_path, "bottle")
    assert first != second
    assert (first / "sentinel").read_text() == "preserve"


def test_baseline_results_are_not_overwritten(config: Config):
    run_baseline(config)
    original = config.model_path.read_bytes()
    with pytest.raises(FileExistsError, match="preserved"):
        run_baseline(config)
    assert config.model_path.read_bytes() == original


def test_prepare_preserves_existing_experiment(config: Config):
    run_baseline(config)
    path = config.output_dir / "manifest.csv"
    original = path.read_bytes()
    save_inventory(config)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="preserved"):
        save_inventory(replace(config, seed=12))
    assert path.read_bytes() == original


def test_provenance_detects_changed_mask(config: Config):
    relative = "bottle/ground_truth/scratch/000_mask.png"
    path = config.data_root / relative
    payload = path.read_bytes()
    provenance = {"files": [{"path": relative, "sha256": hashlib.sha256(payload).hexdigest()}]}
    verify_source_files(config.data_root, provenance)
    path.write_bytes(payload + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_source_files(config.data_root, provenance)


def test_leakage_gate(config: Config):
    samples, _ = prepare(config)
    assert len(normal_paths(samples, "train", config.data_root)) == 12
    with pytest.raises(ValueError):
        normal_paths(samples, "test", config.data_root)
    injected = Sample("bottle/test/good/000.png", "train", 0, "good", None, "hash")
    with pytest.raises(ValueError, match="Leakage"):
        normal_paths([injected], "train", config.data_root)
    injected = replace(injected, path="bottle/train/good/a.png", label=1)
    with pytest.raises(ValueError, match="Leakage"):
        normal_paths([injected], "train", config.data_root)


def test_comparison_rejects_mismatched_test():
    first = pd.DataFrame({"path": ["a", "b"], "label": [0, 1]})
    compare_results(first, first.copy())
    with pytest.raises(ValueError):
        compare_results(first, first.iloc[::-1])
    with pytest.raises(ValueError):
        compare_results(first, pd.concat([first, first]))


def test_patchcore_config_and_reference(config: Config, tmp_path: Path):
    config_path = tmp_path / "base.yaml"
    config.save(config_path)
    run_baseline(config)
    patch = PatchcoreConfig(config_path, config.output_dir, tmp_path / "runs")
    _, samples, _, _ = verify_reference(patch)
    assert len(samples) == 23
    import yaml

    path = tmp_path / "patch.yaml"
    path.write_text(yaml.safe_dump(patch.to_dict()), "utf-8")
    assert load_patchcore_config(path) == patch
    altered = pd.read_csv(config.output_dir / "manifest.csv")
    altered.loc[0, "split"] = "test"
    altered.to_csv(config.output_dir / "manifest.csv", index=False)
    with pytest.raises(ValueError, match="manifest"):
        verify_reference(patch)


def test_common_inspection_inputs(config: Config, tmp_path: Path):
    run_baseline(config)
    inspector = Inspector.load(config.model_path)
    path = config.data_root / "bottle/test/good/000.png"
    a, b = inspector.predict(path), inspector.predict(load_image(path))
    assert a.score == b.score
    assert a.category == "bottle" and a.device == "cpu"
    assert sum(a.timings_ms.values()) == pytest.approx(a.latency_ms)
    json.dumps(a.to_dict(), allow_nan=False)
    with pytest.raises(ValueError, match="Unsupported image"):
        inspector.predict(tmp_path / "not-supported.txt")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"broken")
    with pytest.raises(ValueError, match="decode"):
        inspector.predict(bad)


def test_shared_evaluation_rejects_training(config: Config, tmp_path: Path):
    run_baseline(config)
    samples, _ = prepare(config)
    inspector = Inspector.load(config.model_path)
    with pytest.raises(ValueError, match="held-out test"):
        evaluate_model(inspector, samples, config.data_root, tmp_path / "eval", [], 32)
