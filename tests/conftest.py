import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qualitypilot.cli import run_baseline
from qualitypilot.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Tiny synthetic fixtures: never a substitute for MVTec evaluation."""
    rng = np.random.default_rng(123)
    root = tmp_path / "data" / "bottle"
    for split, defect, count in (
        ("train", "good", 15),
        ("test", "good", 4),
        ("test", "scratch", 4),
    ):
        folder = root / split / defect
        folder.mkdir(parents=True)
        for i in range(count):
            pixels = rng.integers(80, 180, (24, 24, 3), dtype=np.uint8)
            if defect != "good":
                pixels[5:12, 5:12] = 255
                mask = np.zeros((24, 24), dtype=np.uint8)
                mask[5:12, 5:12] = 255
                masks = root / "ground_truth" / defect
                masks.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(masks / f"{i:03d}_mask.png")
            Image.fromarray(pixels).save(folder / f"{i:03d}.png")
    return Config(
        tmp_path / "data",
        "bottle",
        tmp_path / "reports",
        tmp_path / "artifacts" / "baseline.joblib",
        image_size=32,
        n_estimators=10,
    )


@pytest.fixture
def ui_project(config: Config, tmp_path: Path) -> tuple[Path, Path]:
    """Synthetic reports, never presented as a real quality experiment."""
    run_baseline(config)
    root = tmp_path
    config.save(root / "base.yaml")
    (root / "configs").mkdir()
    (root / "configs/ui.yaml").write_text(
        "baseline_config: base.yaml\nruns_root: runs\nselected_run: bottle-patchcore-fixture\n",
        "utf-8",
    )
    run = root / "runs/bottle-patchcore-fixture"
    run.mkdir(parents=True)
    for name in ("baseline", "patchcore"):
        (run / name).mkdir()
        shutil.copyfile(config.output_dir / "results.csv", run / name / "results.csv")
    for name in ("manifest.csv", "dataset_summary.json"):
        shutil.copyfile(config.output_dir / name, run / name)
    summary = json.loads((run / "dataset_summary.json").read_text())
    metrics = json.loads((config.output_dir / "metrics.json").read_text())
    metrics.update(
        model="baseline",
        category="bottle",
        latency_ms={
            "device": "cpu",
            "count": 8,
            "batch_size": 1,
            "warmup_images": 3,
            "cuda_synchronized": False,
            "mean": 1.0,
            "median": 1.0,
            "p95": 2.0,
        },
    )
    patch = copy.deepcopy(metrics)
    patch["model"] = "patchcore"
    for name, values in (("baseline", metrics), ("patchcore", patch)):
        (run / name / "metrics.json").write_text(json.dumps(values))
    (run / "comparison.json").write_text(
        json.dumps(
            {"manifest_sha256": summary["manifest_sha256"], "baseline": metrics, "patchcore": patch}
        )
    )
    (run / "status.json").write_text('{"status": "COMPLETE"}')
    (run / "config.yaml").write_text("category: bottle\n")
    (run / "experiment.json").write_text(json.dumps({"dataset": summary}))
    (run / "model.pt").write_bytes(b"fixture: tests must mock its loader")
    return root, run
