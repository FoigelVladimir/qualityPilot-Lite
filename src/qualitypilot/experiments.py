"""Frozen PatchCore experiment with explicit normal calibration and matched comparison."""

import hashlib
import importlib.metadata
import json
import logging
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from qualitypilot.config import Config, load_config
from qualitypilot.data.pipeline import Sample, prepare
from qualitypilot.decision.policy import calibrate
from qualitypilot.evaluation.report import write_json
from qualitypilot.evaluation.runner import (
    benchmark,
    compare_results,
    evaluate_model,
    save_examples,
)
from qualitypilot.inference.service import Inspector
from qualitypilot.models.baseline import Baseline
from qualitypilot.models.patchcore_config import PatchcoreConfig

LOGGER = logging.getLogger(__name__)


def create_run(root: Path, category: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{category}-patchcore-{stamp}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def normal_paths(samples: list[Sample], split: str, data_root: Path) -> list[Path]:
    if split not in {"train", "calibration"}:
        raise ValueError("Only normal train/calibration can be used for fitting or thresholds")
    selected = [sample for sample in samples if sample.split == split]
    if not selected:
        raise ValueError(f"Empty {split} split")
    for sample in selected:
        if sample.label != 0 or PurePosixPath(sample.path).parts[1:3] != ("train", "good"):
            raise ValueError(f"Leakage: {split} contains non-training-normal image {sample.path}")
    return [data_root / sample.path for sample in selected]


def verify_source_files(data_root: Path, provenance: dict) -> None:
    """Include masks/licenses in provenance checks; image-only manifests cannot cover them."""
    for entry in provenance.get("files", []):
        path = (data_root / entry["path"]).resolve()
        if not path.is_relative_to(data_root.resolve()):
            raise ValueError("Provenance path escapes data root")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"Source file checksum mismatch: {entry['path']}")


def verify_reference(config: PatchcoreConfig) -> tuple[Config, list[Sample], dict, dict]:
    base = load_config(config.baseline_config)
    if config.category != base.category or config.seed != base.seed:
        raise ValueError("Category and split seed must match the baseline reference")
    samples, summary = prepare(base)
    old = pd.read_csv(config.reference_run / "manifest.csv", keep_default_na=False)
    current = pd.DataFrame([asdict(s) for s in samples]).fillna("")
    if not old.equals(current):
        raise ValueError("Dataset or split differs from the baseline manifest")
    previous = json.loads((config.reference_run / "dataset_summary.json").read_text("utf-8"))
    if previous["manifest_sha256"] != summary["manifest_sha256"]:
        raise ValueError("Baseline manifest SHA256 mismatch")
    _, metadata = Baseline.load(base.model_path)
    if metadata["dataset"]["manifest_sha256"] != summary["manifest_sha256"]:
        raise ValueError("Baseline artifact was fitted on a different manifest")
    if metadata.get("data_provenance"):
        verify_source_files(base.data_root, metadata["data_provenance"])
    return base, samples, summary, metadata


def environment(device: str) -> dict:
    from qualitypilot.models.patchcore import torch_runtime

    torch = torch_runtime()
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torchvision",
            "anomalib",
            "lightning",
            "pytorch-lightning",
            "torchmetrics",
            "timm",
            "numpy",
            "scikit-learn",
            "opencv-python",
            "Pillow",
            "pandas",
            "scipy",
        )
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory
        if torch.cuda.is_available()
        else None,
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _comparison_table(baseline: dict, patchcore: dict) -> str:
    rows = [
        ("Image AUROC", baseline["image_auroc"]["value"], patchcore["image_auroc"]["value"]),
        (
            "Average Precision",
            baseline["image_average_precision"]["value"],
            patchcore["image_average_precision"]["value"],
        ),
    ]
    for name in ("precision", "recall", "f1"):
        rows.append(
            (
                name,
                baseline["single_threshold"][name]["value"],
                patchcore["single_threshold"][name]["value"],
            )
        )
    for name in ("median", "p95"):
        rows.append(
            (f"{name} latency (ms)", baseline["latency_ms"][name], patchcore["latency_ms"][name])
        )
    for name in ("pixel_auroc", "dice", "iou"):
        rows.append((name, baseline[name]["value"], patchcore[name]["value"]))

    def show(value):
        return "N/A" if value is None else f"{value:.6f}"

    return "| Metric | Baseline (CPU) | PatchCore |\n|---|---:|---:|\n" + "\n".join(
        f"| {name} | {show(a)} | {show(b)} |" for name, a, b in rows
    )


def run_patchcore(config: PatchcoreConfig) -> Path:
    from qualitypilot.models.patchcore import PatchcoreBackend, seed_runtime, select_device

    base, samples, summary, baseline_metadata = verify_reference(config)
    train = normal_paths(samples, "train", base.data_root)
    calibration = normal_paths(samples, "calibration", base.data_root)
    test = [sample for sample in samples if sample.split == "test"]
    run = create_run(config.output_root, config.category)
    write_json(run / "status.json", {"status": "NOT RUN", "configuration_frozen_before_test": True})
    (run / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), "utf-8")
    pd.DataFrame([asdict(s) for s in samples]).to_csv(run / "manifest.csv", index=False)
    write_json(run / "dataset_summary.json", summary)
    try:
        seed_runtime(config.seed, config.cpu_threads)
        device = select_device(config.device)
        runtime = environment(device)
        write_json(run / "environment.json", runtime)
        write_json(run / "status.json", {"status": "TRAINING", "test_used_for_tuning": False})
        LOGGER.info("Run directory: %s; device: %s", run, device)
        start = perf_counter()
        backend = PatchcoreBackend(config.model_options(), device)
        bank = backend.fit(train, config.train_batch_size, config.coreset_sampling_ratio)
        preparation_seconds = perf_counter() - start
        LOGGER.info("Memory bank ready in %.2f s: %s", preparation_seconds, bank)
        uncalibrated = Inspector(backend, None, config.category)
        calibration_scores, maxima = [], []
        for path in calibration:
            result = uncalibrated.predict(path, map_resolution="model")
            calibration_scores.append(result.score)
            maxima.append(float(result.anomaly_map.max()))
        thresholds = calibrate(
            np.array(calibration_scores), config.accept_quantile, config.reject_quantile
        )
        pixel_threshold = float(np.quantile(maxima, config.pixel_quantile, method="higher"))
        pd.DataFrame(
            {
                "path": [p.relative_to(base.data_root).as_posix() for p in calibration],
                "score": calibration_scores,
                "map_max": maxima,
            }
        ).to_csv(run / "calibration.csv", index=False)
        metadata = {
            "category": config.category,
            "config": config.to_dict(),
            "thresholds": asdict(thresholds),
            "pixel_threshold": pixel_threshold,
            "calibration": {
                "split": "normal-only held out from train/good",
                "count": len(calibration),
                "image_quantiles": [config.accept_quantile, config.reject_quantile],
                "pixel_policy": "higher quantile of per-image native map maxima",
                "pixel_quantile": config.pixel_quantile,
                "limitation": "Empirical, not optimal; no recall/FPR guarantee.",
            },
            "dataset": summary,
            "environment": runtime,
            "memory_bank": bank,
            "preparation_seconds": preparation_seconds,
            "data_provenance": baseline_metadata.get("data_provenance"),
            "test_used_for_tuning": False,
            "comparison_note": "Descriptive post-hoc comparison, baseline was already known.",
        }
        backend.save(run / "model.pt", metadata)
        write_json(run / "experiment.json", metadata)
        with (run / "model.pt").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        write_json(run / "model_checksum.json", {"sha256": digest})
        # Reload on calibration BEFORE the first test inference. No network/pretrained fetch.
        inspector = Inspector.load(run / "model.pt", device)
        original = uncalibrated.predict(calibration[0], map_resolution="model")
        reloaded = inspector.predict(calibration[0], map_resolution="model")
        np.testing.assert_allclose(original.score, reloaded.score, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(original.anomaly_map, reloaded.anomaly_map, rtol=1e-6, atol=1e-6)
        write_json(
            run / "reload_check.json",
            {
                "status": "PASSED",
                "split": "calibration",
                "score": reloaded.score,
                "map_shape": list(reloaded.anomaly_map.shape),
                "score_max_abs_diff": abs(original.score - reloaded.score),
                "map_max_abs_diff": float(
                    np.max(np.abs(original.anomaly_map - reloaded.anomaly_map))
                ),
            },
        )
        del uncalibrated, backend
        warmup = [calibration[i % len(calibration)] for i in range(config.warmup_images)]
        write_json(run / "status.json", {"status": "EVALUATING", "thresholds_frozen": True})
        patch_metrics, patch_rows = evaluate_model(
            inspector, test, base.data_root, run / "patchcore", warmup, config.image_size
        )
        baseline = Inspector.load(base.model_path)
        baseline_metrics, baseline_rows = evaluate_model(
            baseline, test, base.data_root, run / "baseline", warmup, config.image_size
        )
        compare_results(baseline_rows, patch_rows)
        historical = pd.read_csv(config.reference_run / "results.csv")
        compare_results(historical, baseline_rows)
        np.testing.assert_allclose(historical.score, baseline_rows.score, rtol=0, atol=1e-12)
        examples = save_examples(inspector, test, base.data_root, patch_rows, run / "examples")
        cpu_latency = patch_metrics["latency_ms"]
        if device == "cuda":
            cpu = Inspector.load(run / "model.pt", "cpu")
            cpu_latency = benchmark(cpu, [base.data_root / s.path for s in test], warmup)
            del cpu
        comparison = {
            "manifest_sha256": summary["manifest_sha256"],
            "category": config.category,
            "same_test_paths_and_labels": True,
            "baseline_scores_match_sprint1": True,
            "baseline": baseline_metrics,
            "patchcore": patch_metrics,
            "patchcore_cpu_latency_ms": cpu_latency,
            "examples": examples,
            "preparation_seconds": preparation_seconds,
        }
        write_json(run / "comparison.json", comparison)
        table = _comparison_table(baseline_metrics, patch_metrics)
        (run / "comparison.md").write_text(
            "# Fixed-configuration comparison\n\n" + table + "\n\n"
            "Batch size 1, three calibration-image warmups; CUDA synchronized when used.\n"
            "Exact pixel metrics on native input grid with nearest-neighbour masks.\n"
            "Image/pixel thresholds frozen on normal calibration, not optimal thresholds.\n"
            "Descriptive post-hoc comparison: baseline test results were already known.\n",
            "utf-8",
        )
        write_json(
            run / "status.json",
            {
                "status": "COMPLETE",
                "test_images": len(test),
                "reload_check": "PASSED",
                "test_used_for_tuning": False,
            },
        )
        LOGGER.info("Complete: %s", run)
        return run
    except Exception as exc:
        # Failed experiments are retained, including their predeclared config and diagnostics.
        write_json(
            run / "status.json",
            {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)},
        )
        LOGGER.exception("Experiment failed; preserved run: %s", run)
        raise
