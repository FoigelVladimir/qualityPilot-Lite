"""Commands for data preparation and a complete reproducible baseline experiment."""

import argparse
import importlib.metadata
import json
import logging
import platform
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd

from qualitypilot.config import Config, load_config
from qualitypilot.data.download import check_access, import_archive
from qualitypilot.data.mirror import download_mirror
from qualitypilot.data.pipeline import prepare
from qualitypilot.decision.policy import calibrate
from qualitypilot.evaluation.metrics import evaluate
from qualitypilot.evaluation.report import plot_confusion, write_json
from qualitypilot.inference.service import Inspector
from qualitypilot.models.baseline import Baseline

LOGGER = logging.getLogger(__name__)


def save_inventory(config: Config) -> tuple[list, dict]:
    samples, summary = prepare(config)
    if (config.output_dir / "metrics.json").exists():
        previous = pd.read_csv(config.output_dir / "manifest.csv", keep_default_na=False)
        current = pd.DataFrame([asdict(s) for s in samples]).fillna("")
        if not previous.equals(current):
            raise ValueError("Existing experiment is preserved; choose a new output_dir")
        return samples, summary
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(s) for s in samples]).to_csv(
        config.output_dir / "manifest.csv", index=False
    )
    write_json(config.output_dir / "dataset_summary.json", summary)
    config.save(config.output_dir / "config.yaml")
    return samples, summary


def run_baseline(config: Config) -> dict:
    if config.model_path.exists() or (config.output_dir / "metrics.json").exists():
        raise FileExistsError(
            "Existing baseline artifacts are preserved. Choose new output_dir and model_path."
        )
    samples, summary = save_inventory(config)
    train = [config.data_root / s.path for s in samples if s.split == "train"]
    calibration = [config.data_root / s.path for s in samples if s.split == "calibration"]
    test = [s for s in samples if s.split == "test"]
    model = Baseline(config.image_size, config.seed, config.n_estimators)
    model.fit(train)
    calibration_scores = model.scores(calibration)
    thresholds = calibrate(calibration_scores, config.accept_quantile, config.reject_quantile)
    warnings = []
    if len(calibration) < 100:
        warnings.append("Fewer than 100 calibration images: upper quantiles are coarse/unstable.")
    if thresholds.accept == thresholds.reject:
        warnings.append("Thresholds coincide; the REVIEW region is empty.")
    for message in warnings:
        LOGGER.warning(message)
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "qualitypilot-lite",
            "numpy",
            "Pillow",
            "scikit-learn",
            "pandas",
            "matplotlib",
            "PyYAML",
        )
    }
    metadata = {
        "model": "baseline",
        "thresholds": asdict(thresholds),
        "config": config.to_dict(),
        "dataset": summary,
        "versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "warnings": warnings,
        "calibration_note": "Empirical normal quantiles; no defect-recall guarantee.",
    }
    provenance = config.data_root / f"provenance-{config.category}.json"
    if provenance.is_file():
        metadata["data_provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
    # Freeze artifact and operating points BEFORE the first test prediction.
    model.save(config.model_path, metadata)
    write_json(config.output_dir / "experiment.json", metadata)
    pd.DataFrame(
        {
            "path": [p.relative_to(config.data_root).as_posix() for p in calibration],
            "score": calibration_scores,
        }
    ).to_csv(config.output_dir / "calibration.csv", index=False)
    inspector = Inspector.load(config.model_path)
    inspector.predict(config.data_root / test[0].path)  # Untimed warmup; no fitting.
    rows = []
    start = perf_counter()
    for sample in test:
        prediction = inspector.predict(config.data_root / sample.path)
        rows.append(
            {
                "path": sample.path,
                "label": sample.label,
                "defect": sample.defect,
                "score": prediction.score,
                "decision": prediction.decision,
                "latency_ms": prediction.latency_ms,
                "binary_prediction": int(prediction.score > thresholds.accept),
            }
        )
    elapsed = perf_counter() - start
    frame = pd.DataFrame(rows)
    metrics = evaluate(frame.label.to_numpy(), frame.score.to_numpy(), thresholds)
    metrics["cpu_latency_ms"] = {
        "mean": float(frame.latency_ms.mean()),
        "median": float(frame.latency_ms.median()),
        "p95": float(frame.latency_ms.quantile(0.95)),
        "scope": "decode + resize + features + inference + decision",
        "warmup_images": 1,
        "test_wall_seconds": elapsed,
    }
    metrics["manifest_sha256"] = summary["manifest_sha256"]
    metrics["model"] = "baseline"
    write_json(config.output_dir / "metrics.json", metrics)
    frame.to_csv(config.output_dir / "results.csv", index=False)
    plot_confusion(
        metrics["single_threshold"]["confusion_matrix"], config.output_dir / "confusion_matrix.png"
    )
    # Exports are dataset derivatives; stay inside the ignored runs directory.
    examples = config.output_dir / "examples"
    examples.mkdir(exist_ok=True)
    for label, predicted, name in ((1, 1, "tp"), (0, 1, "fp"), (1, 0, "fn")):
        selected = frame[(frame.label == label) & (frame.binary_prediction == predicted)].head(3)
        for index, row in selected.iterrows():
            source = config.data_root / row.path
            shutil.copyfile(source, examples / f"{name}_{index}{source.suffix}")
    LOGGER.info(
        "Baseline evaluated on %s untouched test images. Reports: %s", len(test), config.output_dir
    )
    return metrics


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/bottle.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("access", help="Check the official dataset access page")
    importer = commands.add_parser("import-data", help="Import a vendor tar archive")
    importer.add_argument("archive", type=Path)
    commands.add_parser("prepare", help="Validate dataset and save split manifest")
    commands.add_parser("download-mirror", help="Explicitly use the pinned foersben HF mirror")
    commands.add_parser("baseline", help="Train, calibrate, freeze, then evaluate baseline")
    commands.add_parser("patchcore", help="Fit, calibrate and compare PatchCore in a unique run")
    inspect = commands.add_parser("inspect", help="Inspect one image using a frozen model")
    inspect.add_argument("image", type=Path)
    inspect.add_argument("--model", type=Path, help="PatchCore .pt or baseline .joblib")
    inspect.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        if args.command == "access":
            print(check_access())
            return 0
        if args.command == "patchcore":
            from qualitypilot.experiments import run_patchcore
            from qualitypilot.models.patchcore_config import load_patchcore_config

            print(run_patchcore(load_patchcore_config(args.config)))
            return 0
        if args.command == "inspect" and args.model is not None:
            print(Inspector.load(args.model, args.device).predict(args.image).to_dict())
            return 0
        config = load_config(args.config)
        if args.command == "import-data":
            checksum = import_archive(args.archive, config.data_root, config.category)
            print(f"Imported archive SHA256: {checksum}")
            save_inventory(config)
        elif args.command == "prepare":
            _, summary = save_inventory(config)
            print(summary)
        elif args.command == "download-mirror":
            print(download_mirror(config.data_root, config.category))
            save_inventory(config)
        elif args.command == "baseline":
            run_baseline(config)
        elif args.command == "inspect":
            print(Inspector.load(config.model_path).predict(args.image).to_dict())
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
