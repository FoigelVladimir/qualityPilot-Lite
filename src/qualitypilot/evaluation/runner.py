"""Shared test protocol for a frozen baseline and a frozen PatchCore artifact."""

from pathlib import Path

import pandas as pd

from qualitypilot.data.pipeline import Sample
from qualitypilot.evaluation.latency import summarize_latency
from qualitypilot.evaluation.metrics import evaluate
from qualitypilot.evaluation.pixels import evaluate_pixels, read_mask
from qualitypilot.evaluation.report import plot_confusion, write_json
from qualitypilot.inference.images import resize_map
from qualitypilot.inference.service import Inspector


def evaluate_model(
    inspector: Inspector,
    samples: list[Sample],
    data_root: Path,
    output: Path,
    warmup_paths: list[Path],
    pixel_size: int,
) -> tuple[dict, pd.DataFrame]:
    if not samples or any(s.split != "test" for s in samples):
        raise ValueError("Evaluation accepts only the held-out test split")
    if inspector.thresholds is None:
        raise ValueError("Frozen image thresholds are required before evaluation")
    if not warmup_paths:
        raise ValueError("Warmup images are required")
    output.mkdir(parents=True, exist_ok=False)
    for path in warmup_paths:
        inspector.predict(path)
    rows, masks, maps = [], [], []
    for sample in samples:
        result = inspector.predict(data_root / sample.path, map_resolution="model")
        rows.append(
            {
                "path": sample.path,
                "label": sample.label,
                "defect": sample.defect,
                "score": result.score,
                "decision": result.decision,
                "binary_prediction": int(result.score > inspector.thresholds.accept),
                "latency_ms": result.latency_ms,
                **{f"{k}_ms": v for k, v in result.timings_ms.items()},
            }
        )
        if result.anomaly_map is not None:
            # Exact metrics at declared model-input grid; nearest-neighbour GT labels.
            maps.append(resize_map(result.anomaly_map, (pixel_size, pixel_size)))
            masks.append(
                read_mask(
                    data_root / sample.mask_path if sample.mask_path else None,
                    (pixel_size, pixel_size),
                )
            )
    frame = pd.DataFrame(rows)
    metrics = evaluate(frame.label.to_numpy(), frame.score.to_numpy(), inspector.thresholds)
    metrics["model"] = inspector.model.name
    metrics["category"] = inspector.category
    metrics["map_resolution"] = "native model grid; no crop, masks nearest-neighbour resized"
    metrics["latency_ms"] = summarize_latency(
        frame.latency_ms.tolist(), str(inspector.model.device), len(warmup_paths)
    )
    metrics["latency_components_ms"] = {
        k: summarize_latency(
            frame[f"{k}_ms"].tolist(), str(inspector.model.device), len(warmup_paths)
        )
        for k in ("preprocessing", "inference", "postprocessing")
    }
    if maps:
        metrics.pop("gpu_latency_ms", None)
        pixels = evaluate_pixels(masks, maps, inspector.pixel_threshold)
        metrics.update({key: pixels[key] for key in ("pixel_auroc", "dice", "iou")})
        metrics["pixel_evaluation"] = pixels
    frame.to_csv(output / "results.csv", index=False)
    write_json(output / "metrics.json", metrics)
    plot_confusion(
        metrics["single_threshold"]["confusion_matrix"],
        output / "confusion_matrix.png",
        title=f"{inspector.model.name} / frozen calibration threshold",
    )
    return metrics, frame


def compare_results(baseline: pd.DataFrame, patchcore: pd.DataFrame) -> None:
    for frame in (baseline, patchcore):
        if frame.path.duplicated().any() or frame.empty:
            raise ValueError("Empty or duplicate comparison paths")
    if not baseline[["path", "label"]].equals(patchcore[["path", "label"]]):
        raise ValueError("Cannot compare different test paths, order or labels")


def save_examples(
    inspector: Inspector, samples: list[Sample], data_root: Path, frame: pd.DataFrame, output: Path
) -> dict:
    from qualitypilot.visualization.inspection import save_inspection

    examples = {}
    indexed = {sample.path: sample for sample in samples}
    for name, label, predicted in (("tp", 1, 1), ("fp", 0, 1), ("fn", 1, 0), ("tn", 0, 0)):
        matches = frame[(frame.label == label) & (frame.binary_prediction == predicted)]
        if matches.empty:
            examples[name] = {"count": 0, "example": None, "reason": "No such cases in test"}
            continue
        row = matches.iloc[0]  # Deterministic first case; no visual cherry-picking.
        sample = indexed[row.path]
        result = inspector.predict(data_root / sample.path)
        path = output / f"{name}.png"
        save_inspection(
            data_root / sample.path,
            result,
            data_root / sample.mask_path if sample.mask_path else None,
            path,
        )
        examples[name] = {
            "count": len(matches),
            "example": path.name,
            "path": sample.path,
            "score": result.score,
            "decision": result.decision,
        }
    write_json(output / "examples.json", examples)
    return examples


def benchmark(inspector: Inspector, paths: list[Path], warmup_paths: list[Path]) -> dict:
    for path in warmup_paths:
        inspector.predict(path)
    predictions = [inspector.predict(path, map_resolution="model") for path in paths]
    return summarize_latency(
        [p.latency_ms for p in predictions], str(inspector.model.device), len(warmup_paths)
    )
