"""Exact pixel AUROC and micro Dice/IoU on an explicitly declared evaluation grid."""

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from qualitypilot.evaluation.metrics import metric, ratio


def read_mask(path: Path | None, size: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(size[::-1], dtype=bool)
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize(size, Image.Resampling.NEAREST)) > 0


def evaluate_pixels(
    masks: list[np.ndarray], maps: list[np.ndarray], threshold: float | None
) -> dict:
    if not masks or len(masks) != len(maps):
        raise ValueError("Expected paired nonempty masks and maps")
    for mask, values in zip(masks, maps, strict=True):
        if mask.ndim != 2 or mask.shape != values.shape or not np.isfinite(values).all():
            raise ValueError("Mask/map shapes must match and maps must be finite")
        if not set(np.unique(mask)) <= {0, 1}:
            raise ValueError("Expected binary pixel labels")
    labels = np.concatenate([m.ravel() for m in masks]).astype(bool)
    scores = np.concatenate([m.ravel() for m in maps]).astype(np.float32)
    result = {
        "pixel_auroc": metric(float(roc_auc_score(labels, scores)))
        if np.unique(labels).size == 2
        else metric(None, "Pixel AUROC requires both classes"),
        "pixel_count": labels.size,
        "aggregation": "micro across all pixels of all official test images, including normal",
        "grid_shapes": sorted({str(m.shape) for m in masks}),
        "pixel_threshold": threshold,
    }
    if threshold is None:
        result.update(
            dice=metric(None, "No independent pixel threshold"),
            iou=metric(None, "No independent pixel threshold"),
        )
        return result
    if not np.isfinite(threshold):
        raise ValueError("Pixel threshold must be finite")
    predicted = scores > threshold
    tp = int((labels & predicted).sum())
    fp = int((~labels & predicted).sum())
    fn = int((labels & ~predicted).sum())
    result.update(
        dice=ratio(2 * tp, 2 * tp + fp + fn, "No true or predicted defect pixels"),
        iou=ratio(tp, tp + fp + fn, "No true or predicted defect pixels"),
        pixel_tp=tp,
        pixel_fp=fp,
        pixel_fn=fn,
    )
    return result
