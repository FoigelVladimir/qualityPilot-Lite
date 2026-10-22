"""Summaries contain measured observations, device and measurement scope."""

import numpy as np


def summarize_latency(
    values: list[float], device: str, warmup_images: int, batch_size: int = 1
) -> dict:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Latency requires a nonempty finite, nonnegative vector")
    if warmup_images < 0 or batch_size < 1:
        raise ValueError("Invalid warmup or batch size")
    return {
        "count": len(array),
        "device": device,
        "warmup_images": warmup_images,
        "batch_size": batch_size,
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "scope": "decode + resize/normalize/features + transfer + model + map/decision",
        "cuda_synchronized": device == "cuda",
    }
