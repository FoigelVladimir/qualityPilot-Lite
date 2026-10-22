"""Deterministic input decoding and geometry shared with future UI clients."""

from pathlib import Path

import numpy as np
from PIL import Image

from qualitypilot.data.pipeline import EXTENSIONS, read_rgb

ImageInput = Path | str | Image.Image


def load_image(source: ImageInput) -> Image.Image:
    if isinstance(source, Image.Image):
        source.load()
        return source.convert("RGB")
    path = Path(source)
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    return read_rgb(path)


def resize_map(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a floating map to PIL (width, height), with no crop or rotation."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite two-dimensional anomaly map")
    return np.asarray(Image.fromarray(values).resize(size, Image.Resampling.BILINEAR)).copy()
