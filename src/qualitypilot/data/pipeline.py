"""Keep official test data untouched; hold out only normal training images."""

import hashlib
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from qualitypilot.config import Config

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    path: str
    split: str
    label: int
    defect: str
    mask_path: str | None
    pixel_sha256: str


def read_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"Cannot decode image: {path}: {exc}") from exc


def pixel_hash(image: Image.Image) -> str:
    """Hash decoded pixels as well as shape, detecting copies across file encodings."""
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def assert_no_leakage(samples: list[Sample]) -> None:
    paths: set[str] = set()
    hashes: dict[str, str] = {}
    for sample in samples:
        if sample.path in paths:
            raise ValueError(f"Repeated image path: {sample.path}")
        paths.add(sample.path)
        previous = hashes.get(sample.pixel_sha256)
        if previous is not None:
            raise ValueError(f"Duplicate image content: {previous} and {sample.path}")
        hashes[sample.pixel_sha256] = sample.path


def inventory(config: Config) -> list[Sample]:
    root = config.data_root / config.category
    for required in (root / "train" / "good", root / "test" / "good"):
        if not required.is_dir():
            raise ValueError(f"Missing MVTec AD directory: {required}. See docs/data.md")
    if any(p.is_dir() and p.name != "good" for p in (root / "train").iterdir()):
        raise ValueError("Training directory must contain only train/good")
    samples: list[Sample] = []
    expected_masks: set[Path] = set()
    for split in ("train", "test"):
        folders = sorted(p for p in (root / split).iterdir() if p.is_dir())
        for folder in folders:
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
            if not files:
                raise ValueError(f"No images in {folder}")
            for path in files:
                image = read_rgb(path)
                label = int(folder.name != "good")
                mask_path = None
                if label:
                    mask = root / "ground_truth" / folder.name / f"{path.stem}_mask.png"
                    if not mask.is_file():
                        raise ValueError(f"Missing ground truth mask: {mask}")
                    try:
                        with Image.open(mask) as mask_image:
                            mask_image.load()
                            pixels = np.asarray(mask_image)
                            if mask_image.size != image.size:
                                raise ValueError(f"Mask size mismatch: {mask}")
                            if pixels.ndim != 2 or not set(np.unique(pixels)) <= {0, 1, 255}:
                                raise ValueError(f"Expected a binary grayscale mask: {mask}")
                            if not np.any(pixels):
                                raise ValueError(f"Defective image has an empty mask: {mask}")
                    except OSError as exc:
                        raise ValueError(f"Cannot decode mask: {mask}") from exc
                    expected_masks.add(mask.resolve())
                    mask_path = mask.relative_to(config.data_root).as_posix()
                samples.append(
                    Sample(
                        path.relative_to(config.data_root).as_posix(),
                        split,
                        label,
                        folder.name,
                        mask_path,
                        pixel_hash(image),
                    )
                )
    actual_masks = {
        p.resolve()
        for p in (root / "ground_truth").rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }
    if actual_masks != expected_masks:
        raise ValueError("Orphan or unexpected ground truth masks")
    if not any(s.label for s in samples):
        raise ValueError("No defective test images found")
    assert_no_leakage(samples)
    return samples


def prepare(config: Config) -> tuple[list[Sample], dict]:
    samples = inventory(config)
    normal = [sample for sample in samples if sample.split == "train"]
    if len(normal) < 3:
        raise ValueError("At least 3 normal training images are required")
    order = np.random.default_rng(config.seed).permutation(len(normal))
    count = max(1, math.ceil(len(normal) * config.calibration_fraction))
    if count >= len(normal):
        raise ValueError("Calibration split leaves no training images")
    held_out = {normal[int(i)].path for i in order[:count]}
    samples = [replace(s, split="calibration") if s.path in held_out else s for s in samples]
    assert_no_leakage(samples)
    counts = Counter(f"{s.split}/{s.defect}" for s in samples)
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(repr(asdict(sample)).encode())
    summary = {
        "category": config.category,
        "seed": config.seed,
        "counts": dict(sorted(counts.items())),
        "manifest_sha256": digest.hexdigest(),
        "duplicate_check": "decoded RGB pixels",
        "limitation": "Exact duplicates checked; near-duplicate images are not detected.",
        "calibration": "Normal images held out from official train/good only; no test tuning.",
    }
    return samples, summary
