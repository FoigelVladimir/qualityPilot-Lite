"""Configuration frozen before test, separately from the existing baseline settings."""

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PatchcoreConfig:
    baseline_config: Path
    reference_run: Path
    output_root: Path
    category: str = "bottle"
    backbone: str = "resnet18"
    layers: tuple[str, ...] = ("layer2", "layer3")
    pre_trained: bool = True
    image_size: int = 224
    coreset_sampling_ratio: float = 0.01
    num_neighbors: int = 9
    train_batch_size: int = 4
    device: str = "auto"
    seed: int = 42
    accept_quantile: float = 0.95
    reject_quantile: float = 0.99
    pixel_quantile: float = 0.99
    warmup_images: int = 3
    cpu_threads: int = 4

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if self.image_size < 32 or self.image_size % 8:
            raise ValueError("image_size must be >=32 and divisible by 8")
        if not self.layers or not self.backbone:
            raise ValueError("backbone/layers must not be empty")
        if not 0 < self.coreset_sampling_ratio <= 1:
            raise ValueError("Invalid coreset_sampling_ratio")
        if min(self.num_neighbors, self.train_batch_size, self.warmup_images, self.cpu_threads) < 1:
            raise ValueError("Counts must be positive")
        if self.seed < 0 or not 0 < self.accept_quantile < self.reject_quantile < 1:
            raise ValueError("Invalid seed or image quantiles")
        if not 0 < self.pixel_quantile < 1:
            raise ValueError("Invalid pixel_quantile")

    def to_dict(self) -> dict:
        return {
            k: str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v
            for k, v in asdict(self).items()
        }

    def model_options(self) -> dict:
        return {
            "backbone": self.backbone,
            "layers": list(self.layers),
            "pre_trained": self.pre_trained,
            "image_size": self.image_size,
            "num_neighbors": self.num_neighbors,
        }


def load_patchcore_config(path: Path) -> PatchcoreConfig:
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Expected a YAML mapping")
        for key in ("baseline_config", "reference_run", "output_root"):
            values[key] = (path.resolve().parent / Path(values[key]).expanduser()).resolve()
        if "layers" in values:
            values["layers"] = tuple(values["layers"])
        return PatchcoreConfig(**values)
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid PatchCore configuration {path}: {exc}") from exc
