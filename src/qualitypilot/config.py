"""Validated YAML settings with paths independent of the working directory."""

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    data_root: Path
    category: str
    output_dir: Path
    model_path: Path
    image_size: int = 128
    calibration_fraction: float = 0.2
    seed: int = 42
    n_estimators: int = 200
    accept_quantile: float = 0.95
    reject_quantile: float = 0.99

    def __post_init__(self) -> None:
        if not self.category or Path(self.category).name != self.category:
            raise ValueError("category must be a single directory name")
        if self.category in {".", ".."} or "/" in self.category or "\\" in self.category:
            raise ValueError("Invalid category")
        if not 0 < self.calibration_fraction < 1:
            raise ValueError("calibration_fraction must be between 0 and 1")
        if not 0 < self.accept_quantile < self.reject_quantile < 1:
            raise ValueError("Require 0 < accept_quantile < reject_quantile < 1")
        if self.image_size < 8 or self.n_estimators < 1 or self.seed < 0:
            raise ValueError("Invalid image_size, n_estimators or seed")

    def to_dict(self) -> dict:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def load_config(path: Path) -> Config:
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f", строка {mark.line + 1}, столбец {mark.column + 1}" if mark else ""
        raise ValueError(f"Исправьте синтаксис YAML в {path}{location}") from exc
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for key in ("data_root", "output_dir", "model_path"):
        value = Path(values[key]).expanduser()
        values[key] = (path.resolve().parent / value).resolve()
    return Config(**values)
