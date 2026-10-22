from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from qualitypilot.decision.policy import Thresholds
from qualitypilot.inference.images import ImageInput, load_image, resize_map
from qualitypilot.models.baseline import Baseline


class InferenceBackend(Protocol):
    name: str
    device: str

    def prepare_image(self, image: Any) -> Any: ...

    def predict_prepared(self, inputs: Any) -> tuple[float, np.ndarray | None]: ...

    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class Prediction:
    score: float
    decision: str | None
    latency_ms: float
    anomaly_map: np.ndarray | None = None
    model: str = "baseline"
    category: str = "unknown"
    device: str = "cpu"
    timings_ms: dict[str, float] = field(default_factory=dict)
    original_size: tuple[int, int] = (0, 0)
    pixel_threshold: float | None = None

    def to_dict(self, include_map: bool = False) -> dict:
        """Small JSON-safe response by default; full maps are opt-in."""
        values = self.anomaly_map
        return {
            "score": self.score,
            "decision": self.decision,
            "latency_ms": self.latency_ms,
            "anomaly_map": None
            if values is None
            else values.tolist()
            if include_map
            else {"shape": list(values.shape), "dtype": str(values.dtype)},
            "model": self.model,
            "category": self.category,
            "device": self.device,
            "timings_ms": self.timings_ms,
            "original_size": list(self.original_size),
            "pixel_threshold": self.pixel_threshold,
        }


class Inspector:
    def __init__(
        self,
        model: InferenceBackend,
        thresholds: Thresholds | None,
        category: str = "unknown",
        pixel_threshold: float | None = None,
    ):
        self.model = model
        self.thresholds = thresholds
        self.category = category
        self.pixel_threshold = pixel_threshold

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "Inspector":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model not found: {path}. Run qualitypilot baseline or patchcore."
            )
        if path.suffix == ".pt":
            from qualitypilot.models.patchcore import PatchcoreBackend

            model, metadata = PatchcoreBackend.load(path, device)
        elif path.suffix == ".joblib":
            model, metadata = Baseline.load(path)
        else:
            raise ValueError("Unsupported model format; expected .pt or .joblib")
        try:
            category = metadata.get(
                "category", metadata.get("config", {}).get("category", "unknown")
            )
            return cls(
                model,
                Thresholds(**metadata["thresholds"]),
                category,
                metadata.get("pixel_threshold"),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Model artifact lacks compatible calibrated metadata") from exc

    def predict(self, path: ImageInput, map_resolution: str = "original") -> Prediction:
        if map_resolution not in {"original", "model"}:
            raise ValueError("map_resolution must be original or model")
        self.model.synchronize()
        start = perf_counter()
        image = load_image(path)
        inputs = self.model.prepare_image(image)
        self.model.synchronize()
        prepared = perf_counter()
        score, anomaly_map = self.model.predict_prepared(inputs)
        self.model.synchronize()
        inferred = perf_counter()
        if anomaly_map is not None:
            anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
            if anomaly_map.ndim != 2 or not np.isfinite(anomaly_map).all():
                raise ValueError("Model produced an invalid anomaly map")
            if map_resolution == "original":
                anomaly_map = resize_map(anomaly_map, image.size)
        if not np.isfinite(score):
            raise ValueError("Model produced a nonfinite score")
        decision = self.thresholds.decide(score) if self.thresholds is not None else None
        end = perf_counter()
        timings = {
            "preprocessing": (prepared - start) * 1000,
            "inference": (inferred - prepared) * 1000,
            "postprocessing": (end - inferred) * 1000,
        }
        return Prediction(
            float(score),
            decision,
            (end - start) * 1000,
            anomaly_map,
            self.model.name,
            self.category,
            str(self.model.device),
            timings,
            image.size,
            self.pixel_threshold,
        )
