"""Thin adapter around Anomalib's supported PyTorch PatchcoreModel.

No Engine/datamodule/postprocessor is involved: explicit normal-only manifests
control feature collection and calibration. The PatchCore algorithm itself,
coreset selection, nearest-neighbour scoring and map generation are Anomalib.
"""

import importlib.metadata
import logging
import os
import pickle
import random
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)
FORMAT_VERSION = 1
ANOMALIB_VERSION = "2.2.0"

if TYPE_CHECKING:
    import torch


def torch_runtime() -> ModuleType:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the ML extra: uv sync --extra dev --extra ml") from exc
    return torch


def select_device(requested: str) -> str:
    torch = torch_runtime()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA explicitly requested but unavailable; select cpu or auto")
    return "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"


def seed_runtime(seed: int, cpu_threads: int) -> None:
    torch = torch_runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class PatchcoreBackend:
    name = "patchcore"

    def __init__(self, options: dict, device: str = "auto"):
        torch = torch_runtime()
        version = importlib.metadata.version("anomalib")
        if version != ANOMALIB_VERSION:
            raise ValueError(f"Expected anomalib {ANOMALIB_VERSION}, found {version}")
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel
        from torchvision.transforms import v2

        self.options = dict(options)
        self.device = select_device(device)
        self.model = PatchcoreModel(
            layers=options["layers"],
            backbone=options["backbone"],
            pre_trained=options["pre_trained"],
            num_neighbors=options["num_neighbors"],
        ).to(self.device)
        self.model.requires_grad_(False)
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((options["image_size"], options["image_size"]), antialias=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def prepare_image(self, image: Image.Image) -> "torch.Tensor":
        return self.transform(image).unsqueeze(0).to(self.device)

    def synchronize(self) -> None:
        if self.device == "cuda":
            torch_runtime().cuda.synchronize()

    def fit(self, paths: list[Path], batch_size: int, sampling_ratio: float) -> dict:
        from qualitypilot.inference.images import load_image

        torch = torch_runtime()
        if not paths:
            raise ValueError("No normal training images")
        self.model.train()
        self.model.feature_extractor.eval()
        count = 0
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                inputs = torch.cat(
                    [self.prepare_image(load_image(p)) for p in paths[start : start + batch_size]]
                )
                embeddings = self.model(inputs)
                count += embeddings.shape[0]
                if start % (batch_size * 10) == 0:
                    LOGGER.info("Training normal features: %s/%s images", start, len(paths))
            LOGGER.info("Coreset selection from %s patches at ratio %s", count, sampling_ratio)
            self.model.subsample_embedding(sampling_ratio=sampling_ratio)
        self.model.eval()
        self.synchronize()
        bank = self.model.memory_bank
        if bank.ndim != 2 or not bank.numel():
            raise ValueError("Empty or malformed PatchCore memory bank")
        return {
            "training_images": len(paths),
            "patches_before_coreset": count,
            "memory_bank_shape": list(bank.shape),
            "memory_bank_bytes": bank.numel() * bank.element_size(),
        }

    def predict_prepared(self, inputs: "torch.Tensor") -> tuple[float, np.ndarray]:
        torch = torch_runtime()
        self.model.eval()
        with torch.inference_mode():
            output = self.model(inputs)
        scores, maps = output.pred_score, output.anomaly_map
        if scores.shape != (1,) or maps.shape != (1, 1, *inputs.shape[-2:]):
            raise ValueError(f"Unexpected PatchCore output shapes: {scores.shape}, {maps.shape}")
        return float(scores[0].cpu()), maps[0, 0].cpu().numpy().copy()

    def save(self, path: Path, metadata: dict) -> None:
        torch = torch_runtime()
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite model: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
        torch.save(
            {
                "format_version": FORMAT_VERSION,
                "anomalib_version": ANOMALIB_VERSION,
                "options": self.options,
                "metadata": metadata,
                "state_dict": state,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> tuple["PatchcoreBackend", dict]:
        torch = torch_runtime()
        if not path.is_file():
            raise FileNotFoundError(f"PatchCore model not found: {path}")
        try:
            bundle = torch.load(path, map_location="cpu", weights_only=True)
            if (
                not isinstance(bundle, dict)
                or bundle.get("format_version") != FORMAT_VERSION
                or bundle.get("anomalib_version") != ANOMALIB_VERSION
            ):
                raise ValueError("Unsupported checkpoint format or Anomalib version")
            options = dict(bundle["options"])
            metadata = bundle["metadata"]
            seed_runtime(
                metadata.get("config", {}).get("seed", 42),
                metadata.get("environment", {}).get("cpu_threads", 4),
            )
            options["pre_trained"] = False  # All backbone weights are in state_dict; no download.
            backend = cls(options, device)
            backend.options = bundle["options"]
            backend.model.load_state_dict(bundle["state_dict"], strict=True)
            backend.model.eval()
            if backend.model.memory_bank.numel() == 0:
                raise ValueError("Checkpoint memory bank is empty")
            return backend, metadata
        except (
            KeyError,
            TypeError,
            RuntimeError,
            ValueError,
            EOFError,
            pickle.UnpicklingError,
        ) as exc:
            raise ValueError(f"Incompatible PatchCore checkpoint {path}: {exc}") from exc
