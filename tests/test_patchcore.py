"""Optional ML tests use random local features; they are not quality experiments."""

import importlib.util
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from qualitypilot.config import Config
from qualitypilot.decision.policy import calibrate
from qualitypilot.inference.images import load_image
from qualitypilot.inference.service import Inspector

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="ML extra not installed; install with uv sync --extra ml",
)


def test_cpu_patchcore_roundtrip_and_shapes(config: Config, tmp_path: Path):
    from qualitypilot.models.patchcore import PatchcoreBackend, seed_runtime, torch_runtime

    seed_runtime(42, 2)
    options = {
        "backbone": "resnet18",
        "layers": ["layer2", "layer3"],
        "pre_trained": False,
        "image_size": 32,
        "num_neighbors": 1,
    }
    backend = PatchcoreBackend(options, "cpu")
    paths = sorted((config.data_root / "bottle/train/good").glob("*.png"))[:4]
    stats = backend.fit(paths, 2, 0.25)
    assert stats["memory_bank_shape"] == [16, 384]
    assert not any(p.requires_grad for p in backend.model.parameters())
    batch = backend.prepare_image(load_image(paths[0]))
    assert batch.shape == (1, 3, 32, 32)
    score, values = backend.predict_prepared(batch)
    assert values.shape == (32, 32) and np.isfinite(values).all()
    thresholds = calibrate(np.array([score, score + 1, score + 2]), 0.5, 0.9)
    artifact = tmp_path / "model.pt"
    backend.save(
        artifact,
        {
            "category": "bottle",
            "thresholds": asdict(thresholds),
            "pixel_threshold": float(values.max()),
        },
    )
    # A fresh process can have different TF32 defaults; reload must restore runtime policy.
    torch_runtime().backends.cudnn.allow_tf32 = True
    torch_runtime().backends.cuda.matmul.allow_tf32 = True
    inspector = Inspector.load(artifact, "cpu")
    assert not torch_runtime().backends.cudnn.allow_tf32
    assert not torch_runtime().backends.cuda.matmul.allow_tf32
    prediction = inspector.predict(paths[0])
    assert prediction.anomaly_map.shape == (24, 24)
    assert prediction.score == pytest.approx(score, abs=1e-6)
    native = inspector.predict(paths[0], map_resolution="model")
    np.testing.assert_allclose(native.anomaly_map, values, rtol=1e-6, atol=1e-6)
    json.dumps(prediction.to_dict(), allow_nan=False)
    assert len(prediction.to_dict(include_map=True)["anomaly_map"]) == 24
    with pytest.raises(FileExistsError):
        backend.save(artifact, {})


def test_unavailable_cuda_falls_back(monkeypatch):
    from qualitypilot.models.patchcore import select_device, torch_runtime

    monkeypatch.setattr(torch_runtime().cuda, "is_available", lambda: False)
    assert select_device("auto") == "cpu"
    assert select_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        select_device("cuda")


def test_incompatible_checkpoint(tmp_path: Path):
    from qualitypilot.models.patchcore import PatchcoreBackend, torch_runtime

    artifact = tmp_path / "bad.pt"
    torch_runtime().save({"format_version": -1}, artifact)
    with pytest.raises(ValueError, match="Incompatible"):
        Inspector.load(artifact, "cpu")
    artifact.write_bytes(b"not a torch file")
    with pytest.raises(ValueError, match="Incompatible"):
        Inspector.load(artifact, "cpu")
    with pytest.raises(FileNotFoundError):
        PatchcoreBackend.load(tmp_path / "missing.pt", "cpu")


@pytest.mark.integration
def test_real_patchcore_smoke():
    artifact = os.environ.get("QUALITYPILOT_SMOKE_MODEL")
    image = os.environ.get("QUALITYPILOT_SMOKE_IMAGE")
    if not artifact or not image:
        pytest.skip("Real artifact smoke NOT RUN: set QUALITYPILOT_SMOKE_MODEL and _IMAGE")
    assert Path(artifact).is_file() and Path(image).is_file()
    inspector = Inspector.load(Path(artifact), "cpu")
    first = inspector.predict(Path(image), map_resolution="model")
    second = inspector.predict(Path(image), map_resolution="model")
    assert first.score == pytest.approx(second.score, abs=1e-6)
    assert first.category == "bottle" and first.model == "patchcore"
    assert first.anomaly_map.shape == (224, 224)
    np.testing.assert_allclose(first.anomaly_map, second.anomaly_map, rtol=1e-6, atol=1e-6)
