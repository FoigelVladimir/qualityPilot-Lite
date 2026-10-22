"""One model per session; presentation controls never call inference."""

import gc
import sys
from collections.abc import MutableMapping
from pathlib import Path
from time import perf_counter
from typing import Any

from qualitypilot.decision.policy import Thresholds
from qualitypilot.inference.service import Inspector, Prediction


def reset_thresholds(state: MutableMapping[str, Any]) -> None:
    state["experimental"] = False
    state.pop("manual_accept", None)
    state.pop("manual_reject", None)


def release_model(state: MutableMapping[str, Any]) -> None:
    old = state.pop("_inspector", None)
    cuda = old is not None and str(old.model.device) == "cuda"
    del old
    gc.collect()
    torch = sys.modules.get("torch")
    if cuda and torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    state.pop("_load_ms", None)


def sync_model(state: MutableMapping[str, Any], key: tuple) -> None:
    if state.get("_model_key") != key:
        release_model(state)
        state.pop("_prediction", None)
        state["_model_key"] = key
        reset_thresholds(state)


def sync_image(state: MutableMapping[str, Any], token: str) -> bool:
    changed = state.get("_image_key") != token
    if changed:
        state["_image_key"] = token
        state.pop("_image", None)
        state.pop("_prediction", None)
    return changed


def run_inspection(state: MutableMapping[str, Any], artifact: Path, device: str) -> Prediction:
    if "_image" not in state:
        raise ValueError("Сначала выберите корректное изображение.")
    if "_inspector" not in state:
        started = perf_counter()
        state["_inspector"] = Inspector.load(artifact, device)
        state["_load_ms"] = (perf_counter() - started) * 1000
    result = state["_inspector"].predict(state["_image"])
    state["_prediction"] = result
    return result


def experimental_decision(score: float, saved: Thresholds, accept: float, reject: float) -> str:
    try:
        policy = Thresholds(float(accept), float(reject), saved.calibration_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("Нужны конечные пороги accept ≤ reject") from exc
    return policy.decide(score)
