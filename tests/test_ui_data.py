import shutil
from dataclasses import asdict
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from qualitypilot.decision.policy import Thresholds
from qualitypilot.ui.artifacts import discover_runs, load_reports, load_settings
from qualitypilot.ui.inspection import decode_upload, map_views
from qualitypilot.ui.state import experimental_decision, reset_thresholds, sync_image, sync_model


def test_run_discovery_and_no_model_reports(ui_project):
    root, run = ui_project
    settings = load_settings(root)
    assert settings.selected_run == run.name
    assert discover_runs(settings.runs_root) == [run]
    failed = root / "runs/failed"
    shutil.copytree(run, failed)
    (failed / "status.json").write_text('{"status": "FAILED"}')
    with pytest.raises(ValueError, match="не завершён"):
        load_reports(failed)
    incomplete = root / "runs/incomplete"
    incomplete.mkdir()
    (incomplete / "status.json").write_text('{"status": "COMPLETE"}')
    (root / "runs/audit").mkdir()
    (run / "model.pt").unlink()
    assert discover_runs(settings.runs_root) == [run]
    assert len(load_reports(run)["tables"]["baseline"]) == 8


def test_mismatched_results_are_rejected(ui_project):
    _, run = ui_project
    path = run / "patchcore/results.csv"
    import pandas as pd

    frame = pd.read_csv(path)
    frame.loc[0, "label"] = 1 - frame.loc[0, "label"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="labels"):
        load_reports(run)


def test_corrupted_manifest_is_rejected(ui_project):
    _, run = ui_project
    path = run / "manifest.csv"
    path.write_text(path.read_text().replace("calibration", "train", 1))
    with pytest.raises(ValueError, match="manifest"):
        load_reports(run)


@pytest.mark.parametrize("kind", ["PNG", "JPEG"])
def test_upload_decode_rgb_geometry(kind):
    source = Image.new("RGB", (13, 7), (40, 80, 120))
    data = BytesIO()
    source.save(data, format=kind)
    image = decode_upload(data.getvalue())
    assert image.mode == "RGB" and image.size == (13, 7)


def test_upload_invalid_and_limits():
    with pytest.raises(ValueError, match="декодировать"):
        decode_upload(b"broken.png")
    with pytest.raises(ValueError, match="10 МБ"):
        decode_upload(b"x" * (10 * 1024 * 1024 + 1))
    data = BytesIO()
    Image.new("1", (5000, 4001)).save(data, format="PNG")
    with pytest.raises(ValueError, match="20 млн"):
        decode_upload(data.getvalue())
    data = BytesIO()
    Image.new("RGB", (4, 4)).save(data, format="GIF")
    with pytest.raises(ValueError, match="JPG"):
        decode_upload(data.getvalue())


def test_display_alpha_and_temporary_policy_are_pure():
    image = Image.new("RGB", (10, 8), (40, 40, 40))
    values = np.linspace(0, 1, 80, dtype=np.float32).reshape(8, 10)
    original = values.copy()
    low, high = map_views(image, values, 0.7, 0.2), map_views(image, values, 0.7, 0.8)
    assert np.array_equal(low["mask"], high["mask"])
    assert np.array_equal(low["heatmap"], high["heatmap"])
    assert not np.array_equal(low["overlay"], high["overlay"])
    assert np.array_equal(values, original)
    saved = Thresholds(0.5, 0.8, 42)
    snapshot = asdict(saved)
    assert experimental_decision(0.6, saved, 0.7, 0.7) == "ACCEPT"
    assert saved.decide(0.6) == "REVIEW" and asdict(saved) == snapshot
    for a, b in ((1, 0), (np.nan, 1), (0, np.inf), (None, 1)):
        with pytest.raises(ValueError):
            experimental_decision(0.6, saved, a, b)


def test_session_reset_does_not_keep_stale_result():
    state = {
        "_model_key": ("a",),
        "_prediction": object(),
        "experimental": True,
        "manual_accept": 1.0,
        "manual_reject": 2.0,
    }
    sync_model(state, ("b",))
    assert "_prediction" not in state and "manual_accept" not in state
    assert not state["experimental"]
    state["_prediction"] = object()
    sync_image(state, "new-image")
    assert "_prediction" not in state
    reset_thresholds(state)
    assert not state["experimental"]
