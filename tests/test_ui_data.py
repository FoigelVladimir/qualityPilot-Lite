import shutil
from dataclasses import asdict
from io import BytesIO
from math import ceil

import numpy as np
import pytest
from PIL import Image

from qualitypilot.decision.policy import Thresholds
from qualitypilot.ui.artifacts import discover_runs, load_reports, load_settings
from qualitypilot.ui.inspection import decode_upload, map_views
from qualitypilot.ui.presentation import (
    CASES,
    DECISIONS,
    DEFECT_NAMES,
    VIEWS,
    prediction_table,
    run_label,
)
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


@pytest.mark.parametrize("kind", ["empty", "full", "mixed"])
def test_contour_is_boundary_of_unchanged_mask(kind):
    image = Image.new("RGB", (6, 5), (40, 50, 60))
    values = np.zeros((5, 6), dtype=np.float32)
    if kind == "full":
        values[:] = 1
    elif kind == "mixed":
        values[1:4, 1:5] = 1
        values[2, 2] = 0
    original = values.copy()
    views = map_views(image, values, 0.5, 0.4)
    mask = values > 0.5
    expected = np.zeros_like(mask)
    for y, x in np.ndindex(mask.shape):
        if mask[y, x]:
            expected[y, x] = any(
                yy < 0 or xx < 0 or yy >= 5 or xx >= 6 or not mask[yy, xx]
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
            )
    np.testing.assert_array_equal(views["boundary"], expected)
    np.testing.assert_array_equal(views["mask"], mask.astype(np.uint8) * 255)
    np.testing.assert_array_equal(values, original)
    # Independently define a clipped square neighbourhood around the exact boundary.
    outer = np.zeros_like(mask)
    inner = np.zeros_like(mask)
    for y, x in np.argwhere(expected):
        outer[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = True
        inner[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = True
    np.testing.assert_array_equal(views["contour"][~outer], np.asarray(image)[~outer])
    assert np.all(views["contour"][outer & ~inner] == (23, 44, 60))
    assert np.all(views["contour"][inner] == (255, 210, 0))
    assert np.all(views["contour"][expected] == (255, 210, 0))
    assert views["contour"].shape == (5, 6, 3) and views["maximum"] == 0.75


@pytest.mark.parametrize("size", [240, 900])
@pytest.mark.parametrize("background", [(0, 0, 0), (255, 255, 255)])
def test_contour_has_two_tone_scaled_stroke(size, background):
    image = Image.new("RGB", (size, size), background)
    values = np.zeros((size, size), dtype=np.float32)
    lo, hi = size // 3, 2 * size // 3
    values[lo:hi, lo:hi] = 1
    before = values.copy()
    views = map_views(image, values, 0.5, 0.45)
    y = size // 2
    outer, inner = max(2, ceil(size / 320)), max(1, ceil(size / 640))
    np.testing.assert_array_equal(views["contour"][y, lo - outer], (23, 44, 60))
    np.testing.assert_array_equal(views["contour"][y, lo - inner], (255, 210, 0))
    np.testing.assert_array_equal(views["contour"][y, lo - outer - 1], background)
    np.testing.assert_array_equal(views["contour"][y, y], background)
    np.testing.assert_array_equal(values, before)
    np.testing.assert_array_equal(views["mask"], (before > 0.5).astype(np.uint8) * 255)


def test_prediction_numbers_sort_numerically_without_rounding():
    import pandas as pd

    rows = pd.DataFrame(
        {
            "path": ["a.png", "b.png", "c.png"],
            "defect": ["good"] * 3,
            "label": [0] * 3,
            "binary_prediction": [0] * 3,
            "decision": ["ACCEPT"] * 3,
            **{
                key: [8.04, 19.77, 9.10]
                for key in (
                    "score",
                    "latency_ms",
                    "preprocessing_ms",
                    "inference_ms",
                    "postprocessing_ms",
                )
            },
        }
    )
    before = rows.copy(deep=True)
    display = prediction_table(rows)
    for name in (
        "Оценка аномальности",
        "Обработка, мс",
        "Подготовка, мс",
        "Модель, мс",
        "Решение, мс",
    ):
        assert pd.api.types.is_numeric_dtype(display[name])
        assert display.sort_values(name)[name].tolist() == [8.04, 9.10, 19.77]
    pd.testing.assert_frame_equal(rows, before)


def test_presentation_does_not_mutate_artifact_rows(ui_project):
    import pandas as pd

    _, run = ui_project
    rows = load_reports(run)["tables"]["baseline"]
    before = rows.copy(deep=True)
    display = prediction_table(rows)
    pd.testing.assert_frame_equal(rows, before)
    assert "Оценка аномальности" in display and "score" not in display
    assert set(display["Разметка"]) == {"Исправно", "Дефект"}
    assert all(
        value.endswith(("(ACCEPT)", "(REVIEW)", "(REJECT)")) for value in display["Рекомендация"]
    )
    assert set(DEFECT_NAMES) == {"good", "broken_large", "broken_small", "contamination"}
    assert set(DECISIONS) == {"ACCEPT", "REVIEW", "REJECT"}
    assert set(CASES) == {"TP", "FP", "FN", "TN"} and set(VIEWS) == {
        "overlay",
        "heatmap",
        "contour",
        "mask",
    }
    first = run.with_name("bottle-patchcore-20261021T103450Z-11111111")
    second = run.with_name("bottle-patchcore-20261021T103450Z-22222222")
    assert run_label(first) != run_label(second) and "21.10.2026" in run_label(first)
