import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qualitypilot.decision.policy import Thresholds
from qualitypilot.inference.service import Prediction

pytest.importorskip("streamlit", reason="UI extra not installed")
from streamlit.testing.v1 import AppTest


def launch(root: str):
    from pathlib import Path

    from qualitypilot.ui.app import main

    main(Path(root))


def test_navigation_reports_about_without_loader(ui_project, monkeypatch):
    root, run = ui_project
    (run / "model.pt").unlink()
    from qualitypilot.ui.state import Inspector

    def forbidden(*args, **kwargs):
        raise AssertionError("Reports/About must not load models")

    monkeypatch.setattr(Inspector, "load", forbidden)
    at = AppTest.from_function(launch, args=(str(root),)).run()
    assert not at.exception and at.warning
    at.radio(key="page").set_value("Результаты эксперимента").run()
    assert not at.exception and len(at.dataframe) >= 1 and len(at.table) == 2
    at.selectbox(key="case").select("FN").run()
    assert not at.exception
    at.radio(key="page").set_value("О проекте").run()
    assert not at.exception and "О проекте" in [h.value for h in at.header]


def test_missing_config_keeps_about_available(tmp_path: Path):
    at = AppTest.from_function(launch, args=(str(tmp_path),)).run()
    assert at.error and not at.exception
    at.radio(key="page").set_value("О проекте").run()
    assert not at.exception and at.warning


def test_cuda_error_and_mismatched_reports_do_not_break_navigation(ui_project, monkeypatch):
    root, run = ui_project
    from qualitypilot.ui.state import Inspector

    def unavailable(*args, **kwargs):
        raise ValueError("CUDA explicitly requested but unavailable")

    monkeypatch.setattr(Inspector, "load", unavailable)
    at = AppTest.from_function(launch, args=(str(root),)).run()
    at.selectbox(key="device").select("cuda").run()
    at.button(key="inspect").click().run()
    assert not at.exception and any("CUDA" in item.value for item in at.error)
    assert "_prediction" not in at.session_state
    path = run / "patchcore/results.csv"
    path.write_text(path.read_text().replace("bottle/test/", "other/test/", 1))
    at.radio(key="page").set_value("Результаты эксперимента").run()
    assert not at.exception and at.error and not at.table
    at.radio(key="page").set_value("О проекте").run()
    assert not at.exception


def test_baseline_state_changes_and_no_map(ui_project, monkeypatch):
    root, _ = ui_project
    counts = {"load": 0, "predict": 0}
    from qualitypilot.ui.state import Inspector

    class Fake:
        model = SimpleNamespace(name="baseline", device="cpu")
        thresholds = Thresholds(0.5, 0.8, 42)

        def predict(self, image):
            counts["predict"] += 1
            return Prediction(
                0.6, "REVIEW", 1.0, None, "baseline", "bottle", "cpu", original_size=image.size
            )

    def load(*args, **kwargs):
        counts["load"] += 1
        return Fake()

    monkeypatch.setattr(Inspector, "load", load)
    at = AppTest.from_function(launch, args=(str(root),)).run()
    at.selectbox(key="model").select("Baseline").run()
    at.button(key="inspect").click().run()
    assert not at.exception and counts == {"load": 1, "predict": 1}
    assert any("не выделяет области" in item.value for item in at.info)
    assert at.metric[0].label == "Оценка аномальности" and at.metric[0].value == "0,60"
    at.toggle(key="experimental").set_value(True).run()
    at.number_input(key="manual_accept").set_value(0.7).run()
    assert counts == {"load": 1, "predict": 1}
    assert any("Рекомендуется принять" in item.value for item in at.success)
    assert any("Код рекомендации: ACCEPT" in item.value for item in at.caption)
    assert at.session_state["_prediction"].decision == "REVIEW"
    at.button(key="reset_thresholds").click().run()
    assert not at.toggle(key="experimental").value
    at.selectbox(key="example").select_index(1).run()
    assert "_prediction" not in at.session_state and counts["predict"] == 1
    at.button(key="inspect").click().run()
    assert counts == {"load": 1, "predict": 2}
    at.selectbox(key="model").select("PatchCore").run()
    assert "_inspector" not in at.session_state and "_prediction" not in at.session_state


def test_patchcore_alpha_and_thresholds_do_not_infer(ui_project, monkeypatch):
    root, _ = ui_project
    counts = {"load": 0, "predict": 0}
    from qualitypilot.ui.state import Inspector

    class Fake:
        model = SimpleNamespace(name="patchcore", device="cpu")
        thresholds = Thresholds(0.5, 0.8, 42)

        def predict(self, image):
            counts["predict"] += 1
            return Prediction(
                0.6,
                "REVIEW",
                2.0,
                np.full((image.height, image.width), 0.7, dtype=np.float32),
                "patchcore",
                "bottle",
                "cpu",
                original_size=image.size,
                pixel_threshold=0.65,
            )

    def load(*args, **kwargs):
        counts["load"] += 1
        return Fake()

    monkeypatch.setattr(Inspector, "load", load)
    at = AppTest.from_function(launch, args=(str(root),)).run()
    at.button(key="inspect").click().run()
    assert not at.exception and counts["predict"] == 1
    original = at.session_state["_prediction"].anomaly_map.copy()
    prediction = at.session_state["_prediction"]
    inspector = at.session_state["_inspector"]
    assert at.selectbox(key="view_mode").value == "overlay"
    for mode in ("heatmap", "contour", "mask", "overlay"):
        at.selectbox(key="view_mode").select(mode).run()
        assert not at.exception and at.session_state["_prediction"] is prediction
        assert at.session_state["_inspector"] is inspector
        assert prediction.score == 0.6 and prediction.decision == "REVIEW"
        np.testing.assert_array_equal(prediction.anomaly_map, original)
    at.toggle(key="large_view").set_value(True).run()
    at.selectbox(key="large_target").select("original").run()
    at.selectbox(key="large_target").select("result").run()
    at.toggle(key="large_view").set_value(False).run()
    at.slider(key="alpha").set_value(0.8).run()
    at.toggle(key="experimental").set_value(True).run()
    at.number_input(key="manual_accept").set_value(0.9).run()
    assert not at.exception and at.error  # accept > reject is rejected, not used.
    assert counts == {"load": 1, "predict": 1}
    assert np.array_equal(original, at.session_state["_prediction"].anomaly_map)
    at.selectbox(key="device").select("cuda").run()
    assert "_prediction" not in at.session_state and not at.session_state["experimental"]


def test_russian_labels_keep_internal_values_and_reports(ui_project):
    root, run = ui_project
    at = AppTest.from_function(launch, args=(str(root),)).run()
    assert not at.exception
    assert at.sidebar.selectbox(key="model").value == "PatchCore"
    assert at.sidebar.selectbox(key="device").value == "auto"
    assert at.sidebar.selectbox(key="run").value == run
    assert at.selectbox(key="example_type").value == "good"
    assert "Исправная бутылка" in at.selectbox(key="example_type").options
    assert at.selectbox(key="example").options[0] == "000.png"
    at.radio(key="page").set_value("Результаты эксперимента").run()
    assert not at.exception and not at.error
    assert "Базовая модель" in at.dataframe[0].value.columns
    predictions = at.dataframe[1]
    for name in ("Оценка аномальности", "Обработка, мс"):
        assert pd.api.types.is_numeric_dtype(predictions.value[name])
        options = json.loads(predictions.proto.columns)[name]
        assert options["type_config"] == {"type": "number", "format": "%.2f"}
    assert "Прогноз: дефект" in at.table[0].value.columns
    assert any("FN — дефект пропущен" == item for item in at.selectbox(key="case").options)
    at.selectbox(key="case").select("FN").run()
    assert at.selectbox(key="case").value == "FN"


def test_run_change_and_navigation_reset_cached_state(ui_project, monkeypatch):
    import shutil

    root, run = ui_project
    other = run.with_name("bottle-patchcore-second")
    shutil.copytree(run, other)
    from qualitypilot.ui.state import Inspector

    counts = {"load": 0, "predict": 0}

    class Fake:
        model = SimpleNamespace(name="baseline", device="cpu")
        thresholds = Thresholds(0.5, 0.8, 42)

        def predict(self, image):
            counts["predict"] += 1
            return Prediction(0.6, "REVIEW", 1.0, model="baseline", original_size=image.size)

    def load(*args, **kwargs):
        counts["load"] += 1
        return Fake()

    monkeypatch.setattr(Inspector, "load", load)
    at = AppTest.from_function(launch, args=(str(root),)).run()
    at.selectbox(key="model").select("Baseline").run()
    at.button(key="inspect").click().run()
    at.toggle(key="experimental").set_value(True).run()
    at.number_input(key="manual_accept").set_value(0.7).run()
    pred = at.session_state["_prediction"]
    at.radio(key="page").set_value("Результаты эксперимента").run()
    assert not at.exception and counts == {"load": 1, "predict": 1}
    at.radio(key="page").set_value("Проверка изображения").run()
    assert at.session_state["_prediction"] is pred
    at.selectbox(key="run").select(other).run()
    assert not at.exception and "_prediction" not in at.session_state
    assert "_inspector" not in at.session_state and not at.session_state["experimental"]
    at.button(key="inspect").click().run()
    assert counts == {"load": 2, "predict": 2}


def test_optional_ground_truth_is_display_only(ui_project, monkeypatch):
    root, _ = ui_project
    from qualitypilot.ui.state import Inspector

    counts = {"load": 0, "predict": 0}

    class Fake:
        model = SimpleNamespace(name="patchcore", device="cpu")
        thresholds = Thresholds(0.5, 0.8, 42)

        def predict(self, image):
            counts["predict"] += 1
            return Prediction(
                0.6,
                "REVIEW",
                1.0,
                np.zeros((image.height, image.width), np.float32),
                "patchcore",
                "bottle",
                "cpu",
                original_size=image.size,
                pixel_threshold=0.65,
            )

    def load(*args, **kwargs):
        counts["load"] += 1
        return Fake()

    monkeypatch.setattr(Inspector, "load", load)
    at = AppTest.from_function(launch, args=(str(root),)).run()
    at.selectbox(key="example_type").select("scratch").run()
    assert not at.exception and not at.error
    assert at.selectbox(key="example").value == "bottle/test/scratch/000.png"
    at.button(key="inspect").click().run()
    pred = at.session_state["_prediction"]
    assert not at.toggle(key="show_gt").value
    at.toggle(key="show_gt").set_value(True).run()
    at.selectbox(key="view_mode").select("contour").run()
    at.toggle(key="show_gt").set_value(False).run()
    assert not at.exception and at.session_state["_prediction"] is pred
    assert counts == {"load": 1, "predict": 1} and not pred.anomaly_map.any()


def assert_report_error_keeps_navigation(root, filename, detail):
    at = AppTest.from_function(launch, args=(str(root),)).run()
    at.radio(key="page").set_value("Результаты эксперимента").run()
    assert not at.exception and not at.dataframe and not at.table
    message = " ".join(item.value for item in at.error)
    assert "Сравнение недоступно" in message and filename in message and detail in message
    assert "Traceback" not in message
    at.radio(key="page").set_value("Проверка изображения").run()
    assert not at.exception and not at.error and at.button(key="inspect")
    at.radio(key="page").set_value("О проекте").run()
    assert not at.exception and "О проекте" in [item.value for item in at.header]


@pytest.mark.parametrize(
    "column", ["path", "label", "defect", "score", "decision", "binary_prediction", "latency_ms"]
)
def test_missing_csv_columns_have_friendly_error(ui_project, column):
    root, run = ui_project
    path = run / "patchcore/results.csv"
    frame = pd.read_csv(path)
    frame.rename(columns={column: f"wrong_{column}"}).to_csv(path, index=False)
    assert_report_error_keeps_navigation(root, "results.csv", column)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("score", "bad"),
        ("score", "inf"),
        ("binary_prediction", 2),
        ("decision", "MAYBE"),
        ("latency_ms", "bad"),
        ("path", ""),
    ],
)
def test_invalid_csv_values_have_friendly_error(ui_project, column, value):
    root, run = ui_project
    path = run / "baseline/results.csv"
    frame = pd.read_csv(path).astype({column: object})
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    assert_report_error_keeps_navigation(root, "results.csv", column)


@pytest.mark.parametrize("content", ["", 'path,label\n"unterminated'])
def test_unreadable_csv_has_friendly_error(ui_project, content):
    root, run = ui_project
    (run / "patchcore/results.csv").write_text(content, "utf-8")
    assert_report_error_keeps_navigation(root, "results.csv", "CSV")


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        ("patchcore/metrics.json", "{"),
        ("baseline/metrics.json", "[]"),
        ("comparison.json", '{"manifest_sha256": "invalid"}'),
        ("experiment.json", '{"dataset": []}'),
        ("dataset_summary.json", '{"manifest_sha256": null}'),
    ],
)
def test_invalid_json_files_have_friendly_error(ui_project, relative, content):
    root, run = ui_project
    (run / relative).write_text(content, "utf-8")
    if relative == "dataset_summary.json":
        # This artifact is also required by local examples; upload remains available.
        at = AppTest.from_function(launch, args=(str(root),)).run()
        assert at.error and not at.exception
        at.radio(key="source").set_value("Загрузить JPG/PNG").run()
        assert not at.exception and not at.error
        at.radio(key="page").set_value("О проекте").run()
        assert not at.exception
        at.radio(key="page").set_value("Результаты эксперимента").run()
        assert not at.exception and at.error and not at.dataframe
        assert "dataset_summary.json" in at.error[0].value
    else:
        assert_report_error_keeps_navigation(root, Path(relative).name, "")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_auroc", {}),
        ("image_auroc", {"value": "1"}),
        ("pixel_auroc", {"value": None}),
        ("dice", {"value": float("nan")}),
        ("single_threshold", []),
        ("single_threshold.confusion_matrix", [[1, 2, 3]]),
        ("single_threshold.threshold", "bad"),
        ("single_threshold.precision", None),
        ("two_thresholds", {}),
        ("two_thresholds.review_count", "4"),
        ("two_thresholds.review_fraction", 2),
        ("latency_ms", []),
        ("latency_ms.median", "slow"),
        ("latency_ms.cuda_synchronized", "false"),
        ("pixel_evaluation", []),
        ("pixel_evaluation", {"pixel_threshold": "bad"}),
    ],
)
def test_invalid_nested_metrics_rejected_before_render(ui_project, field, value):
    root, run = ui_project
    path = run / "patchcore/metrics.json"
    data = json.loads(path.read_text())
    target = data
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    path.write_text(json.dumps(data), "utf-8")
    # Keep both copies consistent: the schema, rather than equality, must catch the problem.
    comparison = json.loads((run / "comparison.json").read_text())
    comparison["patchcore"] = data
    (run / "comparison.json").write_text(json.dumps(comparison), "utf-8")
    assert_report_error_keeps_navigation(root, "metrics.json", field.split(".")[0])


def test_invalid_optional_cpu_json_rejected_before_render(ui_project):
    root, run = ui_project
    path = run / "comparison.json"
    data = json.loads(path.read_text())
    data["patchcore_cpu_latency_ms"] = [1, 2]
    path.write_text(json.dumps(data), "utf-8")
    assert_report_error_keeps_navigation(root, "comparison.json", "patchcore_cpu_latency_ms")


def test_invalid_manifest_keeps_upload_and_about_available(ui_project):
    root, run = ui_project
    path = run / "manifest.csv"
    path.write_text(path.read_text().replace("path,", "image_path,", 1), "utf-8")
    at = AppTest.from_function(launch, args=(str(root),)).run()
    assert not at.exception and "manifest.csv" in at.error[0].value
    assert "path" in at.error[0].value
    at.radio(key="source").set_value("Загрузить JPG/PNG").run()
    assert not at.exception and not at.error
    at.radio(key="page").set_value("Результаты эксперимента").run()
    assert not at.exception and at.error and not at.dataframe
    at.radio(key="page").set_value("О проекте").run()
    assert not at.exception


def test_syntax_error_baseline_yaml_keeps_reports_and_about_available(ui_project):
    root, _ = ui_project
    (root / "base.yaml").write_text("data_root: [", "utf-8")
    at = AppTest.from_function(launch, args=(str(root),)).run()
    assert not at.exception and at.error
    assert "base.yaml" in at.error[0].value and "синтаксис YAML" in at.error[0].value
    assert "Traceback" not in at.error[0].value
    at.radio(key="page").set_value("Результаты эксперимента").run()
    assert not at.exception and not at.error and at.dataframe and len(at.table) == 2
    assert any("base.yaml" in item.value and "синтаксис YAML" in item.value for item in at.warning)
    at.radio(key="page").set_value("О проекте").run()
    assert not at.exception and not at.error


def test_corrupt_optional_examples_keeps_reports_available(ui_project):
    root, run = ui_project
    folder = run / "examples"
    folder.mkdir()
    (folder / "examples.json").write_text('{"tp": []}', "utf-8")
    at = AppTest.from_function(launch, args=(str(root),)).run()
    at.radio(key="page").set_value("Результаты эксперимента").run()
    assert not at.exception and not at.error and at.dataframe and len(at.table) == 2
    assert any("examples.json" in item.value for item in at.warning)
    at.radio(key="page").set_value("Проверка изображения").run()
    assert not at.exception and not at.error
    at.radio(key="page").set_value("О проекте").run()
    assert not at.exception
