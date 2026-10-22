"""Streamlit widgets over read-only artifacts and the shared Inspector service."""

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from qualitypilot.config import load_config
from qualitypilot.ui import validation
from qualitypilot.ui.artifacts import (
    Settings,
    discover_runs,
    load_manifest,
    load_reports,
    load_settings,
    metric_table,
    read_json,
    safe_data_path,
)
from qualitypilot.ui.inspection import decode_upload, map_views, user_error
from qualitypilot.ui.state import (
    experimental_decision,
    reset_thresholds,
    run_inspection,
    sync_image,
    sync_model,
)

PAGES = ("Проверка изображения", "Результаты эксперимента", "О проекте")
EXPLANATIONS = {
    "ACCEPT": "Низкая аномальность по сохранённым порогам. Отсутствие дефекта не гарантируется.",
    "REVIEW": "Неоднозначный случай: требуется ручная проверка изделия.",
    "REJECT": "Высокая аномальность: отклонить изделие и проверить предполагаемый дефект.",
}


def main(root: Path) -> None:
    st.set_page_config(page_title="QualityPilot Lite", page_icon="🔬", layout="wide")
    st.title("QualityPilot Lite")
    st.caption("Визуальный контроль качества · локально · bottle")
    page = st.sidebar.radio("Раздел", PAGES, key="page")
    st.sidebar.caption(
        "Score — не вероятность дефекта. Решение поддерживает, а не заменяет контроль."
    )
    try:
        settings = load_settings(root)
    except ValueError as exc:
        st.error(str(exc))
        settings = None
    runs = discover_runs(settings.runs_root) if settings else []
    run = None
    if runs:
        index = next((i for i, p in enumerate(runs) if p.name == settings.selected_run), 0)
        run = st.sidebar.selectbox(
            "Завершённый эксперимент", runs, index=index, format_func=lambda p: p.name, key="run"
        )
    elif page != PAGES[2]:
        st.warning(
            "Нет завершённых PatchCore runs с полным набором отчётов. "
            "Проверьте runs_root в configs/ui.yaml. "
            "Загрузка изображения для baseline остаётся доступной."
        )
    context = str(run) if run else "no-run"
    if st.session_state.get("_run_context") != context:
        sync_model(st.session_state, ("run-change", context))
        st.session_state["_run_context"] = context
    if page == PAGES[2]:
        about()
    elif settings is None:
        st.info("Исправьте configs/ui.yaml; раздел «О проекте» остаётся доступным.")
    elif page == PAGES[1]:
        if run is not None:
            results_page(run, settings)
        else:
            st.info("Для сравнения нужен завершённый локальный эксперимент; модель не требуется.")
    else:
        inspection_page(settings, run)


def inspection_page(settings: Settings, run: Path | None) -> None:
    st.header("Проверка изображения")
    try:
        base = load_config(settings.baseline_config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Не удалось прочитать baseline config: {exc}. Проверьте configs/ui.yaml.")
        return
    model = st.selectbox("Модель", ["PatchCore", "Baseline"], key="model")
    device = (
        st.selectbox("Устройство PatchCore", ["auto", "cpu", "cuda"], key="device")
        if model == "PatchCore"
        else "cpu"
    )
    artifact = run / "model.pt" if model == "PatchCore" and run else base.model_path
    available = model == "Baseline" or run is not None
    stamp = (artifact.stat().st_size, artifact.stat().st_mtime_ns) if artifact.is_file() else None
    sync_model(st.session_state, (str(run), model, device, str(artifact), stamp))
    if not available:
        st.warning(
            "Для PatchCore выберите COMPLETE run. Baseline можно использовать с локальными весами."
        )
    elif not artifact.is_file():
        st.warning(
            "Весов модели нет. Восстановите доверенный локальный артефакт по указанному пути. "
            "Сохранённые результаты эксперимента доступны без весов."
        )
        st.code(str(artifact))
    source = st.radio(
        "Источник изображения",
        ["Локальный тестовый пример", "Загрузить JPG/PNG"],
        index=0 if run else 1,
        horizontal=True,
        key="source",
    )
    mask_path = None
    try:
        if source == "Загрузить JPG/PNG":
            uploaded = st.file_uploader(
                "Одно изображение JPG или PNG", type=["jpg", "jpeg", "png"], key="upload"
            )
            st.caption(
                "До 10 МБ и 20 млн пикселей. Изображение обрабатывается в памяти и не сохраняется."
            )
            if uploaded is None:
                sync_image(st.session_state, "upload:none")
                st.info("Выберите JPG/PNG, затем нажмите «Проверить».")
                return
            payload = uploaded.getvalue()
            token = "upload:" + uploaded.name + ":" + hashlib.sha256(payload).hexdigest()
            if sync_image(st.session_state, token) or "_image" not in st.session_state:
                st.session_state["_image"] = decode_upload(payload)
        else:
            if run is None:
                sync_image(st.session_state, "local:none")
                st.info("Нет test manifest. Выберите загрузку JPG/PNG.")
                return
            manifest = load_manifest(run)
            test = manifest[manifest.split == "test"].sort_values("path")
            types = sorted(test.defect.unique())
            if "good" in types:
                types.remove("good")
                types.insert(0, "good")
            defect = st.selectbox(
                "Тип примера",
                types,
                format_func=lambda s: "Нормальные (good)" if s == "good" else s,
                key="example_type",
            )
            rows = test[test.defect == defect].set_index("path")
            relative = st.selectbox("Известный тестовый пример", list(rows.index), key="example")
            st.warning(
                "Известный тестовый пример из сохранённого manifest. "
                "Это не новая независимая проверка качества."
            )
            path = safe_data_path(base.data_root, relative)
            if not path.is_file():
                sync_image(st.session_state, "missing:" + relative)
                raise FileNotFoundError(f"Пример отсутствует: {path}. Восстановите data/mvtec_ad.")
            token = f"local:{path}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
            if sync_image(st.session_state, token) or "_image" not in st.session_state:
                st.session_state["_image"] = decode_upload(path.read_bytes())
            mask_relative = rows.loc[relative, "mask_path"]
            if mask_relative:
                mask_path = safe_data_path(base.data_root, mask_relative)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        st.session_state.pop("_prediction", None)
        st.error(user_error(exc))
        return
    image = st.session_state["_image"]
    with st.expander("Исходное изображение", expanded=True):
        st.image(image, width=420)
        st.caption(f"RGB · {image.width} × {image.height} · без изменения ориентации или обрезки")
    if st.button(
        "Проверить", type="primary", disabled=not available or not artifact.is_file(), key="inspect"
    ):
        st.session_state.pop("_prediction", None)
        try:
            with st.spinner("Загрузка модели и проверка…"):
                run_inspection(st.session_state, artifact, device)
        except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            st.error(user_error(exc))
    result = st.session_state.get("_prediction")
    if result is None:
        return
    inspector = st.session_state["_inspector"]
    saved = inspector.thresholds
    first, second, third = st.columns(3)
    first.metric("Anomaly score", f"{result.score:.6f}")
    second.metric("Рекомендация по артефакту", result.decision)
    third.metric("Обработка, ms", f"{result.latency_ms:.2f}")
    st.info(EXPLANATIONS[result.decision])
    st.caption(
        f"Модель: {result.model} · категория: {result.category} · фактическое устройство: "
        f"{result.device} · загрузка модели отдельно: {st.session_state['_load_ms']:.2f} ms"
    )
    st.caption(
        f"Сохранённые пороги: accept = {saved.accept:.10g}; reject = {saved.reject:.10g}. "
        "score ≤ accept — ACCEPT; accept < score ≤ reject — REVIEW; score > reject — REJECT."
    )
    if saved.accept == saved.reject:
        st.info("Сохранённые пороги равны: зона REVIEW пуста.")
    st.toggle("Экспериментальные пороги", key="experimental")
    if st.session_state["experimental"]:
        st.warning(
            "ЭКСПЕРИМЕНТ: только временный пересчёт рекомендации. "
            "Настройка по известным test-примерам не является независимой оценкой. "
            "Модель, pixel threshold и исторические метрики не меняются."
        )
        st.session_state.setdefault("manual_accept", saved.accept)
        st.session_state.setdefault("manual_reject", saved.reject)
        left, right = st.columns(2)
        accept = left.number_input(
            "Временный accept", key="manual_accept", format="%.10f", step=0.01
        )
        reject = right.number_input(
            "Временный reject", key="manual_reject", format="%.10f", step=0.01
        )
        try:
            decision = experimental_decision(result.score, saved, accept, reject)
            st.metric("Экспериментальная рекомендация", decision)
            st.caption(EXPLANATIONS[decision])
            if accept == reject:
                st.info("Пороги равны: зона REVIEW пуста.")
        except ValueError:
            st.error("Нужны конечные пороги accept ≤ reject. Исправьте значения.")
        st.button(
            "Сбросить к порогам артефакта",
            on_click=reset_thresholds,
            args=(st.session_state,),
            key="reset_thresholds",
        )
    if result.anomaly_map is None:
        st.info(
            "Baseline использует глобальные признаки и не поддерживает локализацию. "
            "Карта и маска недоступны."
        )
        return
    st.subheader("Локализация предполагаемого дефекта")
    alpha = st.slider("Прозрачность наложения", 0.0, 1.0, 0.45, 0.05, key="alpha")
    views = map_views(image, result.anomaly_map, result.pixel_threshold, alpha)
    cols = st.columns(3)
    for column, key, title in zip(
        cols,
        ("heatmap", "overlay", "mask"),
        ("Карта аномальности", "Наложение", "Predicted mask"),
        strict=True,
    ):
        column.image(views[key], caption=title, width="stretch")
    st.image(views["legend"], width="stretch")
    st.caption(
        f"inferno: 0 слева → {views['maximum']:.6f} справа. "
        f"Общая шкала фиксирована; сохранённый pixel threshold = {result.pixel_threshold:.6f}. "
        "Прозрачность меняет только отображение; score и карта остаются прежними."
    )
    if source == "Локальный тестовый пример" and mask_path is not None:
        try:
            with Image.open(mask_path) as mask:
                if mask.size != image.size:
                    raise ValueError("Размер GT не совпадает с изображением")
                st.image(
                    mask.copy(),
                    caption="Ground truth — только известный размеченный пример",
                    width=320,
                )
        except (ValueError, OSError) as exc:
            st.warning(f"GT недоступна: {exc}")
    elif source == "Загрузить JPG/PNG":
        st.caption("Ground truth для загруженного изображения не предоставлена.")


def results_page(run: Path, settings: Settings) -> None:
    st.header("Результаты эксперимента")
    st.caption(
        "Только сохранённые отчёты. Модель не загружается, эксперимент не запускается заново."
    )
    try:
        reports = load_reports(run)
    except ValueError as exc:
        st.error(str(exc))
        return
    metrics = reports["metrics"]
    st.dataframe(metric_table(metrics), hide_index=True, width="stretch", height=460)
    st.caption(
        "Precision/Recall/F1/FPR — при одном заранее сохранённом accept. "
        "Pixel metrics — native 224×224; baseline локализацию не поддерживает."
    )
    columns = st.columns(2)
    for column, name in zip(columns, ("baseline", "patchcore"), strict=True):
        data = metrics[name]
        single, two = data["single_threshold"], data["two_thresholds"]
        column.subheader(name)
        column.caption(
            f"Один порог: {single['threshold']:.8g}; score > threshold — подозрительное."
        )
        column.table(
            pd.DataFrame(
                single["confusion_matrix"],
                index=["True normal", "True defect"],
                columns=["Pred normal", "Pred defect"],
            )
        )
        column.write(f"Один порог: FP = {single['false_rejects']}, FN = {single['false_accepts']}.")
        column.write(
            f"Два порога: ложных ACCEPT = {two['false_accepts']}, "
            f"ложных REJECT = {two['false_rejects']}."
        )
        column.caption(f"Два порога: accept = {two['accept']:.8g}, reject = {two['reject']:.8g}.")
        column.metric("REVIEW", f"{two['review_count']} ({two['review_fraction']:.2%})")
        timing = data["latency_ms"]
        column.caption(
            f"Latency: {timing['device']}, batch {timing['batch_size']}, "
            f"n={timing['count']}, warmup={timing['warmup_images']}; "
            f"CUDA synchronize={timing['cuda_synchronized']}. "
            "Включает decode/preprocessing, передачу, модель и postprocessing; без загрузки весов."
        )
    st.info(
        "REVIEW — нагрузка на ручную проверку, а не автоматически исправленная ошибка. "
        "Временные пороги страницы проверки не изменяют эти метрики."
    )
    pixel = metrics["patchcore"].get("pixel_evaluation", {})
    if pixel.get("pixel_threshold") is not None:
        st.caption(
            f"Сохранённый pixel threshold: {pixel['pixel_threshold']:.8g}. "
            "Dice/IoU — micro по всем test pixels; порог не оптимизировался по test masks."
        )
    cpu = reports["comparison"].get("patchcore_cpu_latency_ms")
    if cpu:
        with st.expander("Дополнительный CPU benchmark PatchCore"):
            st.write(
                f"Median {cpu['median']:.2f} ms · p95 {cpu['p95']:.2f} ms · "
                f"устройство {cpu['device']} · batch {cpu['batch_size']} · "
                f"n={cpu['count']} · warmup={cpu['warmup_images']}."
            )
    st.subheader("Примеры и строки результатов")
    model = st.selectbox("Результаты модели", ["patchcore", "baseline"], key="results_model")
    case = st.selectbox("Тип результата", ["TP", "FP", "FN", "TN"], key="case")
    labels = {"TP": (1, 1), "FP": (0, 1), "FN": (1, 0), "TN": (0, 0)}
    label, predicted = labels[case]
    table = reports["tables"][model]
    rows = table[(table["label"] == label) & (table["binary_prediction"] == predicted)].sort_values(
        "path"
    )
    if rows.empty:
        st.info(f"{case}: таких случаев в сохранённом test нет. Фиктивные примеры не создаются.")
        return
    st.dataframe(rows, hide_index=True, width="stretch")
    relative = st.selectbox(
        "Строка test для просмотра", rows["path"].tolist(), key="result_example"
    )
    try:
        base = load_config(settings.baseline_config)
        source = safe_data_path(base.data_root, relative)
        with Image.open(source) as image:
            st.image(image.copy(), caption=relative, width=320)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.warning(
            f"Не удалось открыть изображение: {exc}. "
            "Таблица и сохранённые отчёты остаются доступны."
        )
    examples_path = run / "examples/examples.json"
    if model == "patchcore" and examples_path.is_file():
        try:
            example = validation.examples(read_json(examples_path), str(examples_path)).get(
                case.lower(), {}
            )
            if example.get("path") == relative and example.get("example"):
                path = safe_data_path(run / "examples", example["example"])
                if path.is_file():
                    st.image(
                        str(path), caption="Сохранённая иллюстрация эксперимента", width="stretch"
                    )
        except ValueError as exc:
            st.warning(str(exc))


def about() -> None:
    st.header("О проекте")
    st.write(
        "QualityPilot Lite — учебная система обнаружения визуальных аномалий "
        "промышленных изделий и поддержки решения контролёра."
    )
    st.subheader("Как устроена проверка")
    st.markdown(
        "**Baseline:** 60 глобальных цветовых/текстурных признаков и Isolation Forest. "
        "**PatchCore:** предобученные признаки ResNet18 и memory bank нормальных патчей; "
        "сеть с нуля не обучается. Карта показывает расстояния до нормальных признаков."
    )
    st.write(
        "Разбиение bottle: 167 train normal, 42 normal calibration и 83 official test. "
        "Калибровка не использует дефекты и test masks. Пороги не гарантируют заданный recall/FPR."
    )
    st.markdown(
        "**ACCEPT:** низкая аномальность. **REVIEW:** ручная проверка. "
        "**REJECT:** высокая аномальность. Anomaly score не является вероятностью дефекта."
    )
    st.subheader("Данные и ограничения")
    st.markdown(
        "MVTec AD, © MVTec Software GmbH, Bergmann et al. "
        "[Оригинальный источник](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). "
        "Использована [сторонняя копия](https://huggingface.co/datasets/foersben/mvtec-ad), "
        "commit c75b39616f84db43677bcc8228caaafaf5096d7f. Лицензия **CC BY-NC-SA 4.0**, "
        "только некоммерческое использование. Хеши копии проверены; идентичность "
        "недоступному официальному архиву независимо не подтверждена."
    )
    st.warning(
        "Только bottle и 83 test-изображения; AUROC 1.0 не отменяет 4 FP при одном пороге "
        "и умеренную локализацию (Dice ≈0.5604). Нет промышленной сертификации, проверки "
        "дрейфа или гарантий для других камер и категорий."
    )
    st.caption(
        "Приложение использует только доверенные локальные артефакты. "
        "Checkpoints не принимаются через браузер. "
        "Обучение, скачивание и сохранение порогов не выполняются."
    )
