"""Russian Streamlit workflow over the unchanged Inspector and validated reports."""

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
from qualitypilot.ui.presentation import (
    CASES,
    DECISIONS,
    MODEL_NAMES,
    VIEWS,
    defect_name,
    example_label,
    prediction_table,
    run_label,
)
from qualitypilot.ui.state import (
    experimental_decision,
    reset_thresholds,
    run_inspection,
    sync_image,
    sync_model,
)
from qualitypilot.ui.style import CSS

PAGES = ("Проверка изображения", "Результаты эксперимента", "О проекте")
SOURCES = {
    "Локальный тестовый пример": "Пример из датасета",
    "Загрузить JPG/PNG": "Загрузить фотографию",
}
EXPLANATIONS = {
    "ACCEPT": "Низкая аномальность по сохранённым порогам. Отсутствие дефекта не гарантируется.",
    "REVIEW": "Неоднозначный случай: требуется ручная проверка изделия.",
    "REJECT": "Высокая аномальность: отклонить изделие и проверить подозрительную область.",
}


def main(root: Path) -> None:
    st.set_page_config(page_title="QualityPilot Lite", page_icon="🔬", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("QualityPilot Lite")
    st.write("Визуальная проверка бутылок · локальное приложение")
    page = st.sidebar.radio("Раздел", PAGES, key="page")
    try:
        settings = load_settings(root)
    except ValueError as exc:
        st.error(str(exc))
        settings = None
    runs = discover_runs(settings.runs_root) if settings else []
    with st.sidebar.expander("Настройки проверки"):
        model = st.selectbox(
            "Модель", ["PatchCore", "Baseline"], format_func=MODEL_NAMES.__getitem__, key="model"
        )
        device = (
            st.selectbox(
                "Устройство",
                ["auto", "cpu", "cuda"],
                format_func={"auto": "Автоматически", "cpu": "CPU", "cuda": "CUDA"}.__getitem__,
                key="device",
            )
            if model == "PatchCore"
            else "cpu"
        )
        run = None
        if runs:
            index = next(
                (i for i, path in enumerate(runs) if path.name == settings.selected_run), 0
            )
            run = st.selectbox(
                "Сохранённый эксперимент", runs, index=index, format_func=run_label, key="run"
            )
        else:
            st.write("Сохранённых завершённых экспериментов пока нет.")
    st.sidebar.markdown(f"**Активная модель: {MODEL_NAMES[model]}**")
    st.sidebar.write(
        "Ищет подозрительные области."
        if model == "PatchCore"
        else "Оценивает снимок целиком, без выделения областей."
    )
    st.sidebar.write("Рекомендация помогает контролёру принять решение.")
    context = (str(run), model, device)
    if st.session_state.get("_run_context") != context:
        sync_model(st.session_state, ("selection-change", context))
        st.session_state["_run_context"] = context
    if page == PAGES[2]:
        about()
    elif settings is None:
        st.info("Исправьте configs/ui.yaml; раздел «О проекте» остаётся доступным.")
    elif page == PAGES[1]:
        if run is not None:
            results_page(run, settings)
        else:
            st.info("Для сравнения нужен завершённый сохранённый эксперимент. Модель не требуется.")
    else:
        inspection_page(settings, run)


def choose_image(base, run: Path | None):
    source = st.radio(
        "1. Выберите фотографию",
        list(SOURCES),
        index=0 if run else 1,
        format_func=SOURCES.__getitem__,
        horizontal=True,
        key="source",
    )
    mask_path = None
    try:
        if source == "Загрузить JPG/PNG":
            st.write(
                "Нажмите Browse files или перетащите JPG/PNG. До 10 МиБ и 20 млн пикселей; "
                "файл обрабатывается в памяти и не сохраняется."
            )
            uploaded = st.file_uploader(
                "Фотография JPG или PNG", type=["jpg", "jpeg", "png"], key="upload"
            )
            if uploaded is None:
                sync_image(st.session_state, "upload:none")
                st.info("Выберите фотографию, затем нажмите «Проверить изображение».")
                return None
            payload = uploaded.getvalue()
            token = "upload:" + uploaded.name + ":" + hashlib.sha256(payload).hexdigest()
            if sync_image(st.session_state, token) or "_image" not in st.session_state:
                st.session_state["_image"] = decode_upload(payload)
        else:
            if run is None:
                sync_image(st.session_state, "local:none")
                st.info("Локальных примеров пока нет. Выберите «Загрузить фотографию».")
                return None
            test = load_manifest(run).query("split == 'test'").sort_values("path")
            types = sorted(test.defect.unique(), key=lambda value: (value != "good", value))
            defect = st.selectbox("Тип примера", types, format_func=defect_name, key="example_type")
            rows = test[test.defect == defect].set_index("path")
            # Equal short labels in different folders must not retain an old path.
            context = (str(run), defect, tuple(rows.index))
            if st.session_state.get("_example_context") != context:
                st.session_state.pop("example", None)
                sync_image(st.session_state, f"example-group:{context!r}")
                st.session_state["_example_context"] = context
            relative = st.selectbox(
                "Фотография",
                list(rows.index),
                format_func=lambda value: Path(value).name,
                key="example",
            )
            st.write("Известный тестовый пример для демонстрации — не новая оценка качества.")
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
        return None
    return st.session_state["_image"], source, mask_path


def recommendation(decision: str, *, temporary=False) -> None:
    icon, label = DECISIONS[decision]
    if temporary:
        st.markdown("**Временная рекомендация**")
    message = f"{icon} **{label}**"
    {"ACCEPT": st.success, "REVIEW": st.warning, "REJECT": st.error}[decision](message)
    st.caption(f"Код рекомендации: {decision}")
    st.write(EXPLANATIONS[decision])


def inspection_page(settings: Settings, run: Path | None) -> None:
    st.header("Проверка изображения")
    st.write("Выберите фотографию → запустите проверку → осмотрите результат.")
    try:
        base = load_config(settings.baseline_config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(
            f"Не удалось прочитать настройки базовой модели: {exc}. Проверьте configs/ui.yaml."
        )
        return
    model = st.session_state["model"]
    device = st.session_state.get("device", "auto") if model == "PatchCore" else "cpu"
    artifact = run / "model.pt" if model == "PatchCore" and run else base.model_path
    available = model == "Baseline" or run is not None
    stamp = (artifact.stat().st_size, artifact.stat().st_mtime_ns) if artifact.is_file() else None
    sync_model(st.session_state, (str(run), model, device, str(artifact), stamp))
    if not available:
        st.warning(
            "Нет завершённого эксперимента PatchCore. Выберите его в настройках проверки "
            "или используйте базовую модель с локальными весами."
        )
    elif not artifact.is_file():
        st.warning(
            "Весов модели нет. Восстановите доверенный локальный файл модели. "
            "Сохранённые отчёты можно читать без весов."
        )
        with st.expander("Где ожидается файл модели"):
            st.code(str(artifact))
    with st.container(border=True):
        choices, preview = st.columns(2, gap="medium")
        with choices:
            selected = choose_image(base, run)
        if selected is None:
            return
        image, source, mask_path = selected
        with preview:
            st.image(image, caption="Выбранная фотография", width=260)
            inspect = st.button(
                "Проверить изображение",
                type="primary",
                width="stretch",
                disabled=not available or not artifact.is_file(),
                key="inspect",
            )
    if inspect:
        st.session_state.pop("_prediction", None)
        try:
            with st.spinner("Проверяем фотографию…"):
                run_inspection(st.session_state, artifact, device)
        except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            st.error(user_error(exc))
    result = st.session_state.get("_prediction")
    if result is None:
        return
    st.subheader("2. Результат проверки")
    recommendation(result.decision)
    st.metric("Оценка аномальности", f"{result.score:.2f}".replace(".", ","))
    st.write("Оценка аномальности не является вероятностью дефекта.")
    st.write(f"Фактическое устройство: {result.device.upper()}.")
    inspect_details(settings, run, artifact, image, result)
    experimental_policy(result)
    view_result(image, result, source, mask_path)


def inspect_details(settings, run, artifact, image, result) -> None:
    saved = st.session_state["_inspector"].thresholds
    with st.expander("Технические подробности"):
        st.write(f"Точная оценка аномальности: {result.score!r}")
        st.write(
            f"Сохранённые пороги: принятие (accept) {saved.accept!r}; "
            f"отклонение (reject) {saved.reject!r}."
        )
        st.write(
            "Оценка ≤ accept: ACCEPT; accept < оценка ≤ reject: REVIEW; "
            "оценка > reject: REJECT. Пороги выбраны на исправных калибровочных снимках."
        )
        if saved.accept == saved.reject:
            st.info("Сохранённые пороги равны: зона ручной проверки (REVIEW) пуста.")
        st.write(
            f"Модель: {MODEL_NAMES[st.session_state['model']]} · "
            f"устройство: {result.device.upper()} · RGB {image.width} × {image.height}."
        )
        st.write(
            f"Обработка: {result.latency_ms:.2f} мс; "
            f"загрузка модели отдельно: {st.session_state['_load_ms']:.2f} мс."
        )
        st.write(
            "Время одного снимка включает обработку в памяти и модель; "
            "оно не заменяет сравнительное измерение с прогревом."
        )
        st.write(f"Файл модели: {artifact}")
        st.write(f"Настройки базовой модели: {settings.baseline_config}")
        if run:
            st.write(f"Полный идентификатор эксперимента: {run.name}")
            st.write(f"Папка эксперимента: {run}")
        if result.pixel_threshold is not None:
            st.write(f"Сохранённый порог карты: {result.pixel_threshold!r}")


def experimental_policy(result) -> None:
    with st.expander("Экспериментальные пороги"):
        st.toggle("Включить временные пороги", key="experimental")
        if not st.session_state["experimental"]:
            st.write("По умолчанию используются сохранённые пороги модели.")
            return
        st.warning(
            "Только временный пересчёт рекомендации. Настройка по известному тестовому "
            "примеру не является независимой оценкой. Модель, карта, её порог и "
            "исторические метрики не меняются."
        )
        saved = st.session_state["_inspector"].thresholds
        st.session_state.setdefault("manual_accept", saved.accept)
        st.session_state.setdefault("manual_reject", saved.reject)
        accept = st.number_input(
            "Временный порог принятия (accept)", key="manual_accept", format="%.10f", step=0.01
        )
        reject = st.number_input(
            "Временный порог отклонения (reject)", key="manual_reject", format="%.10f", step=0.01
        )
        try:
            recommendation(
                experimental_decision(result.score, saved, accept, reject), temporary=True
            )
            if accept == reject:
                st.info("Пороги равны: зона ручной проверки (REVIEW) пуста.")
        except ValueError:
            st.error("Нужны конечные пороги accept ≤ reject. Исправьте значения.")
        st.button(
            "Сбросить к порогам модели",
            on_click=reset_thresholds,
            args=(st.session_state,),
            key="reset_thresholds",
        )


def view_result(image, result, source, mask_path) -> None:
    st.subheader("3. Осмотр фотографии")
    if source == "Загрузить JPG/PNG":
        st.write("Разметка дефекта для загруженной фотографии неизвестна.")
    if result.anomaly_map is None:
        st.info(
            "Эта модель (Isolation Forest) не выделяет области: используются глобальные "
            "признаки снимка. Тепловая карта и маска недоступны."
        )
        if st.toggle("Крупный просмотр фотографии", key="large_view"):
            st.image(image, caption="Исходная фотография", width="stretch")
        return
    mode = st.selectbox(
        "Режим результата", list(VIEWS), format_func=VIEWS.__getitem__, key="view_mode"
    )
    alpha = st.slider(
        "Прозрачность наложения",
        0.0,
        1.0,
        0.45,
        0.05,
        key="alpha",
        disabled=mode != "overlay",
        help="Меняет только наложение, без повторной проверки.",
    )
    views = map_views(image, result.anomaly_map, result.pixel_threshold, alpha)
    left, right = st.columns(2, gap="medium")
    left.image(image, caption="Исходная фотография", width="stretch")
    right.image(views[mode], caption=VIEWS[mode], width="stretch")
    st.image(views["legend"], width="stretch")
    st.write(
        f"Шкала тепловой карты: 0,00 — тёмный цвет, меньше отклонение; "
        f"{views['maximum']:.6f} — яркий жёлтый, больше отклонение. "
        "Одна шкала для всех фотографий, без подстройки под максимум снимка."
    )
    st.write(
        "Выделенная область и жёлтый контур используют один сохранённый порог карты. "
        "Контур — только способ просмотра, не новая сегментация."
    )
    if st.toggle("Крупный просмотр", key="large_view"):
        target = st.selectbox(
            "Что показать крупно",
            ["result", "original"],
            format_func={
                "result": "Выбранный результат",
                "original": "Исходная фотография",
            }.__getitem__,
            key="large_target",
        )
        st.image(
            views[mode] if target == "result" else image,
            caption=VIEWS[mode] if target == "result" else "Исходная фотография",
            width="stretch",
        )
    if source == "Локальный тестовый пример" and mask_path is not None:
        if st.toggle("Сравнить с разметкой дефекта", key="show_gt"):
            try:
                with Image.open(mask_path) as mask:
                    if mask.size != image.size:
                        raise ValueError("Размер разметки не совпадает с фотографией")
                    left, right = st.columns(2, gap="medium")
                    left.image(views["mask"], caption="Выделенная область", width="stretch")
                    right.image(mask.copy(), caption="Разметка дефекта", width="stretch")
            except (ValueError, OSError) as exc:
                st.warning(f"Разметка недоступна: {exc}")


def results_page(run: Path, settings: Settings) -> None:
    st.header("Результаты эксперимента")
    st.write("Сохранённые результаты. Модель не загружается, эксперимент не запускается заново.")
    try:
        reports = load_reports(run)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.write(run_label(run))
    metrics = reports["metrics"]
    display = metric_table(metrics).rename(
        columns={"baseline": "Базовая модель", "patchcore": "PatchCore"}
    )
    st.dataframe(display, hide_index=True, width="stretch", height=470, row_height=38)
    with st.expander("Как читать метрики"):
        st.write(
            "Точность тревог (Precision): какая доля отмеченных снимков "
            "действительно содержит дефект. "
            "Полнота (Recall): какая доля размеченных дефектов найдена. "
            "Доля ложных тревог (FPR): какая доля исправных снимков ошибочно отмечена."
        )
        st.write(
            "AUROC оценивает ранжирование по всем порогам, а не вероятность дефекта. "
            "Остальные показатели используют заранее сохранённые пороги. "
            "Пиксельные метрики — на сетке 224 × 224; "
            "недоступные значения показаны как N/A с причиной."
        )
    columns = st.columns(2, gap="medium")
    for column, name in zip(columns, ("baseline", "patchcore"), strict=True):
        data = metrics[name]
        single, two = data["single_threshold"], data["two_thresholds"]
        column.subheader("Базовая модель" if name == "baseline" else "PatchCore")
        column.write("Матрица ошибок при одном сохранённом пороге")
        column.table(
            pd.DataFrame(
                single["confusion_matrix"],
                index=["На самом деле: исправно", "На самом деле: дефект"],
                columns=["Прогноз: исправно", "Прогноз: дефект"],
            )
        )
        column.write(
            f"Ложные тревоги (FP): {single['false_rejects']}. "
            f"Пропущенные дефекты (FN): {single['false_accepts']}."
        )
        column.write(
            f"При двух порогах: ошибочно принятых дефектов {two['false_accepts']}, "
            f"ошибочно отклонённых исправных {two['false_rejects']}."
        )
        column.metric(
            "Ручная проверка (REVIEW)", f"{two['review_count']} ({two['review_fraction']:.2%})"
        )
    st.info(
        "Ручная проверка — дополнительная работа контролёра, а не исправленная ошибка. "
        "Временные пороги на странице проверки не изменяют эти результаты."
    )
    with st.expander("Технические подробности эксперимента"):
        st.write(f"Полное имя: {run.name}")
        st.write(f"Папка: {run}")
        for name, data in metrics.items():
            single, two, timing = (
                data["single_threshold"],
                data["two_thresholds"],
                data["latency_ms"],
            )
            st.write(
                f"{'Базовая модель' if name == 'baseline' else 'PatchCore'}: "
                f"один порог {single['threshold']!r}; принятие {two['accept']!r}; "
                f"отклонение {two['reject']!r}."
            )
            st.write(
                f"Устройство {timing['device'].upper()}, размер пакета {timing['batch_size']}, "
                f"измерений {timing['count']}, прогревов {timing['warmup_images']}; "
                f"синхронизация CUDA: {'да' if timing['cuda_synchronized'] else 'нет'}."
            )
            st.write("Включены чтение, подготовка, передача, модель и решение; без загрузки весов.")
        pixel = metrics["patchcore"].get("pixel_evaluation", {})
        if pixel.get("pixel_threshold") is not None:
            st.write(
                f"Сохранённый порог карты: {pixel['pixel_threshold']!r}; "
                "Dice/IoU рассчитаны по всем тестовым пикселям, "
                "без подбора порога по разметке дефектов."
            )
        cpu = reports["comparison"].get("patchcore_cpu_latency_ms")
        if cpu:
            st.write(
                f"Дополнительное измерение PatchCore на CPU: медиана {cpu['median']:.2f} мс, "
                f"p95 {cpu['p95']:.2f} мс; пакет {cpu['batch_size']}, "
                f"измерений {cpu['count']}, прогревов {cpu['warmup_images']}."
            )
    st.subheader("Отдельные предсказания")
    model = st.selectbox(
        "Результаты модели",
        ["patchcore", "baseline"],
        format_func={"patchcore": "PatchCore", "baseline": "Базовая модель"}.__getitem__,
        key="results_model",
    )
    case = st.selectbox("Тип результата", list(CASES), format_func=CASES.__getitem__, key="case")
    label, predicted = {"TP": (1, 1), "FP": (0, 1), "FN": (1, 0), "TN": (0, 0)}[case]
    table = reports["tables"][model]
    rows = table[(table["label"] == label) & (table["binary_prediction"] == predicted)].sort_values(
        "path"
    )
    if rows.empty:
        st.info(
            f"{CASES[case]}: таких случаев в сохранённом наборе нет. "
            "Фиктивные примеры не создаются."
        )
        return
    st.dataframe(
        prediction_table(rows),
        hide_index=True,
        width="stretch",
        row_height=38,
        column_config={
            name: st.column_config.NumberColumn(format="%.2f")
            for name in (
                "Оценка аномальности",
                "Обработка, мс",
                "Подготовка, мс",
                "Модель, мс",
                "Решение, мс",
            )
        },
    )
    relative = st.selectbox(
        "Фотография для просмотра",
        rows["path"].tolist(),
        format_func=example_label,
        key="result_example",
    )
    with st.expander("Точные значения выбранной строки"):
        row = rows.loc[rows["path"] == relative].iloc[0]
        st.write(f"Путь: {relative}")
        st.write(f"Точная оценка аномальности: {row['score']!r}")
        st.write(f"Рекомендация: {row['decision']}")
    try:
        base = load_config(settings.baseline_config)
        source = safe_data_path(base.data_root, relative)
        with Image.open(source) as image:
            st.image(image.copy(), caption=example_label(relative), width=320)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.warning(
            f"Не удалось открыть фотографию: {exc}. Таблица и сохранённые отчёты остаются доступны."
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
                    with st.expander("Сохранённая техническая иллюстрация"):
                        st.write("Историческая иллюстрация может содержать английские подписи.")
                        st.image(
                            str(path),
                            caption="Сохранённая иллюстрация эксперимента",
                            width="stretch",
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
        "**Базовая модель (Isolation Forest):** оценивает 60 глобальных "
        "цветовых и текстурных признаков. "
        "**PatchCore:** использует предобученные признаки ResNet18 и банк "
        "признаков исправных участков. Сеть с нуля не обучается. "
        "Карта показывает отклонение от нормальных признаков."
    )
    st.write(
        "Категория bottle: 167 исправных снимков для подготовки модели, "
        "42 исправных для калибровки и 83 тестовых. Калибровка не использует "
        "дефекты и их разметку. Пороги не гарантируют будущую полноту "
        "обнаружения или долю ложных тревог."
    )
    st.markdown(
        "**Рекомендуется принять (ACCEPT):** низкая аномальность, без гарантии отсутствия дефекта. "
        "**Нужна ручная проверка (REVIEW):** решение принимает контролёр. "
        "**Рекомендуется отклонить (REJECT):** высокая аномальность. "
        "Оценка аномальности не является вероятностью дефекта."
    )
    st.subheader("Данные и ограничения")
    st.markdown(
        "MVTec AD, © MVTec Software GmbH, Bergmann et al. "
        "[Оригинальный источник](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). "
        "Использована [сторонняя копия](https://huggingface.co/datasets/foersben/mvtec-ad), "
        "версия c75b39616f84db43677bcc8228caaafaf5096d7f. Лицензия **CC BY-NC-SA 4.0**, "
        "только некоммерческое использование. Хеши копии проверены; идентичность "
        "недоступному официальному архиву независимо не подтверждена."
    )
    st.warning(
        "Только bottle и 83 тестовых снимка; AUROC 1.0 не отменяет 4 ложные тревоги "
        "и умеренную локализацию (Dice ≈0.5604). Нет промышленной сертификации, проверки "
        "дрейфа или гарантий для других камер и категорий."
    )
    st.caption(
        "Приложение использует только доверенные локальные артефакты. "
        "Файлы моделей не принимаются через браузер. "
        "Обучение, скачивание и сохранение порогов не выполняются."
    )
