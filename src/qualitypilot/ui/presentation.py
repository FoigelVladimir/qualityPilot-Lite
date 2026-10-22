"""Russian display labels over unchanged artifact values. No model imports."""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

MODEL_NAMES = {"Baseline": "Базовая модель (Isolation Forest)", "PatchCore": "PatchCore"}
DEFECT_NAMES = {
    "good": "Исправная бутылка",
    "broken_large": "Крупное повреждение",
    "broken_small": "Мелкое повреждение",
    "contamination": "Загрязнение",
}
DECISIONS = {
    "ACCEPT": ("✓", "Рекомендуется принять"),
    "REVIEW": ("✋", "Нужна ручная проверка"),
    "REJECT": ("!", "Рекомендуется отклонить"),
}
VIEWS = {
    "overlay": "Наложение",
    "heatmap": "Тепловая карта",
    "contour": "Контур подозрительной области",
    "mask": "Бинарная маска",
}
CASES = {
    "TP": "TP — дефект найден",
    "FP": "FP — ложная тревога",
    "FN": "FN — дефект пропущен",
    "TN": "TN — исправная бутылка определена верно",
}
REASONS = {
    "Global baseline has no anomaly map": "эта модель не выделяет области",
    "AUROC requires both classes": "нужны исправные и дефектные примеры",
    "Average precision requires positive labels": "нет размеченных дефектов",
    "No predicted positives": "нет прогнозов дефекта",
    "No true positives in subset": "нет размеченных дефектов в поднаборе",
    "No true or predicted positives": "нет размеченных или предсказанных дефектов",
    "No normal images in subset": "нет исправных примеров в поднаборе",
    "No independent pixel threshold": "нет независимо выбранного порога карты",
    "No true or predicted defect pixels": "нет размеченных или выделенных пикселей дефекта",
}


def defect_name(value: str) -> str:
    return DEFECT_NAMES.get(value, f"Другой тип дефекта ({value})")


def decision_name(value: str) -> str:
    return f"{DECISIONS[value][1]} ({value})"


def run_label(path: Path) -> str:
    match = re.fullmatch(r"bottle-patchcore-(\d{8}T\d{6}Z)-(.+)", path.name)
    if match:
        try:
            stamp = datetime.strptime(match[1], "%Y%m%dT%H%M%SZ")
            return f"Бутылки · {stamp:%d.%m.%Y %H:%M:%S} UTC · {match[2]}"
        except ValueError:
            pass
    return f"Эксперимент · {path.name}"


def example_label(relative: str) -> str:
    path = Path(relative)
    return f"{path.name} · {defect_name(path.parent.name)}"


def prediction_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Localize a copy; filtering and metrics still consume the original columns."""
    names = {
        "path": "Фотография",
        "defect": "Тип примера",
        "label": "Разметка",
        "binary_prediction": "Прогноз при одном пороге",
        "score": "Оценка аномальности",
        "decision": "Рекомендация",
        "latency_ms": "Обработка, мс",
        "preprocessing_ms": "Подготовка, мс",
        "inference_ms": "Модель, мс",
        "postprocessing_ms": "Решение, мс",
    }
    display = frame[[key for key in names if key in frame]].copy(deep=True)
    display["path"] = display["path"].map(lambda value: Path(value).name)
    display["defect"] = display["defect"].map(defect_name)
    for key in ("label", "binary_prediction"):
        display[key] = display[key].map({0: "Исправно", 1: "Дефект"})
    display["decision"] = display["decision"].map(decision_name)
    return display.rename(columns=names)
