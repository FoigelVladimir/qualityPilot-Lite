"""Validate the report fields consumed by the UI before rendering them."""

import math
from numbers import Real


def invalid(source: str, field: str, expected: str) -> None:
    raise ValueError(f"{source}: поле {field} — ожидается {expected}")


def fields(value, required: tuple[str, ...], source: str, field: str = "JSON") -> dict:
    if not isinstance(value, dict):
        invalid(source, field, "объект")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"{source}: {field}: отсутствуют обязательные поля: {', '.join(missing)}")
    return value


def number(value, source: str, field: str, *, count=False, fraction=False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        invalid(source, field, "конечное число")
    if count and (not isinstance(value, int) or value < 0):
        invalid(source, field, "целое число ≥ 0")
    if fraction and not 0 <= value <= 1:
        invalid(source, field, "число от 0 до 1")


def text(value, source: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        invalid(source, field, "непустая строка")


def metric(value, source: str, field: str) -> None:
    item = fields(value, ("value",), source, field)
    if item["value"] is None:
        text(item.get("reason"), source, f"{field}.reason")
    else:
        number(item["value"], source, f"{field}.value", fraction=True)


def latency(value, source: str, field: str, *, synchronized=True) -> None:
    keys = ("median", "p95", "device", "batch_size", "count", "warmup_images")
    item = fields(value, keys, source, field)
    for key in ("median", "p95"):
        number(item[key], source, f"{field}.{key}")
        if item[key] < 0:
            invalid(source, f"{field}.{key}", "число ≥ 0")
    for key in ("batch_size", "count", "warmup_images"):
        number(item[key], source, f"{field}.{key}", count=True)
    text(item["device"], source, f"{field}.device")
    if synchronized and type(item.get("cuda_synchronized")) is not bool:
        invalid(source, f"{field}.cuda_synchronized", "логическое значение")


def metrics(value: dict, source: str) -> None:
    keys = ("image_auroc", "image_average_precision", "pixel_auroc", "dice", "iou")
    fields(value, (*keys, "single_threshold", "two_thresholds", "latency_ms"), source)
    for key in keys:
        metric(value[key], source, key)
    names = ("precision", "recall", "f1", "false_positive_rate")
    single = fields(
        value["single_threshold"],
        (*names, "threshold", "confusion_matrix", "false_accepts", "false_rejects"),
        source,
        "single_threshold",
    )
    for key in names:
        metric(single[key], source, f"single_threshold.{key}")
    number(single["threshold"], source, "single_threshold.threshold")
    matrix = single["confusion_matrix"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
    ):
        invalid(source, "single_threshold.confusion_matrix", "матрица 2×2")
    for row in matrix:
        for cell in row:
            number(cell, source, "single_threshold.confusion_matrix", count=True)
    two = fields(
        value["two_thresholds"],
        ("accept", "reject", "review_count", "review_fraction", "false_accepts", "false_rejects"),
        source,
        "two_thresholds",
    )
    for key in ("accept", "reject"):
        number(two[key], source, f"two_thresholds.{key}")
    if two["accept"] > two["reject"]:
        invalid(source, "two_thresholds", "accept ≤ reject")
    for item, label in ((single, "single_threshold"), (two, "two_thresholds")):
        for key in ("false_accepts", "false_rejects"):
            number(item[key], source, f"{label}.{key}", count=True)
    number(two["review_count"], source, "two_thresholds.review_count", count=True)
    number(two["review_fraction"], source, "two_thresholds.review_fraction", fraction=True)
    latency(value["latency_ms"], source, "latency_ms")
    if "pixel_evaluation" in value:
        pixel = fields(value["pixel_evaluation"], (), source, "pixel_evaluation")
        if pixel.get("pixel_threshold") is not None:
            number(pixel["pixel_threshold"], source, "pixel_evaluation.pixel_threshold")


def examples(value: dict, source: str) -> dict:
    for case, item in value.items():
        fields(item, ("example",), source, case)
        if item["example"] is not None:
            text(item["example"], source, f"{case}.example")
            text(item.get("path"), source, f"{case}.path")
    return value
