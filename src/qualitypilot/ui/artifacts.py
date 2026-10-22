"""Read-only run discovery/validation. Does not import torch or deserialize models."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml

from qualitypilot.data.pipeline import Sample, assert_no_leakage
from qualitypilot.ui import validation

REQUIRED = (
    "status.json",
    "config.yaml",
    "experiment.json",
    "manifest.csv",
    "dataset_summary.json",
    "comparison.json",
    "baseline/results.csv",
    "baseline/metrics.json",
    "patchcore/results.csv",
    "patchcore/metrics.json",
)


@dataclass(frozen=True)
class Settings:
    baseline_config: Path
    runs_root: Path
    selected_run: str


def load_settings(root: Path, config_path: Path | None = None) -> Settings:
    path = config_path or root / "configs/ui.yaml"
    try:
        values = yaml.safe_load(path.read_text("utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Ожидалось YAML-отображение")
        paths = [(root / values[key]).resolve() for key in ("baseline_config", "runs_root")]
        if not all(p.is_relative_to(root.resolve()) for p in paths):
            raise ValueError("Пути UI должны находиться внутри проекта")
        name = values["selected_run"]
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("selected_run должен быть именем папки")
        return Settings(*paths, name)
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"Проверьте configs/ui.yaml: {exc}") from exc


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Ожидался JSON-объект")
        return value
    except (OSError, ValueError) as exc:
        raise ValueError(f"Не удалось прочитать отчёт {path}: {exc}") from exc


def read_csv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, keep_default_na=False)
        missing = [key for key in required if key not in frame.columns]
        if missing:
            raise ValueError(f"отсутствуют обязательные столбцы: {', '.join(missing)}")
        if frame.empty:
            raise ValueError("нет строк данных")
        return frame
    except (OSError, ValueError) as exc:
        raise ValueError(f"Не удалось прочитать CSV {path}: {exc}") from exc


def manifest_hash(run: Path) -> str:
    path = run / "dataset_summary.json"
    summary = validation.fields(read_json(path), ("manifest_sha256",), str(path))
    value = summary["manifest_sha256"]
    validation.text(value, str(path), "manifest_sha256")
    return value


def validate_rows(frame: pd.DataFrame, path: Path, *, results: bool) -> None:
    for key in ("path", "defect"):
        if not frame[key].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            validation.invalid(str(path), key, "непустые строки")
    for key in ("label", "binary_prediction") if results else ("label",):
        if not frame[key].isin([0, 1]).all():
            validation.invalid(str(path), key, "метки 0 или 1")
    if results:
        if not frame["decision"].isin(["ACCEPT", "REVIEW", "REJECT"]).all():
            validation.invalid(str(path), "decision", "ACCEPT, REVIEW или REJECT")
        for key in ("score", "latency_ms", "preprocessing_ms", "inference_ms", "postprocessing_ms"):
            if key in frame:
                for value in frame[key]:
                    validation.number(value, str(path), key)


def discover_runs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    runs = []
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if not path.is_dir() or not all((path / item).is_file() for item in REQUIRED):
            continue
        try:
            if read_json(path / "status.json").get("status") == "COMPLETE":
                # model.pt is deliberately optional: saved reports work without model weights.
                # Keep damaged reports selectable so load_reports can explain the exact error.
                runs.append(path)
        except ValueError:
            continue
    return runs


def load_manifest(run: Path) -> pd.DataFrame:
    try:
        path = run / "manifest.csv"
        frame = read_csv(path, ("path", "split", "label", "defect", "mask_path", "pixel_sha256"))
        validate_rows(frame, path, results=False)
        if not frame["mask_path"].map(lambda value: isinstance(value, str)).all():
            validation.invalid(str(path), "mask_path", "строки путей или пустые строки")
        records = []
        for row in frame.to_dict("records"):
            records.append(
                Sample(
                    str(row["path"]),
                    str(row["split"]),
                    int(row["label"]),
                    str(row["defect"]),
                    row["mask_path"] or None,
                    str(row["pixel_sha256"]),
                )
            )
        assert_no_leakage(records)
        if any(
            s.split not in {"train", "calibration", "test"}
            or s.label not in {0, 1}
            or (s.split != "test" and (s.label != 0 or "/train/good/" not in s.path))
            for s in records
        ):
            raise ValueError("Недопустимое разбиение manifest")
        digest = hashlib.sha256()
        for sample in records:
            digest.update(repr(asdict(sample)).encode())
        expected = manifest_hash(run)
        if digest.hexdigest() != expected:
            raise ValueError("Контрольная сумма manifest не совпадает")
        return frame
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Не удалось проверить manifest: {exc}") from exc


def load_reports(run: Path) -> dict:
    """Validate identities before presenting any comparison, even with no dataset/model."""
    try:
        if read_json(run / "status.json").get("status") != "COMPLETE":
            raise ValueError("Эксперимент не завершён")
        manifest = load_manifest(run)
        test = manifest.loc[manifest["split"] == "test", ["path", "label"]].reset_index(drop=True)
        comparison_path = run / "comparison.json"
        comparison = validation.fields(
            read_json(comparison_path),
            ("manifest_sha256", "baseline", "patchcore"),
            str(comparison_path),
        )
        experiment_path = run / "experiment.json"
        experiment = validation.fields(
            read_json(experiment_path), ("dataset",), str(experiment_path)
        )
        dataset = validation.fields(
            experiment["dataset"], ("manifest_sha256",), str(experiment_path), "dataset"
        )
        expected = manifest_hash(run)
        if comparison["manifest_sha256"] != expected or dataset["manifest_sha256"] != expected:
            raise ValueError("Разные manifest в сохранённых отчётах")
        tables, metrics = {}, {}
        for name in ("baseline", "patchcore"):
            csv_path = run / name / "results.csv"
            frame = read_csv(
                csv_path,
                ("path", "label", "defect", "score", "decision", "binary_prediction", "latency_ms"),
            )
            validate_rows(frame, csv_path, results=True)
            if (
                frame["path"].duplicated().any()
                or frame.empty
                or not frame[["path", "label"]].reset_index(drop=True).equals(test)
            ):
                raise ValueError(f"{name}: paths, порядок или labels отличаются от test manifest")
            tables[name] = frame
            metrics_path = run / name / "metrics.json"
            metrics[name] = read_json(metrics_path)
            if metrics[name].get("model") != name:
                validation.invalid(str(metrics_path), "model", name)
            validation.metrics(metrics[name], str(metrics_path))
            validation.metrics(comparison[name], f"{comparison_path}: {name}")
            if comparison[name] != metrics[name]:
                raise ValueError(f"{name}: comparison и metrics.json не согласованы")
        cpu = comparison.get("patchcore_cpu_latency_ms")
        if cpu is not None:
            validation.latency(
                cpu, str(comparison_path), "patchcore_cpu_latency_ms", synchronized=False
            )
        return {
            "manifest": manifest,
            "tables": tables,
            "metrics": metrics,
            "comparison": comparison,
        }
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Сравнение недоступно: {exc}") from exc


def safe_data_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Путь примера выходит за пределы датасета")
    return path


def metric_table(metrics: dict) -> pd.DataFrame:
    names = {
        "image_auroc": "Image AUROC",
        "image_average_precision": "Average Precision",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "false_positive_rate": "False positive rate",
        "median": "Median latency, ms",
        "p95": "p95 latency, ms",
        "pixel_auroc": "Pixel AUROC",
        "dice": "Micro Dice",
        "iou": "Micro IoU",
    }
    rows = []
    for key, label in names.items():
        row = {"Метрика": label}
        for model, data in metrics.items():
            if key in {"precision", "recall", "f1", "false_positive_rate"}:
                item = data["single_threshold"][key]
            elif key in {"median", "p95"}:
                item = {"value": data["latency_ms"][key]}
            else:
                item = data[key]
            reason = item.get("reason", "недоступно")
            if reason == "Global baseline has no anomaly map":
                reason = "нет карты дефектов"
            row[model] = f"{item['value']:.6f}" if item["value"] is not None else f"N/A — {reason}"
        rows.append(row)
    return pd.DataFrame(rows)
