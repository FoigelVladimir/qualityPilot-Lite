"""In-memory upload validation and display-only map preparation."""

import warnings
from io import BytesIO
from math import ceil

import matplotlib
import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


def decode_upload(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Изображение должно быть непустым и не больше 10 МБ.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise ValueError("Поддерживаются только фактически декодируемые JPG и PNG.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Разрешено не больше 20 млн пикселей.")
                image.load()
                return image.convert("RGB")  # No EXIF rotation, crop or resize.
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Не удалось декодировать JPG/PNG. Проверьте целостность файла.") from exc


def map_views(image: Image.Image, values: np.ndarray, threshold: float, alpha: float) -> dict:
    if (
        values.shape != (image.height, image.width)
        or not np.isfinite(values).all()
        or not np.isfinite(threshold)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("Некорректная геометрия карты, порог или прозрачность")
    maximum = max(1.5 * threshold, 1e-6)
    heatmap = (
        matplotlib.colormaps["inferno"](np.clip(values / maximum, 0, 1))[..., :3] * 255
    ).astype(np.uint8)
    original = np.asarray(image, dtype=np.float32)
    overlay = ((1 - alpha) * original + alpha * heatmap).clip(0, 255).astype(np.uint8)
    # A display contour from the exact same thresholded mask, including image borders.
    # The predicted mask and boundary stay exact; only their display stroke is expanded.
    selected = values > threshold
    padded = np.pad(selected, 1, constant_values=False)
    interior = (
        selected & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    boundary = selected & ~interior
    stroke = Image.fromarray(boundary.astype(np.uint8) * 255)
    short_side = min(image.size)
    outer_radius = max(2, ceil(short_side / 320))
    inner_radius = max(1, ceil(short_side / 640))
    outer = np.asarray(stroke.filter(ImageFilter.MaxFilter(2 * outer_radius + 1))) > 0
    inner = np.asarray(stroke.filter(ImageFilter.MaxFilter(2 * inner_radius + 1))) > 0
    contour = np.asarray(image).copy()
    contour[outer] = (23, 44, 60)
    contour[inner] = (255, 210, 0)
    strip = (
        (matplotlib.colormaps["inferno"](np.linspace(0, 1, 512))[None, :, :3] * 255)
        .astype(np.uint8)
        .repeat(14, axis=0)
    )
    return {
        "heatmap": heatmap,
        "overlay": overlay,
        "mask": selected.astype(np.uint8) * 255,
        "contour": contour,
        "boundary": boundary,
        "legend": strip,
        "maximum": maximum,
    }


def user_error(error: Exception) -> str:
    text = str(error)
    if "CUDA" in text and ("unavailable" in text or "недоступ" in text):
        return (
            "CUDA запрошена, но недоступна. Выберите cpu или auto; настройки не заменялись молча."
        )
    if isinstance(error, ImportError) or "ML extra" in text or "anomalib" in text.lower():
        return (
            "ML-зависимости недоступны или несовместимы. "
            "Выполните uv sync --locked --extra dev --extra ml --extra ui."
        )
    if "checkpoint" in text.lower():
        return (
            "Несовместимый или повреждённый checkpoint. Восстановите доверенный model.pt "
            "и закреплённое окружение проекта. Детали: " + text
        )
    if "out of memory" in text.lower():
        return "Недостаточно памяти. Освободите память или выберите устройство cpu. " + text
    if isinstance(error, FileNotFoundError):
        return (
            "Локальный файл не найден. Проверьте configs/ui.yaml и локальные веса/данные. " + text
        )
    return text
