from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qualitypilot.evaluation.pixels import read_mask
from qualitypilot.inference.images import load_image, resize_map
from qualitypilot.inference.service import Prediction


def save_inspection(
    source: Path, prediction: Prediction, mask_path: Path | None, output: Path
) -> None:
    if prediction.anomaly_map is None or prediction.pixel_threshold is None:
        raise ValueError("Visualization requires a map and a calibrated pixel threshold")
    image = load_image(source)
    values = resize_map(prediction.anomaly_map, image.size)
    mask = read_mask(mask_path, image.size)
    binary = values > prediction.pixel_threshold
    maximum = max(prediction.pixel_threshold * 1.5, 1e-6)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    axes[0].imshow(image)
    axes[1].imshow(values, cmap="inferno", vmin=0, vmax=maximum)
    axes[2].imshow(image)
    axes[2].imshow(values, cmap="inferno", vmin=0, vmax=maximum, alpha=0.5)
    axes[3].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[4].imshow(binary, cmap="gray", vmin=0, vmax=1)
    titles = ["Original", "Anomaly map", "Overlay", "Ground truth", "Predicted mask"]
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"{source.name} | score {prediction.score:.4f} | {prediction.decision}\n"
        f"pixel threshold {prediction.pixel_threshold:.4f}; fixed color range 0–{maximum:.2f}"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)
