import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def plot_confusion(
    matrix: list[list[int]], path: Path, title: str = "Baseline / frozen calibration threshold"
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row][column]), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "Defect"],
        yticklabels=["Normal", "Defect"],
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
