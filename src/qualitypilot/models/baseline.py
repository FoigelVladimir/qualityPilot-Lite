"""Global RGB histograms and elementary texture statistics + Isolation Forest."""

from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest

from qualitypilot.inference.images import ImageInput, load_image


def features(path: ImageInput, image_size: int) -> np.ndarray:
    image = load_image(path).resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255
    values = []
    for channel in range(3):
        plane = pixels[:, :, channel]
        histogram, _ = np.histogram(plane, bins=16, range=(0, 1))
        values.extend(histogram / plane.size)
        values.extend([plane.mean(), plane.std()])
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
    for axis in (0, 1):
        differences = np.abs(np.diff(gray, axis=axis))
        values.extend([differences.mean(), differences.std(), np.quantile(differences, 0.95)])
    return np.asarray(values, dtype=np.float32)


class Baseline:
    name = "baseline"
    device = "cpu"

    def __init__(self, image_size: int = 128, seed: int = 42, n_estimators: int = 200):
        self.image_size = image_size
        self.estimator = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=1)

    def fit(self, paths: list[Path]) -> None:
        if not paths:
            raise ValueError("No training images")
        self.estimator.fit(np.stack([features(p, self.image_size) for p in paths]))

    def scores(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            return np.empty(0, dtype=np.float64)
        matrix = np.stack([features(p, self.image_size) for p in paths])
        # sklearn uses the opposite sign: higher is more normal.
        return -self.estimator.score_samples(matrix)

    def prepare_image(self, image: Image.Image) -> np.ndarray:
        return features(image, self.image_size)[None, :]

    def predict_prepared(self, inputs: np.ndarray) -> tuple[float, None]:
        return float(-self.estimator.score_samples(inputs)[0]), None

    def synchronize(self) -> None:
        pass

    def save(self, path: Path, metadata: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self, "metadata": metadata}, path)

    @staticmethod
    def load(path: Path) -> tuple["Baseline", dict]:
        """Load only locally produced, trusted artifacts (joblib uses pickle)."""
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}. Run qualitypilot baseline first.")
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or not isinstance(bundle.get("model"), Baseline):
            raise ValueError("Not a QualityPilot baseline artifact")
        return bundle["model"], bundle["metadata"]
