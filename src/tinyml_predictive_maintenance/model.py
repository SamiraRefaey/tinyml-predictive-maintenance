from __future__ import annotations

from dataclasses import dataclass
import math


FEATURES = (
    "vibration_mean",
    "vibration_stdev",
    "vibration_max",
    "temperature_mean",
    "temperature_stdev",
    "temperature_max",
    "current_mean",
    "current_stdev",
    "current_max",
)


@dataclass
class AnomalyModel:
    center: dict[str, float]
    scale: dict[str, float]
    threshold: float

    @classmethod
    def fit(cls, rows: list[dict[str, float]]) -> "AnomalyModel":
        normal = [row for row in rows if row.get("label", 0.0) == 0.0]
        center = {feature: sum(row[feature] for row in normal) / len(normal) for feature in FEATURES}
        scale = {}
        scores = []
        for feature in FEATURES:
            variance = sum((row[feature] - center[feature]) ** 2 for row in normal) / len(normal)
            scale[feature] = math.sqrt(variance) or 1.0
        for row in normal:
            scores.append(cls(center, scale, 0).score(row))
        threshold = max(scores) * 1.35
        return cls(center=center, scale=scale, threshold=threshold)

    def score(self, row: dict[str, float]) -> float:
        return sum(abs(row[feature] - self.center[feature]) / self.scale[feature] for feature in FEATURES) / len(FEATURES)

    def predict(self, row: dict[str, float]) -> int:
        return int(self.score(row) > self.threshold)
