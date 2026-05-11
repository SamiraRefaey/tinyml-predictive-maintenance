from __future__ import annotations

import math
import random


def generate_sensor_stream(samples: int = 240, anomaly_start: int = 180, seed: int = 7) -> list[dict[str, float]]:
    rng = random.Random(seed)
    rows: list[dict[str, float]] = []
    for index in range(samples):
        drift = 1.0 if index < anomaly_start else 2.2
        rows.append(
            {
                "vibration": drift * (0.55 + 0.12 * math.sin(index / 6) + rng.uniform(-0.03, 0.03)),
                "temperature": 38 + drift * 3.2 + rng.uniform(-0.8, 0.8),
                "current": 1.8 + drift * 0.22 + rng.uniform(-0.04, 0.04),
                "label": 0.0 if index < anomaly_start else 1.0,
            }
        )
    return rows
