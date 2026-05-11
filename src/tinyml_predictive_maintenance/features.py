from __future__ import annotations

import statistics


SENSOR_COLUMNS = ("vibration", "temperature", "current")


def window_features(rows: list[dict[str, float]], window_size: int = 24) -> list[dict[str, float]]:
    windows: list[dict[str, float]] = []
    for start in range(0, len(rows) - window_size + 1, window_size):
        window = rows[start : start + window_size]
        features: dict[str, float] = {}
        for column in SENSOR_COLUMNS:
            values = [row[column] for row in window]
            features[f"{column}_mean"] = statistics.fmean(values)
            features[f"{column}_stdev"] = statistics.pstdev(values)
            features[f"{column}_max"] = max(values)
        features["label"] = max(row["label"] for row in window)
        windows.append(features)
    return windows
