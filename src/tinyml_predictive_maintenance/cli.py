from __future__ import annotations

from .data import generate_sensor_stream
from .features import window_features
from .model import AnomalyModel


def main() -> None:
    stream = generate_sensor_stream()
    windows = window_features(stream)
    model = AnomalyModel.fit(windows)
    predictions = [model.predict(row) for row in windows]
    alerts = sum(predictions)
    print(f"Processed windows: {len(windows)}")
    print(f"Anomaly alerts: {alerts}")
    print(f"Threshold: {model.threshold:.3f}")


if __name__ == "__main__":
    main()
