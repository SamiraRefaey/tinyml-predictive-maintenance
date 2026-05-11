from src.tinyml_predictive_maintenance.data import generate_sensor_stream
from src.tinyml_predictive_maintenance.features import window_features
from src.tinyml_predictive_maintenance.model import AnomalyModel
from src.tinyml_predictive_maintenance.quantize import quantize_int8


def test_predictive_maintenance_detects_anomaly_windows() -> None:
    windows = window_features(generate_sensor_stream())
    model = AnomalyModel.fit(windows)
    predictions = [model.predict(row) for row in windows]

    assert sum(predictions) >= 1
    assert predictions[-1] == 1
    assert -128 <= quantize_int8(0.5, 0.0, 1.0) <= 127
