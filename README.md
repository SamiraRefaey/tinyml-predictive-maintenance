# TinyML Predictive Maintenance System

End-to-end TinyML project for predictive maintenance using industrial sensor streams and anomaly detection. The repository includes data simulation, feature extraction, threshold-based anomaly modeling, INT8-style quantization, and embedded deployment notes.

## Features

- Synthetic vibration/current/temperature stream generation
- Windowed feature extraction
- Lightweight anomaly model suitable as a TinyML baseline
- Quantization utilities for int8 deployment simulation
- CLI demo and tests

## Run Locally

```bash
python -m pip install pytest
python -m pytest
python -m src.tinyml_predictive_maintenance.cli
```

## Production Extension

- Replace synthetic data with real sensor logs
- Train TensorFlow/Keras or scikit-learn models
- Convert to TensorFlow Lite INT8
- Deploy on microcontroller firmware
- Stream alerts to maintenance dashboards

## Metrics to Track

- Precision and recall for failure alerts
- False alert rate per hour
- Inference latency
- RAM and flash footprint
- Power usage during inference
