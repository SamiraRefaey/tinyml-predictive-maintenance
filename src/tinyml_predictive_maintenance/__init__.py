"""TinyML predictive maintenance package."""

from .data import generate_sensor_stream, load_sensor_data, add_noise_to_data
from .features import window_features, extract_frequency_features, normalize_features
from .model import (
    AnomalyDetector,
    MahalanobisDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
    create_detector,
    FEATURES
)
from .quantize import (
    quantize_int8,
    dequantize_int8,
    quantize_array_int8,
    quantize_row,
    compute_quantization_ranges,
    simulate_tinyml_inference
)

__version__ = "0.1.0"

__all__ = [
    # Data utilities
    "generate_sensor_stream",
    "load_sensor_data",
    "add_noise_to_data",

    # Feature extraction
    "window_features",
    "extract_frequency_features",
    "normalize_features",

    # Models
    "AnomalyDetector",
    "MahalanobisDetector",
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "create_detector",
    "FEATURES",

    # Quantization
    "quantize_int8",
    "dequantize_int8",
    "quantize_array_int8",
    "quantize_row",
    "compute_quantization_ranges",
    "simulate_tinyml_inference",
]
