"""Quantization utilities for TinyML deployment."""

from __future__ import annotations

from typing import Dict, Tuple, Union, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)


def quantize_int8(value: float, min_value: float, max_value: float) -> int:
    """
    Quantize a float value to int8.

    Args:
        value: Value to quantize
        min_value: Minimum value in range
        max_value: Maximum value in range

    Returns:
        Quantized int8 value
    """
    if max_value <= min_value:
        return 0

    # Scale to 0-255 range first, then shift to -128 to 127
    scaled = (value - min_value) / (max_value - min_value) * 255
    quantized = round(scaled) - 128

    return max(-128, min(127, quantized))


def dequantize_int8(quantized_value: int, min_value: float, max_value: float) -> float:
    """
    Dequantize an int8 value back to float.

    Args:
        quantized_value: Quantized int8 value
        min_value: Minimum value in original range
        max_value: Maximum value in original range

    Returns:
        Dequantized float value
    """
    if max_value <= min_value:
        return min_value

    # Shift back to 0-255 range
    scaled = quantized_value + 128

    # Scale back to original range
    return min_value + (scaled / 255) * (max_value - min_value)


def quantize_array_int8(values: np.ndarray, min_value: Optional[float] = None, max_value: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    """
    Quantize an array of float values to int8.

    Args:
        values: Array of float values
        min_value: Minimum value (if None, computed from data)
        max_value: Maximum value (if None, computed from data)

    Returns:
        Tuple of (quantized array, min_value, max_value)
    """
    if min_value is None:
        min_value = float(np.min(values))
    if max_value is None:
        max_value = float(np.max(values))

    quantized = np.array([quantize_int8(float(v), min_value, max_value) for v in values])
    return quantized, min_value, max_value


def quantize_row(row: Dict[str, float], ranges: Dict[str, Tuple[float, float]]) -> Dict[str, int]:
    """
    Quantize a row of features.

    Args:
        row: Dictionary of feature values
        ranges: Dictionary mapping feature names to (min, max) tuples

    Returns:
        Dictionary of quantized values
    """
    quantized = {}
    for key, value in row.items():
        if key in ranges:
            min_val, max_val = ranges[key]
            quantized[key] = quantize_int8(value, min_val, max_val)
        else:
            logger.warning(f"No quantization range found for feature: {key}")
            quantized[key] = 0

    return quantized


def compute_quantization_ranges(df: np.ndarray, feature_names: list[str]) -> Dict[str, Tuple[float, float]]:
    """
    Compute quantization ranges for features.

    Args:
        df: DataFrame or array with features
        feature_names: List of feature names

    Returns:
        Dictionary mapping feature names to (min, max) tuples
    """
    ranges = {}

    if isinstance(df, np.ndarray):
        for i, name in enumerate(feature_names):
            values = df[:, i]
            ranges[name] = (float(np.min(values)), float(np.max(values)))
    else:
        for name in feature_names:
            if name in df.columns:
                values = df[name].values
                ranges[name] = (float(np.min(values)), float(np.max(values)))
            else:
                logger.warning(f"Feature {name} not found in data")
                ranges[name] = (0.0, 1.0)

    return ranges


def simulate_tinyml_inference(model_params: Dict, quantized_input: Dict[str, int], ranges: Dict[str, Tuple[float, float]]) -> float:
    """
    Simulate TinyML inference with quantized inputs.

    Args:
        model_params: Model parameters (center, scale, threshold)
        quantized_input: Quantized input features
        ranges: Quantization ranges

    Returns:
        Anomaly score
    """
    # Dequantize inputs
    dequantized = {}
    for feature, q_val in quantized_input.items():
        if feature in ranges:
            min_val, max_val = ranges[feature]
            dequantized[feature] = dequantize_int8(q_val, min_val, max_val)
        else:
            dequantized[feature] = 0.0

    # Simple anomaly score calculation (Mahalanobis-like)
    center = model_params.get("center", {})
    scale = model_params.get("scale", {})

    score = 0.0
    count = 0
    for feature in dequantized:
        if feature in center and feature in scale:
            diff = abs(dequantized[feature] - center[feature])
            if scale[feature] > 0:
                score += diff / scale[feature]
                count += 1

    return score / count if count > 0 else 0.0
