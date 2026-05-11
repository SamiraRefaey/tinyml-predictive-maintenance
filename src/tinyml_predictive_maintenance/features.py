"""Feature extraction utilities for sensor data processing."""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

SENSOR_COLUMNS = ("vibration", "temperature", "current")


def window_features(
    df: pd.DataFrame,
    window_size: int = 24,
    stride: int = 24,
    feature_functions: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract statistical features from sliding windows of sensor data.

    Args:
        df: Input DataFrame with sensor columns
        window_size: Size of sliding window
        stride: Step size for sliding window
        feature_functions: List of feature functions to apply

    Returns:
        DataFrame with extracted features
    """
    if feature_functions is None:
        feature_functions = ["mean", "std", "max", "min", "rms", "skew", "kurtosis"]

    windows = []

    for start in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[start : start + window_size]
        features = {"window_start": start, "window_end": start + window_size - 1}

        for column in SENSOR_COLUMNS:
            if column not in window.columns:
                continue

            values = window[column].values

            for func_name in feature_functions:
                feature_name = f"{column}_{func_name}"
                try:
                    if func_name == "mean":
                        features[feature_name] = np.mean(values)
                    elif func_name == "std":
                        features[feature_name] = np.std(values, ddof=1)  # Sample std
                    elif func_name == "max":
                        features[feature_name] = np.max(values)
                    elif func_name == "min":
                        features[feature_name] = np.min(values)
                    elif func_name == "rms":
                        features[feature_name] = np.sqrt(np.mean(values**2))
                    elif func_name == "skew":
                        features[feature_name] = stats.skew(values)
                    elif func_name == "kurtosis":
                        features[feature_name] = stats.kurtosis(values)
                    elif func_name == "range":
                        features[feature_name] = np.max(values) - np.min(values)
                    elif func_name == "iqr":
                        features[feature_name] = stats.iqr(values)
                except Exception as e:
                    logger.warning(f"Failed to compute {feature_name}: {e}")
                    features[feature_name] = 0.0

        # Label is the maximum anomaly label in the window
        features["label"] = window["label"].max() if "label" in window.columns else 0.0

        windows.append(features)

    return pd.DataFrame(windows)


def extract_frequency_features(
    df: pd.DataFrame,
    window_size: int = 24,
    sampling_rate: float = 1.0
) -> pd.DataFrame:
    """
    Extract frequency domain features using FFT.

    Args:
        df: Input DataFrame
        window_size: Size of analysis window
        sampling_rate: Sampling rate in Hz

    Returns:
        DataFrame with frequency features
    """
    windows = []

    for start in range(0, len(df) - window_size + 1, window_size):
        window = df.iloc[start : start + window_size]
        features = {"window_start": start, "window_end": start + window_size - 1}

        for column in SENSOR_COLUMNS:
            if column not in window.columns:
                continue

            values = window[column].values

            # Remove DC component
            values_detrended = values - np.mean(values)

            # Compute FFT
            fft = np.fft.fft(values_detrended)
            freqs = np.fft.fftfreq(len(values), 1/sampling_rate)

            # Get magnitude spectrum
            magnitude = np.abs(fft)

            # Extract frequency domain features
            features[f"{column}_fft_mean"] = np.mean(magnitude)
            features[f"{column}_fft_std"] = np.std(magnitude)
            features[f"{column}_fft_max"] = np.max(magnitude)

            # Dominant frequency
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            if len(positive_magnitude) > 0:
                dominant_idx = np.argmax(positive_magnitude)
                features[f"{column}_dominant_freq"] = positive_freqs[dominant_idx]
                features[f"{column}_dominant_power"] = positive_magnitude[dominant_idx]

        features["label"] = window["label"].max() if "label" in window.columns else 0.0
        windows.append(features)

    return pd.DataFrame(windows)


def normalize_features(df: pd.DataFrame, method: str = "zscore") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalize feature DataFrame.

    Args:
        df: Input DataFrame
        method: Normalization method ('zscore', 'minmax', 'robust')

    Returns:
        Tuple of (normalized DataFrame, normalization parameters)
    """
    df_norm = df.copy()
    params = {}

    feature_cols = [col for col in df.columns if col not in ["label", "window_start", "window_end"]]

    for col in feature_cols:
        if method == "zscore":
            mean_val = df[col].mean()
            std_val = df[col].std()
            df_norm[col] = (df[col] - mean_val) / std_val
            params[col] = {"mean": mean_val, "std": std_val}
        elif method == "minmax":
            min_val = df[col].min()
            max_val = df[col].max()
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            params[col] = {"min": min_val, "max": max_val}
        elif method == "robust":
            median_val = df[col].median()
            mad_val = stats.median_abs_deviation(df[col])
            df_norm[col] = (df[col] - median_val) / mad_val
            params[col] = {"median": median_val, "mad": mad_val}

    return df_norm, params
