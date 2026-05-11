"""Data generation and loading utilities for TinyML predictive maintenance."""

from __future__ import annotations

import math
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sensor_stream(
    samples: int = 240,
    anomaly_start: int = 180,
    seed: int = 7,
    noise_level: float = 0.1
) -> pd.DataFrame:
    """
    Generate synthetic sensor data stream with normal and anomalous periods.

    Args:
        samples: Number of data points to generate
        anomaly_start: Index where anomalies begin
        seed: Random seed for reproducibility
        noise_level: Amount of noise to add to signals

    Returns:
        DataFrame with sensor readings and labels
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    data = []
    for index in range(samples):
        # Normal operation vs anomaly drift
        drift = 1.0 if index < anomaly_start else 2.2

        # Generate sensor readings with realistic patterns
        vibration = drift * (
            0.55 + 0.12 * math.sin(index / 6) +
            rng.uniform(-noise_level, noise_level)
        )

        temperature = 38 + drift * 3.2 + rng.uniform(-0.8, 0.8)

        current = 1.8 + drift * 0.22 + rng.uniform(-0.04, 0.04)

        # Add some realistic correlations and patterns
        vibration += 0.05 * temperature  # Temperature affects vibration
        current += 0.01 * vibration  # Vibration affects current slightly

        data.append({
            "timestamp": index,
            "vibration": max(0, vibration),  # Ensure non-negative
            "temperature": temperature,
            "current": current,
            "label": 0.0 if index < anomaly_start else 1.0,
        })

    return pd.DataFrame(data)


def load_sensor_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load sensor data from CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with sensor data
    """
    df = pd.read_csv(file_path)

    # Validate required columns
    required_cols = ["vibration", "temperature", "current"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Add label column if missing
    if "label" not in df.columns:
        df["label"] = 0.0

    return df


def add_noise_to_data(df: pd.DataFrame, noise_level: float = 0.05) -> pd.DataFrame:
    """
    Add Gaussian noise to sensor readings.

    Args:
        df: Input DataFrame
        noise_level: Standard deviation of noise

    Returns:
        DataFrame with added noise
    """
    df_noisy = df.copy()
    sensor_cols = ["vibration", "temperature", "current"]

    for col in sensor_cols:
        if col in df_noisy.columns:
            noise = np.random.normal(0, noise_level * df_noisy[col].std(), len(df_noisy))
            df_noisy[col] += noise

    return df_noisy
