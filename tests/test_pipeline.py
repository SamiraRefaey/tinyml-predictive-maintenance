"""Tests for TinyML predictive maintenance pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.tinyml_predictive_maintenance.data import generate_sensor_stream
from src.tinyml_predictive_maintenance.features import window_features
from src.tinyml_predictive_maintenance.model import (
    MahalanobisDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
    create_detector
)
from src.tinyml_predictive_maintenance.quantize import (
    quantize_int8,
    dequantize_int8,
    compute_quantization_ranges,
    quantize_row
)


class TestDataGeneration:
    """Test data generation utilities."""

    def test_generate_sensor_stream(self):
        """Test synthetic data generation."""
        df = generate_sensor_stream(samples=100, anomaly_start=80)

        assert len(df) == 100
        assert "vibration" in df.columns
        assert "temperature" in df.columns
        assert "current" in df.columns
        assert "label" in df.columns

        # Check anomaly labels
        normal_labels = df.iloc[:80]["label"].sum()
        anomaly_labels = df.iloc[80:]["label"].sum()

        assert normal_labels == 0
        assert anomaly_labels == 20  # All should be 1.0


class TestFeatureExtraction:
    """Test feature extraction."""

    def test_window_features(self):
        """Test windowed feature extraction."""
        df = generate_sensor_stream(samples=96)  # Multiple of 24
        features_df = window_features(df, window_size=24)

        assert len(features_df) == 4  # 96/24 = 4 windows

        # Check expected feature columns
        expected_features = [
            "vibration_mean", "vibration_std", "vibration_max",
            "temperature_mean", "temperature_std", "temperature_max",
            "current_mean", "current_std", "current_max"
        ]

        for feature in expected_features:
            assert feature in features_df.columns

    def test_window_features_with_stride(self):
        """Test windowed features with stride."""
        df = generate_sensor_stream(samples=72)  # 72/12 = 6 windows with stride 12
        features_df = window_features(df, window_size=24, stride=12)

        assert len(features_df) == 5  # Overlapping windows


class TestAnomalyDetectors:
    """Test anomaly detection models."""

    def setup_method(self):
        """Setup test data."""
        df = generate_sensor_stream(samples=240, anomaly_start=180)
        self.features_df = window_features(df, window_size=24)
        self.X = self.features_df.drop(columns=["label", "window_start", "window_end"]).values
        self.y = self.features_df["label"].values

    def test_mahalanobis_detector(self):
        """Test Mahalanobis distance detector."""
        detector = MahalanobisDetector()
        detector = detector.fit(self.X, self.y)
        predictions = detector.predict(self.X)

        assert len(predictions) == len(self.X)
        assert np.sum(predictions) >= 1  # Should detect some anomalies

        # Test scoring
        scores = detector.score(self.X)
        assert len(scores) == len(self.X)
        assert all(score >= 0 for score in scores)

    def test_isolation_forest_detector(self):
        """Test Isolation Forest detector."""
        detector = IsolationForestDetector(contamination=0.1)
        detector = detector.fit(self.X, self.y)
        predictions = detector.predict(self.X)

        assert len(predictions) == len(self.X)
        assert 0 <= np.sum(predictions) <= len(self.X)

    def test_one_class_svm_detector(self):
        """Test One-Class SVM detector."""
        detector = OneClassSVMDetector(nu=0.1)
        detector = detector.fit(self.X, self.y)
        predictions = detector.predict(self.X)

        assert len(predictions) == len(self.X)
        assert 0 <= np.sum(predictions) <= len(self.X)

    def test_create_detector(self):
        """Test detector factory function."""
        detector = create_detector("mahalanobis")
        assert isinstance(detector, MahalanobisDetector)

        detector = create_detector("isolation_forest")
        assert isinstance(detector, IsolationForestDetector)

        detector = create_detector("one_class_svm")
        assert isinstance(detector, OneClassSVMDetector)

        with pytest.raises(ValueError):
            create_detector("invalid_type")


class TestQuantization:
    """Test quantization utilities."""

    def test_quantize_int8(self):
        """Test int8 quantization."""
        # Test basic quantization
        quantized = quantize_int8(0.5, 0.0, 1.0)
        assert -128 <= quantized <= 127

        # Test edge cases
        assert quantize_int8(0.0, 0.0, 1.0) == -128  # Min value
        assert quantize_int8(1.0, 0.0, 1.0) == 127   # Max value

    def test_quantize_dequantize_roundtrip(self):
        """Test quantization/dequantization roundtrip."""
        original = 0.7
        min_val, max_val = 0.0, 1.0

        quantized = quantize_int8(original, min_val, max_val)
        dequantized = dequantize_int8(quantized, min_val, max_val)

        # Should be close (within quantization error)
        assert abs(original - dequantized) < 0.01

    def test_compute_quantization_ranges(self):
        """Test computation of quantization ranges."""
        X = np.random.rand(100, 3)
        feature_names = ["feat1", "feat2", "feat3"]

        ranges = compute_quantization_ranges(X, feature_names)

        assert len(ranges) == 3
        for name in feature_names:
            assert name in ranges
            min_val, max_val = ranges[name]
            assert min_val <= max_val

    def test_quantize_row(self):
        """Test row quantization."""
        row = {"vibration": 0.5, "temperature": 0.8}
        ranges = {"vibration": (0.0, 1.0), "temperature": (0.0, 1.0)}

        quantized = quantize_row(row, ranges)

        assert "vibration" in quantized
        assert "temperature" in quantized
        assert isinstance(quantized["vibration"], int)
        assert isinstance(quantized["temperature"], int)


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self):
        """Test complete anomaly detection pipeline."""
        # Generate data
        df = generate_sensor_stream(samples=240, anomaly_start=180)

        # Extract features
        features_df = window_features(df, window_size=24)

        # Train detector
        X = features_df.drop(columns=["label", "window_start", "window_end"]).values
        y = features_df["label"].values
        detector = MahalanobisDetector()
        detector = detector.fit(X, y)

        # Make predictions
        predictions = detector.predict(X)

        # Should detect anomalies in the later windows
        assert np.sum(predictions) > 0

        # Most anomalies should be in the second half
        half_point = len(predictions) // 2
        anomalies_first_half = np.sum(predictions[:half_point])
        anomalies_second_half = np.sum(predictions[half_point:])

        assert anomalies_second_half >= anomalies_first_half
