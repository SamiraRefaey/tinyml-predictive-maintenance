"""Advanced anomaly detection models for TinyML predictive maintenance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import logging
import math
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib

logger = logging.getLogger(__name__)

FEATURES = (
    "vibration_mean",
    "vibration_stdev",
    "vibration_max",
    "temperature_mean",
    "temperature_stdev",
    "temperature_max",
    "current_mean",
    "current_stdev",
    "current_max",
)


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detection models."""

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or list(FEATURES)

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AnomalyDetector":
        """Fit the model to the training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores or labels for input data."""
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "AnomalyDetector":
        """Load a model from disk."""
        pass


class MahalanobisDetector(AnomalyDetector):
    """Mahalanobis distance-based anomaly detector."""

    def __init__(self, center: Optional[Dict[str, float]] = None, scale: Optional[Dict[str, float]] = None,
                 threshold: float = 3.0, feature_names: Optional[List[str]] = None):
        super().__init__(feature_names)
        self.center = center or {}
        self.scale = scale or {}
        self.threshold = threshold
        self.center_array = np.array([self.center.get(feat, 0.0) for feat in self.feature_names])
        self.scale_array = np.array([self.scale.get(feat, 1.0) for feat in self.feature_names])

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MahalanobisDetector":
        """Fit the model using training data."""
        if y is not None:
            # Use only normal data for fitting
            normal_mask = y == 0
            X_normal = X[normal_mask]
        else:
            X_normal = X

        center = np.mean(X_normal, axis=0)
        scale = np.std(X_normal, axis=0)
        scale = np.where(scale == 0, 1.0, scale)  # Avoid division by zero

        # Calculate threshold as max score on normal data * 1.35
        scores = self._mahalanobis_distance(X_normal, center, scale)
        threshold = np.max(scores) * 1.35

        self.center = {feat: float(center[i]) for i, feat in enumerate(self.feature_names)}
        self.scale = {feat: float(scale[i]) for i, feat in enumerate(self.feature_names)}
        self.threshold = float(threshold)
        
        # Update arrays
        self.center_array = center
        self.scale_array = scale
        
        return self

    @staticmethod
    def _mahalanobis_distance(X: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Calculate Mahalanobis distance."""
        X_centered = X - center
        X_scaled = X_centered / scale
        return np.mean(np.abs(X_scaled), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        scores = self._mahalanobis_distance(X, self.center_array, self.scale_array)
        return (scores > self.threshold).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores."""
        return self._mahalanobis_distance(X, self.center_array, self.scale_array)

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MahalanobisDetector":
        """Load model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest-based anomaly detector."""

    def __init__(self, contamination: float = 0.1, random_state: int = 42, feature_names: Optional[List[str]] = None):
        super().__init__(feature_names)
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.threshold = -contamination

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IsolationForestDetector":
        """Fit the Isolation Forest model."""
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        scores = self.model.decision_function(X)
        return (scores < self.threshold).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (lower is more anomalous)."""
        return -self.model.decision_function(X)

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IsolationForestDetector":
        """Load model from disk."""
        model = joblib.load(path)
        return cls(model=model)


class OneClassSVMDetector(AnomalyDetector):
    """One-Class SVM-based anomaly detector."""

    def __init__(self, nu: float = 0.1, kernel: str = "rbf", gamma: str = "scale", feature_names: Optional[List[str]] = None):
        super().__init__(feature_names)
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.threshold = 0.0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OneClassSVMDetector":
        """Fit the One-Class SVM model."""
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        return (self.model.predict(X) == -1).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores."""
        return -self.model.decision_function(X)

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OneClassSVMDetector":
        """Load model from disk."""
        model = joblib.load(path)
        return cls(model=model)


def create_detector(detector_type: str = "mahalanobis", **kwargs) -> AnomalyDetector:
    """Factory function to create anomaly detectors."""
    if detector_type == "mahalanobis":
        return MahalanobisDetector(**kwargs)
    elif detector_type == "isolation_forest":
        return IsolationForestDetector(**kwargs)
    elif detector_type == "one_class_svm":
        return OneClassSVMDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


# Backward compatibility
AnomalyModel = MahalanobisDetector
