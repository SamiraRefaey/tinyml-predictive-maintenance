"""Command-line interface for TinyML predictive maintenance system."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data import generate_sensor_stream, load_sensor_data
from .features import window_features, normalize_features
from .model import create_detector, FEATURES
from .quantize import compute_quantization_ranges, quantize_row

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TinyML Predictive Maintenance Anomaly Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data-source",
        choices=["generate", "csv"],
        default="generate",
        help="Source of sensor data"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        help="Path to CSV file (required if data-source is csv)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=240,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--anomaly-start",
        type=int,
        default=180,
        help="Index where anomalies begin"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=24,
        help="Size of sliding window for feature extraction"
    )
    parser.add_argument(
        "--detector",
        choices=["mahalanobis", "isolation_forest", "one_class_svm"],
        default="mahalanobis",
        help="Type of anomaly detector to use"
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Expected proportion of anomalies (for isolation_forest)"
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.1,
        help="Anomaly proportion parameter (for one_class_svm)"
    )
    parser.add_argument(
        "--save-model",
        type=str,
        help="Path to save trained model"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        help="Path to load pre-trained model"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize model for TinyML deployment"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        # Load or generate data
        if args.data_source == "generate":
            logger.info("Generating synthetic sensor data...")
            df = generate_sensor_stream(args.samples, args.anomaly_start)
        else:
            if not args.csv_file:
                parser.error("--csv-file is required when data-source is csv")
            logger.info(f"Loading data from {args.csv_file}...")
            df = load_sensor_data(args.csv_file)

        logger.info(f"Loaded {len(df)} data points")

        # Extract features
        logger.info(f"Extracting features with window size {args.window_size}...")
        features_df = window_features(df, window_size=args.window_size)
        logger.info(f"Extracted {len(features_df)} feature windows")

        # Prepare feature matrix
        feature_cols = [col for col in features_df.columns if col.startswith(tuple(f"{s}_" for s in ["vibration", "temperature", "current"]))]
        X = features_df[feature_cols].values
        y = features_df["label"].values if "label" in features_df.columns else None

        # Create and train detector
        if args.load_model:
            logger.info(f"Loading model from {args.load_model}...")
            # For now, assume Mahalanobis detector
            from .model import MahalanobisDetector
            detector = MahalanobisDetector.load(args.load_model)
        else:
            logger.info(f"Training {args.detector} detector...")
            if args.detector == "mahalanobis":
                detector = create_detector("mahalanobis", feature_names=feature_cols)
                detector = detector.fit(X, y)
            elif args.detector == "isolation_forest":
                detector = create_detector("isolation_forest", contamination=args.contamination)
                detector = detector.fit(X, y)
            elif args.detector == "one_class_svm":
                detector = create_detector("one_class_svm", nu=args.nu)
                detector = detector.fit(X, y)

        # Make predictions
        predictions = detector.predict(X)
        anomaly_count = np.sum(predictions)

        logger.info(f"Detected {anomaly_count} anomalous windows out of {len(predictions)}")

        # Quantization demo
        if args.quantize:
            logger.info("Demonstrating quantization...")
            ranges = compute_quantization_ranges(X, feature_cols)

            # Quantize first window as example
            sample_features = {col: X[0, i] for i, col in enumerate(feature_cols)}
            quantized = quantize_row(sample_features, ranges)

            logger.info(f"Sample quantized features: {quantized}")

        # Save model if requested
        if args.save_model:
            logger.info(f"Saving model to {args.save_model}...")
            detector.save(args.save_model)

        # Print summary
        print("\n" + "="*50)
        print("PREDICTIVE MAINTENANCE RESULTS")
        print("="*50)
        print(f"Data points: {len(df)}")
        print(f"Feature windows: {len(features_df)}")
        print(f"Detector type: {args.detector}")
        print(f"Anomalous windows detected: {anomaly_count}")
        print(".1f")
        print("="*50)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
