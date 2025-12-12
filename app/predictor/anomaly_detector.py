"""
app/predictor/anomaly_detector.py

Provides the core anomaly detection logic used during streaming inference.

Responsibilities:
    - Accept scaled/unscaled window data
    - Apply RobustScaler and Isolation Forest model
    - Detect anomaly rows (where prediction == -1)
    - Compute anomaly summary (e.g., top contributing features)
    - Return a structured result for Flink operators

Used by:
    - flink/operators.py → anomaly detection during streaming

This module does NOT:
    - Load models from disk (model_loader)
    - Cache models (model_cache)
    - Train models (model_builder)
    - Fetch historical data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from app.predictor.feature_mapping import FEATURES
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------------
# Main inference method
# -------------------------------------------------------------------
def detect_anomalies(
    df: pd.DataFrame,
    model: Any,
    scaler: Any,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform anomaly detection on an input feature window.

    Args:
        df: DataFrame containing the feature columns defined in FEATURES.
        model: Trained IsolationForest instance.
        scaler: Trained RobustScaler instance.
        metadata: Model metadata dictionary.

    Returns:
        Dictionary containing:
            - is_anomaly (bool)
            - anomaly_indices (list)
            - top_features (list)
            - model_metadata (dict)
            - reason (optional, when anomaly detection is skipped)
    """

    # ---------------------------------------------------------------
    # 1. Validate input
    # ---------------------------------------------------------------
    if df.empty:
        logger.warning("Anomaly detection skipped: empty DataFrame.")
        return {"is_anomaly": False, "reason": "EMPTY_DATAFRAME"}

    # ---------------------------------------------------------------
    # 2. Scale data
    # ---------------------------------------------------------------
    try:
        X_scaled = scaler.transform(df)
    except Exception as exc:
        logger.error(f"Scaler failed to transform DataFrame: {exc}")
        return {"is_anomaly": False, "reason": "SCALER_FAILURE"}

    # ---------------------------------------------------------------
    # 3. Isolation Forest predictions
    # ---------------------------------------------------------------
    try:
        preds = model.predict(X_scaled)
    except Exception as exc:
        logger.error(f"Model prediction failed: {exc}")
        return {"is_anomaly": False, "reason": "MODEL_FAILURE"}

    # Identify anomaly rows (IsolationForest uses -1 for anomalies)
    anomaly_indices = np.where(preds == -1)[0].tolist()

    if not anomaly_indices:
        logger.info("No anomalies detected in this window.")
        return {
            "is_anomaly": False,
            "anomaly_indices": [],
            "top_features": [],
            "model_metadata": metadata,
        }

    logger.warning(f"Anomalies detected at indices: {anomaly_indices}")

    # ---------------------------------------------------------------
    # 4. Compute top contributing features
    # ---------------------------------------------------------------
    top_features = _find_top_features(X_scaled, anomaly_indices)

    return {
        "is_anomaly": True,
        "anomaly_indices": anomaly_indices,
        "top_features": top_features,
        "model_metadata": metadata,
    }


# -------------------------------------------------------------------
# Feature contribution logic
# -------------------------------------------------------------------
def _find_top_features(
    X_scaled: np.ndarray,
    anomaly_indices: List[int],
) -> List[str]:
    """
    Identify features contributing most to anomalies.

    Method:
        Compute mean absolute deviation between anomaly rows
        and overall mean of scaled data. Return the top-2 features.

    Args:
        X_scaled: Scaled feature matrix (N x F)
        anomaly_indices: Positions where model predicted -1

    Returns:
        List of top 2 feature names.
    """

    if len(anomaly_indices) == 0:
        return []

    mean_vals = X_scaled.mean(axis=0)
    deviations = np.abs(X_scaled[anomaly_indices] - mean_vals)

    # Average deviation across anomaly rows
    avg_dev = deviations.mean(axis=0)

    # Get top two deviating feature indices
    top_idx = np.argsort(-avg_dev)[:2]

    return [FEATURES[i] for i in top_idx]
