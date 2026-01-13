"""
app/predictor/anomaly_detector.py

Streaming-safe anomaly detection logic.

Responsibilities:
- Scale input window
- Run IsolationForest inference
- Identify anomaly rows
- Compute top contributing features
- Return structured, non-throwing results
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def detect_anomalies(
    df: pd.DataFrame,
    model: Any,
    scaler: Any,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform anomaly detection on an input feature window.
    """

    # ---------------------------------------------------
    # 1. Input validation
    # ---------------------------------------------------
    if df.empty:
        logger.warning("Anomaly detection skipped: empty DataFrame")
        return {"is_anomaly": False, "reason": "EMPTY_DATAFRAME"}

    feature_names = metadata.get("feature_names")
    if not feature_names:
        logger.error("Missing feature_names in model metadata")
        return {"is_anomaly": False, "reason": "INVALID_METADATA"}

    if df.shape[1] != len(feature_names):
        logger.error(
            "Feature mismatch | df_cols=%d metadata_cols=%d",
            df.shape[1],
            len(feature_names),
        )
        return {"is_anomaly": False, "reason": "FEATURE_MISMATCH"}

    # ---------------------------------------------------
    # 2. Scaling
    # ---------------------------------------------------
    try:
        X_scaled = scaler.transform(df)
    except Exception as exc:
        logger.error(f"Scaler transform failed: {exc}")
        return {"is_anomaly": False, "reason": "SCALER_FAILURE"}

    # ---------------------------------------------------
    # 3. Prediction
    # ---------------------------------------------------
    try:
        preds = model.predict(X_scaled)
    except Exception as exc:
        logger.error(f"Model prediction failed: {exc}")
        return {"is_anomaly": False, "reason": "MODEL_FAILURE"}

    anomaly_indices = np.where(preds == -1)[0].tolist()

    if not anomaly_indices:
        logger.debug("No anomalies detected in window")
        return {
            "is_anomaly": False,
            "anomaly_indices": [],
            "top_features": [],
            "model_metadata": metadata,
        }

    monitor_id = metadata.get("monitor_id", "UNKNOWN")

    logger.warning(
        "Anomalies detected | MONITORID=%s | count=%d | indices=%s",
        monitor_id,
        len(anomaly_indices),
        anomaly_indices,
    )

    # ---------------------------------------------------
    # 4. Feature attribution
    # ---------------------------------------------------
    top_features = _find_top_features(
        X_scaled,
        anomaly_indices,
        feature_names,
    )

    return {
        "is_anomaly": True,
        "anomaly_indices": anomaly_indices,
        "top_features": top_features,
        "model_metadata": metadata,
    }


def _find_top_features(
    X_scaled: np.ndarray,
    anomaly_indices: List[int],
    feature_names: List[str],
) -> List[str]:
    """
    Identify top contributing features using mean absolute deviation.
    """

    if not anomaly_indices:
        return []

    mean_vals = X_scaled.mean(axis=0)
    deviations = np.abs(X_scaled[anomaly_indices] - mean_vals)
    avg_dev = deviations.mean(axis=0)

    top_idx = np.argsort(-avg_dev)[:2]

    return [feature_names[i] for i in top_idx]
