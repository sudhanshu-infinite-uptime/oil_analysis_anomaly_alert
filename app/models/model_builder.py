"""
app/models/model_builder.py

Handles end-to-end model training for a specific MONITORID.

Responsibilities:
- Fetch historical data via TrendAPIClient
- Build feature DataFrame
- Train IsolationForest + RobustScaler
- Persist model, scaler, metadata to S3

This module does NOT:
- Perform anomaly detection
- Handle Kafka or Flink logic
"""

from __future__ import annotations

import pandas as pd
import pickle
from datetime import datetime
from typing import Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from app.api.trend_api_client import TrendAPIClient
from app.models.model_store import (
    save_binary,
    save_metadata,
)
from app.utils.logging_utils import get_logger
from app.utils.exceptions import TrainingFailedError
from app.config import CONFIG

logger = get_logger(__name__)


def build_model_for_monitor(
    monitor_id: str,
    df: Optional[pd.DataFrame] = None,
    months: int = 3,
):
    """
    Build and persist a model for a given MONITORID.

    Args:
        monitor_id: Monitor identifier
        df: Optional live data (ignored for now)
        months: How much historical data to fetch

    Raises:
        TrainingFailedError on any failure
    """

    logger.info(f"Starting model training | MONITORID={monitor_id}")

    try:
        client = TrendAPIClient()
        records = client.get_history(
            monitor_id=monitor_id,
            months=months,
            access_token=CONFIG.TREND_API_TOKEN,
        )
    except Exception as exc:
        raise TrainingFailedError(f"Trend API failure: {exc}")

    try:
        rows = []
        for record in records:
            params = record.get("PROCESS_PARAMETER") or record.get("processParameter")
            if isinstance(params, dict):
                rows.append(params)

        if not rows:
            raise TrainingFailedError("No usable PROCESS_PARAMETER data")

        train_df = pd.DataFrame(rows).dropna(axis=1, how="all")
    except Exception as exc:
        raise TrainingFailedError(f"Data preparation failed: {exc}")

    try:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(train_df)

        model = IsolationForest(
            n_estimators=200,
            contamination=CONFIG.ANOMALY_CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled)
    except Exception as exc:
        raise TrainingFailedError(f"Model training failed: {exc}")

    metadata = {
        "monitor_id": monitor_id,
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": list(train_df.columns),
        "rows_used": len(train_df),
        "algorithm": "IsolationForest",
    }

    try:
        save_binary(
            f"{monitor_id}/model.pkl",
            pickle.dumps(model),
        )
        save_binary(
            f"{monitor_id}/scaler.pkl",
            pickle.dumps(scaler),
        )
        save_metadata(monitor_id, metadata)
    except Exception as exc:
        raise TrainingFailedError(f"S3 persistence failed: {exc}")

    logger.info(f"Model training completed | MONITORID={monitor_id}")
