"""
app/models/model_builder.py

Handles end-to-end model training for a specific MONITORID.

Responsibilities:
- Fetch historical data using TrendAPIClient
- Build training DataFrame from PROCESS_PARAMETER payload
- Train IsolationForest with RobustScaler
- Persist model artifacts and metadata

Design principles:
- Never crash Flink streaming jobs
- Fail gracefully on external dependencies
- Log every failure with context
- Skip training instead of poisoning the pipeline

This module does NOT:
- Perform anomaly detection
- Handle Kafka or Flink execution logic
"""

from __future__ import annotations

import pandas as pd
import pickle
from datetime import datetime
from typing import Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from app.api.trend_api_client import TrendAPIClient
from app.models.model_store import save_binary, save_metadata
from app.utils.logging_utils import get_logger
from app.config import CONFIG

logger = get_logger(__name__)


def build_model_for_parameter_group(
    parameter_group_id: int,
    start_datetime: str,
    end_datetime: str,
) -> None:
    """
    Train and persist a model using Trend API data.

    IMPORTANT:
    - parameter_group_id is INPUT
    - monitor_id is DERIVED from Trend API
    """

    logger.info(
        "Model training started | parameter_group_id=%s",
        parameter_group_id,
    )

    # --------------------------------------------------
    # Step 1: Fetch historical data
    # --------------------------------------------------
    try:
        client = TrendAPIClient()
        records = client.get_history(
            parameter_group_id=parameter_group_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
    except Exception as exc:
        logger.error(
            "Trend API failure | parameter_group_id=%s | error=%s",
            parameter_group_id,
            exc,
            exc_info=True,
        )
        return

    if not records:
        logger.warning(
            "No records returned | parameter_group_id=%s",
            parameter_group_id,
        )
        return

    # ✅ REAL monitorId from API
    monitor_id = records[0]["MONITORID"]

    logger.info(
        "Resolved monitor_id=%s from parameter_group_id=%s",
        monitor_id,
        parameter_group_id,
    )

    # --------------------------------------------------
    # Step 2: Extract PROCESS_PARAMETER
    # --------------------------------------------------
    rows = []
    for record in records:
        params = record.get("PROCESS_PARAMETER")
        if isinstance(params, dict):
            rows.append(params)

    if not rows:
        logger.warning(
            "No usable PROCESS_PARAMETER data | MONITORID=%s",
            monitor_id,
        )
        return

    train_df = pd.DataFrame(rows).dropna(axis=1, how="all")

    if train_df.empty:
        logger.warning(
            "Training DataFrame empty | MONITORID=%s",
            monitor_id,
        )
        return

    # --------------------------------------------------
    # Step 3: Train model
    # --------------------------------------------------
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
        logger.error(
            "Model training failed | MONITORID=%s | error=%s",
            monitor_id,
            exc,
            exc_info=True,
        )
        return

    metadata = {
        "monitor_id": monitor_id,
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": list(train_df.columns),
        "rows_used": len(train_df),
        "algorithm": "IsolationForest",
        "parameter_group_id": parameter_group_id,
    }

    # --------------------------------------------------
    # Step 4: Persist artifacts (KEYED BY monitor_id)
    # --------------------------------------------------
    try:
        save_binary(f"{monitor_id}/model.pkl", pickle.dumps(model))
        save_binary(f"{monitor_id}/scaler.pkl", pickle.dumps(scaler))
        save_metadata(monitor_id, metadata)
    except Exception as exc:
        logger.error(
            "Model persistence failed | MONITORID=%s | error=%s",
            monitor_id,
            exc,
            exc_info=True,
        )
        return

    logger.info(
        "Model training completed successfully | MONITORID=%s",
        monitor_id,
    )
