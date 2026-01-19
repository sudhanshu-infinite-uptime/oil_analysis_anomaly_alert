"""
app/models/model_builder.py

End-to-end model training module for Oil Anomaly Detection (Trend API v2).

Responsibilities:
- Train IsolationForest models
- Persist model + scaler + metadata
- Never crash Flink jobs
- Fail fast and log clearly

IMPORTANT (v2):
- Training uses Trend API v2
- deviceIdentifier is passed directly
- parameter_group_id is NO LONGER USED
"""

from __future__ import annotations

import pickle
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from app.api.trend_api_client import TrendAPIClient
from app.models.model_store import save_binary, save_metadata
from app.utils.logging_utils import get_logger
from app.config import (
    CONFIG,
    MODEL_FEATURE_CODES,
    MODEL_FEATURE_NAME_MAP,
    MODEL_FEATURE_NAMES_ORDERED,
)

logger = get_logger(__name__)


# =====================================================================
# PUBLIC ENTRYPOINT
# DEVICE ID → MODEL (Trend API v2)
# =====================================================================
def build_model_for_device_v2(
    device_id: str,
    start_datetime: str,
    end_datetime: str,
    interval_value: int = 1,
    interval_unit: str = "seconds",
) -> None:
    """
    Train and persist model using Trend API v2.

    Flow:
    deviceIdentifier → Trend API v2 → monitorId → model
    """

    logger.info(
        "Model training started (v2) | DEVICEID=%s",
        device_id,
    )

    try:
        client = TrendAPIClient()

        records = client.get_history(
            device_identifier=device_id,
            feature_codes=MODEL_FEATURE_CODES,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            interval_value=interval_value,
            interval_unit=interval_unit,
        )

    except Exception as exc:
        logger.error(
            "Trend API v2 failure | DEVICEID=%s | error=%s",
            device_id,
            exc,
            exc_info=True,
        )
        return

    if not records:
        logger.warning(
            "No training records returned | DEVICEID=%s",
            device_id,
        )
        return

    monitor_id = records[0].get("MONITORID")
    if not monitor_id:
        logger.error(
            "Missing MONITORID in Trend API v2 response | DEVICEID=%s",
            device_id,
        )
        return

    param_rows = [
        r["PROCESS_PARAMETER"]
        for r in records
        if isinstance(r.get("PROCESS_PARAMETER"), dict)
    ]

    if not param_rows:
        logger.warning(
            "No usable training rows | MONITORID=%s",
            monitor_id,
        )
        return

    _train_single_monitor_v2(
        monitor_id=monitor_id,
        param_rows=param_rows,
    )


# =====================================================================
# INTERNAL CORE TRAINING LOGIC (UNCHANGED ML)
# =====================================================================
def _train_single_monitor_v2(
    monitor_id: int,
    param_rows: List[Dict[str, Any]],
) -> None:
    """
    Core training routine.
    Single monitor → single model.
    """

    try:
        logger.info(
            "Training model | MONITORID=%s | raw_rows=%d",
            monitor_id,
            len(param_rows),
        )

        rows: List[Dict[str, Any]] = []

        for params in param_rows:
            filtered = {}
            for code in MODEL_FEATURE_CODES:
                value = params.get(code)
                if value is None:
                    break
                filtered[MODEL_FEATURE_NAME_MAP[code]] = value
            else:
                rows.append(filtered)

        if not rows:
            logger.warning(
                "No valid rows after feature validation | MONITORID=%s",
                monitor_id,
            )
            return

        train_df = pd.DataFrame(rows)[MODEL_FEATURE_NAMES_ORDERED]

        if train_df.empty:
            logger.warning(
                "Empty training DataFrame | MONITORID=%s",
                monitor_id,
            )
            return

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(train_df)

        model = IsolationForest(
            n_estimators=CONFIG.MODEL_TREES,
            contamination=CONFIG.ANOMALY_CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled)

        metadata = {
            "monitor_id": monitor_id,
            "trained_at": datetime.utcnow().isoformat(),
            "algorithm": "IsolationForest",
            "feature_codes": MODEL_FEATURE_CODES,
            "feature_names": MODEL_FEATURE_NAMES_ORDERED,
            "rows_used": len(train_df),
            "scaler": "RobustScaler",
            "trend_api": "v2",
        }

        save_binary(f"{monitor_id}/model.pkl", pickle.dumps(model))
        save_binary(f"{monitor_id}/scaler.pkl", pickle.dumps(scaler))
        save_metadata(monitor_id, metadata)

        logger.info(
            "Model training completed successfully | MONITORID=%s",
            monitor_id,
        )

    except Exception as exc:
        logger.error(
            "Model training failed | MONITORID=%s | error=%s",
            monitor_id,
            exc,
            exc_info=True,
        )
