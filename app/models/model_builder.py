"""
app/models/model_builder.py

End-to-end model training module for Oil Anomaly Detection.

This module is responsible ONLY for model training and persistence.
It is intentionally decoupled from Flink streaming logic.

Supported training entry points:
1. Parameter Group  → Train models for ALL MONITORIDs
2. Device ID        → Resolve MONITORID and train ONE model
3. Direct MONITORID → Train ONE model (used by bootstrap flows)

Design principles:
- Never crash Flink jobs
- Fail fast and log clearly
- Skip bad monitors instead of poisoning the pipeline
- Single source of truth for training logic
"""

from __future__ import annotations

import pickle
from datetime import datetime
from typing import Dict, Any, List, DefaultDict
from collections import defaultdict

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from app.api.trend_api_client import TrendAPIClient
from app.api.device_api_client import DeviceAPIClient
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
# PUBLIC ENTRYPOINT 1
# PARAMETER GROUP → ALL MONITORS
# =====================================================================
def build_model_for_parameter_group(
    parameter_group_id: int,
    start_datetime: str,
    end_datetime: str,
) -> None:
    """
    Train and persist models for ALL MONITORIDs
    returned under a given parameter group.
    """

    logger.info(
        "Model training started | parameter_group_id=%s",
        parameter_group_id,
    )

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

    grouped: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)

    for r in records:
        monitor_id = r.get("MONITORID")
        params = r.get("PROCESS_PARAMETER")

        if monitor_id is None or not isinstance(params, dict):
            continue

        grouped[monitor_id].append(params)

    logger.info(
        "Fetched data | parameter_group_id=%s | monitors=%d",
        parameter_group_id,
        len(grouped),
    )

    for monitor_id, param_rows in grouped.items():
        _train_single_monitor(
            monitor_id=monitor_id,
            parameter_group_id=parameter_group_id,
            param_rows=param_rows,
        )

# =====================================================================
# PUBLIC ENTRYPOINT 2
# DEVICE ID → MONITOR → MODEL
# =====================================================================
def build_model_for_device(
    device_id: str,
    parameter_group_id: int,
    start_datetime: str,
    end_datetime: str,
) -> None:
    """
    Resolve MONITORID from DEVICE ID and train model for that monitor.
    """

    try:
        trend_client = TrendAPIClient()
        token = trend_client.token_manager.get_token()

        device_client = DeviceAPIClient()
        monitor_id = device_client.get_monitor_id(device_id, token)

    except Exception as exc:
        logger.error(
            "Failed to resolve monitor from device | device=%s | error=%s",
            device_id,
            exc,
            exc_info=True,
        )
        return

    build_model_for_monitor(
        monitor_id=monitor_id,
        parameter_group_id=parameter_group_id,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

# =====================================================================
# PUBLIC ENTRYPOINT 3
# SINGLE MONITOR
# =====================================================================
def build_model_for_monitor(
    monitor_id: int,
    parameter_group_id: int,
    start_datetime: str,
    end_datetime: str,
) -> None:
    """
    Train and persist model for a SINGLE MONITORID.
    """

    logger.info(
        "Model training started | MONITORID=%s | parameter_group_id=%s",
        monitor_id,
        parameter_group_id,
    )

    try:
        client = TrendAPIClient()
        records = client.get_history(
            parameter_group_id=parameter_group_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
    except Exception as exc:
        logger.error(
            "Trend API failure | MONITORID=%s | error=%s",
            monitor_id,
            exc,
            exc_info=True,
        )
        return

    param_rows = [
        r["PROCESS_PARAMETER"]
        for r in records
        if r.get("MONITORID") == monitor_id
        and isinstance(r.get("PROCESS_PARAMETER"), dict)
    ]

    if not param_rows:
        logger.warning("No training data | MONITORID=%s", monitor_id)
        return

    _train_single_monitor(
        monitor_id=monitor_id,
        parameter_group_id=parameter_group_id,
        param_rows=param_rows,
    )

# =====================================================================
# INTERNAL CORE TRAINING LOGIC
# ONE MONITOR → ONE MODEL
# =====================================================================
def _train_single_monitor(
    monitor_id: int,
    parameter_group_id: int,
    param_rows: List[Dict[str, Any]],
) -> None:
    """
    Core training routine.
    This is the ONLY place where model training happens.
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
            "parameter_group_id": parameter_group_id,
            "trained_at": datetime.utcnow().isoformat(),
            "algorithm": "IsolationForest",
            "feature_codes": MODEL_FEATURE_CODES,
            "feature_names": MODEL_FEATURE_NAMES_ORDERED,
            "rows_used": len(train_df),
            "scaler": "RobustScaler",
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
