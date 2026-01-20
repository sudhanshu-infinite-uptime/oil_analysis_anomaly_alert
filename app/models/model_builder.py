from __future__ import annotations

import pickle
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from app.api.trend_api_client import TrendAPIClient
from app.models.model_store import save_binary, save_metadata, mark_model_success
from app.utils.logging_utils import get_logger
from app.config import (
    CONFIG,
    MODEL_FEATURE_CODES,
    MODEL_FEATURE_NAME_MAP,
    MODEL_FEATURE_NAMES_ORDERED,
)

logger = get_logger(__name__)


class ModelTrainingFailed(Exception):
    pass


def build_model_for_device_v2(
    device_id: str,
    start_datetime: str,
    end_datetime: str,
    interval_value: int = 1,
    interval_unit: str = "seconds",
) -> None:

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
        raise ModelTrainingFailed("Trend API v2 call failed") from exc

    if not records:
        raise ModelTrainingFailed(
            f"No training records returned | DEVICEID={device_id}"
        )

    monitor_id = records[0].get("MONITORID")
    if not monitor_id:
        raise ModelTrainingFailed(
            f"Missing MONITORID in Trend API v2 response | DEVICEID={device_id}"
        )

    param_rows = [
        r["PROCESS_PARAMETER"]
        for r in records
        if isinstance(r.get("PROCESS_PARAMETER"), dict)
    ]

    if not param_rows:
        raise ModelTrainingFailed(
            f"No usable training rows | MONITORID={monitor_id}"
        )

    _train_single_monitor_v2(
        monitor_id=monitor_id,
        param_rows=param_rows,
    )


def _train_single_monitor_v2(
    monitor_id: int,
    param_rows: List[Dict[str, Any]],
) -> None:

    try:
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
            raise ModelTrainingFailed(
                f"No valid rows after feature validation | MONITORID={monitor_id}"
            )

        train_df = pd.DataFrame(rows)[MODEL_FEATURE_NAMES_ORDERED]

        if train_df.empty:
            raise ModelTrainingFailed(
                f"Empty training DataFrame | MONITORID={monitor_id}"
            )

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
            "training_status": "SUCCESS",
        }

        save_binary(f"{monitor_id}/model.pkl", pickle.dumps(model))
        save_binary(f"{monitor_id}/scaler.pkl", pickle.dumps(scaler))
        save_metadata(monitor_id, metadata)
        mark_model_success(monitor_id)

        logger.info(
            "Model training completed successfully | MONITORID=%s",
            monitor_id,
        )

    except Exception as exc:
        if not isinstance(exc, ModelTrainingFailed):
            exc = ModelTrainingFailed(
                f"Unexpected training failure | MONITORID={monitor_id}"
            )
        logger.error(
            "Model training failed | MONITORID=%s | %s",
            monitor_id,
            exc,
            exc_info=True,
        )
        raise
