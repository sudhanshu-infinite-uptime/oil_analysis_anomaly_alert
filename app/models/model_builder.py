from __future__ import annotations

import pickle
from datetime import datetime, timedelta
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


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------
class ModelTrainingFailed(Exception):
    """Raised when model training cannot be completed safely."""
    pass


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CHUNK_MINUTES = 60  # Safe for Trend API with 6 features @ 1s interval


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def build_model_for_device_v2(
    device_id: str,
    start_datetime: str,
    end_datetime: str,
    interval_value: int = 1,
    interval_unit: str = "seconds",
) -> None:
    """
    Train an IsolationForest model for ONE monitor resolved by deviceId.

    STRICT CONTRACT:
    - Uses EXACTLY 6 features
    - Training fails if ANY feature is missing
    - Uses chunked Trend API fetching
    """

    logger.info(
        "Model training started (v2) | DEVICEID=%s | window=%s → %s",
        device_id,
        start_datetime,
        end_datetime,
    )

    client = TrendAPIClient()

    try:
        records = _fetch_trend_history_chunked(
            client=client,
            device_id=device_id,
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
            f"Missing MONITORID in Trend API response | DEVICEID={device_id}"
        )

    param_rows = [
        r["PROCESS_PARAMETER"]
        for r in records
        if isinstance(r.get("PROCESS_PARAMETER"), dict)
    ]

    if not param_rows:
        raise ModelTrainingFailed(
            f"No usable PROCESS_PARAMETER rows | MONITORID={monitor_id}"
        )

    _train_single_monitor_strict(
        monitor_id=monitor_id,
        param_rows=param_rows,
    )


# ------------------------------------------------------------------
# Trend API chunked fetch (STRICT 6 FEATURES)
# ------------------------------------------------------------------
def _fetch_trend_history_chunked(
    client: TrendAPIClient,
    device_id: str,
    start_datetime: str,
    end_datetime: str,
    interval_value: int,
    interval_unit: str,
) -> List[Dict[str, Any]]:

    start = datetime.fromisoformat(start_datetime.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_datetime.replace("Z", "+00:00"))

    all_records: List[Dict[str, Any]] = []
    cursor = start

    while cursor < end:
        chunk_end = min(cursor + timedelta(minutes=CHUNK_MINUTES), end)

        logger.info(
            "Fetching trend chunk | DEVICEID=%s | %s → %s | features=6",
            device_id,
            cursor.isoformat(),
            chunk_end.isoformat(),
        )

        records = client.get_history(
            device_identifier=device_id,
            feature_codes=MODEL_FEATURE_CODES,  # EXACTLY 6
            start_datetime=cursor.isoformat().replace("+00:00", "Z"),
            end_datetime=chunk_end.isoformat().replace("+00:00", "Z"),
            interval_value=interval_value,
            interval_unit=interval_unit,
        )

        if not records:
            raise ModelTrainingFailed(
                f"Empty trend chunk | {cursor} → {chunk_end}"
            )

        all_records.extend(records)
        cursor = chunk_end

    if not all_records:
        raise ModelTrainingFailed("No trend data collected after chunking")

    logger.info(
        "Trend history collected | DEVICEID=%s | records=%d",
        device_id,
        len(all_records),
    )

    return all_records


# ------------------------------------------------------------------
# Training logic (STRICT FEATURE ENFORCEMENT)
# ------------------------------------------------------------------
def _train_single_monitor_strict(
    monitor_id: int,
    param_rows: List[Dict[str, Any]],
) -> None:

    logger.info(
        "Training model | MONITORID=%s | raw_rows=%d",
        monitor_id,
        len(param_rows),
    )

    rows: List[Dict[str, float]] = []

    for params in param_rows:
        row: Dict[str, float] = {}

        for code in MODEL_FEATURE_CODES:
            value = params.get(code)
            if value is None:
                break  # STRICT: drop row
            row[MODEL_FEATURE_NAME_MAP[code]] = float(value)
        else:
            rows.append(row)

    if not rows:
        raise ModelTrainingFailed(
            f"No valid rows after feature validation | MONITORID={monitor_id}"
        )

    train_df = pd.DataFrame(rows)[MODEL_FEATURE_NAMES_ORDERED]
    # logger.info("Training columns: %s", list(train_df.columns))


    if train_df.empty:
        raise ModelTrainingFailed(
            f"Empty training DataFrame | MONITORID={monitor_id}"
        )

    # ------------------------------
    # Scaling
    # ------------------------------
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(train_df)

    # ------------------------------
    # Model
    # ------------------------------
    model = IsolationForest(
        n_estimators=CONFIG.MODEL_TREES,
        contamination=CONFIG.ANOMALY_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # ------------------------------
    # Metadata
    # ------------------------------
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

    # ------------------------------
    # Persist
    # ------------------------------
    save_binary(f"{monitor_id}/model.pkl", pickle.dumps(model))
    save_binary(f"{monitor_id}/scaler.pkl", pickle.dumps(scaler))
    save_metadata(monitor_id, metadata)

    logger.info(
        "Model training completed successfully | MONITORID=%s | rows=%d",
        monitor_id,
        len(train_df),
    )
