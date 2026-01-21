from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from app.api.trend_api_client import TrendAPIClient
from app.models.model_store import (
    save_binary,
    save_metadata,
    mark_model_success,
    model_exists,
)
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
    pass


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CHUNK_MINUTES = 60  # safe for Trend API


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
    FINAL PRODUCTION CONTRACT

    - Resolve monitorId via Trend API
    - If model already exists → EXIT immediately
    - Fetch trend data chunk-by-chunk
    - Skip bad chunks and continue
    - Train using all valid collected data
    - Save model ONCE
    """

    logger.info(
        "Model training requested | DEVICEID=%s | window=%s → %s",
        device_id,
        start_datetime,
        end_datetime,
    )

    client = TrendAPIClient()

    records = _fetch_trend_history_chunked_skip_bad(
        client=client,
        device_id=device_id,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        interval_value=interval_value,
        interval_unit=interval_unit,
    )

    if not records:
        raise ModelTrainingFailed(
            f"No usable trend data collected | DEVICEID={device_id}"
        )

    monitor_id = records[0].get("MONITORID")
    if not monitor_id:
        raise ModelTrainingFailed(
            f"Missing MONITORID in Trend API response | DEVICEID={device_id}"
        )

    # --------------------------------------------------------------
    # HARD STOP: model already exists
    # --------------------------------------------------------------
    if model_exists(str(monitor_id)):
        logger.info(
            "Model already exists → skipping training | MONITORID=%s",
            monitor_id,
        )
        return

    param_rows = [
        r["PROCESS_PARAMETER"]
        for r in records
        if isinstance(r.get("PROCESS_PARAMETER"), dict)
    ]

    if not param_rows:
        raise ModelTrainingFailed(
            f"No usable PROCESS_PARAMETER rows | MONITORID={monitor_id}"
        )

    _train_and_persist_monitor(
        monitor_id=monitor_id,
        param_rows=param_rows,
    )


# ------------------------------------------------------------------
# Chunked fetch (SKIP BAD CHUNKS, CONTINUE)
# ------------------------------------------------------------------
def _fetch_trend_history_chunked_skip_bad(
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
            "Fetching trend chunk | DEVICEID=%s | %s → %s",
            device_id,
            cursor.isoformat(),
            chunk_end.isoformat(),
        )

        try:
            records = client.get_history(
                device_identifier=device_id,
                feature_codes=MODEL_FEATURE_CODES,
                start_datetime=cursor.isoformat().replace("+00:00", "Z"),
                end_datetime=chunk_end.isoformat().replace("+00:00", "Z"),
                interval_value=interval_value,
                interval_unit=interval_unit,
            )
        except Exception as exc:
            logger.error(
                "Trend chunk failed → skipping chunk | DEVICEID=%s | %s",
                device_id,
                exc,
            )
            cursor = chunk_end
            continue

        if not records:
            logger.warning(
                "Empty trend chunk → skipping | DEVICEID=%s | %s → %s",
                device_id,
                cursor.isoformat(),
                chunk_end.isoformat(),
            )
            cursor = chunk_end
            continue

        all_records.extend(records)
        cursor = chunk_end

    logger.info(
        "Trend collection completed | DEVICEID=%s | collected=%d",
        device_id,
        len(all_records),
    )

    return all_records


# ------------------------------------------------------------------
# Training + Persistence (STRICT FEATURES)
# ------------------------------------------------------------------
def _train_and_persist_monitor(
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
                break
            row[MODEL_FEATURE_NAME_MAP[code]] = float(value)
        else:
            rows.append(row)

    if not rows:
        raise ModelTrainingFailed(
            f"No valid rows after feature validation | MONITORID={monitor_id}"
        )

    train_df = pd.DataFrame(rows)[MODEL_FEATURE_NAMES_ORDERED]

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
    mark_model_success(str(monitor_id))

    logger.info(
        "Model trained and saved | MONITORID=%s | rows=%d",
        monitor_id,
        len(train_df),
    )
