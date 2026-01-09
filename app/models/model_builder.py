"""
app/models/model_builder.py

Handles end-to-end model training for ALL MONITORIDs
returned for a parameter group.

Responsibilities:
- Fetch historical data using TrendAPIClient
- Group records by MONITORID
- Build training DataFrame from PROCESS_PARAMETER payload
- Train IsolationForest with RobustScaler
- Persist model artifacts and metadata per MONITORID

Design principles:
- Never crash Flink streaming jobs
- Fail gracefully on external dependencies
- Log every failure with context
- Skip bad monitors instead of poisoning the pipeline
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
from app.models.model_store import save_binary, save_metadata
from app.utils.logging_utils import get_logger
from app.config import (
    CONFIG,
    MODEL_FEATURE_CODES,
    MODEL_FEATURE_NAME_MAP,
    MODEL_FEATURE_NAMES_ORDERED,
)

logger = get_logger(__name__)


def build_model_for_parameter_group(
    parameter_group_id: int,
    start_datetime: str,
    end_datetime: str,
) -> None:
    """
    Train and persist models for ALL monitors
    under the given parameter_group_id.
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

    # --------------------------------------------------
    # Step 2: Group records by MONITORID
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Step 3: Train model PER monitor
    # --------------------------------------------------
    for monitor_id, param_rows in grouped.items():
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
                continue

            train_df = pd.DataFrame(rows)

            # Enforce strict feature order
            train_df = train_df[MODEL_FEATURE_NAMES_ORDERED]

            if train_df.empty:
                logger.warning(
                    "Empty training DataFrame | MONITORID=%s",
                    monitor_id,
                )
                continue

            logger.info(
                "Training data ready | MONITORID=%s | rows=%d | features=%s",
                monitor_id,
                len(train_df),
                MODEL_FEATURE_NAMES_ORDERED,
            )

            # --------------------------------------------------
            # Model training
            # --------------------------------------------------
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(train_df)

            model = IsolationForest(
                n_estimators=CONFIG.MODEL_TREES,
                contamination=CONFIG.ANOMALY_CONTAMINATION,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_scaled)

            # --------------------------------------------------
            # Metadata
            # --------------------------------------------------
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

            # --------------------------------------------------
            # Persist artifacts
            # --------------------------------------------------
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
            continue

