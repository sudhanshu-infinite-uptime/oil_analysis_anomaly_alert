"""
app/models/model_builder.py

Builds IsolationForest models for each MONITORID.
This module:
    - Trains from the window dataframe (if provided)
    - OR fetches last N months history from Trend API when df=None
    - Fits RobustScaler + IsolationForest
    - Saves model, scaler, metadata.json

Used by:
    - operators.py (on-demand model building)
    - manual scripts (build_local_model, benchmark)
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

from app.api.trend_api_client import TrendAPIClient
from app.models.model_store import get_model_paths, save_binary, save_metadata
from app.utils.logging_utils import get_logger
from app.utils.exceptions import TrainingFailedError
from app.config import CONFIG

logger = get_logger(__name__)


def build_model_for_monitor(
    monitor_id: str,
    df: pd.DataFrame | None = None,
    months: int = 3,
) -> None:

    logger.info(f"🔧 Starting model build for MONITORID={monitor_id}")

    # A. Fetch data if df not provided
    if df is None:
        logger.info(f"No DF provided → Fetching {months} months history via Trend API")
        api = TrendAPIClient()
        records = api.get_history(monitor_id, months)

        if not records:
            raise TrainingFailedError(monitor_id, "Trend API returned 0 records")

        df = pd.DataFrame([r.get("PROCESS_PARAMETER", {}) for r in records])

    # B. Validate dataframe  (RESTORED)
    if df is None or df.empty:
        raise TrainingFailedError(monitor_id, "Training dataframe is empty")

    # If window df contains nested structure
    if "PROCESS_PARAMETER" in df.columns:
        df = pd.json_normalize(df["PROCESS_PARAMETER"])

    feature_columns = sorted(df.columns.tolist())
    logger.info(f"Training with features: {feature_columns}")

    # C. Fit scaler
    try:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(df[feature_columns])
    except Exception as exc:
        raise TrainingFailedError(monitor_id, f"Scaler failed: {exc}")

    # D. Train model
    try:
        model = IsolationForest(
            n_estimators=CONFIG.MODEL_TREES,
            contamination=CONFIG.MODEL_CONTAMINATION,
            random_state=42,
        ).fit(X_scaled)
    except Exception as exc:
        raise TrainingFailedError(monitor_id, f"IsolationForest failed: {exc}")

    # E. Save artifacts to S3
    paths = get_model_paths(str(monitor_id))

    logger.info("Saving scaler to S3...")
    save_binary(paths["scaler_path"], pickle.dumps(scaler))

    logger.info("Saving model to S3...")
    save_binary(paths["model_path"], pickle.dumps(model))

    metadata = {
        "monitor_id": str(monitor_id),
        "feature_names": feature_columns,
        "trained_at": datetime.utcnow().isoformat(),
        "trained_from": "API" if df is None else "WINDOW",
        "window_size": CONFIG.WINDOW_COUNT,
    }

    logger.info("Saving metadata.json to S3...")
    save_metadata(monitor_id, metadata)

    logger.info(f"✅ Model built & saved for MONITORID={monitor_id}")
