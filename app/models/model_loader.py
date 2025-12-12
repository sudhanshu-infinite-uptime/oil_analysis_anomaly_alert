"""
app/models/model_loader.py

Responsible for loading model artifacts for a given MONITORID. This includes:
    - Isolation Forest model (pickle)
    - RobustScaler (pickle)
    - metadata.json

This module does NOT:
    - Train models (handled by model_builder)
    - Perform predictions (handled by anomaly_detector)
    - Manage caching (handled by model_cache)

Used by:
    - model_cache.py → to retrieve uncached model files
    - flink/operators.py → for inference during the streaming job
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple, Any, Dict

from app.models.model_store import (
    get_model_paths,
    model_exists,
    load_binary
)
from app.models.model_metadata import load_metadata
from app.utils.logging_utils import get_logger
from app.utils.exceptions import ModelNotFoundError, ModelLoadError

logger = get_logger(__name__)


def load_model_bundle(monitor_id: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load Isolation Forest model, RobustScaler, and metadata for a MONITORID.
    """

    # 1️⃣ Check model exists in S3
    if not model_exists(monitor_id):
        raise ModelNotFoundError(
            monitor_id=monitor_id,
            path=f"models/{monitor_id}/",
        )

    paths = get_model_paths(monitor_id)

    model_path = Path(paths["model_path"])
    scaler_path = Path(paths["scaler_path"])
    metadata_path = Path(paths["metadata_path"])   # kept exactly as before

    logger.info(f"Loading model bundle for MONITORID={monitor_id}")

    try:
        # 2️⃣ Load model from S3 (binary)
        model = pickle.loads(load_binary(model_path))
        logger.info(f"Loaded model → S3:{model_path}")

        # 3️⃣ Load scaler from S3
        scaler = pickle.loads(load_binary(scaler_path))
        logger.info(f"Loaded scaler → S3:{scaler_path}")

        # 4️⃣ Load metadata JSON normally
        metadata = load_metadata(monitor_id)
        logger.info(f"Loaded metadata → S3:{metadata_path}")

    except Exception as exc:
        logger.error(f"Failed loading model bundle for MONITORID={monitor_id}: {exc}")
        raise ModelLoadError(monitor_id, str(exc))

    return model, scaler, metadata
