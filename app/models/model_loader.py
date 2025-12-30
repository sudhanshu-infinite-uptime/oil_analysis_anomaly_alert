"""
app/models/model_loader.py

Loads trained model artifacts for a given MONITORID from S3.

Artifacts loaded:
- IsolationForest model (pickle)
- RobustScaler (pickle)
- metadata.json

Responsibilities:
- Validate model existence
- Deserialize model artifacts
- Return a fully usable model bundle

This module does NOT:
- Train models
- Perform inference
- Cache models in memory

Used by:
- model_cache.py
- flink/operators.py
"""

from __future__ import annotations

import pickle
from typing import Tuple, Any, Dict

from app.models.model_store import (
    get_model_paths,
    model_exists,
    load_binary,
    load_metadata,
)
from app.utils.logging_utils import get_logger
from app.utils.exceptions import ModelNotFoundError, ModelLoadError

logger = get_logger(__name__)


def load_model_bundle(
    monitor_id: str,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load trained model bundle for a MONITORID.

    Returns:
        (model, scaler, metadata)

    Raises:
        ModelNotFoundError: If model does not exist in S3
        ModelLoadError: If loading or deserialization fails
    """

    logger.info("Loading model bundle | MONITORID=%s", monitor_id)

    # ------------------------------------------------------------
    # 1️⃣ Validate existence
    # ------------------------------------------------------------
    if not model_exists(monitor_id):
        raise ModelNotFoundError(
            monitor_id=monitor_id,
            path=f"models/{monitor_id}/",
        )

    paths = get_model_paths(monitor_id)

    try:
        # ------------------------------------------------------------
        # 2️⃣ Load binary artifacts
        # ------------------------------------------------------------
        model_bytes = load_binary(paths["model_path"])
        scaler_bytes = load_binary(paths["scaler_path"])

        model = pickle.loads(model_bytes)
        scaler = pickle.loads(scaler_bytes)

        logger.info("Loaded model & scaler | MONITORID=%s", monitor_id)

        # ------------------------------------------------------------
        # 3️⃣ Load metadata
        # ------------------------------------------------------------
        metadata = load_metadata(monitor_id)
        logger.info("Loaded metadata | MONITORID=%s", monitor_id)

    except Exception as exc:
        logger.exception(
            "Failed loading model bundle | MONITORID=%s", monitor_id
        )
        raise ModelLoadError(monitor_id, str(exc)) from exc

    return model, scaler, metadata
