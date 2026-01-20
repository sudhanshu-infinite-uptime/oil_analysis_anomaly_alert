"""
app/models/model_loader.py

S3-backed model loader.

Responsibilities:
- Validate model bundle existence (atomic)
- Load model, scaler, and metadata from S3
- Deserialize artifacts safely
- Fail fast with explicit domain errors

This module does NOT:
- Train models
- Retry failed loads
- Cache models
- Perform inference
"""

from __future__ import annotations

import pickle
from typing import Tuple, Any, Dict

from app.models.model_store import (
    model_exists,
    get_model_paths,
    load_binary,
    load_metadata,
)
from app.utils.logging_utils import get_logger
from app.utils.exceptions import ModelNotFoundError, ModelLoadError

logger = get_logger(__name__)


def load_model_bundle(
    monitor_id: int,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load a fully trained model bundle for a monitor.

    Returns:
        (model, scaler, metadata)

    Raises:
        ModelNotFoundError:
            - No SUCCESS marker in S3
        ModelLoadError:
            - Corrupt bundle
            - Deserialization failure
            - Partial / inconsistent state
    """

    logger.info("Loading model bundle | MONITORID=%s", monitor_id)

    # ------------------------------------------------------------
    # 1️⃣ Atomic existence check (_SUCCESS marker)
    # ------------------------------------------------------------
    if not model_exists(monitor_id):
        raise ModelNotFoundError(
            monitor_id=monitor_id,
            path="oil-analysis-anomaly-alerts/<monitor_id>/",
        )

    paths = get_model_paths(monitor_id)

    # ------------------------------------------------------------
    # 2️⃣ Load & deserialize artifacts
    # ------------------------------------------------------------
    try:
        model_bytes = load_binary(paths["model_path"])
        scaler_bytes = load_binary(paths["scaler_path"])

        model = pickle.loads(model_bytes)
        scaler = pickle.loads(scaler_bytes)

        metadata = load_metadata(monitor_id)

    except Exception as exc:
        logger.exception(
            "Model bundle load failed | MONITORID=%s",
            monitor_id,
        )
        raise ModelLoadError(
            monitor_id=monitor_id,
            message=str(exc),
        ) from exc

    # ------------------------------------------------------------
    # 3️⃣ Metadata validation (hard guard)
    # ------------------------------------------------------------
    if metadata.get("training_status") != "SUCCESS":
        raise ModelLoadError(
            monitor_id=monitor_id,
            message="Model bundle exists but training_status != SUCCESS",
        )

    if "feature_names" not in metadata:
        raise ModelLoadError(
            monitor_id=monitor_id,
            message="Invalid metadata: missing feature_names",
        )

    logger.info(
        "Model bundle loaded successfully | MONITORID=%s",
        monitor_id,
    )

    return model, scaler, metadata
