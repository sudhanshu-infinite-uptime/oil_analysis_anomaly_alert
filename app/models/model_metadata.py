"""
app/models/model_metadata.py

Handles reading and writing model metadata files stored alongside
each trained Isolation Forest + scaler.

Each monitor gets a directory like:

    models/<MONITORID>/
        iforest_model.pkl
        robust_scaler.pkl
        metadata.json

This module provides:
    - create_metadata()  -> build metadata dict describing training details
    - save_metadata()    -> write metadata.json safely using atomic write
    - load_metadata()    -> read metadata.json for inference

Where this module is used:
    - model_builder.py:
        After training a model, create_metadata() is used to store:
            • training timestamp
            • contamination factor
            • feature order
            • number of samples used
            • API history window
        Then save_metadata() writes metadata.json next to the trained files.

    - model_loader.py:
        During inference, load_metadata() retrieves:
            • training parameters
            • feature order
            • metadata needed for debugging and auditing

    - flink/operators.py:
        When generating anomaly alerts, metadata fields such as model version
        or training timestamp may be included in the alert payload.

This module should NOT:
    - Train models
    - Load Isolation Forest weights
    - Perform inference
    - Call external APIs
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.utils.logging_utils import get_logger
from app.utils.path_utils import atomic_write
from app.utils.exceptions import ModelNotFoundError, ModelLoadError

logger = get_logger(__name__)


# -------------------------------------------------------------------
# Create metadata dictionary
# -------------------------------------------------------------------
def create_metadata(
    monitor_id: str,
    contamination: float,
    feature_list: list,
    n_samples: int,
    api_months: int = 3,
) -> Dict[str, Any]:
    """
    Build a metadata dictionary describing the trained model.

    Args:
        monitor_id: MONITORID used for training.
        contamination: Contamination factor used for Isolation Forest.
        feature_list: Ordered features used for training.
        n_samples: Number of samples used during training.
        api_months: How many months of history were fetched.

    Returns:
        Dictionary of metadata fields for JSON serialization.
    """
    return {
        "monitor_id": monitor_id,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contamination": contamination,
        "feature_order": feature_list,
        "training_samples": n_samples,
        "history_months": api_months,
        "version": "1.0",
     }


# -------------------------------------------------------------------
# Save metadata.json atomically
# -------------------------------------------------------------------
def save_metadata(metadata_path: Path, metadata: Dict[str, Any]) -> None:
    """
    Safely write metadata.json using atomic file write.

    Args:
        metadata_path: Path to metadata.json file.
        metadata: Metadata dictionary.

    Raises:
        ModelLoadError if writing fails.
    """
    try:
        raw = json.dumps(metadata, indent=4).encode("utf-8")
        atomic_write(metadata_path, raw)
        logger.info(f"Saved metadata → {metadata_path}")
    except Exception as exc:
        logger.error(f"Failed to save metadata at {metadata_path}: {exc}")
        raise ModelLoadError(metadata.get("monitor_id", "UNKNOWN"), str(exc))


# Load metadata.json
def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """
    Load model metadata from metadata.json.

    Args:
        metadata_path: Path to metadata.json.

    Returns:
        Parsed metadata dictionary.

    Raises:
        ModelNotFoundError if file missing.
        ModelLoadError if file unreadable or invalid JSON.
    """
    if not metadata_path.exists():
        raise ModelNotFoundError(
            monitor_id="UNKNOWN",
            path=str(metadata_path)
        )

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except json.JSONDecodeError as exc:
        logger.error(f"Corrupted metadata JSON at {metadata_path}: {exc}")
        raise ModelLoadError("UNKNOWN", f"Invalid JSON format: {exc}")

    except Exception as exc:
        logger.error(f"Failed to read metadata from {metadata_path}: {exc}")
        raise ModelLoadError("UNKNOWN", str(exc))
