"""
app/models/model_store.py

S3-backed model artifact store.

Responsibilities:
- Persist and retrieve trained models, scalers, and metadata
- Provide existence checks for trained models
- Abstract S3 key structure from calling code

Design principles:
- No local filesystem usage
- Safe for Flink distributed execution
- Explicit error handling and logging

This module does NOT:
- Train models
- Load models into memory for inference
- Perform anomaly detection
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config

from app.config import CONFIG

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# S3 client (with retries + timeouts)
# ------------------------------------------------------------
if not CONFIG.S3_BUCKET_NAME:
    raise RuntimeError("S3_BUCKET_NAME is not configured")

S3_CLIENT = boto3.client(
    "s3",
    config=Config(
        retries={"max_attempts": 5, "mode": "standard"},
        connect_timeout=5,
        read_timeout=30,
    ),
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _s3_key(monitor_id: str, filename: str) -> str:
    """
    Final S3 key structure (APPROVED BY DEVOPS):

    s3://<bucket>/oil-analysis-anomaly-alerts/<MONITORID>/<filename>
    """
    return f"oil-analysis-anomaly-alerts/{monitor_id}/{filename}"


# ------------------------------------------------------------
# Save / Load binary artifacts
# ------------------------------------------------------------
def save_binary(virtual_path: str, data: bytes) -> None:
    """
    Save a binary artifact (model or scaler) to S3.

    Args:
        virtual_path: "<monitor_id>/<filename>"
        data: Serialized binary data
    """
    monitor_id = Path(virtual_path).parent.name
    filename = Path(virtual_path).name
    key = _s3_key(monitor_id, filename)

    try:
        S3_CLIENT.put_object(
            Bucket=CONFIG.S3_BUCKET_NAME,
            Key=key,
            Body=data,
        )
        logger.info("Saved artifact to S3 | key=%s", key)
    except (ClientError, BotoCoreError) as exc:
        logger.error("Failed to save artifact | key=%s | error=%s", key, exc)
        raise


def load_binary(virtual_path: str) -> bytes:
    """
    Load a binary artifact from S3.
    """
    monitor_id = Path(virtual_path).parent.name
    filename = Path(virtual_path).name
    key = _s3_key(monitor_id, filename)

    try:
        obj = S3_CLIENT.get_object(
            Bucket=CONFIG.S3_BUCKET_NAME,
            Key=key,
        )
        return obj["Body"].read()
    except (ClientError, BotoCoreError) as exc:
        logger.error("Failed to load artifact | key=%s | error=%s", key, exc)
        raise


# ------------------------------------------------------------
# Metadata handling
# ------------------------------------------------------------
def save_metadata(monitor_id: str, metadata: Dict) -> None:
    """
    Save model metadata JSON to S3.
    """
    key = _s3_key(monitor_id, "metadata.json")

    try:
        S3_CLIENT.put_object(
            Bucket=CONFIG.S3_BUCKET_NAME,
            Key=key,
            Body=json.dumps(metadata).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("Saved metadata | key=%s", key)
    except (ClientError, BotoCoreError) as exc:
        logger.error("Failed to save metadata | key=%s | error=%s", key, exc)
        raise


def load_metadata(monitor_id: str) -> Dict:
    """
    Load model metadata JSON from S3.
    """
    key = _s3_key(monitor_id, "metadata.json")

    try:
        obj = S3_CLIENT.get_object(
            Bucket=CONFIG.S3_BUCKET_NAME,
            Key=key,
        )
        return json.loads(obj["Body"].read().decode("utf-8"))
    except (ClientError, BotoCoreError, json.JSONDecodeError) as exc:
        logger.error("Failed to load metadata | key=%s | error=%s", key, exc)
        raise


# ------------------------------------------------------------
# Existence checks
# ------------------------------------------------------------
def model_exists(monitor_id: str) -> bool:
    """
    Check if trained model exists for a monitor.
    """
    key = _s3_key(monitor_id, "model.pkl")

    try:
        S3_CLIENT.head_object(
            Bucket=CONFIG.S3_BUCKET_NAME,
            Key=key,
        )
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
            return False
        logger.error("S3 head_object failed | key=%s | error=%s", key, exc)
        raise


# ------------------------------------------------------------
# Virtual paths (used by Flink operators)
# ------------------------------------------------------------
def get_model_paths(monitor_id: str) -> Dict[str, str]:
    """
    Return virtual model paths used by Flink logic.
    """
    return {
        "model_path": f"{monitor_id}/model.pkl",
        "scaler_path": f"{monitor_id}/scaler.pkl",
        "metadata_path": f"{monitor_id}/metadata.json",
    }
