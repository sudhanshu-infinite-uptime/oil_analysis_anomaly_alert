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

SUCCESS_MARKER = "_SUCCESS"


def _s3_key(monitor_id: str, filename: str) -> str:
    return f"oil-analysis-anomaly-alerts/{monitor_id}/{filename}"


def save_binary(virtual_path: str, data: bytes) -> None:
    monitor_id = Path(virtual_path).parent.name
    filename = Path(virtual_path).name
    key = _s3_key(monitor_id, filename)

    try:
        S3_CLIENT.put_object(
            Bucket=CONFIG.S3_BUCKET_NAME,
            Key=key,
            Body=data,
        )
    except (ClientError, BotoCoreError) as exc:
        logger.error("Failed to save artifact | key=%s | %s", key, exc)
        raise


def load_binary(virtual_path: str) -> bytes:
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
        logger.error("Failed to load artifact | key=%s | %s", key, exc)
        raise


def save_metadata(monitor_id: str, metadata: Dict) -> None:
    key = _s3_key(monitor_id, "metadata.json")

    try:
        S3_CLIENT.put_object(
            Bucket=CONFIG.S3_BUCKET_NAME,
            Key=key,
            Body=json.dumps(metadata).encode("utf-8"),
            ContentType="application/json",
        )
    except (ClientError, BotoCoreError) as exc:
        logger.error("Failed to save metadata | key=%s | %s", key, exc)
        raise


def load_metadata(monitor_id: str) -> Dict:
    key = _s3_key(monitor_id, "metadata.json")

    try:
        obj = S3_CLIENT.get_object(
            Bucket=CONFIG.S3_BUCKET_NAME,
            Key=key,
        )
        return json.loads(obj["Body"].read().decode("utf-8"))
    except (ClientError, BotoCoreError, json.JSONDecodeError) as exc:
        logger.error("Failed to load metadata | key=%s | %s", key, exc)
        raise


def mark_model_success(monitor_id: str) -> None:
    key = _s3_key(monitor_id, SUCCESS_MARKER)

    try:
        S3_CLIENT.put_object(
            Bucket=CONFIG.S3_BUCKET_NAME,
            Key=key,
            Body=b"",
        )
    except (ClientError, BotoCoreError) as exc:
        logger.error("Failed to write SUCCESS marker | %s", exc)
        raise


def model_exists(monitor_id: str) -> bool:
    key = _s3_key(monitor_id, SUCCESS_MARKER)

    try:
        S3_CLIENT.head_object(
            Bucket=CONFIG.S3_BUCKET_NAME,
            Key=key,
        )
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
            return False
        raise


def get_model_paths(monitor_id: str) -> Dict[str, str]:
    return {
        "model_path": f"{monitor_id}/model.pkl",
        "scaler_path": f"{monitor_id}/scaler.pkl",
        "metadata_path": f"{monitor_id}/metadata.json",
    }
