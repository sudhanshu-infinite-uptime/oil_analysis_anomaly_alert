"""
app/utils/json_utils.py

Provides safe JSON utilities for the Oil Anomaly Detection Pipeline.

Responsibilities:
    - Safely parse incoming Kafka message payloads
    - Avoid crashes caused by malformed or partial JSON
    - Log parse failures consistently
    - Provide optional helpers for JSON serialization

Used by:
    - Flink operators
    - API clients (if needed)
    - Model builder
"""

from __future__ import annotations

import json
from typing import Any, Optional, Union

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def safe_json_parse(value: Union[str, bytes, bytearray]) -> Optional[Any]:
    """
    Safely parse a JSON string or binary payload.

    Args:
        value: Raw JSON as string, bytes, or bytearray.

    Returns:
        Parsed Python object (dict/list/etc.) or None if parsing fails.

    Why this exists:
        - Kafka may send corrupted messages
        - Some producers send bytes with encoding issues
        - We don't want Flink operators to crash on malformed JSON
    """
    if value is None:
        logger.warning("Attempted to parse JSON but value is None.")
        return None

    # Convert bytes → string
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception as exc:
            logger.error(f"Failed to decode bytes as UTF-8: {exc}")
            return None

    # Parse JSON
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            logger.error(f"JSON parsing error: {exc} | value={value[:200]!r}")
            return None
        except Exception as exc:
            logger.error(f"Unexpected error parsing JSON: {exc}")
            return None

    logger.error(
        f"safe_json_parse expected str/bytes, but got {type(value).__name__}"
    )
    return None


def safe_json_dumps(data: Any, pretty: bool = False) -> Optional[str]:
    """
    Safely serialize a Python object to JSON string.

    Args:
        data: Any Python object.
        pretty: If True, output is indented & human readable.

    Returns:
        A JSON string or None if serialization fails.

    Notes:
        - Useful when sending alerts to Kafka
        - Ensures serialization errors do not crash the pipeline
    """
    try:
        if pretty:
            return json.dumps(data, indent=4, sort_keys=True)
        return json.dumps(data)
    except Exception as exc:
        logger.error(f"Failed to serialize object to JSON: {exc}")
        return None
