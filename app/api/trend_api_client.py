"""
app/api/trend_api_client.py

Production-ready HTTP client for fetching historical trend data
used during model training.

Responsibilities:
- Call Infinite Uptime Trend History API
- Use DevOps-provided authentication token
- Fetch fixed historical time range (as per business requirement)
- Validate HTTP and JSON responses
- Return complete historical records

This module does NOT:
- Train models
- Perform feature engineering
- Persist data
- Handle Flink or Kafka logic
"""

from __future__ import annotations

from typing import List, Dict, Any

import requests

from app.config import CONFIG
from app.utils.exceptions import APICallError
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrendAPIClient:
    """
    Client wrapper for Infinite Uptime Trend History API.
    """

    def __init__(self) -> None:
        if not CONFIG.TREND_API_BASE_URL:
            raise RuntimeError("TREND_API_BASE_URL is not configured")

        if not CONFIG.TREND_API_TOKEN:
            raise RuntimeError("TREND_API_TOKEN is not configured")

        self.base_url: str = CONFIG.TREND_API_BASE_URL
        self.token: str = CONFIG.TREND_API_TOKEN
        self.timeout: int = 30

    def get_history(
        self,
        monitor_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical trend data for a given MONITORID
        using a fixed date range.

        Args:
            monitor_id: External device / parameter group ID

        Returns:
            List of historical records

        Raises:
            APICallError on failure
        """

        payload = {
            "startDateTime": "2025-11-29T10:05:00.000Z",
            "endDateTime": "2025-12-29T10:05:00.000Z",
            "intervalValue": 6,
            "intervalUnit": "hour",
            "externalDeviceParameterGroupIds": [int(monitor_id)],
        }

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        logger.info(
            "Fetching FIXED trend history | MONITORID=%s",
            monitor_id,
        )

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise APICallError(
                self.base_url,
                -1,
                f"HTTP request failed: {exc}",
            )

        if response.status_code == 401:
            raise APICallError(
                self.base_url,
                401,
                "Unauthorized: invalid or expired TREND_API_TOKEN",
            )

        if response.status_code != 200:
            raise APICallError(
                self.base_url,
                response.status_code,
                response.text,
            )

        try:
            data = response.json()
        except Exception as exc:
            raise APICallError(
                self.base_url,
                response.status_code,
                f"Invalid JSON response: {exc}",
            )

        records = data.get("records") or data.get("data")

        if not isinstance(records, list) or not records:
            raise APICallError(
                self.base_url,
                200,
                "No historical records returned by Trend API",
            )

        logger.info(
            "Trend API success | MONITORID=%s | records=%s",
            monitor_id,
            len(records),
        )

        return records
