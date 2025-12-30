"""
app/api/trend_api_client.py

Production-ready HTTP client for fetching historical trend data
used during model training.

Responsibilities:
- Call Infinite Uptime Trend History API
- Handle authentication and request formatting
- Validate HTTP and JSON responses
- Return normalized historical records

Design principles:
- Fail fast on misconfiguration
- Never return partial / invalid data
- Raise domain-specific APICallError only

This module does NOT:
- Train models
- Perform feature engineering
- Persist data
- Handle Flink or Kafka logic
"""

from __future__ import annotations

import requests
from typing import List, Dict, Any
from datetime import datetime, timedelta

from app.config import CONFIG
from app.utils.logging_utils import get_logger
from app.utils.exceptions import APICallError

logger = get_logger(__name__)


class TrendAPIClient:
    """
    Client wrapper for Infinite Uptime Trend History API.
    """

    def __init__(self) -> None:
        """
        Initialize the Trend API client.

        Raises:
            RuntimeError: If mandatory configuration is missing
        """
        if not CONFIG.TREND_API_BASE_URL:
            raise RuntimeError("TREND_API_BASE_URL is not configured")

        if not CONFIG.TREND_API_TOKEN:
            raise RuntimeError("TREND_API_TOKEN is not configured")

        self.base_url: str = CONFIG.TREND_API_BASE_URL
        self.timeout: int = 30

    def get_history(
        self,
        monitor_id: str,
        months: int,
        access_token: str,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical trend data for a given MONITORID.

        Args:
            monitor_id: External device / parameter group ID
            months: Number of months of history to fetch
            access_token: Bearer token for API authentication

        Returns:
            List of historical records from the API

        Raises:
            APICallError: On any HTTP, network, or data validation failure
        """

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30 * months)

        payload = {
            "startDateTime": start_time.isoformat() + "Z",
            "endDateTime": end_time.isoformat() + "Z",
            "intervalValue": 6,
            "intervalUnit": "hour",
            "externalDeviceParameterGroupIds": [int(monitor_id)],
        }

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        logger.info(
            "Fetching trend history | MONITORID=%s | months=%s",
            monitor_id,
            months,
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
                "Unauthorized: invalid or expired token",
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

        if not isinstance(data, dict):
            raise APICallError(
                self.base_url,
                200,
                "Response payload must be a JSON object",
            )

        records = data.get("records") or data.get("data")

        if not isinstance(records, list) or not records:
            raise APICallError(
                self.base_url,
                200,
                "No historical records returned",
            )

        logger.info(
            "Trend API success | MONITORID=%s | records=%s",
            monitor_id,
            len(records),
        )

        return records
