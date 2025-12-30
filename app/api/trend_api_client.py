"""
app/api/trend_api_client.py

Production-ready client for fetching historical trend data required
for model training.

Responsibilities:
- Call Infinite Uptime Trend History API
- Handle authentication
- Validate response schema
- Return normalized historical records

This module does NOT:
- Train models
- Perform feature engineering
- Persist data
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

    def __init__(self):
        self.base_url = CONFIG.TREND_API_BASE_URL
        self.timeout = 30

    def get_history(
        self,
        monitor_id: str,
        months: int,
        access_token: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical trend data for a given MONITORID.

        Args:
            monitor_id: External monitor / parameter group ID
            months: Number of months of history to fetch
            access_token: Bearer token for API auth

        Returns:
            List of historical records (raw API payload)

        Raises:
            APICallError on any failure
        """

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30 * months)

        payload = {
            "startDateTime": start_time.isoformat() + "Z",
            "endDateTime": end_time.isoformat() + "Z",
            "intervalValue": 1,
            "intervalUnit": "hour",
            "externalDeviceParameterGroupIds": [int(monitor_id)],
        }

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        logger.info(
            f"Fetching historical data | MONITORID={monitor_id} | months={months}"
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
                self.base_url, -1, f"Request failed: {exc}"
            )

        if response.status_code == 401:
            raise APICallError(self.base_url, 401, "Invalid or expired token")

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
                self.base_url, response.status_code, f"Invalid JSON: {exc}"
            )

        if not isinstance(data, dict):
            raise APICallError(self.base_url, 200, "Response must be JSON object")

        records = data.get("records") or data.get("data")
        if not isinstance(records, list) or not records:
            raise APICallError(
                self.base_url, 200, "No historical records returned"
            )

        logger.info(
            f"Trend API returned {len(records)} records for MONITORID={monitor_id}"
        )

        return records
