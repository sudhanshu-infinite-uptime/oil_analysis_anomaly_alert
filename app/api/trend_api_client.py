"""
app/api/trend_api_client.py

Production-ready client for Trend API v2.

Key behavior (FINAL CONTRACT):
- Input  : deviceIdentifier (from live data)
- Output : ONE monitor trend (already resolved by backend)
- No dependency on External Device API
- No parameterGroupId required
"""

from __future__ import annotations

import json
from typing import List, Dict, Any

import requests

from app.config import CONFIG
from app.api.token_manager import TokenManager
from app.utils.logging_utils import get_logger
from app.utils.exceptions import APICallError

logger = get_logger(__name__)


class TrendAPIClient:
    """
    Trend API v2 client.

    Backend responsibilities:
    - Resolve deviceIdentifier → monitorId
    - Return single monitor trend data

    This client:
    - Sends deviceIdentifier
    - Normalizes response into internal format
    """

    def __init__(self) -> None:
        if not CONFIG.TREND_API_BASE_URL:
            raise RuntimeError("TREND_API_BASE_URL is not configured")

        self.base_url: str = CONFIG.TREND_API_BASE_URL
        self.timeout: int = 30
        self.token_manager = TokenManager()

    def get_history(
        self,
        device_identifier: str,
        feature_codes: List[str],
        start_datetime: str,
        end_datetime: str,
        interval_value: int,
        interval_unit: str,
    ) -> List[Dict[str, Any]]:
        """
        Fetch trend history using deviceIdentifier (Trend API v2).

        Returns normalized records:
        {
            "MONITORID": int,
            "PROCESS_PARAMETER": {...},
            "timestamp": str
        }
        """

        token = self.token_manager.get_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }

        payload = {
            "startDateTime": start_datetime,
            "endDateTime": end_datetime,
            "intervalValue": interval_value,
            "intervalUnit": interval_unit,
            "deviceIdentifier": device_identifier,
            "featureCodes": feature_codes,
        }

        logger.info(
            "Fetching trend history | DEVICEID=%s",
            device_identifier,
        )

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
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
                "Unauthorized: token invalid or expired",
            )

        if response.status_code != 200:
            raise APICallError(
                self.base_url,
                response.status_code,
                response.text,
            )

        try:
            payload_json = response.json()
        except Exception as exc:
            raise APICallError(
                self.base_url,
                response.status_code,
                f"Invalid JSON response: {exc}",
            )

        monitor_trend = (
            payload_json
            .get("data", {})
            .get("monitorTrend")
        )

        if not monitor_trend:
            raise APICallError(
                self.base_url,
                200,
                "No monitorTrend returned by Trend API v2",
            )

        monitor_id = monitor_trend.get("monitorId")
        readings = monitor_trend.get("readings", [])

        if not monitor_id or not readings:
            raise APICallError(
                self.base_url,
                200,
                "Missing monitorId or readings in Trend API v2 response",
            )

        normalized: List[Dict[str, Any]] = []

        for r in readings:
            try:
                raw = json.loads(r.get("jsonavg", "{}"))
            except Exception:
                continue

            normalized.append(
                {
                    "MONITORID": monitor_id,
                    "PROCESS_PARAMETER": {
                        k: float(v) if v not in (None, "", "null") else None
                        for k, v in raw.items()
                    },
                    "timestamp": r.get("time"),
                }
            )

        if not normalized:
            raise APICallError(
                self.base_url,
                200,
                "Trend data parsing produced no usable records",
            )

        logger.info(
            "Trend API v2 success | MONITORID=%s | records=%d",
            monitor_id,
            len(normalized),
        )

        return normalized
