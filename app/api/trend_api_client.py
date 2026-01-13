"""
app/api/trend_api_client.py

Production-ready HTTP client for fetching historical trend data
used during model training.

IMPORTANT:
- externalDeviceParameterGroupIds is REQUIRED
- monitorId is returned by the API, not supplied
- ALL monitorIds returned by the API are processed
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


# -------------------------------------------------------------------
# Trend API Client
# -------------------------------------------------------------------
class TrendAPIClient:
    """
    Client wrapper for Infinite Uptime Trend History API.

    Input  : externalDeviceParameterGroupId
    Output : multiple monitorIds (derived from API)
    """

    def __init__(self) -> None:
        if not CONFIG.TREND_API_BASE_URL:
            raise RuntimeError("TREND_API_BASE_URL is not configured")

        self.base_url: str = CONFIG.TREND_API_BASE_URL
        self.timeout: int = 30
        self.token_manager = TokenManager()

    def get_history(
        self,
        parameter_group_id: int,
        start_datetime: str,
        end_datetime: str,
        interval_value: int = 6,
        interval_unit: str = "hour",
    ) -> List[Dict[str, Any]]:

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
            "externalDeviceParameterGroupIds": [int(parameter_group_id)],
        }

        logger.info(
            "Fetching trend history | parameter_group_id=%s",
            parameter_group_id,
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

        monitors = payload_json.get("data", {}).get("monitors", [])
        if not monitors:
            raise APICallError(
                self.base_url,
                200,
                "No monitors returned by Trend API",
            )

        normalized: List[Dict[str, Any]] = []

        for monitor in monitors:
            monitor_id = monitor.get("monitorId")
            readings = monitor.get("readings", [])

            if not readings:
                continue

            for r in readings:
                try:
                    raw = json.loads(r["jsonavg"])
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
            "Trend API success | monitors=%d | records=%d",
            len({r["MONITORID"] for r in normalized}),
            len(normalized),
        )

        return normalized
