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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import CONFIG
from app.api.token_manager import TokenManager
from app.utils.logging_utils import get_logger
from app.utils.exceptions import APICallError

logger = get_logger(__name__)


class TrendAPIClient:
    """
    Production-ready Trend API v2 client.

    Contract:
    - Input  : deviceIdentifier
    - Output : ONE monitor trend (backend-resolved)
    - Used ONLY for training
    """

    def __init__(self) -> None:
        if not CONFIG.TREND_API_BASE_URL:
            raise RuntimeError("TREND_API_BASE_URL is not configured")

        self.base_url = CONFIG.TREND_API_BASE_URL
        self.timeout = 30
        self.token_manager = TokenManager()

        # Session with limited retries (SAFE for training)
        self.session = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("POST",),
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    # ------------------------------------------------------------------
    def get_history(
        self,
        device_identifier: str,
        feature_codes: List[str],
        start_datetime: str,
        end_datetime: str,
        interval_value: int,
        interval_unit: str,
    ) -> List[Dict[str, Any]]:

        token = self.token_manager.get_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
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
            "Fetching trend history (v2) | DEVICEID=%s | window=%s → %s",
            device_identifier,
            start_datetime,
            end_datetime,
        )

        try:
            response = self.session.post(
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
            ) from exc

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
                response.text[:500],
            )

        try:
            payload_json = response.json()
        except Exception as exc:
            raise APICallError(
                self.base_url,
                200,
                f"Invalid JSON response: {exc}",
            ) from exc

        monitor_trend = payload_json.get("data", {}).get("monitorTrend")
        if not monitor_trend:
            raise APICallError(
                self.base_url,
                200,
                "No monitorTrend returned by Trend API v2",
            )

        monitor_id = monitor_trend.get("monitorId")
        readings = monitor_trend.get("readings") or []

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
                "Trend API v2 returned no usable records",
            )

        logger.info(
            "Trend API v2 success | MONITORID=%s | records=%d",
            monitor_id,
            len(normalized),
        )

        return normalized
