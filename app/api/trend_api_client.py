"""
app/api/trend_api_client.py

Production-ready HTTP client for fetching historical trend data
used during model training.

Key features:
- Automatic JWT token refresh (session-safe for Flink jobs)
- Uses DevOps-provided username/password credentials
- Fetches historical trend data from Infinite Uptime API
- Strict validation of HTTP and response payloads

This module DOES:
- Handle authentication lifecycle
- Call Trend History API
- Return normalized historical readings (jsonavg)

This module DOES NOT:
- Train models
- Perform feature engineering
- Persist data
- Interact with Kafka or Flink APIs directly
"""

from __future__ import annotations

import json
import time
from typing import List, Dict, Any

import requests
from jose import jwt

from app.config import CONFIG
from app.utils.logging_utils import get_logger
from app.utils.exceptions import APICallError

logger = get_logger(__name__)


# -------------------------------------------------------------------
# Token Manager (session-safe)
# -------------------------------------------------------------------
class TokenManager:
    """
    Handles access-token generation and refresh using username/password.
    """

    def __init__(self) -> None:
        if not CONFIG.TOKEN_URL:
            raise RuntimeError("TOKEN_URL is not configured")
        if not CONFIG.TOKEN_USERNAME or not CONFIG.TOKEN_PASSWORD:
            raise RuntimeError("TOKEN_USERNAME / TOKEN_PASSWORD not configured")

        self.token_url = CONFIG.TOKEN_URL
        self.username = CONFIG.TOKEN_USERNAME
        self.password = CONFIG.TOKEN_PASSWORD

        self.access_token: str | None = None
        self.expiry: int = 0

    def _generate_token(self) -> None:
        logger.info("Generating new Trend API access token")

        response = requests.post(
            self.token_url,
            json={
                "username": self.username,
                "password": self.password,
            },
            headers={"Content-Type": "application/json"},
            timeout=20,
        )
        response.raise_for_status()

        token_response = response.json()

        try:
            token = token_response["data"]["accessToken"]
        except KeyError:
            raise RuntimeError(
                f"Unexpected token response format: {token_response}"
            )

        decoded = jwt.get_unverified_claims(token)

        self.access_token = token
        self.expiry = decoded["exp"]

        logger.info("Trend API token generated successfully")

    def get_token(self) -> str:
        now = int(time.time())

        # Refresh token 60s before expiry
        if not self.access_token or now >= self.expiry - 60:
            self._generate_token()

        return self.access_token


# -------------------------------------------------------------------
# Trend API Client
# -------------------------------------------------------------------
class TrendAPIClient:
    """
    Client wrapper for Infinite Uptime Trend History API.
    """

    def __init__(self) -> None:
        if not CONFIG.TREND_API_BASE_URL:
            raise RuntimeError("TREND_API_BASE_URL is not configured")

        self.base_url: str = CONFIG.TREND_API_BASE_URL
        self.timeout: int = 30
        self.token_manager = TokenManager()

    def get_history(self, monitor_id: str) -> List[Dict[str, Any]]:
        """
        Fetch historical trend data for a given MONITORID
        using the fixed business-approved time range.

        Args:
            monitor_id: External device / parameter group ID

        Returns:
            List of normalized historical records (jsonavg per timestamp)

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

        token = self.token_manager.get_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }

        logger.info("Fetching trend history | MONITORID=%s", monitor_id)

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
                "Unauthorized: token refresh failed",
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

        readings = monitors[0].get("readings", [])
        if not readings:
            raise APICallError(
                self.base_url,
                200,
                "No readings returned by Trend API",
            )

        normalized: List[Dict[str, Any]] = []

        for r in readings:
            try:
                values = json.loads(r["jsonavg"])
            except Exception:
                continue

            values["timestamp"] = r.get("time")
            values["monitor_id"] = r.get("monitorId")

            normalized.append(values)

        if not normalized:
            raise APICallError(
                self.base_url,
                200,
                "Trend data parsing produced no usable records",
            )

        logger.info(
            "Trend API success | MONITORID=%s | records=%s",
            monitor_id,
            len(normalized),
        )

        return normalized
