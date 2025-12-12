"""
app/api/trend_api_client.py

Client for calling the Trend API to retrieve historical sensor data required
for model training.

Responsibilities:
- Construct the correct Trend API endpoint
- Call API using the safe http_get() helper
- Validate the response structure
- Return parsed historical records to the caller (model_builder)

This module should NOT:
- Perform training
- Build models
- Apply preprocessing
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.api.requests_helper import http_get
from app.utils.logging_utils import get_logger
from app.utils.exceptions import APICallError
from app.config import CONFIG

logger = get_logger(__name__)


class TrendAPIClient:
    """
    Lightweight client wrapper around the Trend API.

    The API is expected to expose an endpoint:

        GET /history?monitorId=<id>&months=3

    Expected response:
    {
        "success": true,
        "records": [ ...list of PROCESS_PARAMETER dicts... ]
    }
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or CONFIG.TREND_API_BASE_URL

    # ---------------------------------------------------------------
    # Public method: fetch last N months of history for a monitor
    # ---------------------------------------------------------------
    def get_history(self, monitor_id: str, months: int = 3) -> List[Dict[str, Any]]:
        """
        Fetch last <months> months of historical data for a monitor.

        Args:
            monitor_id: String ID (e.g., "MONITOR123").
            months: How far back to fetch.

        Returns:
            A list of JSON records (raw sensor messages).

        Raises:
            APICallError if the API returns invalid or missing content.
        """
        url = f"{self.base_url}/history"
        params = {
            "monitorId": monitor_id,
            "months": months,
        }

        logger.info(
            f"Requesting historical data for MONITORID={monitor_id} "
            f"({months} months)"
        )

        # ---- API CALL -------------------------------------------------------
        resp = http_get(url, params=params)

        if resp is None:
            raise APICallError(url, -1, "No response received from Trend API")

        if resp.status_code != 200:
            raise APICallError(
                url, resp.status_code,
                f"Non-200 response from Trend API"
            )

        # ---- JSON PARSING ---------------------------------------------------
        try:
            data = resp.json()
        except Exception as exc:
            raise APICallError(url, resp.status_code, f"Invalid JSON: {exc}")

        # ---- VALIDATION -----------------------------------------------------
        if not isinstance(data, dict):
            raise APICallError(url, resp.status_code, "Response must be a JSON object")

        if "records" not in data:
            raise APICallError(url, resp.status_code, "Missing 'records' field in response")

        records = data.get("records", [])
        if not isinstance(records, list):
            raise APICallError(url, resp.status_code, "'records' must be a list")

        logger.info(
            f"Trend API returned {len(records)} historical entries "
            f"for MONITORID={monitor_id}"
        )

        return records
