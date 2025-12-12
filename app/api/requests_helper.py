"""
app/api/requests_helper.py

A lightweight wrapper around Python's requests library that provides:

Responsibilities:
    - Standardized GET requests with retries
    - Timeout protection to avoid hanging the pipeline
    - Logging for API failures
    - No project-specific logic (purely a reusable HTTP helper)

Used by:
    - trend_api_client.py for retrieving 3-month sensor history
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------------
# HTTP GET Helper with Retries
# -------------------------------------------------------------------
def http_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
    retries: int = 3,
    backoff: float = 1.0,
) -> Optional[requests.Response]:
    """
    Perform a GET request with retry and backoff.

    Args:
        url: Full API endpoint URL.
        params: Optional query parameters.
        timeout: Timeout for each request attempt.
        retries: Number of retry attempts before failing.
        backoff: Seconds to wait between retries (increases linearly).

    Returns:
        Response object if successful, otherwise None.

    Why this exists:
        - API might be temporarily down
        - Network jitter is common in distributed systems
        - Prevents your pipeline from failing immediately
        - Makes Trend API calls stable

    Logs:
        - Warnings on retry attempts
        - Errors on final failure
    """
    attempt = 1

    while attempt <= retries:
        try:
            resp = requests.get(url, params=params, timeout=timeout)

            if resp.status_code == 200:
                return resp

            logger.warning(
                f"GET {url} returned status={resp.status_code} "
                f"(attempt {attempt}/{retries})"
            )

        except requests.exceptions.Timeout:
            logger.warning(
                f"Timeout when calling {url} "
                f"(attempt {attempt}/{retries})"
            )

        except requests.exceptions.RequestException as exc:
            logger.warning(
                f"Network error when calling {url}: {exc} "
                f"(attempt {attempt}/{retries})"
            )

        # Retry backoff
        if attempt < retries:
            time.sleep(backoff * attempt)

        attempt += 1

    logger.error(f"GET {url} failed after {retries} attempts.")
    return None