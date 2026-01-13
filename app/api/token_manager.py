from __future__ import annotations

import time
import requests
from jose import jwt

from app.config import CONFIG
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TokenManager:
    """
    Centralized OAuth token manager.

    - Fetches token from TOKEN_URL
    - Caches token in-memory
    - Auto-refreshes on expiry
    """

    def __init__(self) -> None:
        if not CONFIG.TOKEN_URL:
            raise RuntimeError("TOKEN_URL is not configured")
        if not CONFIG.TOKEN_USERNAME or not CONFIG.TOKEN_PASSWORD:
            raise RuntimeError("TOKEN_USERNAME / TOKEN_PASSWORD not configured")

        self._token: str | None = None
        self._expires_at: int = 0

    def _generate_token(self) -> None:
        logger.info("Generating access token")

        response = requests.post(
            CONFIG.TOKEN_URL,
            json={
                "username": CONFIG.TOKEN_USERNAME,
                "password": CONFIG.TOKEN_PASSWORD,
            },
            timeout=20,
        )
        response.raise_for_status()

        token = response.json()["data"]["accessToken"]

        decoded = jwt.get_unverified_claims(token)
        self._expires_at = int(decoded["exp"])
        self._token = token

        logger.info("Access token generated successfully")

    def get_token(self) -> str:
        now = int(time.time())

        if not self._token or now >= self._expires_at - 30:
            self._generate_token()

        return self._token
