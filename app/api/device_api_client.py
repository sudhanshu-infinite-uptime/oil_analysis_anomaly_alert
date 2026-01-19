import requests

from app.config import CONFIG
from app.utils.logging_utils import get_logger
from app.utils.exceptions import APICallError
from app.api.token_manager import TokenManager

logger = get_logger(__name__)


class DeviceAPIClient:
    def __init__(self):
        if not CONFIG.EXTERNAL_DEVICE_API_BASE_URL:
            raise RuntimeError("EXTERNAL_DEVICE_API_BASE_URL not configured")

        self.base_url = CONFIG.EXTERNAL_DEVICE_API_BASE_URL
        self.token_manager = TokenManager()

    # --------------------------------------------------
    # Legacy method (do not break)
    # --------------------------------------------------
    def get_monitor_id(self, device_id: str, token: str) -> int:
        url = f"{self.base_url}/external-devices/{device_id}/parameters"

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code != 200:
            raise APICallError(url, response.status_code, response.text)

        payload = response.json()
        monitor_id = (payload.get("data") or {}).get("monitorId")

        if not monitor_id:
            raise RuntimeError(
                f"Missing monitorId | DEVICEID={device_id} | response={payload}"
            )

        logger.info(
            "Resolved device → monitor | DEVICEID=%s | MONITORID=%s",
            device_id,
            monitor_id,
        )
        return int(monitor_id)

    # --------------------------------------------------
    # Runtime-safe method (USED BY FLINK)
    # --------------------------------------------------
    def get_monitor_id_runtime(self, device_id: str) -> int:
        token = self.token_manager.get_token()

        url = f"{self.base_url}/external-devices/{device_id}/parameters"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code != 200:
            raise APICallError(url, response.status_code, response.text)

        payload = response.json()
        monitor_id = (payload.get("data") or {}).get("monitorId")

        if not monitor_id:
            raise RuntimeError(
                f"Missing monitorId | DEVICEID={device_id} | response={payload}"
            )

        logger.info(
            "Resolved device → monitor | DEVICEID=%s | MONITORID=%s",
            device_id,
            monitor_id,
        )
        return int(monitor_id)
