import requests
from app.config import CONFIG
from app.utils.logging_utils import get_logger
from app.utils.exceptions import APICallError

logger = get_logger(__name__)


class DeviceAPIClient:
    def __init__(self):
        if not CONFIG.EXTERNAL_DEVICE_API_BASE_URL:
            raise RuntimeError("EXTERNAL_DEVICE_API_BASE_URL not configured")

        self.base_url = CONFIG.EXTERNAL_DEVICE_API_BASE_URL

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

        monitor_id = payload.get("monitorId")
        if not monitor_id:
            raise RuntimeError(f"No monitorId returned for device {device_id}")

        logger.info(
            "Resolved device → monitor | device=%s monitor=%s",
            device_id,
            monitor_id,
        )

        return int(monitor_id)
