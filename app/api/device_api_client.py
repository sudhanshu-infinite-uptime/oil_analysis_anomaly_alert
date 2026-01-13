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

    # ------------------------------------------------------------------
    # EXISTING METHOD (DO NOT BREAK)
    # ------------------------------------------------------------------
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
            "Resolved device → monitor | DEVICEID=%s | MONITORID=%s",
            device_id,
            monitor_id,
        )

        return int(monitor_id)

    # ------------------------------------------------------------------
    # NEW METHOD (USED BY FLINK OPERATOR)
    # ------------------------------------------------------------------
    def get_monitor_and_pg(self, device_id: str):
        """
        Resolve DEVICEID → (MONITORID, PARAMETER_GROUP_ID)

        This method is safe for streaming usage.
        """

        from app.api.token_manager import TokenManager

        token = TokenManager().get_token()

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
        parameter_group_id = payload.get("parameterGroupId")

        if not monitor_id or not parameter_group_id:
            raise RuntimeError(
                f"Incomplete device mapping | DEVICEID={device_id} | response={payload}"
            )

        logger.info(
            "Resolved device → monitor + pg | DEVICEID=%s | MONITORID=%s | PGID=%s",
            device_id,
            monitor_id,
            parameter_group_id,
        )

        return int(monitor_id), int(parameter_group_id)
