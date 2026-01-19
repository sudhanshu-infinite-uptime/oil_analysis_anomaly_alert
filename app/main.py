"""
app/main.py

Top-level entry point for the Oil Anomaly Detection Pipeline.

Responsibilities:
    • Initialize logging
    • Perform lightweight startup sanity checks
    • Launch the Flink streaming job

Design principles:
    • NO model bootstrap at startup
    • NO external API calls
    • NO device / monitor resolution
    • Fully event-driven (Kafka → Flink Operator → On-demand Training → Inference)

All model training and inference happen INSIDE the Flink operator.
"""

from __future__ import annotations

from app.flink.flink_job import run_flink_job
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _startup_sanity_check() -> None:
    """
    Lightweight sanity checks to ensure critical symbols exist
    before starting the Flink job.

    This function intentionally:
    - DOES NOT call external APIs
    - DOES NOT train models
    - DOES NOT touch Kafka
    - DOES NOT access S3

    Purpose:
    - Fail fast if imports or wiring are broken
    """

    from app.api.device_api_client import DeviceAPIClient
    from app.models.model_builder import build_model_for_device_v2
    from app.predictor.anomaly_detector import detect_anomalies
    from app.windows.sliding_window import SlidingWindow

    assert callable(run_flink_job), "run_flink_job is not callable"
    assert callable(build_model_for_device_v2), "build_model_for_device_v2 is not callable"
    assert callable(detect_anomalies), "detect_anomalies is not callable"

    client = DeviceAPIClient()
    assert hasattr(client, "get_monitor_id_runtime"), (
        "DeviceAPIClient.get_monitor_id_runtime missing"
    )

    assert SlidingWindow is not None, "SlidingWindow class missing"

    logger.info("Startup sanity check passed (event-driven mode).")


def main() -> None:
    """
    Application entry point.

    Flow:
        1. Run startup sanity checks
        2. Submit Flink streaming job
        3. Kafka drives everything else

    Notes:
        - Models are trained ON-DEMAND per monitorId
        - No assumptions about pre-existing models
        - Safe for distributed Flink execution
    """

    logger.info("Starting Oil Anomaly Detection Service...")
    _startup_sanity_check()
    run_flink_job()


if __name__ == "__main__":
    main()
