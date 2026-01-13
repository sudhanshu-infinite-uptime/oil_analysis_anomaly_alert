"""
app/main.py

Top-level entry point for the Oil Anomaly Detection Pipeline.

Responsibilities:
    • Initialize global logging
    • Perform startup sanity checks
    • Start the Flink streaming pipeline

Design:
    • NO model bootstrap
    • NO external API calls
    • Fully event-driven (Kafka → Device → Model → Inference)
"""

from __future__ import annotations

from app.flink.flink_job import run_flink_job
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _startup_sanity_check() -> None:
    """
    Lightweight sanity checks to ensure critical symbols exist
    before starting the Flink job.

    This does NOT:
    - Call external APIs
    - Train models
    - Touch Kafka
    """

    from app.api.device_api_client import DeviceAPIClient
    from app.models.model_builder import build_model_for_monitor
    from app.predictor.anomaly_detector import detect_anomalies

    assert callable(run_flink_job), "run_flink_job is not callable"
    assert callable(build_model_for_monitor), "build_model_for_monitor is not callable"
    assert callable(detect_anomalies), "detect_anomalies is not callable"
    assert hasattr(DeviceAPIClient(), "get_monitor_and_pg"), (
        "DeviceAPIClient.get_monitor_and_pg missing"
    )

    logger.info("Startup sanity check passed.")


def main() -> None:
    """
    Application entry point.

    Flow:
    - Run startup sanity checks
    - Start Flink streaming job
    - Models are trained ON-DEMAND inside the operator
    """

    logger.info("🚀 Starting Oil Anomaly Detection Service...")
    _startup_sanity_check()
    run_flink_job()


if __name__ == "__main__":
    main()
