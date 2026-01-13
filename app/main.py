"""
app/main.py

Top-level entry point for the Oil Anomaly Detection Pipeline.

Responsibilities:
    • Initialize global logging
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


def main() -> None:
    """
    Application entry point.

    Flow:
    - Start Flink streaming job
    - Models are trained ON-DEMAND inside the operator
    """

    logger.info("🚀 Starting Oil Anomaly Detection Service...")
    run_flink_job()


if __name__ == "__main__":
    main()
