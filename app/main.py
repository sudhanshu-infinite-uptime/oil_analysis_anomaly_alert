"""
app/main.py

Top-level entry point for the Oil Anomaly Detection Pipeline.

Responsibilities:
    • Initialize global logging
    • Start the Flink streaming pipeline
    • Provide a clean, minimal entry point for running in containers

This file should NOT:
    - Load or train models
    - Contain Flink operators
    - Handle Kafka directly
    - Perform sliding window logic
"""

from __future__ import annotations

from app.flink.flink_job import run_flink_job
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Application entry point. Starts the Flink anomaly detection pipeline."""
    logger.info("🚀 Starting Oil Anomaly Detection Service...")
    run_flink_job()


if __name__ == "__main__":
    main()
