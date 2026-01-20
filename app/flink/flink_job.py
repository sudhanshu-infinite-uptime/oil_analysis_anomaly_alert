"""
app/flink/flink_job.py

Defines and starts the Flink streaming pipeline responsible for:

• Reading sensor messages from Kafka
• Applying MultiModelAnomalyOperator for per-monitor anomaly detection
• Writing anomaly alerts back to Kafka

This module is intentionally minimal. It should NOT:
- Train models
- Load models
- Perform sliding window logic
- Detect anomalies

All business logic is delegated to operators.py and lower-level modules.
This file only assembles the streaming pipeline and executes it.
"""

from __future__ import annotations

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.connectors.kafka import (
    FlinkKafkaConsumer,
    FlinkKafkaProducer,
)
from pyflink.common import Types

from app.flink.operators import MultiModelAnomalyOperator
from app.config import CONFIG
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------------
# Build Flink pipeline (but do NOT execute it)
# -------------------------------------------------------------------
def build_flink_job():
    """
    Constructs the Flink streaming job but does not execute it.
    Useful for testing, dry-run checks, and integration tests.
    """

    logger.info("Setting up Flink StreamExecutionEnvironment...")

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(CONFIG.FLINK_PARALLELISM)

    logger.info("======================================")
    logger.info("Submitting Flink job")
    logger.info("Job Name          : Oil Anomaly Detection Pipeline")
    logger.info(f"Kafka Input Topic : {CONFIG.INPUT_TOPIC}")
    logger.info(f"Kafka Output Topic: {CONFIG.ALERT_TOPIC}")
    logger.info(f"Kafka Brokers     : {CONFIG.BROKERS}")
    logger.info("======================================")


    # --------------------------------------------------------------
    # Kafka Consumer
    # --------------------------------------------------------------
    logger.info(
        f"Configuring Kafka consumer → topic={CONFIG.INPUT_TOPIC}, "
        f"brokers={CONFIG.KAFKA_BROKERS}"
    )

    consumer = FlinkKafkaConsumer(
        topics=CONFIG.INPUT_TOPIC,
        deserialization_schema=SimpleStringSchema(),
        properties={
            "bootstrap.servers": CONFIG.KAFKA_BROKERS,
            "group.id": CONFIG.KAFKA_GROUP,
            "auto.offset.reset": "earliest",
        },
    )

    # --------------------------------------------------------------
    # Kafka Producer
    # --------------------------------------------------------------
    logger.info(
        f"Configuring Kafka producer → topic={CONFIG.ALERT_TOPIC}, "
        f"brokers={CONFIG.KAFKA_BROKERS}"
    )

    producer = FlinkKafkaProducer(
        topic=CONFIG.ALERT_TOPIC,
        serialization_schema=SimpleStringSchema(),
        producer_config={"bootstrap.servers": CONFIG.KAFKA_BROKERS},
    )

    # --------------------------------------------------------------
    # Build pipeline graph
    # --------------------------------------------------------------
    stream = env.add_source(consumer)

    anomaly_stream = stream.flat_map(
        MultiModelAnomalyOperator(),
        output_type=Types.STRING(),
    )

    anomaly_stream.add_sink(producer)

    logger.info("Flink pipeline successfully built.")
    return env


# -------------------------------------------------------------------
# Execute pipeline (called by main.py)
# -------------------------------------------------------------------
def run_flink_job():
    """
    Public entry point for launching the full Flink anomaly detection pipeline.
    This is what your top-level main.py will call.
    """
    env = build_flink_job()

    logger.info("Starting Flink job: Oil Anomaly Detection Pipeline")
    env.execute("Oil Anomaly Detection Pipeline")

if __name__ == "__main__":
    run_flink_job()
