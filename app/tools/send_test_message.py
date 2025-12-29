"""
send_test_message.py

Sends synthetic sensor messages to Kafka for multiple MONITORIDs.

Designed for:
    • Local testing
    • Docker execution
    • UAT / PROD environments

Flink behavior:
    - Detect MONITORID
    - Auto-create or load model
    - Run anomaly detection
    - Publish alerts to ALERT_TOPIC
"""

from kafka import KafkaProducer
from json import dumps
import time
import numpy as np
import random
import logging
import os
import sys


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Environment-driven configuration
# ------------------------------------------------------------------
TOPIC = os.getenv("INPUT_TOPIC", "iu_external_device_data_v1")

BROKERS_ENV = (
    os.getenv("KAFKA_ENDPOINTS")
    or os.getenv("KAFKA_BROKERS")
    or "kafka:9092"   # Docker default
)

BROKERS = [b.strip() for b in BROKERS_ENV.split(",") if b.strip()]

if not BROKERS:
    raise RuntimeError(
        "Kafka brokers not configured. "
        "Set KAFKA_ENDPOINTS or KAFKA_BROKERS."
    )

logger.info(f"Kafka brokers resolved to: {BROKERS}")
logger.info(f"Kafka topic: {TOPIC}")


# ------------------------------------------------------------------
# Test data configuration
# ------------------------------------------------------------------
MONITOR_IDS = [29, 30, 31, 32, 33]
TOTAL_MESSAGES = 10000
SLEEP_SECONDS = 1

MODEL_FEATURE_CODES = [f"001_{chr(65+i)}" for i in range(6)]  # A–F


# ------------------------------------------------------------------
# Kafka Producer
# ------------------------------------------------------------------
try:
    producer = KafkaProducer(
        bootstrap_servers=BROKERS,
        value_serializer=lambda v: dumps(v).encode("utf-8"),
        linger_ms=5,
        retries=5,
        request_timeout_ms=10000
    )
    logger.info("🚀 Kafka producer initialized successfully")
except Exception as exc:
    logger.error(f"❌ Failed to initialize Kafka producer: {exc}")
    sys.exit(1)


# ------------------------------------------------------------------
# Send messages
# ------------------------------------------------------------------
for i in range(TOTAL_MESSAGES):
    monitor_id = random.choice(MONITOR_IDS)

    params = {
        "001_A": float(np.random.normal(50.00, 0.2)),
        "001_B": float(np.random.normal(0.889, 0.002)),
        "001_C": float(np.random.normal(800, 1.0)),
        "001_D": float(np.random.normal(775, 15)),
        "001_E": float(np.random.normal(1030, 2)),
        "001_F": float(np.random.normal(5.18, 0.01)),

        "001_G": float(np.random.uniform(0, 2)),
        "001_H": float(np.random.uniform(0, 2)),
        "001_I": float(np.random.uniform(0, 2)),
        "001_J": float(np.random.uniform(0, 2)),
        "001_K": float(np.random.uniform(0, 2)),
        "001_L": float(np.random.uniform(0, 10)),

        "001_M": float(np.random.uniform(0, 2)),
        "001_N": float(np.random.uniform(0, 2)),
        "001_O": float(np.random.uniform(0, 2)),
        "001_P": float(np.random.uniform(0, 2)),
        "001_Q": float(np.random.uniform(0, 2)),
        "001_R": float(np.random.uniform(0, 10)),
    }

    # Inject anomalies periodically
    if i % 7 == 0:
        anomaly_keys = random.sample(MODEL_FEATURE_CODES, k=2)
        for key in anomaly_keys:
            params[key] *= random.uniform(4, 8)
        logger.warning(
            f"⚠️ Injected anomaly | MONITORID={monitor_id} | fields={anomaly_keys}"
        )

    message = {
        "MONITORID": monitor_id,
        "DEVICEID": f"DEV-{monitor_id}",
        "TIMESTAMP": int(time.time() * 1000),
        "PROCESS_PARAMETER": params
    }

    try:
        producer.send(TOPIC, value=message)
        logger.info(f"📤 Sent message {i + 1}/{TOTAL_MESSAGES} | MONITORID={monitor_id}")
    except Exception as exc:
        logger.error(f"❌ Failed to send message {i + 1}: {exc}")

    time.sleep(SLEEP_SECONDS)


producer.flush()
producer.close()
logger.info("🏁 Finished sending test messages")
