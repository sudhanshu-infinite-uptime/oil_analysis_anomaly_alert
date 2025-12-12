"""
send_test_message.py
Sends synthetic sensor messages for MANY MONITORIDs.

Flink will:
  - detect MONITORID
  - check if model exists
  - auto-train using model_builder.py
  - run anomaly detection
  - publish alerts to oil-analysis-anomaly-alert
"""

from kafka import KafkaProducer
from json import dumps
import time
import numpy as np
import random
import logging

# ------------------------------------------------------
# CONFIG  (fixed for now)
# ------------------------------------------------------
TOPIC = "iu_external_device_data_v1"
BROKER = "172.18.0.2:9092"    # <-- Kafka IP

# Multiple monitors → models should auto-create
MONITOR_IDS = [29, 30, 31, 32, 33]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------------------------------------------
# CONNECT TO KAFKA
# ------------------------------------------------------
try:
    producer = KafkaProducer(
        bootstrap_servers=[BROKER],
        value_serializer=lambda v: dumps(v).encode("utf-8"),
        linger_ms=5,
        retries=5
    )
    logging.info(f"🚀 Connected to Kafka @ {BROKER}")
except Exception as e:
    logging.error(f"❌ Kafka connection error: {e}")
    raise SystemExit(1)

# ------------------------------------------------------
# MODEL FEATURE CODES
# ------------------------------------------------------
MODEL_FEATURE_CODES = [f"001_{chr(65+i)}" for i in range(6)]  # A–F


# ------------------------------------------------------
# SEND TEST MESSAGES
# ------------------------------------------------------
for i in range(10000):   # send 50 messages
    monitor_id = random.choice(MONITOR_IDS)

    # Full 18 features (A–R)
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

    # Random anomaly injection
    if i % 7 == 0:
        anomaly_keys = random.sample(MODEL_FEATURE_CODES, k=2)
        for key in anomaly_keys:
            params[key] *= random.uniform(4, 8)
        logging.warning(f"⚠️ Injected anomaly for MONITORID={monitor_id} in {anomaly_keys}")

    message = {
        "MONITORID": monitor_id,
        "DEVICEID": f"DEV-{monitor_id}",
        "TIMESTAMP": int(time.time() * 1000),
        "PROCESS_PARAMETER": params
    }

    try:
        producer.send(TOPIC, value=message)
        logging.info(f"📤 Sent message #{i+1} → MONITORID={monitor_id}")
    except Exception as e:
        logging.error(f"❌ Failed to send message {i+1}: {e}")

    time.sleep(1)

producer.flush()
logging.info("🏁 Finished sending messages!")
