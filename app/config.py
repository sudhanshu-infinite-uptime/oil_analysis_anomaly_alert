"""
app/config.py

Central configuration module for the Oil Anomaly Detection Pipeline.

Responsibilities:
- Load and validate environment variables
- Provide strongly-typed configuration
- Expose a single CONFIG object used across the system
- Fail fast if required configuration is missing

Usage:
    from app.config import CONFIG
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


# -------------------------------------------------------------------
# Environment parsing helpers
# -------------------------------------------------------------------
def _env_str(key: str, default: str = "") -> str:
    val = os.getenv(key)
    return val if val not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"{key} must be an integer (got {val})")


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"{key} must be a float (got {val})")


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if not val:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _env_list(key: str, default: List[str]) -> List[str]:
    val = os.getenv(key)
    if not val:
        return default
    return [x.strip() for x in val.split(",") if x.strip()]


# -------------------------------------------------------------------
# Default filesystem paths
# -------------------------------------------------------------------
MODEL_BASE_PATH = Path(os.getenv("MODEL_BASE_PATH", "./models"))
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))

MODEL_BASE_PATH.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Feature definitions
# -------------------------------------------------------------------
FEATURE_MAP = {
    "001_A": "Oil Temperature",
    "001_B": "Water Activity",
    "001_C": "Moisture",
    "001_D": "Kinematic Viscosity",
    "001_E": "Oil Density",
    "001_F": "Dielectric Constant",
    "001_G": "Ferrous Particles Level 1",
    "001_H": "Ferrous Particles Level 2",
    "001_I": "Ferrous Particles Level 3",
    "001_J": "Ferrous Particles Level 4",
    "001_K": "Ferrous Particles Level 5",
    "001_L": "Total Ferrous Particles",
    "001_M": "Non-Ferrous Particles 1",
    "001_N": "Non-Ferrous Particles 2",
    "001_O": "Non-Ferrous Particles 3",
    "001_P": "Non-Ferrous Particles 4",
    "001_Q": "Non-Ferrous Particles 5",
    "001_R": "Total Non-Ferrous Particles",
}

# ML feature codes used for training
MODEL_FEATURE_CODES = [
    "001_A",
    "001_B",
    "001_C",
    "001_D",
    "001_E",
    "001_F",
]

# Canonical ML feature order (USED BY MODEL)
FEATURES = [
    "Dielectric Constant",
    "Oil Density",
    "Kinematic Viscosity",
    "Moisture",
    "Water Activity",
    "Oil Temperature",
]


# -------------------------------------------------------------------
# Typed configuration object
# -------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    # Kafka
    INPUT_TOPIC: str
    ALERT_TOPIC: str
    BROKERS: List[str]
    KAFKA_BROKERS: str
    KAFKA_GROUP: str

    # Sliding window
    WINDOW_COUNT: int
    SLIDE_COUNT: int

    # Model training
    MODEL_TREES: int
    ANOMALY_CONTAMINATION: float
    MODEL_CACHE_SIZE: int

    # Trend API
    TREND_API_BASE_URL: str
    TREND_API_TOKEN: str

    # AWS / S3
    S3_BUCKET_NAME: str

    # Runtime
    LOG_DIR: Path
    DEBUG: bool
    FLINK_PARALLELISM: int


# -------------------------------------------------------------------
# Build CONFIG
# -------------------------------------------------------------------
_BROKER_LIST = _env_list("KAFKA_ENDPOINTS", ["kafka:9092"])

CONFIG = Config(
    # Kafka
    INPUT_TOPIC=_env_str("INPUT_TOPIC", "iu_external_device_data_v1"),
    ALERT_TOPIC=_env_str("ALERT_TOPIC", "oil-analysis-anomaly-alert"),
    BROKERS=_BROKER_LIST,
    KAFKA_BROKERS=",".join(_BROKER_LIST),
    KAFKA_GROUP=_env_str("KAFKA_GROUP", "oil-anomaly-consumer-group"),

    # Sliding window
    WINDOW_COUNT=_env_int("WINDOW_COUNT", 10),
    SLIDE_COUNT=_env_int("SLIDE_COUNT", 3),

    # Model
    MODEL_TREES=_env_int("MODEL_TREES", 200),
    ANOMALY_CONTAMINATION=_env_float("ANOMALY_CONTAMINATION", 0.05),
    MODEL_CACHE_SIZE=_env_int("MODEL_CACHE_SIZE", 32),

    # Trend API
    TREND_API_BASE_URL=_env_str(
        "TREND_API_BASE_URL",
        "http://localhost:5000/api/v1/trend",
    ),
    TREND_API_TOKEN=_env_str("TREND_API_TOKEN", ""),

    # AWS
    S3_BUCKET_NAME=_env_str("S3_BUCKET_NAME", ""),

    # Runtime
    LOG_DIR=LOG_DIR,
    DEBUG=_env_bool("DEBUG", False),
    FLINK_PARALLELISM=_env_int("FLINK_PARALLELISM", 1),
)


# -------------------------------------------------------------------
# Fail-fast validation (VERY IMPORTANT FOR PROD)
# -------------------------------------------------------------------
_missing = []

if not CONFIG.S3_BUCKET_NAME:
    _missing.append("S3_BUCKET_NAME")

if not CONFIG.TREND_API_TOKEN:
    _missing.append("TREND_API_TOKEN")

if not CONFIG.BROKERS:
    _missing.append("KAFKA_ENDPOINTS")

if _missing:
    raise RuntimeError(
        "Missing required environment variables: "
        + ", ".join(_missing)
    )


__all__ = [
    "CONFIG",
    "Config",
    "FEATURE_MAP",
    "MODEL_FEATURE_CODES",
    "FEATURES"
]
