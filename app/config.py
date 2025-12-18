"""
app/config.py

Central configuration module for the Oil Anomaly Detection Pipeline.

Responsibilities:
    - Load environment variables safely
    - Provide typed defaults (int, float, str, bool, list)
    - Define feature mappings for model training + inference
    - Define filesystem paths (models and logs)
    - Ensure required directories exist
    - Expose a single CONFIG object used across the entire project

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
def _env_str(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(
            f"Environment variable {key} must be an integer (got: {val!r})"
        )


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(
            f"Environment variable {key} must be a float (got: {val!r})"
        )


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
# Default directories
# -------------------------------------------------------------------
DEFAULT_MODEL_BASE_PATH = Path(os.getenv("MODEL_BASE_PATH", "./models"))
DEFAULT_LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))

DEFAULT_MODEL_BASE_PATH.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)


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

# Raw sensor codes used for ML
MODEL_FEATURE_CODES = [
    "001_A", "001_B", "001_C", "001_D", "001_E", "001_F"
]

# Canonical ML feature order (training + inference)
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
    """
    Typed configuration object used throughout the pipeline.
    All runtime code must read configuration only from this object.
    """

    # Kafka topics
    INPUT_TOPIC: str
    ALERT_TOPIC: str

    # Kafka brokers (list form)
    BROKERS: List[str]

    # Kafka brokers (string form – REQUIRED by Flink Kafka connector)
    KAFKA_BROKERS: str

    # Model storage & caching
    MODEL_BASE_PATH: Path
    MODEL_CACHE_SIZE: int

    # Sliding window parameters
    WINDOW_COUNT: int
    SLIDE_COUNT: int

    # Logging
    LOG_DIR: Path
    DEBUG: bool

    # Trend API
    TREND_API_BASE_URL: str

    # Training parameters
    TRAINING_CONTAMINATION: float

    # Static defaults (not env-driven)
    FLINK_PARALLELISM: int = 1
    KAFKA_GROUP: str = "oil-anomaly-consumer-group"

    # Model hyperparameters
    MODEL_TREES: int = 200
    MODEL_CONTAMINATION: float = 0.05

    # S3 bucket
    S3_BUCKET_NAME: str = ""


# -------------------------------------------------------------------
# Create global CONFIG
# -------------------------------------------------------------------

# Read Kafka brokers ONCE and reuse everywhere
_BROKER_LIST = _env_list("KAFKA_ENDPOINTS", ["kafka:9092"])

CONFIG = Config(
    INPUT_TOPIC=_env_str("INPUT_TOPIC", "iu_external_device_data_v1"),
    ALERT_TOPIC=_env_str("ALERT_TOPIC", "oil-analysis-anomaly-alert"),

    # Keep list form (useful for future extensions)
    BROKERS=_BROKER_LIST,

    # String form (USED by Flink Kafka producer/consumer)
    KAFKA_BROKERS=",".join(_BROKER_LIST),

    MODEL_BASE_PATH=DEFAULT_MODEL_BASE_PATH,
    MODEL_CACHE_SIZE=_env_int("MODEL_CACHE_SIZE", 32),

    WINDOW_COUNT=_env_int("WINDOW_COUNT", 10),
    SLIDE_COUNT=_env_int("SLIDE_COUNT", 3),

    LOG_DIR=DEFAULT_LOG_DIR,
    DEBUG=_env_bool("DEBUG", False),

    TREND_API_BASE_URL=_env_str(
        "TREND_API_BASE_URL",
        "http://localhost:5000/api/v1/trend"
    ),

    TRAINING_CONTAMINATION=_env_float("TRAINING_CONTAMINATION", 0.05),

    S3_BUCKET_NAME=_env_str("S3_BUCKET_NAME", "")
)


__all__ = [
    "CONFIG",
    "Config",
    "FEATURE_MAP",
    "MODEL_FEATURE_CODES",
    "FEATURES",
]
