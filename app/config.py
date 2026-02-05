"""
app/config.py

Central configuration module for the Oil Anomaly Detection Pipeline.

Responsibilities:
- Load and validate environment variables
- Provide strongly-typed configuration
- Expose a single CONFIG object used across the system
- Fail fast if required configuration is missing
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
    "001_F": "Oil Dielectric Constant",
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

#  ONLY THESE FEATURES ARE USED FOR MODEL
MODEL_FEATURE_CODES = [
    "001_A",
    "001_B",
    "001_C",
    "001_D",
    "001_E",
    "001_F",
]


FEATURES = [FEATURE_MAP[c] for c in MODEL_FEATURE_CODES]


# -------------------------------------------------------------------
#  Model feature enforcement
# -------------------------------------------------------------------
MODEL_FEATURE_CODES_ORDERED = tuple(MODEL_FEATURE_CODES)

MODEL_FEATURE_NAME_MAP = {
    code: FEATURE_MAP[code]
    for code in MODEL_FEATURE_CODES
}

MODEL_FEATURE_NAMES_ORDERED = [
    FEATURE_MAP[code] for code in MODEL_FEATURE_CODES
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


    # Trend / Device API
    TREND_API_BASE_URL: str
    TREND_API_TOKEN: str
    EXTERNAL_DEVICE_API_BASE_URL: str

    # Token-based auth
    TOKEN_URL: str
    TOKEN_USERNAME: str
    TOKEN_PASSWORD: str

    # AWS / S3
    S3_BUCKET_NAME: str

    # Runtime
    LOG_DIR: Path
    DEBUG: bool
    FLINK_PARALLELISM: int
    ENVIRONMENT: str


# -------------------------------------------------------------------
# Build CONFIG
# -------------------------------------------------------------------
_BROKER_LIST = _env_list("KAFKA_ENDPOINTS", ["kafka:9092"])

CONFIG = Config(
    # Kafka
    INPUT_TOPIC=_env_str("INPUT_TOPIC", "external_device_data_flink_source"),
    ALERT_TOPIC=_env_str("ALERT_TOPIC", "oil-analysis-anomaly-alert"),
    BROKERS=_BROKER_LIST,
    KAFKA_BROKERS=",".join(_BROKER_LIST),
    KAFKA_GROUP=_env_str("KAFKA_GROUP", "oil-anomaly-consumer-group"),

    # Sliding window
    WINDOW_COUNT=_env_int("WINDOW_COUNT", 20),
    SLIDE_COUNT=_env_int("SLIDE_COUNT", 18),

    # Model
    MODEL_TREES=_env_int("MODEL_TREES", 300),
    ANOMALY_CONTAMINATION=_env_float("ANOMALY_CONTAMINATION", 0.0001),
    MODEL_CACHE_SIZE=_env_int("MODEL_CACHE_SIZE", 32),

    # Trend / Device API
    TREND_API_BASE_URL=_env_str(
        "TREND_API_BASE_URL",
        "https://api.infinite-uptime.com/api/3.0/idap-api/external-monitors/trend-history/v2",
    ),
    TREND_API_TOKEN=_env_str("TREND_API_TOKEN", ""),
    EXTERNAL_DEVICE_API_BASE_URL=_env_str(
        "EXTERNAL_DEVICE_API_BASE_URL",
        "",
    ),

    # Token auth
    TOKEN_URL=_env_str("TOKEN_URL", ""),
    TOKEN_USERNAME=_env_str("TOKEN_USERNAME", ""),
    TOKEN_PASSWORD=_env_str("TOKEN_PASSWORD", ""),

    # AWS
    S3_BUCKET_NAME=_env_str("S3_BUCKET_NAME", "").strip(),

    # Runtime
    LOG_DIR=LOG_DIR,
    DEBUG=_env_bool("DEBUG", False),
    FLINK_PARALLELISM=_env_int("FLINK_PARALLELISM", 1),
    ENVIRONMENT=_env_str("ENVIRONMENT", "local"),
)


# -------------------------------------------------------------------
#  Fail-fast validation
# -------------------------------------------------------------------
_missing = []

if CONFIG.ENVIRONMENT == "prod" and not CONFIG.S3_BUCKET_NAME:
    _missing.append("S3_BUCKET_NAME")

if not CONFIG.BROKERS:
    _missing.append("KAFKA_ENDPOINTS")

if not CONFIG.TOKEN_URL:
    _missing.append("TOKEN_URL")

if not CONFIG.TOKEN_USERNAME:
    _missing.append("TOKEN_USERNAME")

if not CONFIG.TOKEN_PASSWORD:
    _missing.append("TOKEN_PASSWORD")

if not CONFIG.EXTERNAL_DEVICE_API_BASE_URL:
    _missing.append("EXTERNAL_DEVICE_API_BASE_URL")

if _missing:
    raise RuntimeError(
        "Missing required environment variables: "
        + ", ".join(_missing)
    )


# -------------------------------------------------------------------
#  Feature sanity checks
# -------------------------------------------------------------------
if len(MODEL_FEATURE_CODES) != 6:
    raise RuntimeError(
        f"MODEL_FEATURE_CODES must contain exactly 6 features, got {len(MODEL_FEATURE_CODES)}"
    )

for code in MODEL_FEATURE_CODES:
    if code not in FEATURE_MAP:
        raise RuntimeError(
            f"Feature code {code} missing in FEATURE_MAP"
        )


# -------------------------------------------------------------------
# Public exports
# -------------------------------------------------------------------
__all__ = [
    "CONFIG",
    "Config",
    "FEATURE_MAP",
    "MODEL_FEATURE_CODES",
    "FEATURES",
    "MODEL_FEATURE_CODES_ORDERED",
    "MODEL_FEATURE_NAME_MAP",
    "MODEL_FEATURE_NAMES_ORDERED",
]
