"""
app/utils/preprocessing_utils.py

Utility functions for preprocessing sensor data for model training
and anomaly detection.

Responsibilities:
    - Clean numeric values (handle None, empty strings, invalid numbers)
    - Convert lists of records → Pandas DataFrames
    - Fill missing values using median imputation
    - Extract model-required features in correct order
    - Fit & apply scalers (RobustScaler recommended)
    - Validate data shapes before training or inference

Used by:
    - model_builder.py (training)
    - anomaly_detector.py (inference)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from app.utils.logging_utils import get_logger
from app.config import FEATURES, FEATURE_MAP, MODEL_FEATURE_CODES

logger = get_logger(__name__)


# -----------------------------------------------------------
# Cleaning and normalization helpers
# -----------------------------------------------------------
def clean_numeric(value: Any) -> Optional[float]:
    """
    Safely convert a raw sensor input into a float.

    Args:
        value: Any incoming value (string, int, float, None, etc.)

    Returns:
        float or None

    Notes:
        - Empty strings → None
        - Non-numeric strings → None
        - Valid numbers → float
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    return None


# -----------------------------------------------------------
# DataFrame builders
# -----------------------------------------------------------
def rows_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of API or Kafka records into a DataFrame,
    keeping only model-relevant features.

    Args:
        records: List of JSON-like dicts with PROCESS_PARAMETER blocks.

    Returns:
        Pandas DataFrame with columns matching FEATURE_MAP.
    """
    rows = []

    for r in records:
        params = r.get("PROCESS_PARAMETER", {})
        row = {}

        for code in MODEL_FEATURE_CODES:
            feature_name = FEATURE_MAP.get(code)
            if feature_name:
                row[feature_name] = clean_numeric(params.get(code))

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning("rows_to_dataframe: received empty records list.")

    return df


# -----------------------------------------------------------
# Handle missing values
# -----------------------------------------------------------
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing (None / NaN) values using column medians.

    Args:
        df: DataFrame with numeric features.

    Returns:
        Cleaned DataFrame.
    """
    if df.empty:
        return df

    try:
        return df.fillna(df.median(numeric_only=True))
    except Exception as exc:
        logger.error(f"Median fill failed: {exc}")
        return df.fillna(0)


# -----------------------------------------------------------
# Feature ordering
# -----------------------------------------------------------
def reorder_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder DataFrame columns according to canonical FEATURES list.

    Args:
        df: Unordered DataFrame.

    Returns:
        Ordered DataFrame.
    """
    try:
        return df.reindex(columns=FEATURES)
    except Exception as exc:
        logger.error(f"Failed to reorder features: {exc}")
        return df


# -----------------------------------------------------------
# Scaler functions
# -----------------------------------------------------------
def fit_scaler(df: pd.DataFrame) -> RobustScaler:
    """
    Fit a RobustScaler to numeric features.

    Args:
        df: Cleaned DataFrame used for model training.

    Returns:
        Fitted RobustScaler.
    """
    if df.empty:
        raise ValueError("Cannot fit scaler: empty DataFrame.")

    scaler = RobustScaler()
    scaler.fit(df.values)

    return scaler


def apply_scaler(df: pd.DataFrame, scaler: RobustScaler) -> np.ndarray:
    """
    Apply a pre-fitted RobustScaler to a DataFrame.

    Args:
        df: DataFrame.
        scaler: Fitted RobustScaler.

    Returns:
        Numpy array of scaled values.
    """
    try:
        return scaler.transform(df.values)
    except Exception as exc:
        logger.error(f"Scaler transform failed: {exc}")
        raise


# -----------------------------------------------------------
# Full preprocessing pipeline
# -----------------------------------------------------------
def preprocess_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Complete preprocessing pipeline used by model_builder:

        records → dataframe → clean → fill → reorder

    Args:
        records: Raw historical data from API.

    Returns:
        Clean DataFrame ready for model training.
    """
    df = rows_to_dataframe(records)
    df = fill_missing_values(df)
    df = reorder_features(df)
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline expected by model_builder:

        - Clean numeric values
        - Keep only model features (from FEATURES)
        - Convert values to float
        - Fill missing using median
        - Ensure correct column order

    This MUST exist because model_builder imports it.
    """

    # -------- Keep only required features --------
    df = df[[f for f in FEATURES if f in df.columns]].copy()

    # -------- Clean each value --------
    for col in df.columns:
        df[col] = df[col].apply(clean_numeric)

    # -------- Convert to float --------
    df = df.astype(float)

    # -------- Fill missing values --------
    df = df.fillna(df.median(numeric_only=True))

    # -------- Reorder columns consistently --------
    df = df.reindex(columns=FEATURES)

    return df

