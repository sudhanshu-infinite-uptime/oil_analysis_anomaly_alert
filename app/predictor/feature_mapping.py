"""
app/predictor/feature_mapping.py

Feature mapping utilities for ML pipelines.

Responsibilities:
- Map raw sensor parameter codes to readable feature names
- Extract only model-relevant features from raw payloads
- Provide safe, validated feature alignment for ML models

This module does NOT:
- Define model feature order (comes from model metadata)
- Train or load models
- Perform anomaly detection
"""

from __future__ import annotations

from typing import Dict, Any, List

from app.config import FEATURE_MAP, MODEL_FEATURE_CODES


# -------------------------------------------------------------------
# Feature name lookup
# -------------------------------------------------------------------
def get_feature_name(code: str) -> str | None:
    """
    Convert a raw parameter code to its readable feature name.
    """
    return FEATURE_MAP.get(code)


# -------------------------------------------------------------------
# Extract model-relevant features from raw PROCESS_PARAMETER
# -------------------------------------------------------------------
def extract_model_features(process_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only features required by the ML model.

    Args:
        process_params: Raw PROCESS_PARAMETER dictionary.

    Returns:
        Dict keyed by readable feature names.
    """
    extracted: Dict[str, Any] = {}

    for code in MODEL_FEATURE_CODES:
        readable = FEATURE_MAP.get(code)
        if readable is not None:
            extracted[readable] = process_params.get(code)

    return extracted


# -------------------------------------------------------------------
# Align features to model metadata (SAFE)
# -------------------------------------------------------------------
def align_features_for_model(
    feature_dict: Dict[str, Any],
    feature_names: List[str],
    fill_value: float = 0.0,
) -> List[Any]:
    """
    Align features to the exact order expected by the trained model.

    Args:
        feature_dict: Dict keyed by readable feature names.
        feature_names: Ordered feature list from model metadata.
        fill_value: Value to use for missing features.

    Returns:
        List of feature values in correct order.

    Raises:
        ValueError if feature_names is empty.
    """
    if not feature_names:
        raise ValueError("feature_names cannot be empty")

    return [
        feature_dict.get(name, fill_value)
        for name in feature_names
    ]
