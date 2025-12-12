"""
app/predictor/feature_mapping.py

Defines feature mappings and utilities for converting sensor parameter codes into
human-readable names, and for ensuring features are ordered correctly for ML models.

This module centralizes:
    - FEATURE_MAP (code → readable name)
    - MODEL_FEATURE_CODES (raw params used as model inputs)
    - FEATURES (canonical ML feature order)
    - Helper functions for feature extraction and ordering

Used by:
    - anomaly_detector.py
    - model_builder.py
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

from app.config import FEATURE_MAP, MODEL_FEATURE_CODES, FEATURES


# -------------------------------------------------------------------
# Name lookup
# -------------------------------------------------------------------
def get_feature_name(code: str) -> Optional[str]:
    """
    Convert a parameter code (e.g., '001_A') to its human-readable name.

    Args:
        code: Feature code string.

    Returns:
        Readable name or None if unknown.
    """
    return FEATURE_MAP.get(code)


# -------------------------------------------------------------------
# Extract only model-relevant features
# -------------------------------------------------------------------
def extract_model_features(process_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only the features used by the ML model from a PROCESS_PARAMETER block.

    Args:
        process_params: Raw sensor parameter dictionary.

    Returns:
        Dict { readable_feature_name: value } limited to model inputs.
    """
    output = {}

    for code in MODEL_FEATURE_CODES:
        readable = FEATURE_MAP.get(code)
        if readable:
            output[readable] = process_params.get(code)

    return output


# -------------------------------------------------------------------
# Enforce consistent ML feature ordering
# -------------------------------------------------------------------
def reorder_for_model(feature_dict: Dict[str, Any]) -> List[Any]:
    """
    Order features according to the ML model's expected input order.

    Args:
        feature_dict: Dict keyed by readable feature names.

    Returns:
        List of feature values in the canonical order defined by FEATURES.
    """
    return [feature_dict.get(name) for name in FEATURES]
