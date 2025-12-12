"""
app/models/model_cache.py

Provides an in-memory LRU (Least Recently Used) cache for storing model bundles
used during inference. This avoids repeatedly loading large pickle files from
disk in the Flink operator.

A cache entry contains:
    - isolation forest model object
    - robust scaler object
    - metadata dictionary

Used by:
    - flink/operators.py → during anomaly detection
    - anomaly_detector.py → to retrieve loaded model bundle

This module does NOT:
    - Load models from disk (delegates to model_loader)
    - Train models
    - Save files
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Tuple

from app.config import CONFIG
from app.models.model_loader import load_model_bundle
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ModelCache:
    """
    Simple LRU cache for model bundles.

    Behavior:
        - get(monitor_id): returns (model, scaler, metadata)
        - Loads from disk if not cached
        - Stores entry in LRU order
        - Evicts oldest entry when full
    """

    def __init__(self, max_size: int = None):
        self.max_size = max_size or CONFIG.MODEL_CACHE_SIZE
        self.cache: OrderedDict[str, Tuple[Any, Any, Dict]] = OrderedDict()

        logger.info(f"ModelCache initialized with max_size={self.max_size}")

    # -------------------------------------------------------------------
    # Public method: retrieve model from cache or load from disk
    # -------------------------------------------------------------------
    def get(self, monitor_id: str) -> Tuple[Any, Any, Dict]:
        """
        Retrieve a model bundle for MONITORID.
        Load from disk if not cached.

        Args:
            monitor_id: Monitor identifier string.

        Returns:
            Tuple: (model_object, scaler_object, metadata_dict)
        """

        # -------------------------
        # Cache hit
        # -------------------------
        if monitor_id in self.cache:
            model_bundle = self.cache.pop(monitor_id)  # refresh LRU
            self.cache[monitor_id] = model_bundle
            logger.info(f"Cache hit → MONITORID={monitor_id}")
            return model_bundle

        # -------------------------
        # Cache miss → load from disk
        # -------------------------
        logger.info(f"Cache miss → loading model for MONITORID={monitor_id}")
        model_bundle = load_model_bundle(monitor_id)

        # Insert into LRU cache
        self.cache[monitor_id] = model_bundle

        # Evict if over capacity
        if len(self.cache) > self.max_size:
            evicted_id, _ = self.cache.popitem(last=False)
            logger.warning(f"LRU cache eviction → MONITORID={evicted_id}")

        return model_bundle

    # -------------------------------------------------------------------
    # Optional: clear cache (useful for tests or manual restart)
    # -------------------------------------------------------------------
    def clear(self) -> None:
        """Remove all cached model entries."""
        self.cache.clear()
        logger.info("ModelCache cleared.")
