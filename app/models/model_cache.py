"""
app/models/model_cache.py

In-memory LRU cache for storing trained model bundles used during inference.

Each cache entry contains:
- IsolationForest model
- RobustScaler
- metadata dictionary

Purpose:
- Avoid repeated S3 reads during Flink stream processing
- Improve latency and throughput

Notes:
- Cache is per Python process (per TaskManager worker)
- Thread-safe for concurrent operator access

Used by:
- flink/operators.py
- anomaly_detector.py
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Tuple
from threading import Lock

from app.config import CONFIG
from app.models.model_loader import load_model_bundle
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ModelCache:
    """
    Thread-safe LRU cache for model bundles.

    API:
        get(monitor_id) -> (model, scaler, metadata)
        clear() -> None
    """

    def __init__(self, max_size: int | None = None):
        self.max_size = max_size or CONFIG.MODEL_CACHE_SIZE
        self.cache: OrderedDict[str, Tuple[Any, Any, Dict]] = OrderedDict()
        self._lock = Lock()

        logger.info(
            "ModelCache initialized | max_size=%d | id=%s",
            self.max_size,
            hex(id(self)),
        )

    # ------------------------------------------------------------------
    # Retrieve model bundle (cache-first)
    # ------------------------------------------------------------------
    def get(self, monitor_id: str) -> Tuple[Any, Any, Dict]:
        """
        Retrieve model bundle for MONITORID.

        Loads from S3 if not present in cache.

        Raises:
            ModelNotFoundError
            ModelLoadError
        """

        with self._lock:
            if monitor_id in self.cache:
                bundle = self.cache.pop(monitor_id)
                self.cache[monitor_id] = bundle
                logger.debug("ModelCache HIT | MONITORID=%s", monitor_id)
                return bundle

        logger.info("ModelCache MISS | Loading model | MONITORID=%s", monitor_id)

        # Load outside lock (S3/network I/O)
        bundle = load_model_bundle(monitor_id)

        with self._lock:
            self.cache[monitor_id] = bundle

            if len(self.cache) > self.max_size:
                evicted_id, _ = self.cache.popitem(last=False)
                logger.warning(
                    "ModelCache EVICT | MONITORID=%s | cache_size=%d",
                    evicted_id,
                    len(self.cache),
                )

        return bundle

    # ------------------------------------------------------------------
    # Clear cache
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self.cache.clear()
        logger.info("ModelCache cleared")
