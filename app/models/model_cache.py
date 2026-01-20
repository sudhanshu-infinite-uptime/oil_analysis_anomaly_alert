"""
app/models/model_cache.py

In-memory LRU cache for trained model bundles.

Responsibilities:
- Cache (model, scaler, metadata) per MONITORID
- Reduce repeated S3 reads during streaming inference
- Enforce thread-safe access for Flink operators

This module does NOT:
- Train models
- Retry failed loads
- Perform inference
- Handle persistence
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Tuple

from app.config import CONFIG
from app.models.model_loader import load_model_bundle
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ModelCache:
    """
    Thread-safe LRU cache for model bundles.

    Cache key:
        monitor_id (int)

    Cache value:
        (model, scaler, metadata)
    """

    def __init__(self, max_size: int | None = None):
        self.max_size = max_size or CONFIG.MODEL_CACHE_SIZE
        self._cache: OrderedDict[int, Tuple[Any, Any, Dict]] = OrderedDict()
        self._lock = Lock()

        logger.info(
            "ModelCache initialized | max_size=%d | instance=%s",
            self.max_size,
            hex(id(self)),
        )

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def get(self, monitor_id: int) -> Tuple[Any, Any, Dict]:
        """
        Retrieve model bundle for a monitor.

        Behavior:
        - Cache HIT  → return immediately
        - Cache MISS → load from S3 → insert → return

        Raises:
            ModelNotFoundError
            ModelLoadError
        """

        # ---------- fast path (cache hit)
        with self._lock:
            bundle = self._cache.get(monitor_id)
            if bundle is not None:
                # refresh LRU order
                self._cache.move_to_end(monitor_id)
                logger.debug(
                    "ModelCache HIT | MONITORID=%s | cache_size=%d",
                    monitor_id,
                    len(self._cache),
                )
                return bundle

        logger.info("ModelCache MISS | MONITORID=%s | loading from S3", monitor_id)

        # ---------- slow path (S3 load outside lock)
        bundle = load_model_bundle(monitor_id)

        # ---------- insert into cache
        with self._lock:
            # another thread may have loaded it already
            if monitor_id in self._cache:
                self._cache.move_to_end(monitor_id)
                return self._cache[monitor_id]

            self._cache[monitor_id] = bundle

            if len(self._cache) > self.max_size:
                evicted_id, _ = self._cache.popitem(last=False)
                logger.warning(
                    "ModelCache EVICT | MONITORID=%s | cache_size=%d",
                    evicted_id,
                    len(self._cache),
                )

        return bundle

    def clear(self) -> None:
        """Clear all cached model bundles."""
        with self._lock:
            self._cache.clear()
        logger.info("ModelCache cleared")
