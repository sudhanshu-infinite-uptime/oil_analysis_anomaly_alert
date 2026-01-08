"""
app/flink/operators.py

MultiModelAnomalyOperator (Inference-Only)

Responsibilities:
- Consume Kafka records
- Maintain per-monitor sliding windows
- Load trained models from ModelCache
- Run anomaly detection
- Emit alerts

This operator MUST NOT:
- Train models
- Call external APIs
- Perform blocking I/O
"""

from __future__ import annotations

import json
import pandas as pd
from typing import Dict, Any

from pyflink.datastream.functions import FlatMapFunction, RuntimeContext

from app.models.model_cache import ModelCache
from app.models.model_store import model_exists
from app.predictor.anomaly_detector import detect_anomalies
from app.windows.sliding_window import SlidingWindow

from app.utils.json_utils import safe_json_parse
from app.utils.logging_utils import get_logger
from app.config import CONFIG

logger = get_logger(__name__)


class MultiModelAnomalyOperator(FlatMapFunction):

    def open(self, runtime_context: RuntimeContext):
        logger.info("Initializing MultiModelAnomalyOperator (inference-only)")
        self.model_cache = ModelCache(max_size=CONFIG.MODEL_CACHE_SIZE)
        self.windows: Dict[str, SlidingWindow] = {}

    def flat_map(self, value: str):
        record = safe_json_parse(value)
        if not record:
            return

        monitor_id = record.get("MONITORID")
        if not monitor_id:
            return


        # ---------------------------------------------------------
        # Sliding window setup
        # ---------------------------------------------------------
        if monitor_id not in self.windows:
            self.windows[monitor_id] = SlidingWindow(
                window_size=CONFIG.WINDOW_COUNT,
                slide_size=CONFIG.SLIDE_COUNT,
            )

        window = self.windows[monitor_id]

        
        window.add(record)

        if not window.is_full():
            return

        # ---------------------------------------------------------
        # Model existence check (NO TRAINING HERE)
        # ---------------------------------------------------------
        if not model_exists(monitor_id):
            logger.warning(
                "Model not found → skipping inference | MONITORID=%s",
                monitor_id,
            )
            window.slide()
            return

        # ---------------------------------------------------------
        # Load model bundle
        # ---------------------------------------------------------
        try:
            model, scaler, metadata = self.model_cache.get(monitor_id)
        except Exception as exc:
            logger.error(
                "Model load failed | MONITORID=%s | %s",
                monitor_id,
                exc,
            )
            window.slide()
            return

        feature_names = metadata.get("feature_names")
        if not feature_names:
            logger.error(
                "Invalid metadata (missing feature_names) | MONITORID=%s",
                monitor_id,
            )
            window.slide()
            return

        # ---------------------------------------------------------
        # Prepare data
        # ---------------------------------------------------------
        df = window.to_dataframe()
        df = self._align_features(df, feature_names)

        # ---------------------------------------------------------
        # Inference
        # ---------------------------------------------------------
        try:
            result = detect_anomalies(df, model, scaler, metadata)
        except Exception as exc:
            logger.error(
                "Anomaly detection failed | MONITORID=%s | %s",
                monitor_id,
                exc,
            )
            window.slide()
            return

        window.slide()

        if result.get("is_anomaly"):
            yield json.dumps(self._format_alert(monitor_id, result))

    # ---------------------------------------------------------
    def _align_features(self, df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0
        return df[feature_names]

    # ---------------------------------------------------------
    def _format_alert(self, monitor_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "monitorId": monitor_id,
            "isAnomaly": True,
            "indices": result.get("anomaly_indices", []),
            "topFeatures": result.get("top_features", []),
            "modelMetadata": result.get("model_metadata", {}),
        }
