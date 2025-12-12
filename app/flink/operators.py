"""
app/flink/operators.py

MultiModelAnomalyOperator:

- Receives incoming Kafka JSON messages
- Groups records by MONITORID
- Maintains a SlidingWindow per monitor
- Loads model/scaler/metadata through ModelCache
- Runs anomaly detection when window is full
- Builds a new model if one does not exist (on-demand training)
- Emits anomaly results downstream

This operator should NOT:
- Train models directly (delegates to model_builder)
- Load models from disk (delegates to model_cache → model_loader)
- Handle Kafka consumer/producer configuration
"""

from __future__ import annotations

import json
import pandas as pd
from typing import Dict, Any

from pyflink.datastream.functions import FlatMapFunction, RuntimeContext

from app.models.model_cache import ModelCache
from app.models.model_builder import build_model_for_monitor
from app.models.model_store import model_exists
from app.predictor.anomaly_detector import detect_anomalies
from app.windows.sliding_window import SlidingWindow

from app.utils.json_utils import safe_json_parse
from app.utils.logging_utils import get_logger
from app.utils.exceptions import TrainingFailedError
from app.config import CONFIG

logger = get_logger(__name__)


class MultiModelAnomalyOperator(FlatMapFunction):

    def open(self, runtime_context: RuntimeContext):
        logger.info("Initializing MultiModelAnomalyOperator...")
        self.model_cache = ModelCache(max_size=CONFIG.MODEL_CACHE_SIZE)
        self.windows: Dict[str, SlidingWindow] = {}

    def flat_map(self, value: str):
        record = safe_json_parse(value)
        if not record:
            return

        monitor_id = record.get("MONITORID")
        if not monitor_id:
            return

        # -------------------------------------------------------------
        # 1. Maintain sliding window
        # -------------------------------------------------------------
        if monitor_id not in self.windows:
            self.windows[monitor_id] = SlidingWindow(
                window_size=CONFIG.WINDOW_COUNT,
                slide_size=CONFIG.SLIDE_COUNT,
            )

        window = self.windows[monitor_id]
        window.add(record)

        # Wait until full
        if not window.is_full():
            return

        df = window.to_dataframe()

        # -------------------------------------------------------------
        # 2. Build model if missing → API training happens here
        # -------------------------------------------------------------
        if not model_exists(monitor_id):
            logger.warning(f"No model found for {monitor_id} → Training via API...")

            try:
                build_model_for_monitor(monitor_id, df=None,months=3)
            except TrainingFailedError as exc:
                logger.error(f"Training failed for {monitor_id}: {exc}")
                window.slide()
                return

        # -------------------------------------------------------------
        # 3. Load model bundle
        # -------------------------------------------------------------
        model, scaler, metadata = self.model_cache.get(monitor_id)

        feature_names = metadata.get("feature_names", [])
        if not feature_names:
            logger.error(f"Missing feature_names in metadata for {monitor_id}")
            window.slide()
            return

        # -------------------------------------------------------------
        # 4. Align DataFrame to model's expected features
        # -------------------------------------------------------------
        df = self._align_features(df, feature_names)

        # -------------------------------------------------------------
        # 5. Detect anomalies
        # -------------------------------------------------------------
        result = detect_anomalies(df, model, scaler, metadata)

        window.slide()

        if result.get("is_anomaly"):
            yield json.dumps(self._format_alert(monitor_id, result))

    # -------------------------------------------------------------
    # Force DF to match training-time feature order
    # -------------------------------------------------------------
    def _align_features(self, df: pd.DataFrame, feature_names: list):
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0

        df = df[feature_names]
        return df

    # -------------------------------------------------------------
    def _format_alert(self, monitor_id: str, result: Dict[str, Any]):
        return {
            "monitorId": monitor_id,
            "isAnomaly": True,
            "indices": result.get("anomaly_indices", []),
            "topFeatures": result.get("top_features", []),
            "modelMetadata": result.get("model_metadata", {}),
        }