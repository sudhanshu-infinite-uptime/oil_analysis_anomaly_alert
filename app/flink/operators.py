"""
app/flink/operators.py

MultiModelAnomalyOperator (On-Demand Training + Inference)

Responsibilities:
- Consume Kafka records
- Resolve deviceId → monitorId
- Train model on-demand if missing
- Maintain per-monitor sliding windows
- Run anomaly detection
- Emit alerts

Design:
- Event-driven
- No startup bootstrap
- Safe for streaming systems
"""

from __future__ import annotations

import json
import pandas as pd
from typing import Dict, Any

from pyflink.datastream.functions import FlatMapFunction, RuntimeContext

from app.api.device_api_client import DeviceAPIClient
from app.models.model_cache import ModelCache
from app.models.model_store import model_exists
from app.models.model_builder import build_model_for_monitor
from app.predictor.anomaly_detector import detect_anomalies
from app.windows.sliding_window import SlidingWindow

from app.utils.json_utils import safe_json_parse
from app.utils.logging_utils import get_logger
from app.config import CONFIG

logger = get_logger(__name__)


class MultiModelAnomalyOperator(FlatMapFunction):

    def open(self, runtime_context: RuntimeContext):
        logger.info("Initializing MultiModelAnomalyOperator")
        self.model_cache = ModelCache(max_size=CONFIG.MODEL_CACHE_SIZE)
        self.windows: Dict[int, SlidingWindow] = {}
        self.device_client = DeviceAPIClient()

    def flat_map(self, value: str):
        record = safe_json_parse(value)
        if not record:
            return

        device_id = record.get("DEVICEID")
        if not device_id:
            return

        # ---------------------------------------------------------
        # Resolve device → monitor
        # ---------------------------------------------------------
        try:
            monitor_id, parameter_group_id = (
                self.device_client.get_monitor_and_pg(device_id)
            )
        except Exception as exc:
            logger.error(
                "Device resolution failed | DEVICEID=%s | %s",
                device_id,
                exc,
            )
            return

        # ---------------------------------------------------------
        # Train model ON-DEMAND if missing
        # ---------------------------------------------------------
        if not model_exists(monitor_id):
            logger.info(
                "Model missing → training | DEVICEID=%s | MONITORID=%s",
                device_id,
                monitor_id,
            )
            try:
                build_model_for_monitor(
                    monitor_id=monitor_id,
                    parameter_group_id=parameter_group_id,
                    start_datetime=CONFIG.TRAIN_START_TIME,
                    end_datetime=CONFIG.TRAIN_END_TIME,
                )
            except Exception as exc:
                logger.error(
                    "Model training failed | MONITORID=%s | %s",
                    monitor_id,
                    exc,
                )
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
        # Load model
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

        # ---------------------------------------------------------
        # Prepare data
        # ---------------------------------------------------------
        feature_names = metadata.get("feature_names")
        if not feature_names:
            logger.error(
                "Invalid metadata | MONITORID=%s",
                monitor_id,
            )
            window.slide()
            return

        df = window.to_dataframe()
        df = self._align_features(df, feature_names)

        # ---------------------------------------------------------
        # Inference
        # ---------------------------------------------------------
        try:
            result = detect_anomalies(df, model, scaler, metadata)
        except Exception as exc:
            logger.error(
                "Inference failed | MONITORID=%s | %s",
                monitor_id,
                exc,
            )
            window.slide()
            return

        window.slide()

        if result.get("is_anomaly"):
            logger.warning(
                "Anomaly detected | MONITORID=%s",
                monitor_id,
            )
            yield json.dumps(self._format_alert(monitor_id, result))

    # ---------------------------------------------------------
    def _align_features(self, df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0
        return df[feature_names]

    # ---------------------------------------------------------
    def _format_alert(self, monitor_id: int, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "monitorId": monitor_id,
            "isAnomaly": True,
            "indices": result.get("anomaly_indices", []),
            "modelMetadata": result.get("model_metadata", {}),
        }
