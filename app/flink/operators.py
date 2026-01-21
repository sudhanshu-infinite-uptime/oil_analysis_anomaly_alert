from __future__ import annotations

import json
from typing import Dict, Any
from datetime import datetime, timedelta, timezone

import pandas as pd
from pyflink.datastream.functions import FlatMapFunction, RuntimeContext

from app.api.device_api_client import DeviceAPIClient
from app.models.model_cache import ModelCache
from app.models.model_store import model_exists
from app.models.model_builder import build_model_for_device_v2
from app.predictor.anomaly_detector import detect_anomalies
from app.windows.sliding_window import SlidingWindow
from app.utils.json_utils import safe_json_parse
from app.utils.logging_utils import get_logger
from app.config import CONFIG

logger = get_logger(__name__)


class MultiModelAnomalyOperator(FlatMapFunction):
    """
    Flink operator for multi-monitor anomaly detection.

    Responsibilities:
    - Resolve DEVICEID → MONITORID (runtime)
    - Train model ONCE per monitor (Trend API v2)
    - Maintain per-monitor sliding windows
    - Run anomaly detection
    - Emit alerts
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self, runtime_context: RuntimeContext):
        logger.info("Initializing MultiModelAnomalyOperator")

        self.device_client = DeviceAPIClient()
        self.model_cache = ModelCache(max_size=CONFIG.MODEL_CACHE_SIZE)

        # Per-monitor runtime state
        self.windows: Dict[int, SlidingWindow] = {}
        self.training_state: Dict[int, str] = {}
        # States: NOT_STARTED | IN_PROGRESS | READY | FAILED

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------
    def flat_map(self, value: str):
        record = safe_json_parse(value)
        if not record:
            return

        device_id = record.get("DEVICEID")
        if not device_id:
            return

        # ------------------------------------------------------------
        # Resolve device → monitor
        # ------------------------------------------------------------
        try:
            monitor_id = self.device_client.get_monitor_id_runtime(device_id)
        except Exception as exc:
            logger.error(
                "Device resolution failed | DEVICEID=%s | %s",
                device_id,
                exc,
            )
            return

        # ------------------------------------------------------------
        # Training state machine (ONCE per monitor)
        # ------------------------------------------------------------
        state = self.training_state.get(monitor_id, "NOT_STARTED")

        if state == "READY":
            pass

        elif state in ("IN_PROGRESS", "FAILED"):
            return

        elif state == "NOT_STARTED":
            if model_exists(monitor_id):
                self.training_state[monitor_id] = "READY"
                logger.info(
                    "Model already exists | MONITORID=%s",
                    monitor_id,
                )
            else:
                self.training_state[monitor_id] = "IN_PROGRESS"
                logger.info(
                    "Model missing → training ONCE | DEVICEID=%s | MONITORID=%s",
                    device_id,
                    monitor_id,
                )

                try:
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(days=30)

                    build_model_for_device_v2(
                        device_id=device_id,
                        start_datetime=start_time.isoformat().replace("+00:00", "Z"),
                        end_datetime=end_time.isoformat().replace("+00:00", "Z"),
                    )

                except Exception as exc:
                    self.training_state[monitor_id] = "FAILED"
                    logger.error(
                        "Model training failed permanently | MONITORID=%s | %s",
                        monitor_id,
                        exc,
                    )
                    return

                if model_exists(monitor_id):
                    self.training_state[monitor_id] = "READY"
                    logger.info(
                        "Model training successful | MONITORID=%s",
                        monitor_id,
                    )
                else:
                    self.training_state[monitor_id] = "FAILED"
                    logger.error(
                        "Training completed but model missing | MONITORID=%s",
                        monitor_id,
                    )
                    return

        # ------------------------------------------------------------
        # Sliding window setup
        # ------------------------------------------------------------
        if monitor_id not in self.windows:
            self.windows[monitor_id] = SlidingWindow(
                window_size=CONFIG.WINDOW_COUNT,
                slide_size=CONFIG.SLIDE_COUNT,
            )

        window = self.windows[monitor_id]
        window.add(record)

        if not window.is_full():
            return

        # ------------------------------------------------------------
        # Load model bundle
        # ------------------------------------------------------------
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
                "Invalid model metadata | MONITORID=%s",
                monitor_id,
            )
            window.slide()
            return

        # ------------------------------------------------------------
        # Prepare data
        # ------------------------------------------------------------
        df = window.to_dataframe()
        df = self._align_features(df, feature_names)

        # ------------------------------------------------------------
        # Inference
        # ------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _align_features(
        self,
        df: pd.DataFrame,
        feature_names: list,
    ) -> pd.DataFrame:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].astype(float)
        return df[feature_names]

    def _format_alert(
        self,
        monitor_id: int,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "monitorId": monitor_id,
            "isAnomaly": True,
            "indices": result.get("anomaly_indices", []),
            "modelMetadata": result.get("model_metadata", {}),
        }
