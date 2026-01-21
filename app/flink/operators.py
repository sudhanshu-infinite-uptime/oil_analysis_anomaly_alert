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
    FINAL, IMMUTABLE MODEL BEHAVIOR

    • One model per monitor
    • Never retrain existing models
    • Never replace models
    • Train once, then always infer
    """

    def open(self, runtime_context: RuntimeContext):
        logger.info("Initializing MultiModelAnomalyOperator")

        self.device_client = DeviceAPIClient()
        self.model_cache = ModelCache(max_size=CONFIG.MODEL_CACHE_SIZE)

        self.windows: Dict[int, SlidingWindow] = {}
        self.training_state: Dict[int, str] = {}
        # States: NOT_STARTED | READY | FAILED

    def flat_map(self, value: str):
        record = safe_json_parse(value)
        if not record:
            return

        device_id = record.get("DEVICEID")
        if not device_id:
            return

        try:
            runtime_monitor_id = self.device_client.get_monitor_id_runtime(device_id)
        except Exception as exc:
            logger.error(
                "Device resolution failed | DEVICEID=%s | %s",
                device_id,
                exc,
            )
            return

        state = self.training_state.get(runtime_monitor_id, "NOT_STARTED")

        # --------------------------------------------------
        # ENSURE MODEL EXISTS (ONCE)
        # --------------------------------------------------
        if state == "NOT_STARTED":

            if model_exists(str(runtime_monitor_id)):
                # Validate by loading once
                try:
                    self.model_cache.get(runtime_monitor_id)
                    self.training_state[runtime_monitor_id] = "READY"
                    logger.info(
                        "Using existing model | MONITORID=%s",
                        runtime_monitor_id,
                    )
                except Exception as exc:
                    self.training_state[runtime_monitor_id] = "FAILED"
                    logger.error(
                        "Existing model failed to load | MONITORID=%s | %s",
                        runtime_monitor_id,
                        exc,
                    )
                    return

            else:
                logger.info(
                    "No model found → training ONCE | DEVICEID=%s",
                    device_id,
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
                    self.training_state[runtime_monitor_id] = "FAILED"
                    logger.error(
                        "Model training failed | DEVICEID=%s | %s",
                        device_id,
                        exc,
                    )
                    return

                # Re-validate via cache load
                try:
                    self.model_cache.get(runtime_monitor_id)
                    self.training_state[runtime_monitor_id] = "READY"
                    logger.info(
                        "Model ready for inference | MONITORID=%s",
                        runtime_monitor_id,
                    )
                except Exception as exc:
                    self.training_state[runtime_monitor_id] = "FAILED"
                    logger.error(
                        "Training completed but model load failed | MONITORID=%s | %s",
                        runtime_monitor_id,
                        exc,
                    )
                    return

        if self.training_state.get(runtime_monitor_id) != "READY":
            logger.debug(
                "Model not ready, skipping record | MONITORID=%s",
                runtime_monitor_id,
            )
            return

        # --------------------------------------------------
        # SLIDING WINDOW
        # --------------------------------------------------
        if runtime_monitor_id not in self.windows:
            self.windows[runtime_monitor_id] = SlidingWindow(
                window_size=CONFIG.WINDOW_COUNT,
                slide_size=CONFIG.SLIDE_COUNT,
            )

        window = self.windows[runtime_monitor_id]
        window.add(record)

        if not window.is_full():
            return

        # --------------------------------------------------
        # LOAD MODEL
        # --------------------------------------------------
        try:
            model, scaler, metadata = self.model_cache.get(runtime_monitor_id)
        except Exception as exc:
            logger.error(
                "Model load failed | MONITORID=%s | %s",
                runtime_monitor_id,
                exc,
            )
            window.slide()
            return

        df = self._align_features(
            window.to_dataframe(),
            metadata["feature_names"],
        )

        # --------------------------------------------------
        # INFERENCE
        # --------------------------------------------------
        try:
            result = detect_anomalies(df, model, scaler, metadata)
        except Exception as exc:
            logger.error(
                "Inference failed | MONITORID=%s | %s",
                runtime_monitor_id,
                exc,
            )
            window.slide()
            return

        window.slide()

        if result.get("is_anomaly"):
            yield json.dumps(
                {
                    "monitorId": runtime_monitor_id,
                    "isAnomaly": True,
                    "indices": result.get("anomaly_indices", []),
                    "modelMetadata": result.get("model_metadata", {}),
                }
            )

    def _align_features(self, df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].astype(float)
        return df[feature_names]
