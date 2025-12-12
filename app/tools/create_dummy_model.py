import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from app.models.model_store import get_model_paths
from app.models.model_metadata import create_metadata, save_metadata
from app.utils.logging_utils import get_logger
from app.config import FEATURES

logger = get_logger(__name__)


def create_dummy_model(monitor_id="TEST_MONITOR"):
    print(f"➡ Creating dummy model for {monitor_id}")

    # ------------------------------------------------------------
    # 1. Generate fake training data (100 samples, 6 features)
    # ------------------------------------------------------------
    X = np.random.normal(50, 10, size=(100, len(FEATURES)))

    # ------------------------------------------------------------
    # 2. Fit RobustScaler + Isolation Forest
    # ------------------------------------------------------------
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)

    # ------------------------------------------------------------
    # 3. Get model, scaler & metadata paths
    # ------------------------------------------------------------
    paths = get_model_paths(monitor_id)
    model_dir = paths["model_path"].parent

    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created model directory → {model_dir}")

    # ------------------------------------------------------------
    # 4. Save MODEL using joblib.dump (writes direct to file)
    # ------------------------------------------------------------
    joblib.dump(model, paths["model_path"])
    joblib.dump(scaler, paths["scaler_path"])

    # ------------------------------------------------------------
    # 5. Save metadata
    # ------------------------------------------------------------
    metadata = create_metadata(
        monitor_id=monitor_id,
        contamination=0.05,
        feature_list=FEATURES,
        n_samples=100,
        api_months=0,  # dummy
    )
    save_metadata(paths["metadata_path"], metadata)

    print("✅ Dummy model created successfully!")
    print(paths)


if __name__ == "__main__":
    create_dummy_model()
