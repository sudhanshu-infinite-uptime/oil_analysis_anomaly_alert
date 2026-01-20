"""
REALISTIC DEMO: Train + Predict using ONLY APIs (no hardcoded values)

✔ No hardcoded sensor values
✔ Model trained from Trend API v2
✔ Live data simulated using latest API record
✔ Model saved per monitorId
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.api.trend_api_client import TrendAPIClient
from app.config import MODEL_FEATURE_CODES_ORDERED

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE_IDENTIFIER = "AA:BB:CC:XX:YY:13"

START_DATETIME = "2026-01-06T15:00:00.000Z"
END_DATETIME   = "2026-01-06T16:00:00.000Z"

INTERVAL_VALUE = 1
INTERVAL_UNIT  = "seconds"

FEATURES = list(MODEL_FEATURE_CODES_ORDERED)

MODEL_DIR = "demo_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
print("\n=== TRAINING MODEL FROM TREND API v2 ===\n")

trend_client = TrendAPIClient()

records = trend_client.get_history(
    device_identifier=DEVICE_IDENTIFIER,
    feature_codes=FEATURES,
    start_datetime=START_DATETIME,
    end_datetime=END_DATETIME,
    interval_value=INTERVAL_VALUE,
    interval_unit=INTERVAL_UNIT,
)

if not records:
    raise RuntimeError("No trend data returned")

monitor_id = records[0]["MONITORID"]
print(f"Resolved MONITORID = {monitor_id}")

rows = [r["PROCESS_PARAMETER"] for r in records]
df = pd.DataFrame(rows)

# Ensure all features exist
for col in FEATURES:
    if col not in df.columns:
        df[col] = np.nan

df = df[FEATURES].astype(float)
df = df.ffill().bfill().fillna(0.0)

print("Training data shape:", df.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
)
model.fit(X_scaled)

model_path = os.path.join(MODEL_DIR, f"model_{monitor_id}.joblib")
scaler_path = os.path.join(MODEL_DIR, f"scaler_{monitor_id}.joblib")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"Model saved → {model_path}")
print(f"Scaler saved → {scaler_path}")

# --------------------------------------------------
# SIMULATE LIVE DATA (LATEST RECORD)
# --------------------------------------------------
print("\n=== SIMULATING LIVE DATA (LATEST RECORD) ===\n")

latest_record = records[-1]["PROCESS_PARAMETER"]

live_df = pd.DataFrame([latest_record])

for col in FEATURES:
    if col not in live_df.columns:
        live_df[col] = 0.0

live_df = live_df[FEATURES].astype(float)

X_live = scaler.transform(live_df)
prediction = model.predict(X_live)[0]

status = "ANOMALY" if prediction == -1 else "NORMAL"

print("Live input:")
print(live_df)
print("\nPrediction result:", status)

print("\n=== DEMO COMPLETED SUCCESSFULLY ===\n")
