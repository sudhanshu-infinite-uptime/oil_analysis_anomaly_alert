Kafka → Flink Operator → Sliding Window → Model Cache → Model Loader/Builder 
      → Preprocessing → Scaler → Model Prediction → Anomaly Detector 
      → Alert JSON → Kafka (alert topic)

Trend API → Model Builder → Model Store → Models Folder


OIL_ANOMALY_PIPELINE/
│
├── app/
│   ├── api/
│   ├── flink/
│   ├── models/
│   ├── predictor/
│   ├── utils/
│   ├── config.py
│   └── main.py
│
├── docker/
│
├── logs/                <-- correct (for runtime logs)
│
├── models/              <-- correct (runtime trained models)
│
├── scripts/
│   ├── benchmark_training.py
│   ├── build_local_model.py
│   └── replay_kafka_messages.py
│
├── tests/
│   ├── test_anomaly_detector.py
│   ├── test_integration_pipeline.py
│   ├── test_model_builder.py
│   ├── test_model_loader.py
│   ├── test_sliding_window.py
│   └── test_trend_api_client.py
│
├── windows/
│   └── sliding_window.py
│
├── .gitignore
├── pyproject.toml
└── README.md
|__ requirements.txt


build_model_for_monitor(monitor_id, df=None, months=3)
