**🧠 Real-Time Multi-Model Anomaly Detection Pipeline**

**📘 Overview**<br>

This project implements a real-time anomaly detection pipeline that processes multi-sensor data using Apache Flink, retrieves & stores ML models in AWS S3, and publishes anomaly alerts through Kafka.
Each monitor has its own machine learning model, enabling highly accurate multi-device anomaly detection.


### 🧩 Key Features

⚡ Real-time streaming inference using PyFlink

🧠 Per-monitor ML models stored in AWS S3

📦 Automatic model building & updating inside the Flink pipeline

🔄 Sliding-window computation for anomaly trends

🛰 Kafka-based ingestion + alert publishing

🐳 Fully containerized using Docker Compose<br>

 

### ⚙️ Prerequisites<br>

Install the following before running:

**🐋 Docker Desktop**

🐍 Python 3.10+

**🔑 AWS IAM user with S3 read/write permissions**

**📦 Kafka & Flink provided via Docker Compose**<br>



### 📦 Dependencies

All Python libs are installed inside the Docker image.

🚀 Run the Pipeline
🧰 Step 1 — Build Docker Image <br>
docker compose build --no-cache

🔥 Step 2 — Start Flink + Kafka + Zookeeper <br>
docker compose up -d

Check containers: <br>
docker ps

Step 3- run send_test_message.py file <br>
docker exec -it flink-jobmanager python /opt/flink/app/tools/send_test_message.py

Step 4- check the models folder