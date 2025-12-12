FROM apache/flink:1.19.0-scala_2.12

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ------------------------------------------------------------
# Install Python + build tools
# ------------------------------------------------------------
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip python3-dev build-essential curl netcat-openbsd && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/flink

# ------------------------------------------------------------
# Install PyFlink (without dependencies)
# ------------------------------------------------------------
RUN pip install --no-cache-dir apache-flink==1.19.0 --no-deps

# ------------------------------------------------------------
# PyFlink Required Dependencies (correct versions)
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    py4j==0.10.9.7 \
    apache-beam==2.54.0 \
    avro-python3==1.10.2 \
    fastavro==1.7.4 \
    protobuf==3.20.3

# ------------------------------------------------------------
# ML + Utility Dependencies
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    pyarrow==12.0.1 \
    joblib==1.4.2 \
    kafka-python==2.0.2 \
    requests==2.32.3 \
    grpcio==1.65.0 \
    googleapis-common-protos==1.63.0 \
    google-cloud-core==2.4.1 \
    google-cloud-bigquery-storage==2.25.0

# install boto3 for S3
RUN pip install boto3


# ------------------------------------------------------------
# Kafka Connector JARs
# ------------------------------------------------------------
RUN mkdir -p /opt/flink/lib && \
    curl -fsSL -o /opt/flink/lib/flink-connector-kafka-3.2.0-1.19.jar \
        https://repo1.maven.org/maven2/org/apache/flink/flink-connector-kafka/3.2.0-1.19/flink-connector-kafka-3.2.0-1.19.jar && \
    curl -fsSL -o /opt/flink/lib/kafka-clients-3.5.1.jar \
        https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.5.1/kafka-clients-3.5.1.jar

# ------------------------------------------------------------
# Copy application code
# ------------------------------------------------------------
COPY app /opt/flink/app
COPY models /opt/flink/models
COPY start-flink.sh /opt/flink/start-flink.sh

RUN chmod +x /opt/flink/start-flink.sh && \
    chown -R flink:flink /opt/flink

# ------------------------------------------------------------
# Logging directory permissions
# ------------------------------------------------------------
RUN mkdir -p /opt/flink/logs && \
    chmod -R 777 /opt/flink/logs

# ------------------------------------------------------------
# Environment variables (critical for PyFlink)
# ------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/flink/app:/opt/flink" \
    FLINK_PYTHON_EXECUTABLE=/usr/bin/python3 \
    PYFLINK_CLIENT_EXECUTABLE=/usr/bin/python3 \
    PYFLINK_CREATE_VENV=false

# Trick PyFlink into using system Python
RUN ln -sf /usr/bin/python3 /opt/flink/bin/python3

USER flink

CMD ["/opt/flink/start-flink.sh"]
