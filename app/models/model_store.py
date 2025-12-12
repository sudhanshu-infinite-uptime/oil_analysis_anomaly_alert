import boto3
import json
import pickle
from botocore.exceptions import ClientError
from pathlib import Path
from app.config import CONFIG

# No region needed
S3_CLIENT = boto3.client("s3")

def s3_key(monitor_id: str, filename: str) -> str:
    return f"models/{monitor_id}/{filename}"

# Save binary file to S3
def save_binary(path: str, data: bytes):
    monitor_id = Path(path).parent.name
    filename = Path(path).name
    key = s3_key(monitor_id, filename)
    S3_CLIENT.put_object(Bucket=CONFIG.S3_BUCKET_NAME, Key=key, Body=data)

# Load binary file from S3
def load_binary(path: str) -> bytes:
    monitor_id = Path(path).parent.name
    filename = Path(path).name
    key = s3_key(monitor_id, filename)
    obj = S3_CLIENT.get_object(Bucket=CONFIG.S3_BUCKET_NAME, Key=key)
    return obj["Body"].read()

# Save metadata JSON
def save_metadata(monitor_id: str, metadata: dict):
    key = s3_key(monitor_id, "metadata.json")
    S3_CLIENT.put_object(
        Bucket=CONFIG.S3_BUCKET_NAME,
        Key=key,
        Body=json.dumps(metadata).encode("utf-8")
    )

# Load metadata JSON
def load_metadata(monitor_id: str) -> dict:
    key = s3_key(monitor_id, "metadata.json")
    obj = S3_CLIENT.get_object(Bucket=CONFIG.S3_BUCKET_NAME, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

# Check if a model exists
def model_exists(monitor_id: str) -> bool:
    key = s3_key(monitor_id, "model.pkl")
    try:
        S3_CLIENT.head_object(Bucket=CONFIG.S3_BUCKET_NAME, Key=key)
        return True
    except ClientError:
        return False

# Virtual model paths (Flink will pass these)
def get_model_paths(monitor_id: str):
    return {
        "model_path": f"{monitor_id}/model.pkl",
        "scaler_path": f"{monitor_id}/scaler.pkl",
        "metadata_path": f"{monitor_id}/metadata.json"
    }
