"""
app/utils/exceptions.py

Defines custom exceptions used across the Oil Anomaly Detection Pipeline.

Why custom exceptions?
    - Makes error handling cleaner and more descriptive
    - Enables domain-specific errors (model not found, training failure, etc.)
    - Improves log clarity
    - Avoids generic ValueError / RuntimeError everywhere

Used by:
    - model_store.py
    - model_loader.py
    - model_builder.py
    - anomaly_detector.py
    - flink/operators.py
"""

from __future__ import annotations


# -------------------------------------------------------------------
# Base Project Exception
# -------------------------------------------------------------------
class PipelineError(Exception):
    """Base class for all custom exceptions in the project."""
    pass


# -------------------------------------------------------------------
# Model Loading Errors
# -------------------------------------------------------------------
class ModelNotFoundError(PipelineError):
    """Raised when a model or scaler file is missing."""

    def __init__(self, monitor_id: str, path: str):
        super().__init__(
            f"Model files not found for MONITORID={monitor_id}. Path: {path}"
        )
        self.monitor_id = monitor_id
        self.path = path


class ModelLoadError(PipelineError):
    """Raised when a model or scaler exists but fails to load."""

    def __init__(self, monitor_id: str, reason: str):
        super().__init__(
            f"Failed to load model for MONITORID={monitor_id}. Reason: {reason}"
        )
        self.monitor_id = monitor_id
        self.reason = reason


# -------------------------------------------------------------------
# Training Errors
# -------------------------------------------------------------------
class ModelTrainingError(PipelineError):
    """General training failure exception."""

    def __init__(self, monitor_id: str, reason: str):
        super().__init__(
            f"Model training failed for MONITORID={monitor_id}. Reason: {reason}"
        )
        self.monitor_id = monitor_id
        self.reason = reason


class TrainingFailedError(ModelTrainingError):
    """
    Raised when training fails due to invalid data, insufficient samples,
    preprocessing errors, or other recoverable training issues.
    """
    def __init__(self, monitor_id: str, reason: str):
        super().__init__(monitor_id, reason)


# -------------------------------------------------------------------
# API Errors
# -------------------------------------------------------------------
class APICallError(PipelineError):
    """Raised when Trend API fails or returns bad content."""

    def __init__(self, url: str, status: int, message: str = ""):
        msg = f"API call failed: {url} (status: {status})"
        if message:
            msg += f" | Message: {message}"

        super().__init__(msg)
        self.url = url
        self.status = status
        self.message = message


# -------------------------------------------------------------------
# Sliding Window Errors
# -------------------------------------------------------------------
class WindowError(PipelineError):
    """Raised for invalid operations performed on sliding windows."""

    def __init__(self, reason: str):
        super().__init__(f"Sliding window error: {reason}")
        self.reason = reason
