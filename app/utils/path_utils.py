"""
app/utils/path_utils.py

Utility functions for working with filesystem paths.

Responsibilities:
    - Create directories safely
    - Build consistent file paths for models, metadata, logs, etc.
    - Provide atomic write helpers to prevent partial file corruption
    - Normalize paths to avoid mistakes (strings vs Path objects)

This module is used by:
    - model_store.py
    - model_loader.py
    - model_builder.py
    - logging utilities
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Union, Optional

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------------
# Path normalization
# -------------------------------------------------------------------
def to_path(path: Union[str, Path]) -> Path:
    """
    Normalize a string or Path into a pathlib.Path instance.

    Args:
        path: String or Path

    Returns:
        A pathlib.Path object.

    Why:
        Helps ensure all modules use Path consistently.
    """
    return Path(path) if not isinstance(path, Path) else path


# -------------------------------------------------------------------
# Directory helpers
# -------------------------------------------------------------------
def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists. If not, create it.

    Args:
        path: Directory path.

    Returns:
        The pathlib.Path for the directory.

    Notes:
        - Safe for repeated calls
        - Logs creation on first-time creation
    """
    p = to_path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error(f"Failed to create directory {p}: {exc}")
        raise
    return p


# -------------------------------------------------------------------
# File building helpers (model paths, metadata paths, etc.)
# -------------------------------------------------------------------
def build_model_dir(base_dir: Union[str, Path], monitor_id: str) -> Path:
    """
    Returns the directory where a model for the given MONITORID should be stored.

    Example:
        /models/MON123/
    """
    path = to_path(base_dir) / monitor_id
    ensure_dir(path)
    return path


def build_file_path(base_dir: Union[str, Path], *parts: str) -> Path:
    """
    Create a full file path inside a base directory.

    Example:
        build_file_path("/models/MON123", "iforest_model.pkl")
    """
    p = to_path(base_dir).joinpath(*parts)
    ensure_dir(p.parent)
    return p


# -------------------------------------------------------------------
# Safe atomic write (prevents corrupted files)
# -------------------------------------------------------------------
def atomic_write(path: Union[str, Path], data: bytes) -> None:
    """
    Write binary data to a file atomically.

    Steps:
        - Write to temporary file
        - Move temp file → final file (os.replace)
        - Guarantees file is never partially written

    Args:
        path: Target file path
        data: Bytes to write

    Notes:
        This is the safest way to write model files or metadata.
    """
    path = to_path(path)
    ensure_dir(path.parent)

    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=path.parent) as tmp:
            tmp.write(data)
            temp_path = Path(tmp.name)

        os.replace(temp_path, path)  # atomic move on most OSes
        logger.debug(f"Atomic write completed → {path}")

    except Exception as exc:
        logger.error(f"Atomic write failed for {path}: {exc}")
        raise
