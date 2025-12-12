"""
app/utils/logging_utils.py

Centralized logging setup for the Oil Anomaly Detection Pipeline.

Responsibilities:
    - Provide a consistent logger for every module
    - Write logs to rotating log files inside CONFIG.LOG_DIR
    - Write readable logs to console (STDOUT)
    - Prevent duplicate handlers when modules import logger multiple times
    - Create log directory if missing

This module should be imported by every module as:
    from app.utils.logging_utils import get_logger
    logger = get_logger(__name__)

It does NOT:
    - Perform anomaly detection
    - Load models
    - Make API requests
"""

from __future__ import annotations
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from app.config import CONFIG

# INTERNAL: Create log directory if it does not exist
LOG_DIR:Path=CONFIG.LOG_DIR

try:
    LOG_DIR.mkdir(parents=True,exist_ok=True)
except Exception:
    pass

# INTERNAL: Formatters
FILE_FORMAT = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

CONSOLE_FORMAT = logging.Formatter(
    "%(levelname)s | %(message)s"
)

class ColorFormatter(logging.Formatter):
    """Optional: adds ANSI colors for console output for better readability."""
    COLORS = {
        "DEBUG": "\033[94m",     # Blue
        "INFO": "\033[92m",      # Green
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

# MAIN LOGGER FUNCTION (used by all modules)
def get_logger(name: str) -> logging.Logger:
    """
    Create or return an existing logger with:
        - Rotating file handler
        - Console handler (with optional color)
        - No duplicate handlers on repeated imports

    Args:
        name: Name of the logger, usually __name__.

    Returns:
        A fully configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG if CONFIG.DEBUG else logging.INFO)

    # -----------------------------
    # Console handler
    # -----------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter("%(levelname)s | %(message)s"))
    logger.addHandler(console_handler)

    # -----------------------------
    # Rotating file handler (per module)
    # Logs go to: logs/<module_name>.log
    # -----------------------------
    safe_name = name.replace(".", "_")  # log file must be filesystem-safe
    file_path = LOG_DIR / f"{safe_name}.log"

    file_handler = RotatingFileHandler(
        filename=file_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,             # keep 3 backups
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(FILE_FORMAT)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoids duplicate console logs)
    logger.propagate = False

    return logger