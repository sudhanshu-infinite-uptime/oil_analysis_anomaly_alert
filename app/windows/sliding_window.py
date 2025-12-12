"""
windows/sliding_window.py

A reusable sliding window buffer used by the Flink operator to accumulate
a fixed number of sequential records per MONITORID before performing
anomaly detection.

Responsibilities:
    - Maintain an ordered buffer of the most recent `window_size` records
    - Add new entries
    - Detect when window is "full" (len == window_size)
    - Return window data as list or DataFrame
    - Slide forward by `slide_size` items (remove oldest rows)
    - Reset buffer if needed

This module should NOT:
    - Parse Kafka messages
    - Perform anomaly detection
    - Load or save ML models
    - Call external APIs
"""

from __future__ import annotations

from collections import deque
from typing import List, Dict, Any

import pandas as pd

from app.predictor.feature_mapping import MODEL_FEATURE_CODES, FEATURES


class SlidingWindow:
    """
    Simple sliding window structure.

    Example:
        window = SlidingWindow(window_size=10, slide_size=3)
        window.add(record)

        if window.is_full():
            df = window.to_dataframe()
            window.slide()
    """

    def __init__(self, window_size: int, slide_size: int):
        self.window_size = window_size
        self.slide_size = slide_size
        self.buffer = deque(maxlen=window_size)

    # -----------------------------------------------------------
    # Add new record
    # -----------------------------------------------------------
    def add(self, record: Dict[str, Any]) -> None:
        """Add a new Kafka record (dict) into the window buffer."""
        self.buffer.append(record)

    # -----------------------------------------------------------
    # Check if window is full
    # -----------------------------------------------------------
    def is_full(self) -> bool:
        """Returns True when the buffer is full (`len == window_size`)."""
        return len(self.buffer) == self.window_size

    # -----------------------------------------------------------
    # Convert window to DataFrame
    # -----------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the sliding window buffer into a pandas DataFrame
        containing only model-relevant features.

        Expected record format:
            {
                "PROCESS_PARAMETER": {
                    "001_A": value,
                    "001_B": value,
                    ...
                }
            }

        Returns:
            DataFrame with columns defined in FEATURES (canonical order).
        """
        rows = []

        for entry in self.buffer:
            params = entry.get("PROCESS_PARAMETER", {})
            row = {}

            # Extract only raw model input features (A–F, etc.)
            for code in MODEL_FEATURE_CODES:
                row[code] = params.get(code)

            rows.append(row)

        df = pd.DataFrame(rows)

        # Rename to canonical ML feature names
        df.columns = FEATURES

        return df

    # -----------------------------------------------------------
    # Slide the window (drop oldest items)
    # -----------------------------------------------------------
    def slide(self) -> None:
        """
        Remove the oldest `slide_size` entries from the buffer.

        Example:
            window_size = 10
            slide_size = 3

            After sliding, 3 oldest items are removed → 7 remain.
        """
        for _ in range(min(self.slide_size, len(self.buffer))):
            self.buffer.popleft()

    # -----------------------------------------------------------
    # Reset window
    # -----------------------------------------------------------
    def reset(self) -> None:
        """Clear all records from the window."""
        self.buffer.clear()
