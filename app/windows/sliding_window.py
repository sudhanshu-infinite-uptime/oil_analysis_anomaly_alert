"""
windows/sliding_window.py

A reusable sliding window buffer used by the Flink operator to accumulate
a fixed number of sequential records per MONITORID before performing
anomaly detection.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Any

import pandas as pd

from app.predictor.feature_mapping import MODEL_FEATURE_CODES
from app.config import FEATURES


class SlidingWindow:
    """
    Simple sliding window structure.
    """

    def __init__(self, window_size: int, slide_size: int):
        self.window_size = window_size
        self.slide_size = slide_size
        self.buffer = deque(maxlen=window_size)

    def add(self, record: Dict[str, Any]) -> None:
        """Add a new Kafka record into the window buffer."""
        self.buffer.append(record)

    def is_full(self) -> bool:
        """Return True when the buffer is full."""
        return len(self.buffer) == self.window_size

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert window buffer into a DataFrame aligned with ML features.
        """
        rows = []

        for entry in self.buffer:
            params = entry.get("PROCESS_PARAMETER", {})
            row = {}

            for code in MODEL_FEATURE_CODES:
                row[code] = params.get(code)

            rows.append(row)

        df = pd.DataFrame(rows)
        df.columns = FEATURES
        return df

    def slide(self) -> None:
        """Remove the oldest `slide_size` records."""
        for _ in range(min(self.slide_size, len(self.buffer))):
            self.buffer.popleft()

    def reset(self) -> None:
        """Clear all records from the window."""
        self.buffer.clear()
