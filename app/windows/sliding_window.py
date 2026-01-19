"""
app/windows/sliding_window.py

Sliding window buffer for streaming anomaly detection.

Responsibilities:
- Maintain a fixed-size sliding buffer per MONITORID
- Convert buffered Kafka records into a clean ML-ready DataFrame
- Enforce strict feature presence and ordering
- Never leak schema drift into the model

Design principles:
- Stateless outside buffer
- Deterministic feature alignment
- Safe for Flink distributed execution
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Any, List

import pandas as pd

from app.config import MODEL_FEATURE_CODES_ORDERED


class SlidingWindow:
    """
    Fixed-size sliding window for streaming ML inference.

    Each window holds the most recent `window_size` records.
    After inference, the window slides forward by `slide_size`.
    """

    def __init__(self, window_size: int, slide_size: int):
        self.window_size = window_size
        self.slide_size = slide_size
        self.buffer: deque[Dict[str, Any]] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    def add(self, record: Dict[str, Any]) -> None:
        """
        Add a new Kafka record to the window buffer.

        Expected record shape:
        {
            "DEVICEID": "...",
            "001_A": ...,
            "001_B": ...,
            ...
        }
        """
        self.buffer.append(record)

    # ------------------------------------------------------------------
    def is_full(self) -> bool:
        """Return True when window is ready for inference."""
        return len(self.buffer) == self.window_size

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert buffered records into a strictly-aligned DataFrame.

        Guarantees:
        - Columns == MODEL_FEATURE_CODES_ORDERED
        - Missing values filled with 0.0
        - dtype = float
        """
        rows: List[Dict[str, float]] = []

        for entry in self.buffer:
            row = {}

            for code in MODEL_FEATURE_CODES_ORDERED:
                value = entry.get(code)
                row[code] = float(value) if value not in (None, "", "null") else 0.0

            rows.append(row)

        df = pd.DataFrame(rows, columns=MODEL_FEATURE_CODES_ORDERED)
        df = df.astype(float)

        return df

    # ------------------------------------------------------------------
    def slide(self) -> None:
        """Slide window forward by removing oldest records."""
        for _ in range(min(self.slide_size, len(self.buffer))):
            self.buffer.popleft()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the entire window buffer."""
        self.buffer.clear()
