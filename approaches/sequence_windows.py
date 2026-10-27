"""Fixed-length, past-only sequence windows for future CNN/RNN comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmapss_data import SENSOR_COLUMNS


def build_sequence_windows(
    frame: pd.DataFrame,
    window_size: int = 30,
    sensors: list[str] | None = None,
    label_column: str | None = "rul",
    stride: int = 1,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray | None]:
    """Return full trailing windows and endpoint metadata without crossing engines."""
    if window_size < 2 or stride < 1:
        raise ValueError("window_size must be >= 2 and stride must be >= 1.")
    sensor_columns = SENSOR_COLUMNS if sensors is None else sensors
    required = {"unit", "cycle", *sensor_columns}
    if label_column is not None:
        required.add(label_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input frame is missing required columns: {missing}")

    sequences: list[np.ndarray] = []
    metadata: list[dict] = []
    labels: list[float] = []
    has_label = label_column is not None
    for unit, group in frame.sort_values(["unit", "cycle"], kind="stable").groupby(
        "unit", sort=True
    ):
        values = group[sensor_columns].to_numpy(dtype=np.float32)
        cycles = group["cycle"].to_numpy()
        target = group[label_column].to_numpy() if has_label else None
        for end in range(window_size - 1, len(group), stride):
            start = end - window_size + 1
            sequences.append(values[start : end + 1])
            metadata.append(
                {
                    "unit": unit,
                    "start_cycle": cycles[start],
                    "end_cycle": cycles[end],
                }
            )
            if target is not None:
                labels.append(float(target[end]))

    shape = (0, window_size, len(sensor_columns))
    X = np.stack(sequences).astype(np.float32, copy=False) if sequences else np.empty(shape, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32) if has_label else None
    return X, pd.DataFrame(metadata), y
