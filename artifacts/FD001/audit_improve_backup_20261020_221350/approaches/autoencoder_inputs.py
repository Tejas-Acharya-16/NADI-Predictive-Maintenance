"""Healthy-reference window preparation for an optional anomaly autoencoder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmapss_data import SENSOR_COLUMNS


def build_early_life_windows(
    frame: pd.DataFrame,
    early_fraction: float = 0.2,
    window_size: int = 20,
    sensors: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Prepare early-life training windows, never mixing rows between engines.

    The fraction is defined relative to each complete *training* trajectory.
    Use only development engines for fitting an autoencoder; held-out engines
    must remain validation data.
    """
    if not 0 < early_fraction <= 1:
        raise ValueError("early_fraction must be in (0, 1].")
    if window_size < 2:
        raise ValueError("window_size must be >= 2.")
    sensor_columns = SENSOR_COLUMNS if sensors is None else sensors
    missing = sorted({"unit", "cycle", *sensor_columns} - set(frame.columns))
    if missing:
        raise ValueError(f"Input frame is missing required columns: {missing}")

    windows: list[np.ndarray] = []
    metadata: list[dict] = []
    ordered = frame.sort_values(["unit", "cycle"], kind="stable")
    for unit, group in ordered.groupby("unit", sort=True):
        first_cycle = int(group["cycle"].min())
        last_cycle = int(group["cycle"].max())
        cutoff = first_cycle + int(np.floor((last_cycle - first_cycle) * early_fraction))
        early = group.loc[group["cycle"] <= cutoff]
        values = early[sensor_columns].to_numpy(dtype=np.float32)
        cycles = early["cycle"].to_numpy()
        for end in range(window_size - 1, len(early)):
            start = end - window_size + 1
            windows.append(values[start : end + 1])
            metadata.append(
                {"unit": unit, "start_cycle": cycles[start], "end_cycle": cycles[end]}
            )

    shape = (0, window_size, len(sensor_columns))
    X = np.stack(windows).astype(np.float32, copy=False) if windows else np.empty(shape, dtype=np.float32)
    return X, pd.DataFrame(metadata)
