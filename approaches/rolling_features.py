"""Causal, per-engine rolling features for tabular C-MAPSS RUL experiments."""

from __future__ import annotations

import pandas as pd

from cmapss_data import SENSOR_COLUMNS, SETTING_COLUMNS


def build_causal_features(
    frame: pd.DataFrame,
    windows: tuple[int, ...] = (5, 20),
    sensors: list[str] | None = None,
) -> pd.DataFrame:
    """Build current-value, rolling-statistic, delta, and trend features.

    Every feature uses the current cycle and earlier observations from the same
    engine. No centered windows or future observations are used.
    """
    if not windows or any(window < 2 for window in windows):
        raise ValueError("Each rolling window must contain at least two cycles.")
    sensor_columns = SENSOR_COLUMNS if sensors is None else sensors
    missing = sorted(set(["unit", "cycle", *sensor_columns, *SETTING_COLUMNS]) - set(frame.columns))
    if missing:
        raise ValueError(f"Input frame is missing required columns: {missing}")

    ordered = frame.sort_values(["unit", "cycle"], kind="stable").copy()
    groups = ordered.groupby("unit", sort=False)
    feature_columns = {
        column: ordered[column].to_numpy()
        for column in ["unit", "cycle", *SETTING_COLUMNS]
    }
    feature_columns["cycles_seen"] = (groups.cumcount() + 1).to_numpy()

    for sensor in sensor_columns:
        feature_columns[sensor] = ordered[sensor].to_numpy()
        sensor_group = ordered.groupby("unit", sort=False)[sensor]
        feature_columns[f"{sensor}_delta_1"] = sensor_group.diff(1).fillna(0).to_numpy()
        for window in windows:
            rolling_mean = sensor_group.transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            )
            rolling_std = sensor_group.transform(
                lambda values: values.rolling(window, min_periods=1).std()
            ).fillna(0)
            feature_columns[f"{sensor}_mean_{window}"] = rolling_mean.to_numpy()
            feature_columns[f"{sensor}_std_{window}"] = rolling_std.to_numpy()
            feature_columns[f"{sensor}_delta_{window}"] = (
                sensor_group.diff(window).fillna(0).to_numpy()
            )
            feature_columns[f"{sensor}_slope_{window}"] = (
                rolling_mean.groupby(ordered["unit"], sort=False)
                .diff(window)
                .div(window)
                .fillna(0)
                .to_numpy()
            )
    return pd.DataFrame(feature_columns, index=ordered.index)
