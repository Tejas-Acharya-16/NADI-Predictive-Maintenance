"""Training-fitted causal feature variants for FD001 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from approaches.rolling_features import build_causal_features
from cmapss_data import SENSOR_COLUMNS


@dataclass
class CausalFeatureVariant:
    """Past-only features with optional training-only sensor screening.

    Difference-to-initial-reading values are safe because an engine's first
    observation is available before every later prediction.  Sensor removal is
    fit once on development engines and preserved for validation/test.
    """

    windows: tuple[int, ...] = (5, 20)
    include_initial_differences: bool = False
    include_absolute_cycle: bool = True
    low_variance_threshold: float | None = None
    kept_sensors: list[str] | None = None

    def fit(self, frame: pd.DataFrame) -> "CausalFeatureVariant":
        if self.low_variance_threshold is not None and self.low_variance_threshold < 0:
            raise ValueError("low_variance_threshold must be non-negative or None.")
        if self.low_variance_threshold is None:
            self.kept_sensors = list(SENSOR_COLUMNS)
        else:
            std = frame[SENSOR_COLUMNS].std(axis=0, ddof=1)
            self.kept_sensors = [
                sensor for sensor in SENSOR_COLUMNS if float(std[sensor]) > self.low_variance_threshold
            ]
        if not self.kept_sensors:
            raise ValueError("Feature variant removed every sensor.")
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.kept_sensors is None:
            raise RuntimeError("Fit CausalFeatureVariant before transforming data.")
        features = build_causal_features(
            frame, windows=self.windows, sensors=self.kept_sensors
        )
        if self.include_initial_differences:
            ordered = frame.sort_values(["unit", "cycle"], kind="stable")
            initial = ordered.groupby("unit", sort=False)[self.kept_sensors].transform("first")
            for sensor in self.kept_sensors:
                features[f"{sensor}_minus_initial"] = (
                    ordered[sensor].to_numpy() - initial[sensor].to_numpy()
                )
        columns_to_drop = ["unit"]
        if not self.include_absolute_cycle:
            # `cycles_seen` is identical to absolute cycle for FD001 because
            # each engine starts at cycle one. Drop both to make this a real
            # time-free ablation rather than a duplicate of the control.
            columns_to_drop.extend(["cycle", "cycles_seen"])
        return features.drop(columns=columns_to_drop)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def metadata(self) -> dict[str, object]:
        return asdict(self)


def feature_variant_name(
    windows: tuple[int, ...],
    include_initial_differences: bool,
    low_variance_threshold: float | None,
    include_absolute_cycle: bool = True,
) -> str:
    """Build a compact, reproducible label for an ablation setting."""
    parts = ["causal", "w" + "_".join(str(window) for window in windows)]
    if include_initial_differences:
        parts.append("initial_delta")
    if low_variance_threshold is not None:
        parts.append(f"std_gt_{low_variance_threshold:g}")
    if not include_absolute_cycle:
        parts.append("no_absolute_cycle")
    return "__".join(parts)
