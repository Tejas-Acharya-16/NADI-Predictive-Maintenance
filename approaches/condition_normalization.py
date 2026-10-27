"""Training-fitted operating-condition normalization for FD002/FD004."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from cmapss_data import SENSOR_COLUMNS, SETTING_COLUMNS


@dataclass
class ConditionAwareNormalizer:
    """Assign six FD004 operating regimes using training settings and z-score sensors.

    Both the setting scaler, cluster centres, and per-condition sensor moments
    are fitted only on source training rows. A target subset is merely assigned
    to its nearest source condition at transform time.
    """

    n_conditions: int = 6
    random_state: int = 20261020
    setting_scaler: StandardScaler | None = None
    kmeans: KMeans | None = None
    sensor_means: pd.DataFrame | None = None
    sensor_stds: pd.DataFrame | None = None

    def fit(self, frame: pd.DataFrame) -> "ConditionAwareNormalizer":
        if self.n_conditions < 2:
            raise ValueError("At least two operating conditions are required.")
        self.setting_scaler = StandardScaler().fit(frame[SETTING_COLUMNS])
        scaled = self.setting_scaler.transform(frame[SETTING_COLUMNS])
        self.kmeans = KMeans(n_clusters=self.n_conditions, random_state=self.random_state, n_init=10).fit(scaled)
        condition = self.kmeans.labels_
        grouped = frame[SENSOR_COLUMNS].groupby(condition, sort=True)
        self.sensor_means = grouped.mean().reindex(range(self.n_conditions))
        self.sensor_stds = grouped.std(ddof=1).reindex(range(self.n_conditions)).replace(0.0, 1.0).fillna(1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        if self.setting_scaler is None or self.kmeans is None or self.sensor_means is None or self.sensor_stds is None:
            raise RuntimeError("Fit ConditionAwareNormalizer before transform.")
        result = frame.astype({sensor: float for sensor in SENSOR_COLUMNS})
        # Some raw C-MAPSS sensors are parsed as integer columns.  Normalize
        # into an explicit floating-point copy rather than relying on pandas
        # to silently upcast an integer block.
        condition = self.kmeans.predict(self.setting_scaler.transform(frame[SETTING_COLUMNS]))
        means = self.sensor_means.loc[condition].set_axis(frame.index)
        stds = self.sensor_stds.loc[condition].set_axis(frame.index)
        result.loc[:, SENSOR_COLUMNS] = (frame[SENSOR_COLUMNS] - means) / stds
        if not np.isfinite(result[SENSOR_COLUMNS].to_numpy(dtype=float)).all():
            raise ValueError("Condition normalization produced non-finite sensor values.")
        return result, condition.astype(int)

    def metadata(self) -> dict[str, object]:
        if self.kmeans is None or self.sensor_means is None or self.sensor_stds is None:
            raise RuntimeError("Fit ConditionAwareNormalizer before requesting metadata.")
        return {
            "configuration": asdict(self) | {"setting_scaler": None, "kmeans": None, "sensor_means": None, "sensor_stds": None},
            "centers_scaled_settings": self.kmeans.cluster_centers_.tolist(),
            "sensor_means": self.sensor_means.to_dict(),
            "sensor_stds": self.sensor_stds.to_dict(),
        }
