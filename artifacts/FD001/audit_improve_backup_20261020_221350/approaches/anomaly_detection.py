"""Early-life reconstruction-error anomaly scoring for C-MAPSS demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from approaches.autoencoder_model import EarlyLifeAutoencoder


@dataclass
class EarlyLifeAnomalyScorer:
    """Fit a normal-reference autoencoder and derive a training-only threshold."""

    model: EarlyLifeAutoencoder
    threshold: float | None = None
    threshold_quantile: float = 0.99

    def fit(self, healthy_windows: np.ndarray) -> "EarlyLifeAnomalyScorer":
        if not 0 < self.threshold_quantile < 1:
            raise ValueError("threshold_quantile must be strictly between zero and one.")
        self.model.fit(healthy_windows)
        self.threshold = float(np.quantile(self.model.score_windows(healthy_windows), self.threshold_quantile))
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Fit EarlyLifeAnomalyScorer before scoring windows.")
        return self.model.score_windows(windows)

    def is_anomalous(self, windows: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Fit EarlyLifeAnomalyScorer before classifying windows.")
        return self.score(windows) > self.threshold
