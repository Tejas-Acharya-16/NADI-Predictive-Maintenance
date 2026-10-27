"""Compact dense RUL regressor inspired by the supplied TinyML paper.

The paper's classifier used two small dense layers for vibration classes.  This
version keeps that deployment-friendly shape but changes the final layer to a
single continuous RUL prediction and selects features using development data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from approaches.neural_models import _scale_fit, require_torch


def select_top_rul_features(
    features: pd.DataFrame, targets: np.ndarray, count: int
) -> list[str]:
    """Select development-only features by absolute Pearson correlation."""
    if count < 1:
        raise ValueError("feature count must be positive.")
    if features.empty or len(features) != len(targets):
        raise ValueError("features and targets must be non-empty and aligned.")
    values = features.to_numpy(dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64).reshape(-1)
    labels_centered = labels - labels.mean()
    feature_centered = values - values.mean(axis=0)
    denominator = np.sqrt((feature_centered**2).sum(axis=0) * (labels_centered**2).sum())
    correlations = np.divide(
        np.abs(feature_centered.T @ labels_centered),
        denominator,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=denominator > 1e-12,
    )
    ordered = np.argsort(-correlations, kind="stable")
    return [str(features.columns[index]) for index in ordered[: min(count, len(ordered))]]


class TinyMLRULRegressor:
    """20 -> 10 dense RUL network with a compact feature interface."""

    def __init__(
        self,
        feature_count: int = 32,
        hidden_1: int = 20,
        hidden_2: int = 10,
        epochs: int = 60,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        if min(feature_count, hidden_1, hidden_2, epochs, batch_size) < 1:
            raise ValueError("TinyML dimensions, epochs, and batch size must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        self.feature_count = feature_count
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.feature_names: list[str] | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.network: Any | None = None

    def _build(self, input_size: int) -> Any:
        _, nn = require_torch()
        return nn.Sequential(
            nn.Linear(input_size, self.hidden_1),
            nn.ReLU(),
            nn.Linear(self.hidden_1, self.hidden_2),
            nn.ReLU(),
            nn.Linear(self.hidden_2, 1),
        )

    def fit(self, features: pd.DataFrame, targets: np.ndarray) -> "TinyMLRULRegressor":
        torch, _ = require_torch()
        labels = np.asarray(targets, dtype=np.float32).reshape(-1)
        self.feature_names = select_top_rul_features(features, labels, self.feature_count)
        values = features[self.feature_names].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or not np.isfinite(labels).all():
            raise ValueError("TinyML RUL training data contains non-finite values.")
        self.mean_, self.scale_ = _scale_fit(values)
        standardized = (values - self.mean_) / self.scale_
        torch.manual_seed(self.seed)
        self.network = self._build(values.shape[1])
        data = torch.utils.data.TensorDataset(
            torch.as_tensor(standardized, dtype=torch.float32),
            torch.as_tensor(labels, dtype=torch.float32).reshape(-1, 1),
        )
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        self.network.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.network(batch_x), batch_y)
                loss.backward()
                optimizer.step()
        self.network.eval()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        torch, _ = require_torch()
        if self.network is None or self.mean_ is None or self.scale_ is None or self.feature_names is None:
            raise RuntimeError("TinyMLRULRegressor must be fitted or loaded before prediction.")
        missing = sorted(set(self.feature_names) - set(features.columns))
        if missing:
            raise ValueError(f"Inference data is missing selected features: {missing}")
        values = features[self.feature_names].to_numpy(dtype=np.float32)
        with torch.inference_mode():
            prediction = self.network(
                torch.as_tensor((values - self.mean_) / self.scale_, dtype=torch.float32)
            )
        return np.maximum(prediction.cpu().numpy().reshape(-1), 0.0)

    def save(self, path: str | Path) -> None:
        torch, _ = require_torch()
        if self.network is None or self.mean_ is None or self.scale_ is None or self.feature_names is None:
            raise RuntimeError("Cannot save an unfitted TinyMLRULRegressor.")
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "feature_names": self.feature_names,
                "mean": torch.as_tensor(self.mean_),
                "scale": torch.as_tensor(self.scale_),
                "feature_count": self.feature_count,
                "hidden_1": self.hidden_1,
                "hidden_2": self.hidden_2,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TinyMLRULRegressor":
        torch, _ = require_torch()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            feature_count=int(state["feature_count"]), hidden_1=int(state["hidden_1"]),
            hidden_2=int(state["hidden_2"]), epochs=int(state["epochs"]),
            batch_size=int(state["batch_size"]), learning_rate=float(state["learning_rate"]),
            seed=int(state["seed"]),
        )
        model.feature_names = list(state["feature_names"])
        model.mean_ = state["mean"].cpu().numpy()
        model.scale_ = state["scale"].cpu().numpy()
        model.network = model._build(len(model.feature_names))
        model.network.load_state_dict(state["state_dict"])
        model.network.eval()
        return model
