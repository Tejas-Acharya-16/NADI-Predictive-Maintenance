"""Optional PyTorch MLP regressor; PyTorch is imported only when used."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def require_torch() -> tuple[Any, Any]:
    """Import the optional neural backend or raise an actionable error."""
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise ImportError(
            "PyTorch is required for neural models. Install it separately from "
            "the optional instructions in requirements-models.txt."
        ) from error
    return torch, nn


def _scale_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _train_network(
    network: Any,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Any:
    torch, _ = require_torch()
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x_tensor = torch.as_tensor(features, dtype=torch.float32)
    y_tensor = torch.as_tensor(targets, dtype=torch.float32).reshape(-1, 1)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    network.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    network.eval()
    return network


class MLPRegressor:
    """Small dense neural RUL regressor with serializable state-dict artifacts."""

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (64, 32),
        epochs: int = 40,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        if not hidden_layers or any(width < 1 for width in hidden_layers):
            raise ValueError("hidden_layers must contain positive layer widths.")
        self.hidden_layers = tuple(hidden_layers)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.network: Any | None = None

    def _build(self, input_size: int) -> Any:
        _, nn = require_torch()
        layers: list[Any] = []
        current_size = input_size
        for width in self.hidden_layers:
            layers.extend([nn.Linear(current_size, width), nn.ReLU()])
            current_size = width
        layers.append(nn.Linear(current_size, 1))
        return nn.Sequential(*layers)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "MLPRegressor":
        values = np.asarray(features, dtype=np.float32)
        labels = np.asarray(targets, dtype=np.float32).reshape(-1)
        if values.ndim != 2 or values.shape[0] != labels.size or not labels.size:
            raise ValueError("MLP input must be a non-empty 2D matrix aligned to targets.")
        if not np.isfinite(values).all() or not np.isfinite(labels).all():
            raise ValueError("MLP training data contains non-finite values.")
        torch, _ = require_torch()
        torch.manual_seed(self.seed)
        self.mean_, self.scale_ = _scale_fit(values)
        standardized = (values - self.mean_) / self.scale_
        self.network = self._build(values.shape[1])
        self.network = _train_network(
            self.network,
            standardized,
            labels,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            seed=self.seed,
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        torch, _ = require_torch()
        if self.network is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("MLPRegressor must be fitted or loaded before prediction.")
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.mean_.size:
            raise ValueError("MLP prediction features have an incompatible shape.")
        if not np.isfinite(values).all():
            raise ValueError("MLP prediction features contain non-finite values.")
        standardized = (values - self.mean_) / self.scale_
        with torch.inference_mode():
            predicted = self.network(torch.as_tensor(standardized, dtype=torch.float32))
        return np.maximum(predicted.cpu().numpy().reshape(-1), 0.0)

    def save(self, path: str | Path) -> None:
        torch, _ = require_torch()
        if self.network is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Cannot save an unfitted MLPRegressor.")
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "mean": torch.as_tensor(self.mean_),
                "scale": torch.as_tensor(self.scale_),
                "hidden_layers": list(self.hidden_layers),
                "input_size": int(self.mean_.size),
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLPRegressor":
        torch, _ = require_torch()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            hidden_layers=tuple(state["hidden_layers"]),
            epochs=int(state["epochs"]),
            batch_size=int(state["batch_size"]),
            learning_rate=float(state["learning_rate"]),
            seed=int(state["seed"]),
        )
        model.mean_ = state["mean"].cpu().numpy()
        model.scale_ = state["scale"].cpu().numpy()
        model.network = model._build(int(state["input_size"]))
        model.network.load_state_dict(state["state_dict"])
        model.network.eval()
        return model
