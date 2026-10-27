"""Optional early-life PyTorch autoencoder for health/anomaly scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from approaches.neural_models import _scale_fit, require_torch


class EarlyLifeAutoencoder:
    """Reconstruct fixed sensor windows and return per-window MSE scores."""

    def __init__(
        self,
        window_size: int = 20,
        hidden_size: int = 64,
        bottleneck_size: int = 16,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        if window_size < 2 or hidden_size < 1 or bottleneck_size < 1:
            raise ValueError("window_size and autoencoder layer sizes must be positive.")
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.sensor_count: int | None = None
        self.network: Any | None = None

    def _build(self, flat_size: int) -> Any:
        _, nn = require_torch()
        return nn.Sequential(
            nn.Linear(flat_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.bottleneck_size),
            nn.ReLU(),
            nn.Linear(self.bottleneck_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, flat_size),
        )

    def fit(self, windows: np.ndarray) -> "EarlyLifeAutoencoder":
        torch, _ = require_torch()
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] == 0:
            raise ValueError("Autoencoder input must be non-empty 3D sensor windows.")
        if values.shape[1] != self.window_size:
            raise ValueError(
                f"Expected windows of length {self.window_size}, got {values.shape[1]}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Autoencoder training windows contain non-finite values.")

        torch.manual_seed(self.seed)
        self.sensor_count = int(values.shape[2])
        self.mean_, self.scale_ = _scale_fit(values.reshape(-1, self.sensor_count))
        standardized = (values - self.mean_[None, None, :]) / self.scale_[None, None, :]
        flat = standardized.reshape(len(standardized), -1).copy()
        self.network = self._build(flat.shape[1])

        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive.")
        torch.manual_seed(self.seed)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        tensor = torch.as_tensor(flat, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(tensor, tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.MSELoss()
        self.network.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(self.network(batch_x), batch_y)
                loss.backward()
                optimizer.step()
        self.network.eval()
        return self

    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        torch, _ = require_torch()
        if (
            self.network is None
            or self.mean_ is None
            or self.scale_ is None
            or self.sensor_count is None
        ):
            raise RuntimeError("Autoencoder must be fitted or loaded before scoring.")
        values = np.asarray(windows, dtype=np.float32)
        if (
            values.ndim != 3
            or values.shape[1] != self.window_size
            or values.shape[2] != self.sensor_count
        ):
            raise ValueError("Autoencoder scoring windows have an incompatible shape.")
        if not np.isfinite(values).all():
            raise ValueError("Autoencoder scoring windows contain non-finite values.")
        standardized = (values - self.mean_[None, None, :]) / self.scale_[None, None, :]
        flat = standardized.reshape(len(standardized), -1).copy()
        with torch.inference_mode():
            tensor = torch.as_tensor(flat, dtype=torch.float32)
            reconstruction = self.network(tensor)
            errors = torch.mean(torch.square(reconstruction - tensor), dim=1)
        return errors.cpu().numpy()

    def save(self, path: str | Path) -> None:
        torch, _ = require_torch()
        if (
            self.network is None
            or self.mean_ is None
            or self.scale_ is None
            or self.sensor_count is None
        ):
            raise RuntimeError("Cannot save an unfitted autoencoder.")
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "mean": torch.as_tensor(self.mean_),
                "scale": torch.as_tensor(self.scale_),
                "sensor_count": self.sensor_count,
                "window_size": self.window_size,
                "hidden_size": self.hidden_size,
                "bottleneck_size": self.bottleneck_size,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EarlyLifeAutoencoder":
        torch, _ = require_torch()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            window_size=int(state["window_size"]),
            hidden_size=int(state["hidden_size"]),
            bottleneck_size=int(state["bottleneck_size"]),
            epochs=int(state["epochs"]),
            batch_size=int(state["batch_size"]),
            learning_rate=float(state["learning_rate"]),
            seed=int(state["seed"]),
        )
        model.sensor_count = int(state["sensor_count"])
        model.mean_ = state["mean"].cpu().numpy()
        model.scale_ = state["scale"].cpu().numpy()
        flat_size = model.window_size * model.sensor_count
        model.network = model._build(flat_size)
        model.network.load_state_dict(state["state_dict"])
        model.network.eval()
        return model
