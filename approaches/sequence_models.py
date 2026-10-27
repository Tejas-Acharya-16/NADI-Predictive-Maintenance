"""Optional PyTorch 1D-CNN and LSTM RUL regressors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from copy import deepcopy

from approaches.neural_models import _scale_fit, require_torch


SEQUENCE_ARCHITECTURES = ("cnn", "lstm", "cnn_lstm")


class SequenceRegressor:
    """Past-window sequence regressor with per-sensor train-only scaling."""

    def __init__(
        self,
        architecture: str,
        hidden_size: int = 32,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        seed: int = 42,
        patience: int = 6,
    ) -> None:
        if architecture not in SEQUENCE_ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {SEQUENCE_ARCHITECTURES}."
            )
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        self.architecture = architecture
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.patience = patience
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.target_mean_: float | None = None
        self.target_scale_: float | None = None
        self.trained_epochs_: int | None = None
        self.network: Any | None = None

    def _build(self, sensor_count: int) -> Any:
        _, nn = require_torch()
        if self.architecture == "cnn":
            return nn.Sequential(
                nn.Conv1d(sensor_count, self.hidden_size, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(
                    self.hidden_size, self.hidden_size, kernel_size=3, padding=1
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(self.hidden_size, 1),
            )

        class LSTMHead(nn.Module):
            def __init__(self, input_size: int, hidden_size: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                )
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, sequence: Any) -> Any:
                output, _ = self.lstm(sequence)
                return self.head(output[:, -1, :])

        if self.architecture == "lstm":
            return LSTMHead(sensor_count, self.hidden_size)

        class CNNLSTMHead(nn.Module):
            def __init__(self, input_size: int, hidden_size: int) -> None:
                super().__init__()
                self.convolution = nn.Sequential(
                    nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, sequence: Any) -> Any:
                convolved = self.convolution(sequence.transpose(1, 2)).transpose(1, 2)
                output, _ = self.lstm(convolved)
                return self.head(output[:, -1, :])

        return CNNLSTMHead(sensor_count, self.hidden_size)

    def fit(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        validation_sequences: np.ndarray | None = None,
        validation_targets: np.ndarray | None = None,
    ) -> "SequenceRegressor":
        values = np.asarray(sequences, dtype=np.float32)
        labels = np.asarray(targets, dtype=np.float32).reshape(-1)
        if values.ndim != 3 or values.shape[0] != labels.size or not labels.size:
            raise ValueError(
                "Sequence input must be a non-empty (samples, time, sensors) array."
            )
        if not np.isfinite(values).all() or not np.isfinite(labels).all():
            raise ValueError("Sequence training data contains non-finite values.")
        torch, _ = require_torch()
        torch.manual_seed(self.seed)
        self.mean_, self.scale_ = _scale_fit(values.reshape(-1, values.shape[-1]))
        standardized = (values - self.mean_[None, None, :]) / self.scale_[None, None, :]
        if self.architecture == "cnn":
            standardized = np.transpose(standardized, (0, 2, 1)).copy()
        self.target_mean_ = float(labels.mean())
        self.target_scale_ = float(labels.std()) if float(labels.std()) > 1e-6 else 1.0
        normalized_targets = (labels - self.target_mean_) / self.target_scale_
        self.network = self._build(values.shape[-1])
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0 or self.patience < 1:
            raise ValueError("epochs, batch_size, learning_rate, and patience must be positive.")
        x_tensor = torch.as_tensor(standardized, dtype=torch.float32)
        y_tensor = torch.as_tensor(normalized_targets, dtype=torch.float32).reshape(-1, 1)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
        )
        if (validation_sequences is None) != (validation_targets is None):
            raise ValueError("validation_sequences and validation_targets must be provided together.")
        validation_x = validation_y = None
        if validation_sequences is not None and validation_targets is not None:
            raw_validation = np.asarray(validation_sequences, dtype=np.float32)
            validation_labels = np.asarray(validation_targets, dtype=np.float32).reshape(-1)
            if raw_validation.ndim != 3 or raw_validation.shape[-1] != values.shape[-1]:
                raise ValueError("Validation sequences have an incompatible shape.")
            transformed = (raw_validation - self.mean_[None, None, :]) / self.scale_[None, None, :]
            if self.architecture == "cnn":
                transformed = np.transpose(transformed, (0, 2, 1)).copy()
            validation_x = torch.as_tensor(transformed, dtype=torch.float32)
            validation_y = torch.as_tensor(
                (validation_labels - self.target_mean_) / self.target_scale_, dtype=torch.float32
            ).reshape(-1, 1)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        best_state = None
        best_loss = float("inf")
        stalled = 0
        for epoch in range(self.epochs):
            self.network.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.network(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            if validation_x is not None and validation_y is not None:
                self.network.eval()
                with torch.inference_mode():
                    validation_loss = float(criterion(self.network(validation_x), validation_y))
                if validation_loss < best_loss - 1e-6:
                    best_loss = validation_loss
                    best_state = deepcopy(self.network.state_dict())
                    stalled = 0
                else:
                    stalled += 1
                    if stalled >= self.patience:
                        self.trained_epochs_ = epoch + 1
                        break
            self.trained_epochs_ = epoch + 1
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        return self

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        torch, _ = require_torch()
        if (
            self.network is None or self.mean_ is None or self.scale_ is None
            or self.target_mean_ is None or self.target_scale_ is None
        ):
            raise RuntimeError("SequenceRegressor must be fitted or loaded before prediction.")
        values = np.asarray(sequences, dtype=np.float32)
        if values.ndim != 3 or values.shape[-1] != self.mean_.size:
            raise ValueError("Sequence prediction features have an incompatible shape.")
        if not np.isfinite(values).all():
            raise ValueError("Sequence prediction features contain non-finite values.")
        standardized = (values - self.mean_[None, None, :]) / self.scale_[None, None, :]
        if self.architecture == "cnn":
            standardized = np.transpose(standardized, (0, 2, 1)).copy()
        with torch.inference_mode():
            predicted = self.network(torch.as_tensor(standardized, dtype=torch.float32))
        return np.maximum(
            predicted.cpu().numpy().reshape(-1) * self.target_scale_ + self.target_mean_, 0.0
        )

    def save(self, path: str | Path) -> None:
        torch, _ = require_torch()
        if (
            self.network is None or self.mean_ is None or self.scale_ is None
            or self.target_mean_ is None or self.target_scale_ is None
        ):
            raise RuntimeError("Cannot save an unfitted SequenceRegressor.")
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "mean": torch.as_tensor(self.mean_),
                "scale": torch.as_tensor(self.scale_),
                "target_mean": self.target_mean_,
                "target_scale": self.target_scale_,
                "architecture": self.architecture,
                "sensor_count": int(self.mean_.size),
                "hidden_size": self.hidden_size,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
                "patience": self.patience,
                "trained_epochs": self.trained_epochs_,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SequenceRegressor":
        torch, _ = require_torch()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            architecture=str(state["architecture"]),
            hidden_size=int(state["hidden_size"]),
            epochs=int(state["epochs"]),
            batch_size=int(state["batch_size"]),
            learning_rate=float(state["learning_rate"]),
            seed=int(state["seed"]),
            patience=int(state.get("patience", 6)),
        )
        model.mean_ = state["mean"].cpu().numpy()
        model.scale_ = state["scale"].cpu().numpy()
        model.target_mean_ = float(state.get("target_mean", 0.0))
        model.target_scale_ = float(state.get("target_scale", 1.0))
        model.trained_epochs_ = state.get("trained_epochs")
        model.network = model._build(int(state["sensor_count"]))
        model.network.load_state_dict(state["state_dict"])
        model.network.eval()
        return model
