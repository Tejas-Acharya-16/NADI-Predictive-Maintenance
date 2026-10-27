"""Joint compact autoencoder and RUL regression model for C-MAPSS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from approaches.neural_models import _scale_fit, require_torch
from approaches.tinyml_rul import select_top_rul_features


class HybridTinyMLAutoencoderRUL:
    """Shared 20 -> 10 encoder with reconstruction and normalized-RUL losses."""

    def __init__(
        self,
        feature_count: int = 32,
        hidden_1: int = 20,
        bottleneck: int = 10,
        reconstruction_weight: float = 0.25,
        epochs: int = 60,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        if min(feature_count, hidden_1, bottleneck, epochs, batch_size) < 1:
            raise ValueError("Hybrid dimensions, epochs, and batch size must be positive.")
        if not 0 <= reconstruction_weight <= 1 or learning_rate <= 0:
            raise ValueError("reconstruction_weight must be in [0, 1] and learning_rate positive.")
        self.feature_count, self.hidden_1, self.bottleneck = feature_count, hidden_1, bottleneck
        self.reconstruction_weight = reconstruction_weight
        self.epochs, self.batch_size, self.learning_rate, self.seed = epochs, batch_size, learning_rate, seed
        self.feature_names: list[str] | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.target_mean_: float | None = None
        self.target_scale_: float | None = None
        self.encoder: Any | None = None
        self.decoder: Any | None = None
        self.rul_head: Any | None = None

    def _build(self, input_size: int) -> None:
        _, nn = require_torch()
        self.encoder = nn.Sequential(nn.Linear(input_size, self.hidden_1), nn.ReLU(), nn.Linear(self.hidden_1, self.bottleneck), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(self.bottleneck, self.hidden_1), nn.ReLU(), nn.Linear(self.hidden_1, input_size))
        self.rul_head = nn.Linear(self.bottleneck, 1)

    def fit(self, features: pd.DataFrame, targets: np.ndarray) -> "HybridTinyMLAutoencoderRUL":
        torch, _ = require_torch()
        labels = np.asarray(targets, dtype=np.float32).reshape(-1)
        self.feature_names = select_top_rul_features(features, labels, self.feature_count)
        values = features[self.feature_names].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or not np.isfinite(labels).all():
            raise ValueError("Hybrid training data contains non-finite values.")
        self.mean_, self.scale_ = _scale_fit(values)
        standardized = (values - self.mean_) / self.scale_
        self.target_mean_ = float(labels.mean())
        self.target_scale_ = float(labels.std()) if float(labels.std()) > 1e-6 else 1.0
        normalized_targets = (labels - self.target_mean_) / self.target_scale_
        torch.manual_seed(self.seed)
        self._build(values.shape[1])
        assert self.encoder is not None and self.decoder is not None and self.rul_head is not None
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.rul_head.parameters())
        data = torch.utils.data.TensorDataset(
            torch.as_tensor(standardized, dtype=torch.float32),
            torch.as_tensor(normalized_targets, dtype=torch.float32).reshape(-1, 1),
        )
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device="cpu").manual_seed(self.seed))
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        for _ in range(self.epochs):
            self.encoder.train(); self.decoder.train(); self.rul_head.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                embedding = self.encoder(batch_x)
                loss = criterion(self.rul_head(embedding), batch_y) + self.reconstruction_weight * criterion(self.decoder(embedding), batch_x)
                loss.backward(); optimizer.step()
        self.encoder.eval(); self.decoder.eval(); self.rul_head.eval()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        torch, _ = require_torch()
        if any(value is None for value in (self.encoder, self.decoder, self.rul_head, self.feature_names, self.mean_, self.scale_, self.target_mean_, self.target_scale_)):
            raise RuntimeError("HybridTinyMLAutoencoderRUL must be fitted or loaded before prediction.")
        assert self.feature_names is not None and self.mean_ is not None and self.scale_ is not None
        assert self.target_mean_ is not None and self.target_scale_ is not None and self.encoder is not None and self.rul_head is not None
        missing = sorted(set(self.feature_names) - set(features.columns))
        if missing:
            raise ValueError(f"Inference data is missing selected features: {missing}")
        values = features[self.feature_names].to_numpy(dtype=np.float32)
        with torch.inference_mode():
            normalized = self.rul_head(self.encoder(torch.as_tensor((values - self.mean_) / self.scale_, dtype=torch.float32)))
        return np.maximum(normalized.cpu().numpy().reshape(-1) * self.target_scale_ + self.target_mean_, 0.0)

    def save(self, path: str | Path) -> None:
        torch, _ = require_torch()
        if any(value is None for value in (self.encoder, self.decoder, self.rul_head, self.feature_names, self.mean_, self.scale_, self.target_mean_, self.target_scale_)):
            raise RuntimeError("Cannot save an unfitted hybrid model.")
        assert self.encoder is not None and self.decoder is not None and self.rul_head is not None
        torch.save({"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict(), "rul_head": self.rul_head.state_dict(), "feature_names": self.feature_names, "mean": torch.as_tensor(self.mean_), "scale": torch.as_tensor(self.scale_), "target_mean": self.target_mean_, "target_scale": self.target_scale_, "feature_count": self.feature_count, "hidden_1": self.hidden_1, "bottleneck": self.bottleneck, "reconstruction_weight": self.reconstruction_weight, "epochs": self.epochs, "batch_size": self.batch_size, "learning_rate": self.learning_rate, "seed": self.seed}, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "HybridTinyMLAutoencoderRUL":
        torch, _ = require_torch()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(feature_count=int(state["feature_count"]), hidden_1=int(state["hidden_1"]), bottleneck=int(state["bottleneck"]), reconstruction_weight=float(state["reconstruction_weight"]), epochs=int(state["epochs"]), batch_size=int(state["batch_size"]), learning_rate=float(state["learning_rate"]), seed=int(state["seed"]))
        model.feature_names = list(state["feature_names"]); model.mean_ = state["mean"].cpu().numpy(); model.scale_ = state["scale"].cpu().numpy(); model.target_mean_ = float(state["target_mean"]); model.target_scale_ = float(state["target_scale"])
        model._build(len(model.feature_names))
        assert model.encoder is not None and model.decoder is not None and model.rul_head is not None
        model.encoder.load_state_dict(state["encoder"]); model.decoder.load_state_dict(state["decoder"]); model.rul_head.load_state_dict(state["rul_head"])
        model.encoder.eval(); model.decoder.eval(); model.rul_head.eval()
        return model
