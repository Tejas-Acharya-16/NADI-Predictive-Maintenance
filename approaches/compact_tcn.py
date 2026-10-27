"""Small CPU-bounded temporal convolutional RUL regressor with attention."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np


@dataclass
class CompactTCNRegressor:
    """Dilated TCN with lightweight sensor and temporal attention.

    This is an inspired compact implementation, not a claim of reproducing a
    paper's complete architecture.  Scaling is fit only on development data.
    """
    channels: int = 16
    epochs: int = 12
    batch_size: int = 256
    learning_rate: float = 1e-3
    patience: int = 3
    seed: int = 20261021

    def fit(self, X, y, X_valid, y_valid):
        import torch
        from torch import nn

        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.x_mean_ = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self.x_std_ = np.maximum(X.std(axis=(0, 1), keepdims=True), 1e-6).astype(np.float32)
        self.y_mean_ = float(np.mean(y)); self.y_std_ = float(max(np.std(y), 1e-6))
        X = ((X - self.x_mean_) / self.x_std_).astype(np.float32)
        X_valid = ((X_valid - self.x_mean_) / self.x_std_).astype(np.float32)
        ys = ((y - self.y_mean_) / self.y_std_).astype(np.float32)
        yvs = ((y_valid - self.y_mean_) / self.y_std_).astype(np.float32)
        n_features = X.shape[2]

        class Block(nn.Module):
            def __init__(self, width, dilation):
                super().__init__(); pad = dilation
                self.net = nn.Sequential(nn.Conv1d(width, width, 3, padding=pad, dilation=dilation), nn.ReLU(), nn.Dropout(0.08), nn.Conv1d(width, width, 1), nn.ReLU())
            def forward(self, z): return z + self.net(z)
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.sensor = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(n_features, max(4, n_features // 4), 1), nn.ReLU(), nn.Conv1d(max(4, n_features // 4), n_features, 1), nn.Sigmoid())
                self.inp = nn.Conv1d(n_features, self_channels, 1)
                self.blocks = nn.Sequential(Block(self_channels, 1), Block(self_channels, 2), Block(self_channels, 4))
                self.time = nn.Conv1d(self_channels, 1, 1)
                self.head = nn.Sequential(nn.Linear(self_channels, self_channels), nn.ReLU(), nn.Linear(self_channels, 1))
            def forward(self, z):
                z = z.transpose(1, 2); z = z * self.sensor(z); z = self.blocks(self.inp(z))
                a = torch.softmax(self.time(z), dim=2); return self.head((z * a).sum(dim=2)).squeeze(1)
        self_channels = self.channels
        self.model_ = Net(); opt = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4); loss = nn.MSELoss()
        xt, yt = torch.from_numpy(X), torch.from_numpy(ys); xv, yv = torch.from_numpy(X_valid), torch.from_numpy(yvs)
        best, best_loss, stale = None, float("inf"), 0
        for epoch in range(self.epochs):
            self.model_.train(); order = torch.randperm(len(xt));
            for ids in order.split(self.batch_size):
                opt.zero_grad(); value = loss(self.model_(xt[ids]), yt[ids]); value.backward(); opt.step()
            self.model_.eval()
            with torch.no_grad(): current = float(loss(self.model_(xv), yv))
            if current < best_loss - 1e-5: best_loss, best, stale = current, copy.deepcopy(self.model_.state_dict()), 0
            else: stale += 1
            if stale >= self.patience: break
        self.model_.load_state_dict(best); self.epochs_completed_ = epoch + 1; self.validation_scaled_mse_ = best_loss
        return self

    def predict(self, X):
        import torch
        Z = ((X - self.x_mean_) / self.x_std_).astype(np.float32)
        self.model_.eval()
        with torch.no_grad(): result = self.model_(torch.from_numpy(Z)).numpy()
        return np.maximum(result * self.y_std_ + self.y_mean_, 0.0)
