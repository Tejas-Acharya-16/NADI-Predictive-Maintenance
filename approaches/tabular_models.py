"""Scikit-learn RUL regressors and development-only AFICv selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from approaches.aficv import select_aficv_features


TABULAR_MODEL_NAMES = (
    "mean",
    "ridge",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
)


@dataclass
class TabularModelBundle:
    """Estimator and ordered feature list required for consistent inference."""

    estimator: Any
    feature_names: list[str]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.feature_names) - set(features.columns))
        if missing:
            raise ValueError(f"Inference data is missing model features: {missing}")
        predictions = self.estimator.predict(features[self.feature_names])
        return np.maximum(np.asarray(predictions, dtype=np.float64), 0.0)


def make_tabular_estimator(
    model_name: str,
    seed: int = 42,
    model_params: dict[str, Any] | None = None,
) -> Any:
    """Create one named regressor with conservative, reproducible defaults."""
    params = {} if model_params is None else dict(model_params)
    if model_name == "mean":
        return DummyRegressor(strategy="mean")
    if model_name == "ridge":
        allowed = {"alpha", "fit_intercept"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"Unsupported Ridge parameters: {sorted(unknown)}")
        return make_pipeline(StandardScaler(), Ridge(**params))
    if model_name == "random_forest":
        defaults: dict[str, Any] = {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "max_features": 1.0,
            "n_jobs": -1,
            "random_state": seed,
        }
        defaults.update(params)
        return RandomForestRegressor(**defaults)
    if model_name == "extra_trees":
        defaults = {
            "n_estimators": 250,
            "min_samples_leaf": 2,
            "max_features": 1.0,
            "n_jobs": -1,
            "random_state": seed,
        }
        defaults.update(params)
        return ExtraTreesRegressor(**defaults)
    if model_name == "hist_gradient_boosting":
        defaults = {
            "max_iter": 150,
            "learning_rate": 0.08,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "random_state": seed,
        }
        defaults.update(params)
        return HistGradientBoostingRegressor(**defaults)
    raise ValueError(
        f"Unknown tabular model {model_name!r}; expected one of {TABULAR_MODEL_NAMES}."
    )


def fit_tabular_model(
    model_name: str,
    features: pd.DataFrame,
    targets: np.ndarray,
    engine_ids: np.ndarray,
    seed: int = 42,
    model_params: dict[str, Any] | None = None,
    aficv_threshold: float | None = None,
) -> TabularModelBundle:
    """Fit a tabular model, optionally selecting features within development folds."""
    if features.empty or len(features) != len(targets):
        raise ValueError("features and targets must have matching, non-empty rows.")
    if len(engine_ids) != len(features):
        raise ValueError("engine_ids must align one-to-one with features.")
    if aficv_threshold is not None and model_name != "random_forest":
        raise ValueError("AFICv selection is currently supported only for random_forest.")

    feature_names = list(features.columns)
    if aficv_threshold is not None:
        unique_engines = np.unique(engine_ids)
        if len(unique_engines) < 2:
            raise ValueError("AFICv requires at least two development engines.")
        splitter = GroupKFold(n_splits=min(3, len(unique_engines)))
        fold_importances: list[np.ndarray] = []
        for train_index, _ in splitter.split(features, targets, groups=engine_ids):
            fold_model = make_tabular_estimator(
                model_name, seed=seed, model_params=model_params
            )
            fold_model.fit(features.iloc[train_index], targets[train_index])
            importances = getattr(fold_model, "feature_importances_", None)
            if importances is None:
                raise RuntimeError(
                    f"{model_name} does not expose feature_importances_ for AFICv."
                )
            fold_importances.append(np.asarray(importances, dtype=np.float64))
        importance_frame = pd.DataFrame(fold_importances, columns=feature_names)
        summary = select_aficv_features(
            importance_frame,
            cumulative_threshold=aficv_threshold,
        )
        feature_names = summary.index[summary["selected"]].tolist()
        if not feature_names:
            raise RuntimeError("AFICv selected no features.")

    estimator = make_tabular_estimator(
        model_name, seed=seed, model_params=model_params
    )
    estimator.fit(features[feature_names], targets)
    return TabularModelBundle(estimator=estimator, feature_names=feature_names)
