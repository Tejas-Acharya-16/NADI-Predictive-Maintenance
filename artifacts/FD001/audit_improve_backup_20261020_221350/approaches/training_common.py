"""Shared leakage-safe C-MAPSS training, validation, and artifact utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cmapss_data import (
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    add_training_rul,
    quality_report,
    read_cmapss_file,
    split_train_by_engine,
)
from approaches.rolling_features import build_causal_features


@dataclass(frozen=True)
class TrainingConfig:
    subset: str = "FD001"
    validation_fraction: float = 0.2
    seed: int = 42
    feature_mode: str = "causal"
    rolling_windows: tuple[int, ...] = (5, 20)
    rul_cap: int | None = None
    endpoint_fractions: tuple[float, ...] = (0.5, 0.7, 0.9)
    model_name: str = "random_forest"
    model_params: dict[str, Any] | None = None
    aficv_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.feature_mode not in {"raw", "causal"}:
            raise ValueError("feature_mode must be 'raw' or 'causal'.")
        if self.rul_cap is not None and self.rul_cap <= 0:
            raise ValueError("rul_cap must be a positive integer or None.")
        if not self.endpoint_fractions:
            raise ValueError("At least one endpoint fraction is required.")
        if any(not 0 < fraction < 1 for fraction in self.endpoint_fractions):
            raise ValueError("Endpoint fractions must be strictly between 0 and 1.")
        if len(set(self.endpoint_fractions)) != len(self.endpoint_fractions):
            raise ValueError("Endpoint fractions must not contain duplicates.")
        if not self.rolling_windows or any(window < 2 for window in self.rolling_windows):
            raise ValueError("Rolling windows must be non-empty and at least two cycles.")
        if len(set(self.rolling_windows)) != len(self.rolling_windows):
            raise ValueError("Rolling windows must not contain duplicates.")

    def as_json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.model_params is None:
            result["model_params"] = {}
        return result


@dataclass
class PreparedData:
    development_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    X_development: pd.DataFrame
    y_development: np.ndarray
    X_validation: pd.DataFrame
    y_validation: np.ndarray
    engine_summary: pd.DataFrame
    training_data_sha256: str
    split_sha256: str


def sha256_file(path: str | Path) -> str:
    """Hash a file incrementally so large data files need not fit in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(value: Any) -> str:
    """Return a deterministic hash of JSON-serializable configuration data."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def software_versions(model_name: str) -> dict[str, str | None]:
    """Capture versions that can affect this model family and its artifacts."""
    packages = ["numpy", "pandas", "scikit-learn", "joblib"]
    if model_name in {"mlp", "cnn", "lstm", "cnn_lstm", "autoencoder", "tinyml_rul", "hybrid_tinyml_autoencoder_rul"}:
        packages.append("torch")
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_feature_frame(
    frame: pd.DataFrame,
    feature_mode: str,
    rolling_windows: tuple[int, ...],
) -> pd.DataFrame:
    """Create the saved model's raw or causal features without using labels."""
    required = {"unit", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input frame is missing required feature columns: {missing}")
    ordered = frame.sort_values(["unit", "cycle"], kind="stable")
    if feature_mode == "raw":
        features = ordered[["unit", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]].copy()
    elif feature_mode == "causal":
        features = build_causal_features(ordered, windows=rolling_windows)
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode!r}.")
    return features.drop(columns=["unit"])


def prepare_training_data(
    data_dir: str | Path,
    config: TrainingConfig,
) -> PreparedData:
    """Load only the training file, then make the fixed engine-wise split."""
    training_path = Path(data_dir) / f"train_{config.subset}.txt"
    raw = read_cmapss_file(training_path)
    report = quality_report(raw, config.subset, "train")
    invalid_counts = {
        key: report[key]
        for key in (
            "missing_cells",
            "nonfinite_cells",
            "duplicate_unit_cycle",
            "non_increasing_cycles_sorted",
            "cycle_gaps",
            "invalid_unit_or_cycle",
        )
        if report[key]
    }
    if invalid_counts:
        raise ValueError(
            f"{config.subset} training data failed quality checks: {invalid_counts}"
        )

    development_raw, validation_raw, engine_summary = split_train_by_engine(
        raw,
        validation_fraction=config.validation_fraction,
        random_state=config.seed,
    )
    development = add_training_rul(development_raw, cap=config.rul_cap)
    validation = add_training_rul(validation_raw)

    development = development.sort_values(["unit", "cycle"], kind="stable")
    validation = validation.sort_values(["unit", "cycle"], kind="stable")
    X_development = build_feature_frame(
        development, config.feature_mode, config.rolling_windows
    )
    X_validation = build_feature_frame(
        validation, config.feature_mode, config.rolling_windows
    )
    y_development = development["rul"].to_numpy(dtype=np.float32)
    y_validation = validation["rul"].to_numpy(dtype=np.float32)

    if not np.isfinite(X_development.to_numpy(dtype=np.float64)).all():
        raise ValueError("Development features contain non-finite values.")
    if not np.isfinite(X_validation.to_numpy(dtype=np.float64)).all():
        raise ValueError("Validation features contain non-finite values.")
    if not np.isfinite(y_development).all() or not np.isfinite(y_validation).all():
        raise ValueError("Training or validation RUL targets contain non-finite values.")

    overlap = set(development["unit"].unique()) & set(validation["unit"].unique())
    if overlap:
        raise RuntimeError(f"Engine split leakage detected: {sorted(overlap)}")

    split_identity = {
        "development_units": sorted(development["unit"].unique().tolist()),
        "validation_units": sorted(validation["unit"].unique().tolist()),
        "validation_fraction": config.validation_fraction,
        "seed": config.seed,
    }
    return PreparedData(
        development_frame=development,
        validation_frame=validation,
        X_development=X_development,
        y_development=y_development,
        X_validation=X_validation,
        y_validation=y_validation,
        engine_summary=engine_summary,
        training_data_sha256=sha256_file(training_path),
        split_sha256=stable_fingerprint(split_identity),
    )


def right_censored_endpoint_rows(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    fractions: tuple[float, ...] = (0.5, 0.7, 0.9),
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Select known-label validation endpoints at predeclared life fractions."""
    if len(frame) != len(features):
        raise ValueError("frame and features must have the same number of rows.")
    endpoint_metadata = right_censored_endpoint_metadata(frame, fractions)
    ordered = frame.sort_values(["unit", "cycle"], kind="stable")
    feature_ordered = features.loc[ordered.index]
    selected_indices: list[int] = []
    for row in endpoint_metadata.itertuples(index=False):
        eligible = ordered.loc[
            (ordered["unit"] == row.unit) & (ordered["cycle"] <= row.cutoff_cycle)
        ]
        if eligible.empty:
            raise ValueError(
                f"No validation observation for unit {row.unit} at cutoff "
                f"{row.cutoff_cycle}."
            )
        selected_indices.append(int(eligible.index[-1]))
    selected = feature_ordered.loc[selected_indices].reset_index(drop=True)
    return (
        selected,
        endpoint_metadata["actual_rul"].to_numpy(dtype=np.float32),
        endpoint_metadata,
    )


def right_censored_endpoint_metadata(
    frame: pd.DataFrame,
    fractions: tuple[float, ...] = (0.5, 0.7, 0.9),
) -> pd.DataFrame:
    """Describe known-label right-censored validation points without features."""
    if not {"unit", "cycle", "rul"} <= set(frame.columns):
        raise ValueError("frame must contain unit, cycle, and uncapped rul columns.")
    if any(not 0 < fraction < 1 for fraction in fractions):
        raise ValueError("Endpoint fractions must be strictly between 0 and 1.")
    ordered = frame.sort_values(["unit", "cycle"], kind="stable")
    metadata: list[dict[str, float | int]] = []
    for unit, group in ordered.groupby("unit", sort=True):
        first_cycle = int(group["cycle"].iloc[0])
        last_cycle = int(group["cycle"].iloc[-1])
        for fraction in fractions:
            cutoff = first_cycle + int(np.floor((last_cycle - first_cycle) * fraction))
            eligible = group.loc[group["cycle"] <= cutoff]
            if eligible.empty:
                raise ValueError(
                    f"No validation observation for unit {unit} at cutoff {cutoff}."
                )
            row = eligible.iloc[-1]
            metadata.append(
                {
                    "unit": int(unit),
                    "fraction": float(fraction),
                    "cutoff_cycle": int(row["cycle"]),
                    "actual_rul": int(row["rul"]),
                }
            )
    return pd.DataFrame(metadata)


def phm08_contributions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return PHM08 asymmetric score terms for aligned RUL predictions.

    ``d = prediction - truth``. Underprediction uses the 13-cycle divisor;
    overprediction, which can imply a late intervention, uses the 10-cycle
    divisor.  Inputs are intentionally not clipped before scoring.
    """
    actual = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actual and predicted arrays must be non-empty and same-sized.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual or predicted RUL contains non-finite values.")
    error = predicted - actual
    return np.where(
        error < 0,
        np.expm1(np.clip(-error / 13.0, 0, 50)),
        np.expm1(np.clip(error / 10.0, 0, 50)),
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Calculate aligned RUL metrics, including explicit PHM08 sum and mean."""
    actual = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actual and predicted arrays must be non-empty and same-sized.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual or predicted RUL contains non-finite values.")

    error = predicted - actual
    penalties = phm08_contributions(actual, predicted)
    absolute_error = np.abs(error)
    mape_denominator = np.maximum(np.abs(actual), 1.0)
    smape_denominator = np.maximum(np.abs(actual) + np.abs(predicted), 1.0)
    return {
        "samples": int(actual.size),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "median_absolute_error": float(np.median(absolute_error)),
        "r2": float(r2_score(actual, predicted)),
        "mape_percent": float(100.0 * np.mean(absolute_error / mape_denominator)),
        "smape_percent": float(200.0 * np.mean(absolute_error / smape_denominator)),
        "bias_predicted_minus_actual": float(error.mean()),
        "overprediction_count": int(np.count_nonzero(error > 0)),
        "underprediction_count": int(np.count_nonzero(error < 0)),
        "overprediction_fraction": float(np.mean(error > 0)),
        "absolute_error_p90": float(np.quantile(absolute_error, 0.9)),
        "absolute_error_max": float(absolute_error.max()),
        "phm08_sum": float(penalties.sum()),
        "phm08_mean": float(penalties.mean()),
        # Backward-compatible aliases for existing result notebooks/artifacts.
        "nasa_score": float(penalties.sum()),
        "nasa_score_mean": float(penalties.mean()),
    }


def endpoint_metric_rows(
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
) -> list[dict[str, float | int]]:
    """Return endpoint metrics grouped by simulated censoring fraction."""
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(predicted) != len(metadata):
        raise ValueError("Predictions and endpoint metadata must have equal lengths.")
    results: list[dict[str, float | int]] = []
    for fraction, positions in metadata.groupby("fraction", sort=True).groups.items():
        indices = np.asarray(list(positions), dtype=int)
        scores = regression_metrics(
            metadata.iloc[indices]["actual_rul"].to_numpy(),
            predicted[indices],
        )
        results.append({"fraction": float(fraction), **scores})
    return results


def run_identity(
    config: TrainingConfig,
    data: PreparedData,
) -> dict[str, Any]:
    """Build the immutable identity recorded and checked for every artifact."""
    identity = {
        "config": config.as_json_dict(),
        "training_data_sha256": data.training_data_sha256,
        "split_sha256": data.split_sha256,
        "feature_columns": list(data.X_development.columns),
        "software_versions": software_versions(config.model_name),
    }
    return json.loads(json.dumps(identity, sort_keys=True, default=str))


def artifact_directory(
    root: str | Path,
    config: TrainingConfig,
    data: PreparedData,
) -> Path:
    """Return a config/data-addressed directory so incompatible runs cannot collide."""
    identity = run_identity(config, data)
    fingerprint = stable_fingerprint(identity)
    return Path(root) / config.subset / config.model_name / fingerprint


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Atomically write a JSON manifest next to the model artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(destination)


def validate_manifest(
    manifest_path: str | Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    """Load a manifest and refuse artifacts trained against different inputs."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("identity") != expected_identity:
        raise ValueError(f"Artifact identity mismatch; refusing reuse: {path}")
    return manifest
