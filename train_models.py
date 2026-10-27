"""Train and cache selected C-MAPSS candidates without loading official test data.

Examples:
    python train_models.py --subset FD001 --models mean ridge random_forest hist_gradient_boosting
    python train_models.py --subset FD001 --models mlp cnn lstm
    python train_models.py --subset FD001 --models autoencoder
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from approaches.autoencoder_inputs import build_early_life_windows
from approaches.autoencoder_model import EarlyLifeAutoencoder
from approaches.hybrid_autoencoder_rul import HybridTinyMLAutoencoderRUL
from approaches.neural_models import MLPRegressor, require_torch
from approaches.sequence_models import SequenceRegressor
from approaches.sequence_windows import build_sequence_windows
from approaches.tabular_models import (
    TABULAR_MODEL_NAMES,
    TabularModelBundle,
    fit_tabular_model,
)
from approaches.tinyml_rul import TinyMLRULRegressor
from approaches.training_common import (
    PreparedData,
    TrainingConfig,
    artifact_directory,
    endpoint_metric_rows,
    prepare_training_data,
    regression_metrics,
    right_censored_endpoint_metadata,
    right_censored_endpoint_rows,
    run_identity,
    sha256_file,
    software_versions,
    validate_manifest,
    write_manifest,
)
from cmapss_data import SENSOR_COLUMNS, SUBSETS


MODEL_NAMES = (
    *TABULAR_MODEL_NAMES,
    "mlp",
    "cnn",
    "lstm",
    "cnn_lstm",
    "autoencoder",
    "tinyml_rul",
    "hybrid_tinyml_autoencoder_rul",
)
PROJECT_ROOT = Path(__file__).resolve().parent


def _make_config(
    model_name: str,
    *,
    subset: str,
    data_dir: Path,
    output_dir: Path,
    seed: int,
    validation_fraction: float,
    feature_mode: str,
    rul_cap: int | None,
    rolling_windows: tuple[int, ...],
    endpoint_fractions: tuple[float, ...],
    aficv_threshold: float | None,
    sequence_window_size: int,
    autoencoder_window_size: int,
    early_fraction: float,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    learning_rate: float,
) -> tuple[TrainingConfig, Path, PreparedData]:
    if aficv_threshold is not None and model_name != "random_forest":
        raise ValueError("AFICv threshold is supported only with random_forest.")
    if model_name in {"cnn", "lstm", "cnn_lstm"}:
        model_params = {
            "window_size": sequence_window_size,
            "hidden_size": hidden_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        selected_feature_mode = "raw"
        selected_cap = rul_cap
        selected_aficv = None
    elif model_name == "mlp":
        model_params = {
            "hidden_layers": [hidden_size * 2, hidden_size],
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        selected_feature_mode = feature_mode
        selected_cap = rul_cap
        selected_aficv = None
    elif model_name == "autoencoder":
        model_params = {
            "window_size": autoencoder_window_size,
            "early_fraction": early_fraction,
            "hidden_size": hidden_size,
            "bottleneck_size": max(4, hidden_size // 4),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        selected_feature_mode = "raw"
        selected_cap = None
        selected_aficv = None
    elif model_name in {"tinyml_rul", "hybrid_tinyml_autoencoder_rul"}:
        model_params = {
            "feature_count": 32,
            "hidden_1": 20,
            "hidden_2": 10,
            "reconstruction_weight": 0.25,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        selected_feature_mode = feature_mode
        selected_cap = rul_cap
        selected_aficv = None
    else:
        model_params = {}
        selected_feature_mode = feature_mode
        selected_cap = rul_cap
        selected_aficv = aficv_threshold if model_name == "random_forest" else None
        if aficv_threshold is not None and model_name != "random_forest":
            raise ValueError("AFICv threshold is supported only with random_forest.")

    config = TrainingConfig(
        subset=subset,
        validation_fraction=validation_fraction,
        seed=seed,
        feature_mode=selected_feature_mode,
        rolling_windows=rolling_windows,
        rul_cap=selected_cap,
        endpoint_fractions=endpoint_fractions,
        model_name=model_name,
        model_params=model_params,
        aficv_threshold=selected_aficv,
    )
    data = prepare_training_data(data_dir, config)
    destination = artifact_directory(output_dir, config, data)
    return config, destination, data


def _model_path(directory: Path, model_name: str) -> Path:
    suffix = ".joblib" if model_name in TABULAR_MODEL_NAMES else ".pt"
    return directory / f"model{suffix}"


def _verify_saved_artifact(manifest: dict[str, Any], model_path: Path) -> None:
    expected_hash = manifest.get("artifact_sha256")
    if not expected_hash:
        raise ValueError(f"Artifact manifest has no model checksum: {model_path}")
    actual_hash = sha256_file(model_path)
    if actual_hash != expected_hash:
        raise ValueError(f"Saved model checksum mismatch: {model_path}")


def _save_neural_model(model: Any, model_path: Path) -> None:
    temporary_model = model_path.with_suffix(model_path.suffix + ".tmp")
    model.save(temporary_model)
    temporary_model.replace(model_path)


def _load_model(model_name: str, path: Path) -> Any:
    if model_name in TABULAR_MODEL_NAMES:
        import joblib

        return joblib.load(path)
    if model_name == "mlp":
        return MLPRegressor.load(path)
    if model_name in {"cnn", "lstm", "cnn_lstm"}:
        return SequenceRegressor.load(path)
    if model_name == "autoencoder":
        return EarlyLifeAutoencoder.load(path)
    if model_name == "tinyml_rul":
        return TinyMLRULRegressor.load(path)
    if model_name == "hybrid_tinyml_autoencoder_rul":
        return HybridTinyMLAutoencoderRUL.load(path)
    raise ValueError(f"Unsupported model name: {model_name!r}.")


def load_saved_model(artifact_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a local model after validating its manifest and file checksum.

    The returned manifest includes the training identity and validation report.
    Only load joblib artifacts created by this project or another trusted source.
    """
    directory = Path(artifact_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_name = manifest.get("model_name")
    artifact_file = manifest.get("artifact_file")
    if (
        model_name not in MODEL_NAMES
        or not isinstance(artifact_file, str)
        or Path(artifact_file).name != artifact_file
    ):
        raise ValueError(f"Artifact manifest has an unsupported format: {manifest_path}")
    model_path = directory / artifact_file
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact is missing: {model_path}")
    _verify_saved_artifact(manifest, model_path)
    return _load_model(model_name, model_path), manifest


def _write_score_files(
    directory: Path,
    metrics: dict[str, Any],
    endpoint_predictions: pd.DataFrame | None = None,
    autoencoder_scores: pd.DataFrame | None = None,
) -> None:
    write_manifest(directory / "validation_metrics.json", metrics)
    if endpoint_predictions is not None:
        endpoint_predictions.to_csv(directory / "endpoint_predictions.csv", index=False)
    if autoencoder_scores is not None:
        autoencoder_scores.to_csv(directory / "autoencoder_validation_scores.csv", index=False)


def _fit_tabular(
    config: TrainingConfig,
    directory: Path,
    data: Any,
    *,
    reuse: bool,
) -> dict[str, Any]:
    identity = run_identity(config, data)
    model_path = _model_path(directory, config.model_name)
    manifest_path = directory / "manifest.json"
    if reuse and manifest_path.exists():
        manifest = validate_manifest(manifest_path, identity)
        if not model_path.exists():
            raise FileNotFoundError(f"Manifest exists but model is missing: {model_path}")
        _verify_saved_artifact(manifest, model_path)
        _load_model(config.model_name, model_path)
        return manifest["validation_metrics"]
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"Artifact directory is non-empty but has no reusable manifest: {directory}"
        )

    directory.mkdir(parents=True, exist_ok=True)
    groups = data.development_frame["unit"].to_numpy()
    bundle: TabularModelBundle = fit_tabular_model(
        config.model_name,
        data.X_development,
        data.y_development,
        groups,
        seed=config.seed,
        model_params=config.model_params,
        aficv_threshold=config.aficv_threshold,
    )
    cycle_predictions = bundle.predict(data.X_validation)
    cycle_metrics = regression_metrics(data.y_validation, cycle_predictions)
    endpoint_features, _, endpoint_metadata = right_censored_endpoint_rows(
        data.validation_frame,
        data.X_validation,
        config.endpoint_fractions,
    )
    endpoint_estimates = bundle.predict(endpoint_features)
    endpoints_report = endpoint_metadata.copy()
    endpoints_report["predicted_rul"] = endpoint_estimates
    metrics = {
        "cycle_level": cycle_metrics,
        "cycle_validation_rows": int(len(data.y_validation)),
        "validation_engines": int(data.validation_frame["unit"].nunique()),
        "right_censored_endpoints": endpoint_metric_rows(
            endpoint_estimates, endpoint_metadata
        ),
        "selected_feature_count": len(bundle.feature_names),
        "selected_features": bundle.feature_names,
    }
    import joblib

    temporary_model = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(bundle, temporary_model)
    temporary_model.replace(model_path)
    _write_score_files(directory, metrics, endpoint_predictions=endpoints_report)
    write_manifest(
        manifest_path,
        {
            "model_name": config.model_name,
            "identity": identity,
            "artifact_file": model_path.name,
            "artifact_sha256": sha256_file(model_path),
            "validation_metrics": metrics,
            "software_versions": software_versions(config.model_name),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return metrics


def _fit_mlp(
    config: TrainingConfig,
    directory: Path,
    data: Any,
    *,
    reuse: bool,
) -> dict[str, Any]:
    identity = run_identity(config, data)
    model_path = _model_path(directory, config.model_name)
    manifest_path = directory / "manifest.json"
    if reuse and manifest_path.exists():
        manifest = validate_manifest(manifest_path, identity)
        if not model_path.exists():
            raise FileNotFoundError(f"Manifest exists but model is missing: {model_path}")
        _verify_saved_artifact(manifest, model_path)
        MLPRegressor.load(model_path)
        return manifest["validation_metrics"]
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Incomplete artifact directory: {directory}")

    model_params = config.model_params or {}
    model = MLPRegressor(
        hidden_layers=tuple(model_params["hidden_layers"]),
        epochs=int(model_params["epochs"]),
        batch_size=int(model_params["batch_size"]),
        learning_rate=float(model_params["learning_rate"]),
        seed=config.seed,
    ).fit(data.X_development.to_numpy(dtype=np.float32), data.y_development)
    directory.mkdir(parents=True, exist_ok=True)
    cycle_predictions = model.predict(data.X_validation.to_numpy(dtype=np.float32))
    endpoint_features, _, endpoint_metadata = right_censored_endpoint_rows(
        data.validation_frame, data.X_validation, config.endpoint_fractions
    )
    endpoint_predictions = model.predict(endpoint_features.to_numpy(dtype=np.float32))
    endpoints_report = endpoint_metadata.copy()
    endpoints_report["predicted_rul"] = endpoint_predictions
    metrics = {
        "cycle_level": regression_metrics(data.y_validation, cycle_predictions),
        "cycle_validation_rows": int(len(data.y_validation)),
        "validation_engines": int(data.validation_frame["unit"].nunique()),
        "right_censored_endpoints": endpoint_metric_rows(
            endpoint_predictions, endpoint_metadata
        ),
    }
    _save_neural_model(model, model_path)
    _write_score_files(directory, metrics, endpoint_predictions=endpoints_report)
    write_manifest(
        manifest_path,
        {
            "model_name": config.model_name,
            "identity": identity,
            "artifact_file": model_path.name,
            "artifact_sha256": sha256_file(model_path),
            "validation_metrics": metrics,
            "software_versions": software_versions(config.model_name),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return metrics


def _fit_compact_rul_network(
    config: TrainingConfig,
    directory: Path,
    data: Any,
    *,
    reuse: bool,
) -> dict[str, Any]:
    """Fit one compact dense model with feature selection fit only on development rows."""
    identity = run_identity(config, data)
    model_path = _model_path(directory, config.model_name)
    manifest_path = directory / "manifest.json"
    if reuse and manifest_path.exists():
        manifest = validate_manifest(manifest_path, identity)
        if not model_path.exists():
            raise FileNotFoundError(f"Manifest exists but model is missing: {model_path}")
        _verify_saved_artifact(manifest, model_path)
        _load_model(config.model_name, model_path)
        return manifest["validation_metrics"]
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Incomplete artifact directory: {directory}")

    params = config.model_params or {}
    common = {
        "feature_count": int(params["feature_count"]),
        "hidden_1": int(params["hidden_1"]),
        "epochs": int(params["epochs"]),
        "batch_size": int(params["batch_size"]),
        "learning_rate": float(params["learning_rate"]),
        "seed": config.seed,
    }
    if config.model_name == "tinyml_rul":
        model: Any = TinyMLRULRegressor(hidden_2=int(params["hidden_2"]), **common)
    else:
        model = HybridTinyMLAutoencoderRUL(
            bottleneck=int(params["hidden_2"]),
            reconstruction_weight=float(params["reconstruction_weight"]),
            **common,
        )
    model.fit(data.X_development, data.y_development)
    directory.mkdir(parents=True, exist_ok=True)
    cycle_predictions = model.predict(data.X_validation)
    endpoint_features, _, endpoint_metadata = right_censored_endpoint_rows(
        data.validation_frame, data.X_validation, config.endpoint_fractions
    )
    endpoint_predictions = model.predict(endpoint_features)
    endpoints_report = endpoint_metadata.copy()
    endpoints_report["predicted_rul"] = endpoint_predictions
    metrics = {
        "cycle_level": regression_metrics(data.y_validation, cycle_predictions),
        "cycle_validation_rows": int(len(data.y_validation)),
        "validation_engines": int(data.validation_frame["unit"].nunique()),
        "right_censored_endpoints": endpoint_metric_rows(endpoint_predictions, endpoint_metadata),
        "selected_feature_count": len(model.feature_names),
        "selected_features": model.feature_names,
    }
    _save_neural_model(model, model_path)
    _write_score_files(directory, metrics, endpoint_predictions=endpoints_report)
    write_manifest(
        manifest_path,
        {
            "model_name": config.model_name,
            "identity": identity,
            "artifact_file": model_path.name,
            "artifact_sha256": sha256_file(model_path),
            "validation_metrics": metrics,
            "software_versions": software_versions(config.model_name),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return metrics


def _sequence_data(
    data: Any,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    X_development, _, y_development = build_sequence_windows(
        data.development_frame,
        window_size=window_size,
        sensors=SENSOR_COLUMNS,
        label_column="rul",
    )
    X_validation, validation_metadata, y_validation = build_sequence_windows(
        data.validation_frame,
        window_size=window_size,
        sensors=SENSOR_COLUMNS,
        label_column="rul",
    )
    if y_development is None or y_validation is None:
        raise RuntimeError("Sequence training requires RUL labels.")
    if not len(X_development) or not len(X_validation):
        raise ValueError(
            f"No full sequence windows found at window size {window_size}."
        )
    return (
        X_development,
        y_development,
        validation_metadata,
        X_validation,
        y_validation,
        data.validation_frame,
    )


def _align_sequence_endpoints(
    frame: pd.DataFrame,
    window_metadata: pd.DataFrame,
    predictions: np.ndarray,
    fractions: tuple[float, ...],
) -> tuple[np.ndarray, pd.DataFrame]:
    endpoint_metadata = right_censored_endpoint_metadata(frame, fractions)
    lookup = {
        (int(row.unit), int(row.end_cycle)): index
        for index, row in enumerate(window_metadata.itertuples(index=False))
    }
    indices: list[int] = []
    for row in endpoint_metadata.itertuples(index=False):
        key = (int(row.unit), int(row.cutoff_cycle))
        if key not in lookup:
            raise ValueError(
                f"Sequence endpoint {key} is unavailable; reduce the window size "
                "or inspect the validation trajectories."
            )
        indices.append(lookup[key])
    endpoint_predictions = np.asarray(predictions)[indices]
    report = endpoint_metadata.copy()
    report["predicted_rul"] = endpoint_predictions
    return endpoint_predictions, report


def _fit_sequence(
    config: TrainingConfig,
    directory: Path,
    data: Any,
    *,
    reuse: bool,
) -> dict[str, Any]:
    identity = run_identity(config, data)
    model_path = _model_path(directory, config.model_name)
    manifest_path = directory / "manifest.json"
    if reuse and manifest_path.exists():
        manifest = validate_manifest(manifest_path, identity)
        if not model_path.exists():
            raise FileNotFoundError(f"Manifest exists but model is missing: {model_path}")
        _verify_saved_artifact(manifest, model_path)
        _load_model(config.model_name, model_path)
        return manifest["validation_metrics"]
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Incomplete artifact directory: {directory}")

    params = config.model_params or {}
    (
        X_development,
        y_development,
        validation_metadata,
        X_validation,
        y_validation,
        validation_frame,
    ) = _sequence_data(data, int(params["window_size"]))
    model = SequenceRegressor(
        architecture=config.model_name,
        hidden_size=int(params["hidden_size"]),
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        learning_rate=float(params["learning_rate"]),
        seed=config.seed,
    ).fit(X_development, y_development)
    directory.mkdir(parents=True, exist_ok=True)
    cycle_predictions = model.predict(X_validation)
    endpoint_predictions, endpoint_report = _align_sequence_endpoints(
        validation_frame,
        validation_metadata,
        cycle_predictions,
        config.endpoint_fractions,
    )
    endpoint_metrics_metadata = endpoint_report.drop(columns=["predicted_rul"])
    metrics = {
        "cycle_level": regression_metrics(y_validation, cycle_predictions),
        "cycle_validation_rows": int(len(y_validation)),
        "sequence_window_size": int(params["window_size"]),
        "validation_engines": int(validation_frame["unit"].nunique()),
        "right_censored_endpoints": endpoint_metric_rows(
            endpoint_predictions, endpoint_metrics_metadata
        ),
    }
    _save_neural_model(model, model_path)
    _write_score_files(directory, metrics, endpoint_predictions=endpoint_report)
    write_manifest(
        manifest_path,
        {
            "model_name": config.model_name,
            "identity": identity,
            "artifact_file": model_path.name,
            "artifact_sha256": sha256_file(model_path),
            "validation_metrics": metrics,
            "software_versions": software_versions(config.model_name),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return metrics


def _fit_autoencoder(
    config: TrainingConfig,
    directory: Path,
    data: Any,
    *,
    reuse: bool,
) -> dict[str, Any]:
    identity = run_identity(config, data)
    model_path = _model_path(directory, config.model_name)
    manifest_path = directory / "manifest.json"
    model_params = config.model_params or {}
    if reuse and manifest_path.exists():
        manifest = validate_manifest(manifest_path, identity)
        if not model_path.exists():
            raise FileNotFoundError(f"Manifest exists but model is missing: {model_path}")
        _verify_saved_artifact(manifest, model_path)
        EarlyLifeAutoencoder.load(model_path)
        metrics = manifest["validation_metrics"]
    else:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Incomplete artifact directory: {directory}")
        train_windows, _ = build_early_life_windows(
            data.development_frame,
            early_fraction=float(model_params["early_fraction"]),
            window_size=int(model_params["window_size"]),
            sensors=SENSOR_COLUMNS,
        )
        if not len(train_windows):
            raise ValueError("No development early-life windows are available.")
        model = EarlyLifeAutoencoder(
            window_size=int(model_params["window_size"]),
            hidden_size=int(model_params["hidden_size"]),
            bottleneck_size=int(model_params["bottleneck_size"]),
            epochs=int(model_params["epochs"]),
            batch_size=int(model_params["batch_size"]),
            learning_rate=float(model_params["learning_rate"]),
            seed=config.seed,
        ).fit(train_windows)

        directory.mkdir(parents=True, exist_ok=True)
        validation_windows, validation_meta, _ = build_sequence_windows(
            data.validation_frame,
            window_size=int(model_params["window_size"]),
            sensors=SENSOR_COLUMNS,
            label_column=None,
        )
        if not len(validation_windows):
            raise ValueError("No full validation windows are available.")
        scores = model.score_windows(validation_windows)
        labels = data.validation_frame[["unit", "cycle", "rul"]]
        score_frame = validation_meta.merge(
            labels,
            left_on=["unit", "end_cycle"],
            right_on=["unit", "cycle"],
            how="left",
            validate="one_to_one",
        )
        if score_frame["rul"].isna().any():
            raise RuntimeError("Could not align autoencoder scores with validation RUL.")
        score_frame["reconstruction_mse"] = scores
        spearman = score_frame["reconstruction_mse"].rank().corr(
            score_frame["rul"].rank()
        )
        metrics = {
            "development_early_life_windows": int(len(train_windows)),
            "validation_scored_windows": int(len(validation_windows)),
            "validation_reconstruction_mse_mean": float(scores.mean()),
            "validation_reconstruction_mse_median": float(np.median(scores)),
            "validation_reconstruction_mse_std": float(scores.std()),
            "spearman_reconstruction_error_vs_rul": (
                float(spearman) if pd.notna(spearman) else None
            ),
            "interpretation": (
                "Unsupervised health signal only; C-MAPSS has no cycle-level "
                "anomaly labels."
            ),
        }
        _save_neural_model(model, model_path)
        _write_score_files(directory, metrics, autoencoder_scores=score_frame)
        write_manifest(
            manifest_path,
            {
                "model_name": config.model_name,
                "identity": identity,
                "artifact_file": model_path.name,
                "artifact_sha256": sha256_file(model_path),
                "validation_metrics": metrics,
                "software_versions": software_versions(config.model_name),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return metrics

    if "autoencoder_validation_scores.csv" not in {
        path.name for path in directory.iterdir()
    }:
        raise FileNotFoundError(
            f"Saved autoencoder is missing validation scores: {directory}"
        )
    return metrics


def train_models(
    *,
    data_dir: str | Path = PROJECT_ROOT / "CMAPSSData",
    output_dir: str | Path = PROJECT_ROOT / "artifacts",
    subset: str = "FD001",
    models: tuple[str, ...] = ("mean", "ridge", "random_forest", "hist_gradient_boosting"),
    feature_mode: str = "causal",
    rul_cap: int | None = None,
    seed: int = 42,
    validation_fraction: float = 0.2,
    rolling_windows: tuple[int, ...] = (5, 20),
    endpoint_fractions: tuple[float, ...] = (0.5, 0.7, 0.9),
    aficv_threshold: float | None = None,
    sequence_window_size: int = 30,
    autoencoder_window_size: int = 20,
    early_fraction: float = 0.2,
    epochs: int = 30,
    batch_size: int = 256,
    hidden_size: int = 32,
    learning_rate: float = 1e-3,
    reuse: bool = True,
) -> pd.DataFrame:
    """Fit selected candidates or load matching local artifacts; never load test files."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}.")
    if not models:
        raise ValueError("Select at least one model.")
    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model names {unknown}; expected one of {MODEL_NAMES}.")
    if len(set(models)) != len(models):
        raise ValueError("Model names must be unique within one run.")
    if aficv_threshold is not None and set(models) != {"random_forest"}:
        raise ValueError(
            "AFICv is currently a Random Forest-only run; select only random_forest."
        )
    if set(models) & {"mlp", "cnn", "lstm", "cnn_lstm", "autoencoder", "tinyml_rul", "hybrid_tinyml_autoencoder_rul"}:
        require_torch()

    rows: list[dict[str, Any]] = []
    data_root = Path(data_dir)
    artifacts_root = Path(output_dir)
    for model_name in models:
        config, directory, prepared = _make_config(
            model_name,
            subset=subset,
            data_dir=data_root,
            output_dir=artifacts_root,
            seed=seed,
            validation_fraction=validation_fraction,
            feature_mode=feature_mode,
            rul_cap=rul_cap,
            rolling_windows=rolling_windows,
            endpoint_fractions=endpoint_fractions,
            aficv_threshold=aficv_threshold,
            sequence_window_size=sequence_window_size,
            autoencoder_window_size=autoencoder_window_size,
            early_fraction=early_fraction,
            epochs=epochs,
            batch_size=batch_size,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
        )
        if model_name in TABULAR_MODEL_NAMES:
            metrics = _fit_tabular(config, directory, prepared, reuse=reuse)
        elif model_name == "mlp":
            metrics = _fit_mlp(config, directory, prepared, reuse=reuse)
        elif model_name in {"cnn", "lstm", "cnn_lstm"}:
            metrics = _fit_sequence(config, directory, prepared, reuse=reuse)
        elif model_name in {"tinyml_rul", "hybrid_tinyml_autoencoder_rul"}:
            metrics = _fit_compact_rul_network(config, directory, prepared, reuse=reuse)
        else:
            metrics = _fit_autoencoder(config, directory, prepared, reuse=reuse)
        rows.append(
            {
                "subset": subset,
                "model": model_name,
                "artifact_dir": str(directory),
                "validation_metrics": json.dumps(metrics, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _parse_fractions(value: str) -> tuple[float, ...]:
    try:
        fractions = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated numeric fractions.") from error
    if not fractions or any(not 0 < fraction < 1 for fraction in fractions):
        raise argparse.ArgumentTypeError("Fractions must be strictly between 0 and 1.")
    return fractions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "CMAPSSData")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--subset", choices=SUBSETS, default="FD001")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=["mean", "ridge", "random_forest", "hist_gradient_boosting"],
    )
    parser.add_argument(
        "--feature-mode",
        choices=("raw", "causal"),
        default="causal",
        help="Tabular/MLP feature mode; sequence models use raw sensor windows.",
    )
    parser.add_argument("--rul-cap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--rolling-windows", type=int, nargs="+", default=[5, 20])
    parser.add_argument(
        "--endpoint-fractions", type=_parse_fractions, default=(0.5, 0.7, 0.9)
    )
    parser.add_argument("--aficv-threshold", type=float, default=None)
    parser.add_argument("--sequence-window-size", type=int, default=30)
    parser.add_argument("--autoencoder-window-size", type=int, default=20)
    parser.add_argument("--early-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Refuse to reuse an existing matching artifact (does not overwrite it).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_models(
        data_dir=args.data_dir,
        output_dir=args.artifact_dir,
        subset=args.subset,
        models=tuple(args.models),
        feature_mode=args.feature_mode,
        rul_cap=args.rul_cap,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        rolling_windows=tuple(args.rolling_windows),
        endpoint_fractions=args.endpoint_fractions,
        aficv_threshold=args.aficv_threshold,
        sequence_window_size=args.sequence_window_size,
        autoencoder_window_size=args.autoencoder_window_size,
        early_fraction=args.early_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        reuse=not args.no_reuse,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
