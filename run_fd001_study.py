"""CPU-bounded, validation-only FD001 ablation and tree-model study.

This script deliberately never opens ``test_FD001.txt`` or ``RUL_FD001.txt``.
It selects a candidate from engine-grouped, right-censored validation cases and
saves a full-training artifact for one later, explicit official-test evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from approaches.feature_variants import CausalFeatureVariant, feature_variant_name
from approaches.training_common import (
    endpoint_metric_rows,
    phm08_contributions,
    regression_metrics,
    right_censored_endpoint_rows,
)
from cmapss_data import add_training_rul, read_cmapss_file, split_train_by_engine


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "CMAPSSData"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "FD001" / "validation_study"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    rul_cap: int | None
    windows: tuple[int, ...]
    initial_differences: bool = False
    low_variance_threshold: float | None = None
    model_params: dict[str, Any] | None = None


DEFAULT_HGB_PARAMS: dict[str, Any] = {
    "max_iter": 150,
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


EXPERIMENTS = (
    ExperimentSpec("control_uncapped_w5_20", None, (5, 20)),
    ExperimentSpec("label_cap125", 125, (5, 20)),
    ExperimentSpec("cap125_near_constant_removed", 125, (5, 20), low_variance_threshold=0.01),
    ExperimentSpec("cap125_w10_30", 125, (10, 30)),
    ExperimentSpec("cap125_w20_50", 125, (20, 50)),
    ExperimentSpec("cap125_initial_difference", 125, (5, 20), initial_differences=True),
    ExperimentSpec(
        "cap125_hgb_tuned",
        125,
        (5, 20),
        model_params={
            "max_iter": 250,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 15,
            "l2_regularization": 2.0,
            "early_stopping": False,
        },
    ),
)


def _prepared_split(data_dir: Path, rul_cap: int | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = read_cmapss_file(data_dir / "train_FD001.txt")
    development_raw, validation_raw, summary = split_train_by_engine(
        raw, validation_fraction=0.2, random_state=seed
    )
    development = add_training_rul(development_raw, cap=rul_cap).sort_values(["unit", "cycle"], kind="stable")
    validation = add_training_rul(validation_raw, cap=None).sort_values(["unit", "cycle"], kind="stable")
    return development, validation, summary


def _fit_experiment(spec: ExperimentSpec, data_dir: Path, output_dir: Path, seed: int) -> tuple[dict[str, Any], Path]:
    development, validation, split_summary = _prepared_split(data_dir, spec.rul_cap, seed)
    variant = CausalFeatureVariant(
        windows=spec.windows,
        include_initial_differences=spec.initial_differences,
        low_variance_threshold=spec.low_variance_threshold,
    ).fit(development)
    x_development = variant.transform(development)
    x_validation = variant.transform(validation)
    params = dict(DEFAULT_HGB_PARAMS)
    params.update(spec.model_params or {})
    params["random_state"] = seed
    model = HistGradientBoostingRegressor(**params).fit(
        x_development, development["rul"].to_numpy(dtype=float)
    )
    endpoint_x, endpoint_y, endpoint_metadata = right_censored_endpoint_rows(
        validation, x_validation, (0.5, 0.7, 0.9)
    )
    endpoint_prediction = np.maximum(model.predict(endpoint_x), 0.0)
    endpoint_report = endpoint_metadata.copy()
    endpoint_report["predicted_rul"] = endpoint_prediction
    endpoint_report["error_predicted_minus_actual"] = endpoint_prediction - endpoint_y
    endpoint_report["absolute_error"] = np.abs(endpoint_report["error_predicted_minus_actual"])
    endpoint_report["phm08_contribution"] = phm08_contributions(endpoint_y, endpoint_prediction)
    metrics = regression_metrics(endpoint_y, endpoint_prediction)
    record: dict[str, Any] = {
        "experiment": spec.name,
        "seed": seed,
        "protocol": "fixed engine split; validation prefixes at 50%, 70%, and 90% of each run-to-failure trajectory",
        "rul_cap": spec.rul_cap,
        "feature_variant": feature_variant_name(spec.windows, spec.initial_differences, spec.low_variance_threshold),
        "feature_count": int(x_development.shape[1]),
        "kept_sensor_count": len(variant.kept_sensors or []),
        "development_engines": int(development["unit"].nunique()),
        "validation_engines": int(validation["unit"].nunique()),
        "validation_cases": int(len(endpoint_report)),
        "model_params": json.dumps(params, sort_keys=True),
        **metrics,
    }
    destination = output_dir / f"{spec.name}__seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "variant": variant, "spec": asdict(spec), "seed": seed}, destination / "model.joblib")
    endpoint_report.to_csv(destination / "validation_endpoint_predictions.csv", index=False)
    split_summary.to_csv(destination / "split_engines.csv", index=False)
    (destination / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record, destination


def _fit_final(selected: ExperimentSpec, data_dir: Path, output_dir: Path, seed: int) -> Path:
    raw = read_cmapss_file(data_dir / "train_FD001.txt")
    training = add_training_rul(raw, cap=selected.rul_cap).sort_values(["unit", "cycle"], kind="stable")
    variant = CausalFeatureVariant(
        windows=selected.windows,
        include_initial_differences=selected.initial_differences,
        low_variance_threshold=selected.low_variance_threshold,
    ).fit(training)
    params = dict(DEFAULT_HGB_PARAMS)
    params.update(selected.model_params or {})
    params["random_state"] = seed
    model = HistGradientBoostingRegressor(**params).fit(
        variant.transform(training), training["rul"].to_numpy(dtype=float)
    )
    destination = output_dir.parent / "selected_hgb_full_training"
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "variant": variant, "spec": asdict(selected), "seed": seed}, destination / "model.joblib")
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "selection_protocol": "validation PHM08 mean, then validation RMSE",
                "selected_experiment": asdict(selected),
                "training_engines": 100,
                "test_data_loaded": False,
                "feature_variant": variant.metadata(),
                "model_params": params,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return destination


def run_study(data_dir: Path, report_dir: Path, artifact_dir: Path) -> pd.DataFrame:
    """Run bounded ablations, repeat the selected setting, and fit it on all training engines."""
    report_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for spec in EXPERIMENTS:
        record, _ = _fit_experiment(spec, data_dir, artifact_dir, seed=42)
        records.append(record)
    initial = pd.DataFrame(records).sort_values(["phm08_mean", "rmse", "mae"]).reset_index(drop=True)
    selected_name = str(initial.iloc[0]["experiment"])
    selected = next(spec for spec in EXPERIMENTS if spec.name == selected_name)
    for seed in (17, 73):
        record, _ = _fit_experiment(selected, data_dir, artifact_dir, seed=seed)
        records.append(record)
    summary = pd.DataFrame(records).sort_values(["experiment", "seed"]).reset_index(drop=True)
    summary.to_csv(report_dir / "metrics_summary.csv", index=False)
    repeated = summary.loc[summary["experiment"] == selected.name]
    repeated_summary = repeated[["rmse", "mae", "r2", "phm08_sum", "phm08_mean"]].agg(["mean", "std"]).T
    repeated_summary.to_csv(report_dir / "selected_repeat_summary.csv")
    final_path = _fit_final(selected, data_dir, artifact_dir, seed=42)
    (report_dir / "selected_experiment.json").write_text(
        json.dumps(
            {
                "selected_experiment": asdict(selected),
                "selection_seed": 42,
                "selection_metrics": initial.iloc[0].to_dict(),
                "repeat_metrics_file": "selected_repeat_summary.csv",
                "full_training_artifact": str(final_path),
                "official_test_loaded_during_study": False,
            },
            indent=2,
            default=str,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    summary = run_study(args.data_dir, args.report_dir, args.artifact_dir)
    print(summary[["experiment", "seed", "rmse", "mae", "r2", "phm08_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
