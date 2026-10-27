"""Targeted, CPU-bounded diagnosis and model study for global C-MAPSS RUL.

Selection uses only FD001--FD004 training trajectories. Official endpoint labels
were previously inspected in earlier work; they are loaded only after the
selection is locked and are labelled a historical benchmark audit.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from approaches.condition_normalization import ConditionAwareNormalizer
from approaches.feature_variants import CausalFeatureVariant
from approaches.next_experiment_utils import (
    add_rul_buckets,
    endpoint_feature_matrix,
    enhanced_regression_metrics,
    full_range_endpoint_metadata,
)
from approaches.tabular_models import fit_tabular_model
from approaches.training_common import phm08_contributions, regression_metrics, software_versions
from cmapss_data import SETTING_COLUMNS, SUBSETS, add_training_rul, load_subset


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "CMAPSSData"
REPORTS = ROOT / "reports" / "next_experiment"
ARTIFACTS = ROOT / "artifacts" / "next_experiment"
SEED = 20261021
N_SPLITS = 3
HGB_BASE = {
    "max_iter": 160,
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    model_family: str = "hist_gradient_boosting"
    rul_cap: int | None = 125
    condition_mode: str = "normalized_with_settings"
    windows: tuple[int, ...] = (10, 30)
    initial_differences: bool = False
    drop_settings: bool = False
    model_params: dict | None = None


# The grid is intentionally compact: caps, condition treatment, one causal
# baseline-difference ablation, one window ablation, and bounded tree challengers.
CANDIDATES = (
    Candidate("hgb_cap100_normalized", rul_cap=100),
    Candidate("hgb_cap125_normalized", rul_cap=125),
    Candidate("hgb_cap150_normalized", rul_cap=150),
    Candidate("hgb_cap200_normalized", rul_cap=200),
    Candidate("hgb_uncapped_normalized", rul_cap=None),
    Candidate("hgb_cap125_raw_sensor_control", rul_cap=125, condition_mode="raw"),
    Candidate("hgb_cap125_normalized_no_settings", rul_cap=125, drop_settings=True),
    Candidate("hgb_cap125_initial_baseline_delta", rul_cap=125, initial_differences=True),
    Candidate("hgb_cap125_w5_20", rul_cap=125, windows=(5, 20)),
    Candidate(
        "hgb_tuned_cap125_normalized",
        rul_cap=125,
        model_params={
            "max_iter": 220, "learning_rate": 0.05, "max_leaf_nodes": 63,
            "min_samples_leaf": 12, "l2_regularization": 1.5, "early_stopping": False,
        },
    ),
    Candidate(
        "extra_trees_cpu_cap125_normalized",
        model_family="extra_trees",
        rul_cap=125,
        model_params={
            "n_estimators": 60, "min_samples_leaf": 8, "max_depth": 18,
            "max_features": 0.7, "bootstrap": True, "max_samples": 0.5, "n_jobs": 4,
        },
    ),
    Candidate(
        "random_forest_cpu_cap125_normalized",
        model_family="random_forest",
        rul_cap=125,
        model_params={
            "n_estimators": 60, "min_samples_leaf": 8, "max_depth": 18,
            "max_features": 0.7, "bootstrap": True, "max_samples": 0.5, "n_jobs": 4,
        },
    ),
)


def combined_training_data() -> pd.DataFrame:
    frames = []
    for index, subset in enumerate(SUBSETS, start=1):
        train = load_subset(DATA, subset)["train"].copy()
        train["subset"] = subset
        train["source_unit"] = train["unit"].astype(int)
        train["unit"] = index * 10_000 + train["source_unit"]
        frames.append(train)
    result = pd.concat(frames, ignore_index=True).sort_values(["unit", "cycle"], kind="stable")
    if result.duplicated(["unit", "cycle"]).any():
        raise ValueError("Global engine remapping did not produce unique unit/cycle keys.")
    return result.reset_index(drop=True)


def engine_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("unit", sort=True).agg(
        subset=("subset", "first"), source_unit=("source_unit", "first"), rows=("cycle", "size")
    ).reset_index()


def grouped_folds(inventory: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(splitter.split(inventory, inventory["subset"], groups=inventory["unit"]))


def transformed_features(
    development: pd.DataFrame, target: pd.DataFrame, candidate: Candidate
) -> tuple[CausalFeatureVariant, ConditionAwareNormalizer | None, pd.DataFrame, pd.DataFrame]:
    normalizer = None
    if candidate.condition_mode == "normalized_with_settings":
        normalizer = ConditionAwareNormalizer().fit(development)
        development_input, _ = normalizer.transform(development)
        target_input, _ = normalizer.transform(target)
    elif candidate.condition_mode == "raw":
        development_input, target_input = development, target
    else:
        raise ValueError(f"Unknown condition mode {candidate.condition_mode}")
    variant = CausalFeatureVariant(
        windows=candidate.windows,
        include_initial_differences=candidate.initial_differences,
    ).fit(development_input)
    x_development = variant.transform(development_input)
    x_target = variant.transform(target_input)
    if candidate.drop_settings:
        x_development = x_development.drop(columns=SETTING_COLUMNS)
        x_target = x_target.drop(columns=SETTING_COLUMNS)
    return variant, normalizer, x_development, x_target


def prediction_rows(metadata: pd.DataFrame, prediction: np.ndarray, candidate: Candidate, fold: int) -> pd.DataFrame:
    result = add_rul_buckets(metadata)
    result["candidate"] = candidate.name
    result["model_family"] = candidate.model_family
    result["fold"] = fold
    result["predicted_rul"] = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    result["signed_error"] = result["predicted_rul"] - result["actual_rul"]
    result["absolute_error"] = result["signed_error"].abs()
    result["phm08_contribution"] = phm08_contributions(
        result["actual_rul"].to_numpy(), result["predicted_rul"].to_numpy()
    )
    return result


def group_metrics(predictions: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    for name, block in predictions.groupby(group, sort=True):
        rows.append({
            group: name,
            "cases": len(block),
            "engines": block["unit"].nunique(),
            **enhanced_regression_metrics(
                block["actual_rul"].to_numpy(), block["predicted_rul"].to_numpy(), regression_metrics
            ),
        })
    return pd.DataFrame(rows)


def summary_row(candidate: Candidate, predictions: pd.DataFrame, fit_seconds: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    overall = enhanced_regression_metrics(
        predictions["actual_rul"].to_numpy(), predictions["predicted_rul"].to_numpy(), regression_metrics
    )
    by_subset = group_metrics(predictions, "subset")
    by_bucket = group_metrics(predictions, "rul_bucket")
    by_fold = group_metrics(predictions, "fold")
    return ({
        "candidate": candidate.name,
        "model_family": candidate.model_family,
        "rul_cap": "uncapped" if candidate.rul_cap is None else candidate.rul_cap,
        "condition_mode": candidate.condition_mode,
        "windows": "/".join(map(str, candidate.windows)),
        "initial_differences": candidate.initial_differences,
        "drop_settings": candidate.drop_settings,
        "validation_cases": len(predictions),
        "validation_engines": predictions["unit"].nunique(),
        "fit_seconds_total": fit_seconds,
        **overall,
        "rmse_fold_mean": float(by_fold["rmse"].mean()),
        "rmse_fold_std": float(by_fold["rmse"].std(ddof=1)),
        "phm08_mean_fold_mean": float(by_fold["phm08_mean"].mean()),
        "phm08_mean_fold_std": float(by_fold["phm08_mean"].std(ddof=1)),
        "macro_subset_rmse": float(by_subset["rmse"].mean()),
        "macro_subset_mae": float(by_subset["mae"].mean()),
        "macro_subset_phm08_mean": float(by_subset["phm08_mean"].mean()),
    }, by_subset, by_bucket)


def audit_baseline() -> None:
    """Diagnose the historically inspected baseline from its persisted predictions."""
    source = ROOT / "reports" / "domain_aware_next" / "global_train_all_test_all" / "official_test_predictions_all_subsets.csv"
    predictions = add_rul_buckets(pd.read_csv(source))
    rows = []
    for subset, block in predictions.groupby("subset", sort=True):
        raw = enhanced_regression_metrics(block.actual_rul.to_numpy(), block.predicted_rul.to_numpy(), regression_metrics)
        capped = enhanced_regression_metrics(
            np.minimum(block.actual_rul.to_numpy(), 100), np.minimum(block.predicted_rul.to_numpy(), 100), regression_metrics
        )
        rows.append({"subset": subset, "target_view": "raw_official", **raw})
        rows.append({"subset": subset, "target_view": "capped_at_100_diagnostic", **capped})
    pd.DataFrame(rows).to_csv(REPORTS / "baseline_raw_and_capped_metrics.csv", index=False)
    group_metrics(predictions, "subset").to_csv(REPORTS / "baseline_metrics_by_subset.csv", index=False)
    bucket = predictions.groupby(["subset", "rul_bucket"], sort=True).apply(
        lambda block: pd.Series(enhanced_regression_metrics(block.actual_rul.to_numpy(), block.predicted_rul.to_numpy(), regression_metrics)),
        include_groups=False,
    ).reset_index()
    bucket.to_csv(REPORTS / "baseline_error_by_rul_bucket.csv", index=False)
    label_counts = []
    for subset, block in predictions.groupby("subset", sort=True):
        label_counts.append({
            "subset": subset, "engines": len(block),
            "above_100_count": int((block.actual_rul > 100).sum()), "above_100_percent": float(100 * (block.actual_rul > 100).mean()),
            "above_125_count": int((block.actual_rul > 125).sum()), "above_125_percent": float(100 * (block.actual_rul > 125).mean()),
            "above_150_count": int((block.actual_rul > 150).sum()), "above_150_percent": float(100 * (block.actual_rul > 150).mean()),
        })
    pd.DataFrame(label_counts).to_csv(REPORTS / "baseline_test_target_tail_counts.csv", index=False)
    worst = predictions.sort_values("absolute_error", ascending=False).groupby("subset", sort=True).head(10)
    worst.to_csv(REPORTS / "baseline_worst_10_errors_by_subset.csv", index=False)


def make_plots(predictions: pd.DataFrame) -> None:
    figures = REPORTS / "figures"
    figures.mkdir(exist_ok=True)
    for subset, block in predictions.groupby("subset", sort=True):
        plt.figure(figsize=(5.2, 4.0))
        plt.scatter(block.actual_rul, block.predicted_rul, alpha=0.65, s=18)
        bound = max(block.actual_rul.max(), block.predicted_rul.max()) + 5
        plt.plot([0, bound], [0, bound], "k--", linewidth=1)
        plt.title(f"{subset}: official endpoint audit")
        plt.xlabel("Actual RUL cycles"); plt.ylabel("Predicted RUL cycles")
        plt.tight_layout(); plt.savefig(figures / f"{subset}_actual_vs_predicted.png", dpi=150); plt.close()
        plt.figure(figsize=(5.2, 4.0))
        plt.scatter(block.actual_rul, block.signed_error, alpha=0.65, s=18)
        plt.axhline(0, color="k", linestyle="--", linewidth=1)
        plt.title(f"{subset}: residuals")
        plt.xlabel("Actual RUL cycles"); plt.ylabel("Prediction minus actual")
        plt.tight_layout(); plt.savefig(figures / f"{subset}_residuals.png", dpi=150); plt.close()


def write_report(selection: pd.DataFrame, test_metrics: pd.DataFrame, candidate: Candidate) -> None:
    baseline_tail = pd.read_csv(REPORTS / "baseline_test_target_tail_counts.csv")
    baseline_buckets = pd.read_csv(REPORTS / "baseline_error_by_rul_bucket.csv")
    report = [
        "# Targeted global C-MAPSS RUL experiment\n",
        "## Protocol\n",
        "All candidates use the same three subset-stratified engine-grouped folds. Each held-out engine contributes causal cutoffs at 15%, 30%, 45%, 60%, 75%, and 90% of its observed training lifetime. This expands validation beyond the prior 10-to-125-cycle schedule. Feature transforms, operating-condition clustering, normalization and model fits use development engines only. Official test labels were previously inspected, so the final test stage is a historical benchmark audit and never selects a candidate.\n",
        "The PHM08 contribution uses `d = prediction - truth`: `exp(-d/13)-1` for `d < 0`, and `exp(d/10)-1` otherwise. Tables report both PHM08 total and mean per endpoint. Selection is predeclared as lowest macro-average subset PHM08 mean, then lowest pooled RMSE.\n",
        "## Official-test high-RUL tail in the existing baseline\n\n```text\n",
        baseline_tail.round(3).to_string(index=False), "\n```\n",
        "## Baseline raw-RUL error by target bucket\n\n```text\n",
        baseline_buckets[["subset", "rul_bucket", "samples", "rmse", "mae", "bias_predicted_minus_actual", "phm08_mean"]].round(3).to_string(index=False), "\n```\n",
        "## Validation candidate comparison\n\n```text\n",
        selection[["candidate", "rul_cap", "condition_mode", "rmse", "mae", "macro_subset_rmse", "macro_subset_phm08_mean", "phm08_mean", "fit_seconds_total", "selected_by_validation"]].round(4).to_string(index=False), "\n```\n",
        "## Locked historical official-test audit\n\n",
        f"Validation-selected candidate: `{candidate.name}`.\n\n```text\n",
        test_metrics[["subset", "engines", "rmse", "mae", "r2", "phm08_sum", "phm08_mean", "bias_predicted_minus_actual", "within_5_cycles_fraction", "within_10_cycles_fraction", "within_15_cycles_fraction"]].round(4).to_string(index=False), "\n```\n",
        "## Recommendation\n",
        "Retain the validation-selected global model only if its macro subset metrics and raw-RUL audit do not regress materially from the saved cap-100 baseline. FD002 and FD004 remain the deciding subsets because they include multiple operating conditions and FD004 includes multiple fault modes. The saved anomaly detector and deterministic work-order triage remain separate from RUL evaluation because C-MAPSS has no cycle-level anomaly labels.\n",
    ]
    (REPORTS / "NEXT_EXPERIMENT_REPORT.md").write_text("".join(report), encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    audit_baseline()
    train = combined_training_data()
    inventory = engine_inventory(train)
    inventory.to_csv(REPORTS / "training_engine_inventory.csv", index=False)
    splits = grouped_folds(inventory)
    split_dir = REPORTS / "splits"; split_dir.mkdir(exist_ok=True)
    candidate_rows: dict[str, list[pd.DataFrame]] = {item.name: [] for item in CANDIDATES}
    candidate_seconds: dict[str, float] = {item.name: 0.0 for item in CANDIDATES}

    for fold, (development_i, validation_i) in enumerate(splits, start=1):
        development_ids = set(inventory.iloc[development_i].unit)
        validation_ids = set(inventory.iloc[validation_i].unit)
        if development_ids & validation_ids:
            raise RuntimeError("Validation engine leakage detected.")
        development_raw = train.loc[train.unit.isin(development_ids)].copy()
        validation_raw = train.loc[train.unit.isin(validation_ids)].copy()
        split_inventory = inventory.copy()
        split_inventory["split"] = np.where(split_inventory.unit.isin(validation_ids), "validation", "development")
        split_inventory.to_csv(split_dir / f"fold_{fold}_engines.csv", index=False)
        metadata = full_range_endpoint_metadata(validation_raw)
        metadata.to_csv(split_dir / f"fold_{fold}_endpoints.csv", index=False)
        for candidate in CANDIDATES:
            development = add_training_rul(development_raw, cap=candidate.rul_cap)
            validation = add_training_rul(validation_raw)
            start = time.perf_counter()
            _, _, x_development, x_validation = transformed_features(development, validation, candidate)
            model = fit_tabular_model(
                candidate.model_family, x_development, development.rul.to_numpy(), development.unit.to_numpy(),
                seed=SEED + fold, model_params=HGB_BASE if candidate.model_params is None and candidate.model_family == "hist_gradient_boosting" else candidate.model_params,
            )
            candidate_seconds[candidate.name] += time.perf_counter() - start
            endpoint_x = endpoint_feature_matrix(validation, x_validation, metadata)
            prediction = model.predict(endpoint_x)
            candidate_rows[candidate.name].append(prediction_rows(metadata, prediction, candidate, fold))
        print(f"Completed full-range validation fold {fold}/{N_SPLITS}", flush=True)

    rows = []
    subset_tables = []
    bucket_tables = []
    for candidate in CANDIDATES:
        predictions = pd.concat(candidate_rows[candidate.name], ignore_index=True)
        predictions.to_csv(REPORTS / f"validation_predictions_{candidate.name}.csv", index=False)
        row, by_subset, by_bucket = summary_row(candidate, predictions, candidate_seconds[candidate.name])
        rows.append(row)
        by_subset.insert(0, "candidate", candidate.name); subset_tables.append(by_subset)
        by_bucket.insert(0, "candidate", candidate.name); bucket_tables.append(by_bucket)
    selection = pd.DataFrame(rows).sort_values(["macro_subset_phm08_mean", "rmse"], ignore_index=True)
    selected_name = str(selection.iloc[0].candidate)
    selection["selected_by_validation"] = selection.candidate.eq(selected_name)
    selection.to_csv(REPORTS / "validation_candidate_comparison.csv", index=False)
    pd.concat(subset_tables, ignore_index=True).to_csv(REPORTS / "validation_metrics_by_subset.csv", index=False)
    pd.concat(bucket_tables, ignore_index=True).to_csv(REPORTS / "validation_metrics_by_rul_bucket.csv", index=False)

    chosen = next(item for item in CANDIDATES if item.name == selected_name)
    full = add_training_rul(train, cap=chosen.rul_cap)
    start = time.perf_counter()
    variant, normalizer, x_full, _ = transformed_features(full, full, chosen)
    model = fit_tabular_model(
        chosen.model_family, x_full, full.rul.to_numpy(), full.unit.to_numpy(), seed=SEED,
        model_params=HGB_BASE if chosen.model_params is None and chosen.model_family == "hist_gradient_boosting" else chosen.model_params,
    )
    full_fit_seconds = time.perf_counter() - start
    artifact = ARTIFACTS / "selected_validation_candidate"; artifact.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_variant": variant, "condition_normalizer": normalizer, "candidate": asdict(chosen)}, artifact / "model.joblib")
    (artifact / "manifest.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "candidate": asdict(chosen), "seed": SEED,
        "selection_rule": "lowest macro subset PHM08 mean, then pooled RMSE", "full_fit_seconds": full_fit_seconds,
        "official_test_status": "historically inspected benchmark audit only", "versions": software_versions(chosen.model_family),
    }, indent=2, default=str), encoding="utf-8")

    # Official labels are read only after locking the candidate above.
    official_rows = []
    for subset in SUBSETS:
        target = load_subset(DATA, subset)
        test = target["test"].sort_values(["unit", "cycle"], kind="stable")
        if normalizer is None:
            test_input = test
        else:
            test_input, _ = normalizer.transform(test)
        features = variant.transform(test_input)
        if chosen.drop_settings:
            features = features.drop(columns=SETTING_COLUMNS)
        endpoint = test.groupby("unit", sort=True).tail(1).sort_values("unit")
        start = time.perf_counter()
        prediction = model.predict(features.loc[endpoint.index])
        latency_ms = 1000.0 * (time.perf_counter() - start) / len(endpoint)
        truth = target["test_rul"].sort_values("unit")
        if endpoint.unit.tolist() != truth.unit.tolist():
            raise ValueError(f"{subset}: test endpoint labels are not aligned by unit.")
        result = add_rul_buckets(pd.DataFrame({"subset": subset, "unit": endpoint.unit.to_numpy(), "actual_rul": truth.rul.to_numpy(), "predicted_rul": prediction}))
        result["signed_error"] = result.predicted_rul - result.actual_rul
        result["absolute_error"] = result.signed_error.abs()
        result["phm08_contribution"] = phm08_contributions(result.actual_rul.to_numpy(), result.predicted_rul.to_numpy())
        result["inference_latency_ms_per_engine"] = latency_ms
        result.to_csv(REPORTS / f"official_test_predictions_{subset}.csv", index=False)
        official_rows.append(result)
    official = pd.concat(official_rows, ignore_index=True)
    official.to_csv(REPORTS / "official_test_predictions_all_subsets.csv", index=False)
    official_metrics = group_metrics(official, "subset")
    official_metrics["inference_latency_ms_per_engine"] = official_metrics.subset.map(official.groupby("subset").inference_latency_ms_per_engine.first())
    official_metrics.to_csv(REPORTS / "official_test_metrics_by_subset.csv", index=False)
    group_metrics(official, "rul_bucket").to_csv(REPORTS / "official_test_metrics_by_rul_bucket.csv", index=False)
    pooled = enhanced_regression_metrics(official.actual_rul.to_numpy(), official.predicted_rul.to_numpy(), regression_metrics)
    pd.DataFrame([{"scope": "pooled", "engines": len(official), "model_size_bytes": (artifact / "model.joblib").stat().st_size, **pooled}]).to_csv(REPORTS / "official_test_metrics_pooled.csv", index=False)
    make_plots(official)
    write_report(selection, official_metrics, chosen)
    print(selection[["candidate", "rmse", "macro_subset_phm08_mean", "selected_by_validation"]].to_string(index=False))
    print(official_metrics[["subset", "rmse", "mae", "phm08_mean", "bias_predicted_minus_actual"]].to_string(index=False))


if __name__ == "__main__":
    main()
