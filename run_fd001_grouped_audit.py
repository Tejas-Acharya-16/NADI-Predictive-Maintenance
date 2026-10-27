"""Leakage-controlled FD001 grouped-validation audit and tree-model study.

This runner intentionally reads only ``train_FD001.txt``.  It uses five
engine-level folds and five predeclared right-censoring horizons per held-out
engine.  No official-test file, truth vector, residual, or metric is loaded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

from approaches.feature_variants import CausalFeatureVariant
from approaches.tabular_models import fit_tabular_model
from approaches.training_common import phm08_contributions, regression_metrics, software_versions
from cmapss_data import SENSOR_COLUMNS, add_training_rul, quality_report, read_cmapss_file


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "CMAPSSData"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts" / "FD001" / "grouped_audit"
HORIZONS = (10, 25, 50, 75, 100)
N_SPLITS = 5
SEED = 20261020


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    rul_cap: int | None = 125
    windows: tuple[int, ...] = (10, 30)
    initial_differences: bool = False
    low_variance_threshold: float | None = None
    include_absolute_cycle: bool = True
    model_params: dict[str, Any] | None = None


HGB_BASE = {
    "max_iter": 180, "learning_rate": 0.08, "max_leaf_nodes": 31,
    "min_samples_leaf": 20, "l2_regularization": 1.0, "early_stopping": False,
}

CANDIDATES = (
    Candidate("hgb_control_cap125_w10_30", "hist_gradient_boosting", model_params=HGB_BASE),
    Candidate("hgb_cap100_w10_30", "hist_gradient_boosting", rul_cap=100, model_params=HGB_BASE),
    Candidate("hgb_cap150_w10_30", "hist_gradient_boosting", rul_cap=150, model_params=HGB_BASE),
    Candidate("hgb_uncapped_w10_30", "hist_gradient_boosting", rul_cap=None, model_params=HGB_BASE),
    Candidate("hgb_cap125_w5_20", "hist_gradient_boosting", windows=(5, 20), model_params=HGB_BASE),
    Candidate("hgb_cap125_initial_delta", "hist_gradient_boosting", initial_differences=True, model_params=HGB_BASE),
    Candidate("hgb_cap125_no_absolute_cycle", "hist_gradient_boosting", include_absolute_cycle=False, model_params=HGB_BASE),
    Candidate("hgb_cap125_near_constant_removed", "hist_gradient_boosting", low_variance_threshold=0.01, model_params=HGB_BASE),
    Candidate("hgb_tuned_small_leaves", "hist_gradient_boosting", model_params={
        "max_iter": 240, "learning_rate": 0.05, "max_leaf_nodes": 15,
        "min_samples_leaf": 15, "l2_regularization": 2.0, "early_stopping": False,
    }),
    Candidate("hgb_absolute_error_loss", "hist_gradient_boosting", model_params={**HGB_BASE, "loss": "absolute_error"}),
    Candidate("random_forest_cap125", "random_forest", model_params={"n_estimators": 160, "min_samples_leaf": 2, "n_jobs": 4}),
    Candidate("extra_trees_cap125", "extra_trees", model_params={"n_estimators": 160, "min_samples_leaf": 2, "n_jobs": 4}),
)


def endpoint_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one prefix endpoint for each supported predeclared horizon."""
    rows: list[dict[str, int]] = []
    for unit, group in frame.groupby("unit", sort=True):
        last_cycle = int(group["cycle"].max())
        first_cycle = int(group["cycle"].min())
        for horizon in HORIZONS:
            cutoff = last_cycle - horizon
            if cutoff >= first_cycle:
                rows.append({"unit": int(unit), "cutoff_cycle": cutoff, "actual_rul": horizon, "horizon": horizon})
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No valid validation endpoints were constructed.")
    return result


def feature_endpoint_matrix(frame: pd.DataFrame, features: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["unit", "cycle"], kind="stable")
    table = features.loc[ordered.index].copy()
    table["_unit"] = ordered["unit"].to_numpy()
    table["_cycle"] = ordered["cycle"].to_numpy()
    keyed = table.set_index(["_unit", "_cycle"], drop=True)
    positions = pd.MultiIndex.from_arrays([metadata["unit"].to_numpy(), metadata["cutoff_cycle"].to_numpy()])
    selected = keyed.loc[positions].copy()
    selected.index = metadata.index
    return selected


def bootstrap_by_engine(predictions: pd.DataFrame, repetitions: int = 300) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    engines = np.sort(predictions["unit"].unique())
    values: dict[str, list[float]] = {"rmse": [], "mae": [], "phm08_mean": []}
    for _ in range(repetitions):
        sample = rng.choice(engines, size=len(engines), replace=True)
        blocks = [predictions.loc[predictions["unit"] == unit] for unit in sample]
        boot = pd.concat(blocks, ignore_index=True)
        metrics = regression_metrics(boot["actual_rul"].to_numpy(), boot["predicted_rul"].to_numpy())
        for name in values:
            values[name].append(float(metrics[name]))
    output: dict[str, float] = {}
    for name, value in values.items():
        output[f"{name}_bootstrap_p05"] = float(np.quantile(value, 0.05))
        output[f"{name}_bootstrap_p95"] = float(np.quantile(value, 0.95))
    return output


def prediction_rows(metadata: pd.DataFrame, prediction: np.ndarray, candidate: Candidate, fold: int) -> pd.DataFrame:
    out = metadata.copy()
    out["candidate"] = candidate.name
    out["model_family"] = candidate.family
    out["fold"] = fold
    out["predicted_rul"] = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    out["signed_error"] = out["predicted_rul"] - out["actual_rul"]
    out["absolute_error"] = out["signed_error"].abs()
    out["phm08_contribution"] = phm08_contributions(out["actual_rul"].to_numpy(), out["predicted_rul"].to_numpy())
    return out


def candidate_feature_variant(candidate: Candidate) -> CausalFeatureVariant:
    return CausalFeatureVariant(
        windows=candidate.windows,
        include_initial_differences=candidate.initial_differences,
        include_absolute_cycle=candidate.include_absolute_cycle,
        low_variance_threshold=candidate.low_variance_threshold,
    )


def summary_row(candidate: Candidate, predictions: pd.DataFrame) -> dict[str, Any]:
    metrics = regression_metrics(predictions["actual_rul"].to_numpy(), predictions["predicted_rul"].to_numpy())
    by_fold = []
    for _, fold_rows in predictions.groupby("fold", sort=True):
        by_fold.append(regression_metrics(fold_rows["actual_rul"].to_numpy(), fold_rows["predicted_rul"].to_numpy()))
    result = {
        "candidate": candidate.name,
        "model_family": candidate.family,
        "protocol": "five_fold_engine_grouped_horizons_10_25_50_75_100",
        "rul_cap": "uncapped" if candidate.rul_cap is None else candidate.rul_cap,
        "windows": "/".join(map(str, candidate.windows)),
        "initial_differences": candidate.initial_differences,
        "low_variance_threshold": candidate.low_variance_threshold,
        "include_absolute_cycle": candidate.include_absolute_cycle,
        "distinct_validation_engines": int(predictions["unit"].nunique()),
        "validation_cases": int(len(predictions)),
        **metrics,
    }
    for field in ("rmse", "mae", "r2", "phm08_sum", "phm08_mean"):
        result[f"{field}_fold_mean"] = float(np.mean([row[field] for row in by_fold]))
        result[f"{field}_fold_std"] = float(np.std([row[field] for row in by_fold], ddof=1))
    result.update(bootstrap_by_engine(predictions))
    return result


def draw_error_analysis(predictions: pd.DataFrame, selected: str, report_dir: Path) -> None:
    output = report_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    selected_rows = predictions.loc[predictions["candidate"] == selected].copy()
    plt.figure(figsize=(6.2, 4.6))
    plt.scatter(selected_rows.actual_rul, selected_rows.predicted_rul, s=16, alpha=0.65)
    bound = max(selected_rows.actual_rul.max(), selected_rows.predicted_rul.max()) + 5
    plt.plot([0, bound], [0, bound], "k--", linewidth=1)
    plt.xlabel("Actual RUL cycles")
    plt.ylabel("Predicted RUL cycles")
    plt.title("OOF actual versus predicted RUL")
    plt.tight_layout(); plt.savefig(output / "oof_actual_vs_predicted.png", dpi=150); plt.close()
    plt.figure(figsize=(6.2, 4.6))
    plt.scatter(selected_rows.actual_rul, selected_rows.signed_error, s=16, alpha=0.65)
    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Actual RUL cycles"); plt.ylabel("Prediction minus actual")
    plt.title("OOF residuals by RUL")
    plt.tight_layout(); plt.savefig(output / "oof_residual_vs_true.png", dpi=150); plt.close()
    plt.figure(figsize=(6.2, 4.6))
    plt.hist(selected_rows.signed_error, bins=25, color="#4C78A8", edgecolor="white")
    plt.xlabel("Prediction minus actual cycles"); plt.ylabel("Cases")
    plt.title("OOF residual distribution")
    plt.tight_layout(); plt.savefig(output / "oof_residual_histogram.png", dpi=150); plt.close()
    grouped = selected_rows.groupby("horizon", sort=True)["absolute_error"].mean()
    plt.figure(figsize=(6.2, 4.6))
    plt.bar(grouped.index.astype(str), grouped.values, color="#59A14F")
    plt.xlabel("Actual RUL band / horizon cycles"); plt.ylabel("Mean absolute error cycles")
    plt.title("OOF error by RUL horizon")
    plt.tight_layout(); plt.savefig(output / "oof_error_by_horizon.png", dpi=150); plt.close()
    engine_error = selected_rows.groupby("unit", sort=True)["absolute_error"].mean().sort_values(ascending=False).head(20)
    plt.figure(figsize=(7.0, 4.8))
    plt.bar(engine_error.index.astype(str), engine_error.values, color="#F28E2B")
    plt.xlabel("Engine ID"); plt.ylabel("Mean absolute error cycles")
    plt.title("20 highest OOF per-engine errors")
    plt.tight_layout(); plt.savefig(output / "oof_per_engine_absolute_error.png", dpi=150); plt.close()
    phm = selected_rows.sort_values("phm08_contribution", ascending=False).head(30)
    plt.figure(figsize=(7.0, 4.8))
    plt.bar(phm["unit"].astype(str) + ":" + phm["horizon"].astype(str), phm["phm08_contribution"], color="#E15759")
    plt.xticks(rotation=75, fontsize=7); plt.ylabel("PHM08 contribution")
    plt.title("30 largest OOF PHM08 contributions")
    plt.tight_layout(); plt.savefig(output / "oof_phm08_contributions.png", dpi=150); plt.close()


def write_markdown_reports(train: pd.DataFrame, summary: pd.DataFrame, selected: str, report_dir: Path) -> None:
    quality = quality_report(train, "FD001", "train")
    ranges = train[SENSOR_COLUMNS].agg(["min", "max", "std"]).T
    near_constant = ranges.index[ranges["std"] < 0.01].tolist()
    (report_dir / "data_audit.md").write_text(
        "# FD001 data audit\n\n"
        "FD001 is simulated turbofan telemetry, not a Siemens pump, truck, compressor, or motor-vibration dataset. "
        f"The training file contains {quality['rows']:,} rows, {quality['engines']} engines, and 26 raw columns. "
        "Checks found zero missing/non-finite cells, duplicate unit-cycle records, sorted-cycle faults, cycle gaps, or invalid IDs.\n\n"
        "## Feature audit\n\n"
        "The existing causal builder intentionally includes the current cycle and `cycles_seen`; it excludes engine ID. "
        "The grouped study adds a cycle-free ablation so this available-at-inference signal is measured rather than silently assumed. "
        f"Near-constant training sensors at standard deviation below 0.01: {', '.join(near_constant)}. "
        "Sensor removal is fit within each development fold only.\n\n"
        "## Sensor ranges\n\n```text\n" + ranges.round(6).to_string() + "\n```\n",
        encoding="utf-8",
    )
    (report_dir / "pipeline_map.md").write_text(
        "# Pipeline map\n\n"
        "`cmapss_data.py` reads and validates raw whitespace-delimited telemetry and builds training RUL labels. "
        "`approaches/rolling_features.py` creates current, trailing mean, standard deviation, lag-difference, and trailing-slope features per engine. "
        "`approaches/feature_variants.py` fits feature choices on development engines and transforms held-out engines causally. "
        "`run_fd001_grouped_audit.py` performs five-fold engine-grouped pseudo-censored validation without opening official-test files. "
        "`run_fd001_sequence_study.py` contains bounded CNN/LSTM/CNN-LSTM comparisons. "
        "`run_anomaly_triage_demo.py`, `approaches/anomaly_detection.py`, and `triage.py` produce the separate anomaly and deterministic work-order demo. "
        "`evaluate_selected_fd001.py` is a separate, explicitly labelled historical official-test stage.\n",
        encoding="utf-8",
    )
    (report_dir / "validation_protocol.md").write_text(
        "# Validation protocol\n\n"
        "The grouped audit uses five `GroupKFold` folds over engine ID. Every held-out engine produces one causal endpoint for each predeclared remaining-life horizon 10, 25, 50, 75, and 100 cycles when supported. FD001 training engines all support these horizons, producing 500 correlated endpoint cases from 100 distinct engines.\n\n"
        "Transforms, sensor screening, feature construction choices, and model fitting use development engines only. Endpoint metadata and predictions are saved with fold IDs under `reports/splits/` and `reports/predictions_oof.csv`. Metrics are pooled over endpoints and also summarized across folds; uncertainty is a 300-resample engine-block bootstrap. The official test set is not read by this runner.\n",
        encoding="utf-8",
    )
    best = summary.loc[summary["candidate"] == selected].iloc[0]
    lines = ["# Feature and target ablation\n", "All candidates use identical engine folds and horizons. Lower PHM08 and RMSE are better.\n", "| Candidate | Change | RMSE | MAE | PHM08 mean | Conclusion |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for row in summary.sort_values(["phm08_mean", "rmse"]).itertuples(index=False):
        conclusion = "Selected by validation" if row.candidate == selected else "Kept as controlled comparison"
        lines.append(f"| {row.candidate} | cap={row.rul_cap}; windows={row.windows}; initial_delta={row.initial_differences}; cycle={row.include_absolute_cycle} | {row.rmse:.3f} | {row.mae:.3f} | {row.phm08_mean:.3f} | {conclusion} |")
    lines.extend(["", f"The selected validation candidate is `{selected}` with PHM08 mean {best.phm08_mean:.3f} and RMSE {best.rmse:.3f}. This is a validation-only conclusion."])
    (report_dir / "feature_target_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data_dir: Path = DATA, report_dir: Path = REPORTS, artifact_dir: Path = ARTIFACTS) -> pd.DataFrame:
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = report_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train = read_cmapss_file(data_dir / "train_FD001.txt").sort_values(["unit", "cycle"], kind="stable")
    groups = train["unit"].to_numpy()
    gkf = GroupKFold(n_splits=N_SPLITS)
    candidate_rows: dict[str, list[pd.DataFrame]] = {candidate.name: [] for candidate in CANDIDATES}
    fitted_for_importance: tuple[Any, pd.DataFrame, pd.DataFrame, Candidate] | None = None

    for fold, (development_idx, validation_idx) in enumerate(gkf.split(train, groups=groups), start=1):
        development_raw = train.iloc[development_idx].copy()
        validation_raw = train.iloc[validation_idx].copy()
        split = pd.DataFrame({"unit": sorted(train["unit"].unique())})
        development_units = set(development_raw["unit"].unique())
        split["split"] = np.where(split["unit"].isin(development_units), "development", "validation")
        split.to_csv(splits_dir / f"grouped_audit_fold_{fold}_engines.csv", index=False)
        metadata = endpoint_rows(add_training_rul(validation_raw, cap=None))
        metadata.to_csv(splits_dir / f"grouped_audit_fold_{fold}_endpoints.csv", index=False)

        for candidate in CANDIDATES:
            development = add_training_rul(development_raw, cap=candidate.rul_cap)
            validation = add_training_rul(validation_raw, cap=None)
            variant = candidate_feature_variant(candidate).fit(development)
            x_development = variant.transform(development)
            x_validation = variant.transform(validation)
            bundle = fit_tabular_model(
                candidate.family, x_development, development["rul"].to_numpy(dtype=float),
                development["unit"].to_numpy(), seed=SEED + fold, model_params=candidate.model_params,
            )
            endpoint_x = feature_endpoint_matrix(validation, x_validation, metadata)
            prediction = bundle.predict(endpoint_x)
            candidate_rows[candidate.name].append(prediction_rows(metadata, prediction, candidate, fold))
            if candidate.name == "hgb_control_cap125_w10_30" and fold == 1:
                fitted_for_importance = (bundle, endpoint_x, metadata, candidate)

    all_rows = []
    summaries = []
    for candidate in CANDIDATES:
        predicted = pd.concat(candidate_rows[candidate.name], ignore_index=True)
        predicted.to_csv(report_dir / f"predictions_validation_{candidate.name}.csv", index=False)
        all_rows.append(predicted)
        summaries.append(summary_row(candidate, predicted))
    predictions = pd.concat(all_rows, ignore_index=True)
    predictions.to_csv(report_dir / "predictions_oof.csv", index=False)
    summary = pd.DataFrame(summaries).sort_values(["phm08_mean", "rmse"], ignore_index=True)

    # A predeclared equal blend is evaluated only from existing OOF predictions.
    pivot = predictions.pivot(index=["unit", "cutoff_cycle", "actual_rul", "horizon", "fold"], columns="candidate", values="predicted_rul").reset_index()
    blend_parts = ["hgb_control_cap125_w10_30", "extra_trees_cap125"]
    if set(blend_parts).issubset(pivot.columns):
        blend = pivot[["unit", "cutoff_cycle", "actual_rul", "horizon", "fold"]].copy()
        blend["candidate"] = "equal_blend_hgb_extra_trees"
        blend["model_family"] = "oof_equal_average"
        blend["predicted_rul"] = pivot[blend_parts].mean(axis=1)
        blend["signed_error"] = blend["predicted_rul"] - blend["actual_rul"]
        blend["absolute_error"] = blend["signed_error"].abs()
        blend["phm08_contribution"] = phm08_contributions(blend["actual_rul"].to_numpy(), blend["predicted_rul"].to_numpy())
        blend.to_csv(report_dir / "predictions_validation_equal_blend_hgb_extra_trees.csv", index=False)
        predictions = pd.concat([predictions, blend], ignore_index=True)
        blend_candidate = Candidate("equal_blend_hgb_extra_trees", "oof_equal_average")
        summary = pd.concat([summary, pd.DataFrame([summary_row(blend_candidate, blend)])], ignore_index=True).sort_values(["phm08_mean", "rmse"], ignore_index=True)

    selected = str(summary.iloc[0]["candidate"])
    summary["selected_by_validation"] = summary["candidate"].eq(selected)
    summary.to_csv(report_dir / "feature_target_ablation.csv", index=False)
    summary.to_csv(report_dir / "metrics_summary_grouped_audit.csv", index=False)
    draw_error_analysis(predictions, selected, report_dir)

    # Permutation importance is saved only for the HGB control on a held-out fold.
    if fitted_for_importance is not None:
        bundle, endpoint_x, metadata, _ = fitted_for_importance
        y = metadata["actual_rul"].to_numpy()
        result = permutation_importance(bundle.estimator, endpoint_x[bundle.feature_names], y, n_repeats=5, random_state=SEED, scoring="neg_mean_absolute_error")
        importance = pd.DataFrame({"feature": bundle.feature_names, "permutation_importance_mean": result.importances_mean, "permutation_importance_std": result.importances_std}).sort_values("permutation_importance_mean", ascending=False)
        importance.to_csv(report_dir / "feature_importance_hgb_control_fold1.csv", index=False)

    selected_rows = predictions.loc[predictions["candidate"] == selected].copy()
    selected_rows.to_csv(report_dir / "predictions_oof_selected.csv", index=False)
    selected_config = next((candidate for candidate in CANDIDATES if candidate.name == selected), None)
    if selected_config is not None:
        full = add_training_rul(train, cap=selected_config.rul_cap)
        variant = candidate_feature_variant(selected_config).fit(full)
        full_x = variant.transform(full)
        bundle = fit_tabular_model(selected_config.family, full_x, full["rul"].to_numpy(dtype=float), full["unit"].to_numpy(), seed=SEED, model_params=selected_config.model_params)
        selected_dir = artifact_dir / "selected_validation_candidate"
        selected_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": bundle, "feature_variant": variant, "candidate": asdict(selected_config)}, selected_dir / "model.joblib")
        (selected_dir / "manifest.json").write_text(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "candidate": asdict(selected_config), "software_versions": software_versions(selected_config.family), "selection_protocol": "validation only; official test not loaded"}, indent=2, default=str), encoding="utf-8")

    write_markdown_reports(train, summary, selected, report_dir)
    baseline = summary.loc[summary["candidate"] == "hgb_control_cap125_w10_30"].iloc[0]
    (report_dir / "baseline_reproduction.md").write_text(
        "# Baseline reproduction\n\n"
        "The prior fixed-split HGB control used a capped-125 target and causal 10/30 features. This audit reproduces the same named configuration on a stronger five-fold engine-grouped, five-horizon protocol. "
        f"Its pooled validation results are RMSE {baseline.rmse:.3f}, MAE {baseline.mae:.3f}, R2 {baseline.r2:.3f}, PHM08 mean {baseline.phm08_mean:.3f}. "
        "These numbers are not expected to equal the older fixed 60-case split because the held-out engine folds and endpoint horizons differ.\n",
        encoding="utf-8",
    )
    (report_dir / "experiment_plan.md").write_text(
        "# Audit-improve experiment plan\n\n"
        "Completed validation-only suite: HGB label-cap variants 100/125/150/uncapped; causal 5/20 versus 10/30 features; initial-reading deltas; a cycle-free ablation; development-only near-constant sensor removal; two HGB parameter/loss choices; random forest; ExtraTrees; and one predeclared equal OOF blend. "
        "All candidates use identical five GroupKFold splits and five fixed horizons. Selection ranks PHM08 mean then RMSE. The official benchmark was not loaded by this suite.\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--report-dir", type=Path, default=REPORTS)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    result = run(args.data_dir, args.report_dir, args.artifact_dir)
    print(result[["candidate", "rmse", "mae", "r2", "phm08_mean", "selected_by_validation"]].to_string(index=False))
