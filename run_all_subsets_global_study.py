"""Global C-MAPSS RUL study: train on FD001--FD004 and test every subset.

Model selection uses only combined training trajectories.  Official test labels
are deliberately loaded after the validation-selected configuration is locked.
The runner is CPU-bounded: three HGB candidates, three engine-grouped folds,
and compact, endpoint-only validation predictions are retained.
"""

from __future__ import annotations

import json
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from approaches.condition_normalization import ConditionAwareNormalizer
from approaches.feature_variants import CausalFeatureVariant
from approaches.tabular_models import fit_tabular_model
from approaches.training_common import phm08_contributions, regression_metrics, software_versions
from cmapss_data import SUBSETS, add_training_rul, load_subset


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "CMAPSSData"
REPORTS = ROOT / "reports" / "all_subsets_global"
ARTIFACTS = ROOT / "artifacts" / "all_subsets_global"
HORIZONS = (10, 25, 50, 75, 100, 125)
SEED = 20261022
HGB = {
    "max_iter": 160,
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


@dataclass(frozen=True)
class GlobalCandidate:
    name: str
    rul_cap: int
    condition_normalized: bool


CANDIDATES = (
    GlobalCandidate("global_hgb_cap100_condition_normalized", 100, True),
    GlobalCandidate("global_hgb_cap125_condition_normalized", 125, True),
    GlobalCandidate("global_hgb_cap125_raw_sensor_control", 125, False),
)


def combined_training_data() -> pd.DataFrame:
    """Load every training file and make engine IDs unique across subsets."""
    frames: list[pd.DataFrame] = []
    for subset_index, subset in enumerate(SUBSETS, start=1):
        train = load_subset(DATA, subset)["train"].copy()
        train["subset"] = subset
        train["source_unit"] = train["unit"].astype(int)
        # Rolling features group by ``unit``.  C-MAPSS restarts each file at 1,
        # so a globally unique engine key is necessary before concatenation.
        train["unit"] = subset_index * 10_000 + train["source_unit"]
        frames.append(train)
    result = pd.concat(frames, ignore_index=True).sort_values(["unit", "cycle"], kind="stable")
    if result.duplicated(["unit", "cycle"]).any():
        raise ValueError("Combined training data contains duplicate global engine/cycle keys.")
    return result.reset_index(drop=True)


def endpoint_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Create fixed-horizon pseudo-censored endpoints for each validation engine."""
    rows: list[dict[str, int | str]] = []
    for unit, group in frame.groupby("unit", sort=True):
        first, last = int(group["cycle"].min()), int(group["cycle"].max())
        subset = str(group["subset"].iloc[0])
        source_unit = int(group["source_unit"].iloc[0])
        for horizon in HORIZONS:
            if last - horizon >= first:
                rows.append(
                    {
                        "unit": int(unit),
                        "source_unit": source_unit,
                        "subset": subset,
                        "cutoff_cycle": last - horizon,
                        "actual_rul": horizon,
                        "horizon": horizon,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No validation endpoints were generated.")
    return result


def select_endpoint_features(
    frame: pd.DataFrame, features: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Retrieve the matching causal feature vector for each endpoint."""
    ordered = frame.sort_values(["unit", "cycle"], kind="stable")
    lookup = features.loc[ordered.index].copy()
    lookup["_unit"] = ordered["unit"].to_numpy()
    lookup["_cycle"] = ordered["cycle"].to_numpy()
    keys = pd.MultiIndex.from_arrays([metadata["unit"].to_numpy(), metadata["cutoff_cycle"].to_numpy()])
    selected = lookup.set_index(["_unit", "_cycle"]).loc[keys].copy()
    selected.index = metadata.index
    return selected


def transform_pair(
    development: pd.DataFrame, target: pd.DataFrame, candidate: GlobalCandidate
) -> tuple[CausalFeatureVariant, ConditionAwareNormalizer | None, pd.DataFrame, pd.DataFrame, np.ndarray]:
    normalizer = ConditionAwareNormalizer().fit(development) if candidate.condition_normalized else None
    if normalizer is None:
        development_input, target_input = development, target
        target_conditions = np.full(len(target), -1, dtype=int)
    else:
        development_input, _ = normalizer.transform(development)
        target_input, target_conditions = normalizer.transform(target)
    variant = CausalFeatureVariant(windows=(10, 30)).fit(development_input)
    return variant, normalizer, variant.transform(development_input), variant.transform(target_input), target_conditions


def prediction_frame(
    metadata: pd.DataFrame, prediction: np.ndarray, candidate: GlobalCandidate, fold: int
) -> pd.DataFrame:
    result = metadata.copy()
    result["candidate"] = candidate.name
    result["fold"] = fold
    result["predicted_rul"] = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    result["signed_error"] = result["predicted_rul"] - result["actual_rul"]
    result["absolute_error"] = result["signed_error"].abs()
    result["phm08_contribution"] = phm08_contributions(
        result["actual_rul"].to_numpy(), result["predicted_rul"].to_numpy()
    )
    return result


def engine_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratify engine groups by source subset, while never splitting an engine."""
    engines = (
        frame.groupby("unit", sort=True)
        .agg(subset=("subset", "first"), source_unit=("source_unit", "first"), rows=("cycle", "size"))
        .reset_index()
    )
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    # Each row represents exactly one group; stratification keeps every FD set
    # represented in the held-out engine population of each fold.
    return list(splitter.split(engines, engines["subset"], groups=engines["unit"]))


def aggregate_metrics(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for value, group in predictions.groupby(group_column, sort=True):
        rows.append(
            {
                group_column: value,
                "cases": len(group),
                "engines": group["unit"].nunique(),
                **regression_metrics(group["actual_rul"].to_numpy(), group["predicted_rul"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame.loc[:, columns].round(4).to_string(index=False)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    combined = combined_training_data()
    engines = (
        combined.groupby("unit", sort=True)
        .agg(subset=("subset", "first"), source_unit=("source_unit", "first"), rows=("cycle", "size"))
        .reset_index()
    )
    engines.to_csv(REPORTS / "global_training_engine_inventory.csv", index=False)
    folds = engine_folds(combined)
    split_dir = REPORTS / "splits"
    split_dir.mkdir(exist_ok=True)
    candidate_predictions: dict[str, list[pd.DataFrame]] = {candidate.name: [] for candidate in CANDIDATES}

    for fold, (development_engine_i, validation_engine_i) in enumerate(folds, start=1):
        development_ids = set(engines.iloc[development_engine_i]["unit"])
        validation_ids = set(engines.iloc[validation_engine_i]["unit"])
        if development_ids & validation_ids:
            raise RuntimeError("Engine leakage detected in global grouped validation.")
        development_raw = combined.loc[combined["unit"].isin(development_ids)].copy()
        validation_raw = combined.loc[combined["unit"].isin(validation_ids)].copy()
        fold_inventory = engines.copy()
        fold_inventory["split"] = np.where(fold_inventory["unit"].isin(validation_ids), "validation", "development")
        fold_inventory.to_csv(split_dir / f"global_fold_{fold}_engines.csv", index=False)
        metadata = endpoint_metadata(add_training_rul(validation_raw))
        metadata.to_csv(split_dir / f"global_fold_{fold}_endpoints.csv", index=False)

        for candidate in CANDIDATES:
            development = add_training_rul(development_raw, cap=candidate.rul_cap)
            validation = add_training_rul(validation_raw)
            _, _, x_development, x_validation, _ = transform_pair(development, validation, candidate)
            model = fit_tabular_model(
                "hist_gradient_boosting",
                x_development,
                development["rul"].to_numpy(),
                development["unit"].to_numpy(),
                seed=SEED + fold,
                model_params=HGB,
            )
            endpoint_prediction = model.predict(select_endpoint_features(validation, x_validation, metadata))
            candidate_predictions[candidate.name].append(
                prediction_frame(metadata, endpoint_prediction, candidate, fold)
            )
        print(f"Completed global validation fold {fold}/3", flush=True)

    validation_rows: list[dict[str, object]] = []
    per_subset_validation: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        predictions = pd.concat(candidate_predictions[candidate.name], ignore_index=True)
        predictions.to_csv(REPORTS / f"validation_predictions_{candidate.name}.csv", index=False)
        metrics = regression_metrics(predictions["actual_rul"].to_numpy(), predictions["predicted_rul"].to_numpy())
        fold_metrics = aggregate_metrics(predictions, "fold")
        validation_rows.append(
            {
                "candidate": candidate.name,
                "rul_cap": candidate.rul_cap,
                "condition_normalized": candidate.condition_normalized,
                "validation_cases": len(predictions),
                "validation_engines": predictions["unit"].nunique(),
                **metrics,
                "rmse_fold_mean": float(fold_metrics["rmse"].mean()),
                "rmse_fold_std": float(fold_metrics["rmse"].std(ddof=1)),
                "phm08_mean_fold_mean": float(fold_metrics["phm08_mean"].mean()),
                "phm08_mean_fold_std": float(fold_metrics["phm08_mean"].std(ddof=1)),
            }
        )
        per_subset = aggregate_metrics(predictions, "subset")
        per_subset.insert(0, "candidate", candidate.name)
        per_subset_validation.append(per_subset)
    validation = pd.DataFrame(validation_rows).sort_values(["phm08_mean", "rmse"], ignore_index=True)
    selected_name = str(validation.iloc[0]["candidate"])
    validation["selected_by_global_validation"] = validation["candidate"].eq(selected_name)
    validation.to_csv(REPORTS / "global_validation_model_comparison.csv", index=False)
    validation_by_subset = pd.concat(per_subset_validation, ignore_index=True)
    validation_by_subset.to_csv(REPORTS / "global_validation_by_subset.csv", index=False)

    choice = next(candidate for candidate in CANDIDATES if candidate.name == selected_name)
    full_training = add_training_rul(combined, cap=choice.rul_cap)
    variant, normalizer, x_full, _, _ = transform_pair(full_training, full_training, choice)
    model = fit_tabular_model(
        "hist_gradient_boosting",
        x_full,
        full_training["rul"].to_numpy(),
        full_training["unit"].to_numpy(),
        seed=SEED,
        model_params=HGB,
    )
    artifact_dir = ARTIFACTS / "selected_global_model"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_variant": variant, "condition_normalizer": normalizer, "candidate": choice},
        artifact_dir / "model.joblib",
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_subsets": list(SUBSETS),
        "training_rows": len(combined),
        "training_engines": int(combined["unit"].nunique()),
        "selected_candidate": asdict(choice),
        "selection_protocol": "3-fold stratified engine-grouped validation across FD001-FD004; fixed horizons 10/25/50/75/100/125; rank by PHM08 mean then RMSE",
        "official_test_stage": "all four official test files and labels were read after model selection",
        "software_versions": software_versions("hist_gradient_boosting"),
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # The official test files are intentionally not loaded until the selected
    # configuration and full global model above have been locked.
    official_predictions: list[pd.DataFrame] = []
    for subset in SUBSETS:
        data = load_subset(DATA, subset)
        test = data["test"].sort_values(["unit", "cycle"], kind="stable")
        if normalizer is None:
            test_input = test
            conditions = np.full(len(test), -1, dtype=int)
        else:
            test_input, conditions = normalizer.transform(test)
        features = variant.transform(test_input)
        endpoint = test.groupby("unit", sort=True).tail(1).sort_values("unit")
        prediction = model.predict(features.loc[endpoint.index])
        truth = data["test_rul"].sort_values("unit")
        if endpoint["unit"].tolist() != truth["unit"].tolist():
            raise ValueError(f"{subset}: endpoint and truth engine ordering differs.")
        result = pd.DataFrame(
            {
                "subset": subset,
                "unit": endpoint["unit"].to_numpy(),
                "actual_rul": truth["rul"].to_numpy(),
                "predicted_rul": prediction,
                "assigned_condition": conditions[endpoint.index],
            }
        )
        result["signed_error"] = result["predicted_rul"] - result["actual_rul"]
        result["absolute_error"] = result["signed_error"].abs()
        result["phm08_contribution"] = phm08_contributions(result["actual_rul"].to_numpy(), result["predicted_rul"].to_numpy())
        result.to_csv(REPORTS / f"official_test_predictions_{subset}.csv", index=False)
        official_predictions.append(result)

    official = pd.concat(official_predictions, ignore_index=True)
    official.to_csv(REPORTS / "official_test_predictions_all_subsets.csv", index=False)
    official_summary = aggregate_metrics(official, "subset")
    official_summary.to_csv(REPORTS / "official_test_metrics_by_subset.csv", index=False)
    pooled = regression_metrics(official["actual_rul"].to_numpy(), official["predicted_rul"].to_numpy())
    pd.DataFrame([{"scope": "all_FD001_to_FD004_test_engines", "engines": len(official), **pooled}]).to_csv(
        REPORTS / "official_test_metrics_pooled.csv", index=False
    )

    report = (
        "# Global FD001--FD004 training and official-test evaluation\n\n"
        "## Protocol\n\n"
        "All four C-MAPSS training files were concatenated for model development. Engine IDs were remapped before concatenation, so rolling features never cross an engine or subset boundary. Three candidates were compared with 3-fold subset-stratified, engine-grouped validation. Each held-out engine contributes fixed pseudo-censored endpoints at 10, 25, 50, 75, 100, and 125 cycles before its training failure. The selection rule was lowest mean PHM08 score, then lowest RMSE. No official test file or RUL label was loaded until after that choice.\n\n"
        "## Global validation comparison\n\n```text\n"
        + markdown_table(validation, ["candidate", "rul_cap", "condition_normalized", "rmse", "mae", "r2", "phm08_mean", "selected_by_global_validation"])
        + "\n```\n\n"
        "## Validation detail by source subset\n\n```text\n"
        + markdown_table(validation_by_subset, ["candidate", "subset", "cases", "engines", "rmse", "mae", "r2", "phm08_mean"])
        + "\n```\n\n"
        "## Locked official-test results\n\n```text\n"
        + markdown_table(official_summary, ["subset", "cases", "engines", "rmse", "mae", "r2", "phm08_mean", "bias_predicted_minus_actual"])
        + "\n```\n\n"
        "## Pooled official-test result\n\n```text\n"
        + pd.DataFrame([pooled]).round(4).to_string(index=False)
        + "\n```\n\n"
        "The per-subset rows are the primary result because the four files have different operating-condition and fault-mode coverage. The pooled score is a descriptive total across all 707 official test engines.\n"
    )
    (REPORTS / "GLOBAL_TRAIN_ALL_TEST_ALL_REPORT.md").write_text(report, encoding="utf-8")
    print("\nValidation selection")
    print(validation[["candidate", "rmse", "mae", "r2", "phm08_mean", "selected_by_global_validation"]].to_string(index=False))
    print("\nLocked official-test results")
    print(official_summary[["subset", "rmse", "mae", "r2", "phm08_mean", "bias_predicted_minus_actual"]].to_string(index=False))
    print("\nPooled")
    print(pd.DataFrame([pooled]).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORTS)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    arguments = parser.parse_args()
    REPORTS = arguments.report_dir
    ARTIFACTS = arguments.artifact_dir
    main()
