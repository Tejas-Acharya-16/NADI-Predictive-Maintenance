"""Train on FD004 and test locked source-selected HGB on FD001/2/3.

Model selection uses FD004 engine-grouped validation only. FD001, FD002, and
FD003 official endpoints are used only after the FD004 configuration is locked.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from approaches.condition_normalization import ConditionAwareNormalizer
from approaches.feature_variants import CausalFeatureVariant
from approaches.tabular_models import fit_tabular_model
from approaches.training_common import phm08_contributions, regression_metrics, software_versions
from cmapss_data import add_training_rul, load_subset


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "CMAPSSData"
REPORTS = ROOT / "reports" / "fd004_transfer"
ARTIFACTS = ROOT / "artifacts" / "FD004" / "transfer"
HORIZONS = (10, 25, 50, 75, 100, 125)
SEED = 20261021
HGB = {"max_iter": 180, "learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0, "early_stopping": False}


@dataclass(frozen=True)
class TransferCandidate:
    name: str
    rul_cap: int
    condition_normalized: bool


CANDIDATES = (
    TransferCandidate("fd004_hgb_cap100_condition_normalized", 100, True),
    TransferCandidate("fd004_hgb_cap125_condition_normalized", 125, True),
    TransferCandidate("fd004_hgb_cap125_raw_sensor_control", 125, False),
)


def endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unit, group in frame.groupby("unit", sort=True):
        last = int(group["cycle"].max()); first = int(group["cycle"].min())
        for horizon in HORIZONS:
            if last - horizon >= first:
                rows.append({"unit": int(unit), "cutoff_cycle": last - horizon, "actual_rul": horizon, "horizon": horizon})
    result = pd.DataFrame(rows)
    if result["unit"].nunique() != frame["unit"].nunique():
        raise ValueError("Every FD004 validation engine must support all declared horizons.")
    return result


def select_features(frame: pd.DataFrame, features: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["unit", "cycle"], kind="stable")
    lookup = features.loc[ordered.index].copy()
    lookup["_unit"] = ordered["unit"].to_numpy(); lookup["_cycle"] = ordered["cycle"].to_numpy()
    result = lookup.set_index(["_unit", "_cycle"]).loc[pd.MultiIndex.from_arrays([metadata["unit"].to_numpy(), metadata["cutoff_cycle"].to_numpy()])].copy()
    result.index = metadata.index
    return result


def transform_pair(development: pd.DataFrame, validation: pd.DataFrame, candidate: TransferCandidate):
    normalizer = ConditionAwareNormalizer().fit(development) if candidate.condition_normalized else None
    if normalizer is not None:
        development_input, development_condition = normalizer.transform(development)
        validation_input, validation_condition = normalizer.transform(validation)
    else:
        development_input, validation_input = development, validation
        development_condition = validation_condition = np.full(len(development), -1, dtype=int)
    variant = CausalFeatureVariant(windows=(10, 30)).fit(development_input)
    return variant, normalizer, variant.transform(development_input), variant.transform(validation_input), development_condition, validation_condition


def prediction_frame(metadata: pd.DataFrame, pred: np.ndarray, candidate: TransferCandidate, fold: int) -> pd.DataFrame:
    result = metadata.copy(); result["candidate"] = candidate.name; result["fold"] = fold
    result["predicted_rul"] = np.maximum(np.asarray(pred, dtype=float), 0.0)
    result["signed_error"] = result["predicted_rul"] - result["actual_rul"]
    result["absolute_error"] = result["signed_error"].abs()
    result["phm08_contribution"] = phm08_contributions(result["actual_rul"].to_numpy(), result["predicted_rul"].to_numpy())
    return result


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True); ARTIFACTS.mkdir(parents=True, exist_ok=True)
    source = load_subset(DATA, "FD004")["train"].sort_values(["unit", "cycle"], kind="stable")
    candidate_predictions: dict[str, list[pd.DataFrame]] = {x.name: [] for x in CANDIDATES}
    split_files = REPORTS / "splits"; split_files.mkdir(parents=True, exist_ok=True)
    for fold, (dev_i, val_i) in enumerate(GroupKFold(n_splits=3).split(source, groups=source["unit"]), start=1):
        dev_raw, val_raw = source.iloc[dev_i].copy(), source.iloc[val_i].copy()
        pd.DataFrame({"unit": sorted(source.unit.unique()), "split": np.where(pd.Series(sorted(source.unit.unique())).isin(set(dev_raw.unit)), "development", "validation")}).to_csv(split_files / f"fd004_fold_{fold}_engines.csv", index=False)
        metadata = endpoints(add_training_rul(val_raw))
        metadata.to_csv(split_files / f"fd004_fold_{fold}_endpoints.csv", index=False)
        for candidate in CANDIDATES:
            development = add_training_rul(dev_raw, cap=candidate.rul_cap); validation = add_training_rul(val_raw)
            _, _, x_dev, x_val, _, _ = transform_pair(development, validation, candidate)
            model = fit_tabular_model("hist_gradient_boosting", x_dev, development.rul.to_numpy(), development.unit.to_numpy(), seed=SEED + fold, model_params=HGB)
            pred = model.predict(select_features(validation, x_val, metadata))
            candidate_predictions[candidate.name].append(prediction_frame(metadata, pred, candidate, fold))

    rows = []
    for candidate in CANDIDATES:
        prediction = pd.concat(candidate_predictions[candidate.name], ignore_index=True)
        prediction.to_csv(REPORTS / f"validation_predictions_{candidate.name}.csv", index=False)
        metrics = regression_metrics(prediction.actual_rul.to_numpy(), prediction.predicted_rul.to_numpy())
        folds = [regression_metrics(g.actual_rul.to_numpy(), g.predicted_rul.to_numpy()) for _, g in prediction.groupby("fold")]
        rows.append({"candidate": candidate.name, "rul_cap": candidate.rul_cap, "condition_normalized": candidate.condition_normalized, "distinct_validation_engines": prediction.unit.nunique(), "validation_cases": len(prediction), **metrics, "rmse_fold_mean": float(np.mean([x["rmse"] for x in folds])), "rmse_fold_std": float(np.std([x["rmse"] for x in folds], ddof=1)), "phm08_mean_fold_mean": float(np.mean([x["phm08_mean"] for x in folds])), "phm08_mean_fold_std": float(np.std([x["phm08_mean"] for x in folds], ddof=1))})
    summary = pd.DataFrame(rows).sort_values(["phm08_mean", "rmse"], ignore_index=True)
    selected = str(summary.iloc[0].candidate); summary["selected_by_fd004_validation"] = summary.candidate.eq(selected)
    summary.to_csv(REPORTS / "fd004_validation_model_comparison.csv", index=False)

    choice = next(item for item in CANDIDATES if item.name == selected)
    full = add_training_rul(source, cap=choice.rul_cap)
    variant, normalizer, x_full, _, source_conditions, _ = transform_pair(full, full, choice)
    model = fit_tabular_model("hist_gradient_boosting", x_full, full.rul.to_numpy(), full.unit.to_numpy(), seed=SEED, model_params=HGB)
    artifact = ARTIFACTS / "selected_fd004_source_model"; artifact.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_variant": variant, "condition_normalizer": normalizer, "candidate": choice}, artifact / "model.joblib")
    (artifact / "manifest.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "source_subset": "FD004", "candidate": asdict(choice), "validation_protocol": "3-fold engine grouped, horizons 10/25/50/75/100/125", "software_versions": software_versions("hist_gradient_boosting"), "condition_counts_source": pd.Series(source_conditions).value_counts().sort_index().to_dict()}, indent=2, default=str), encoding="utf-8")

    test_rows = []
    for target_subset in ("FD001", "FD002", "FD003"):
        target = load_subset(DATA, target_subset)
        test = target["test"].sort_values(["unit", "cycle"], kind="stable")
        if normalizer is not None:
            test_input, conditions = normalizer.transform(test)
        else:
            test_input, conditions = test, np.full(len(test), -1, dtype=int)
        features = variant.transform(test_input)
        endpoint = test.groupby("unit", sort=True).tail(1)
        pred = model.predict(features.loc[endpoint.index])
        truth = target["test_rul"].sort_values("unit")
        if endpoint.unit.tolist() != truth.unit.tolist():
            raise ValueError(f"{target_subset}: endpoint / truth units do not align")
        result = pd.DataFrame({"subset": target_subset, "unit": endpoint.unit.to_numpy(), "actual_rul": truth.rul.to_numpy(), "predicted_rul": pred})
        result["signed_error"] = result.predicted_rul - result.actual_rul; result["absolute_error"] = result.signed_error.abs()
        result["phm08_contribution"] = phm08_contributions(result.actual_rul.to_numpy(), result.predicted_rul.to_numpy())
        result.to_csv(REPORTS / f"fd004_to_{target_subset}_endpoint_predictions.csv", index=False)
        metrics = regression_metrics(result.actual_rul.to_numpy(), result.predicted_rul.to_numpy())
        test_rows.append({"source_subset": "FD004", "target_subset": target_subset, "candidate": selected, "source_condition_normalized": choice.condition_normalized, "target_engines": len(result), "target_condition_counts": json.dumps(pd.Series(conditions).value_counts().sort_index().to_dict()), **metrics})
    transfer = pd.DataFrame(test_rows)
    transfer.to_csv(REPORTS / "fd004_to_other_subsets_transfer_metrics.csv", index=False)
    (REPORTS / "fd004_transfer_summary.md").write_text(
        "# FD004 to FD001 FD002 FD003 transfer study\n\n"
        "The source model was selected with FD004 training engines only. Three HGB candidates were compared by 3-fold engine-grouped FD004 validation with predeclared 10/25/50/75/100/125-cycle pseudo-censored endpoints. FD001, FD002, and FD003 official endpoint labels were not read until after source selection.\n\n"
        "## FD004 validation selection\n\n```text\n" + summary.round(4).to_string(index=False) + "\n```\n\n"
        "## Cross-subset official endpoint evaluation\n\n```text\n" + transfer.drop(columns=["target_condition_counts"]).round(4).to_string(index=False) + "\n```\n\n"
        "These are domain-transfer evaluations, not within-subset tuned benchmarks. FD004 has six operating conditions and two fault modes; target subsets differ in condition and fault coverage. The model uses FD004-fitted condition clusters and sensor standardization. No target-subset rows or labels are used to fit the transformer or model.\n",
        encoding="utf-8",
    )
    print(summary[["candidate", "rmse", "mae", "r2", "phm08_mean", "selected_by_fd004_validation"]].to_string(index=False))
    print(transfer[["target_subset", "rmse", "mae", "r2", "phm08_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
