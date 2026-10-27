"""Run the corrected cycle-free HGB ablation on the audit's fixed grouped folds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from approaches.feature_variants import CausalFeatureVariant
from approaches.tabular_models import fit_tabular_model
from cmapss_data import add_training_rul, read_cmapss_file
from run_fd001_grouped_audit import (
    DATA, HGB_BASE, HORIZONS, REPORTS, SEED, Candidate, endpoint_rows,
    feature_endpoint_matrix, prediction_rows, summary_row,
)


def main() -> None:
    candidate = Candidate(
        "hgb_cap125_without_cycle_or_cycles_seen", "hist_gradient_boosting",
        rul_cap=125, windows=(10, 30), include_absolute_cycle=False,
        model_params=HGB_BASE,
    )
    train = read_cmapss_file(DATA / "train_FD001.txt").sort_values(["unit", "cycle"], kind="stable")
    rows = []
    groups = train["unit"].to_numpy()
    for fold, (dev_index, val_index) in enumerate(GroupKFold(n_splits=5).split(train, groups=groups), start=1):
        development = add_training_rul(train.iloc[dev_index].copy(), cap=candidate.rul_cap)
        validation = add_training_rul(train.iloc[val_index].copy(), cap=None)
        variant = CausalFeatureVariant(windows=candidate.windows, include_absolute_cycle=False).fit(development)
        x_dev = variant.transform(development)
        x_val = variant.transform(validation)
        model = fit_tabular_model(candidate.family, x_dev, development["rul"].to_numpy(), development["unit"].to_numpy(), seed=SEED + fold, model_params=candidate.model_params)
        metadata = endpoint_rows(validation)
        prediction = model.predict(feature_endpoint_matrix(validation, x_val, metadata))
        rows.append(prediction_rows(metadata, prediction, candidate, fold))
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(REPORTS / f"predictions_validation_{candidate.name}.csv", index=False)
    updated = pd.read_csv(REPORTS / "metrics_summary_grouped_audit.csv")
    updated = updated.loc[updated["candidate"] != "hgb_cap125_no_absolute_cycle"].copy()
    updated = pd.concat([updated, pd.DataFrame([summary_row(candidate, predictions)])], ignore_index=True)
    updated = updated.sort_values(["phm08_mean", "rmse"], ignore_index=True)
    updated["selected_by_validation"] = updated["candidate"].eq(updated.iloc[0]["candidate"])
    updated.to_csv(REPORTS / "metrics_summary_grouped_audit.csv", index=False)
    updated.to_csv(REPORTS / "feature_target_ablation.csv", index=False)
    print(updated[["candidate", "rmse", "mae", "r2", "phm08_mean", "selected_by_validation"]].to_string(index=False))


if __name__ == "__main__":
    main()
