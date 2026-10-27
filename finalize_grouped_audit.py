"""Materialize reports and final validation-selected artifact after the grouped audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from approaches.feature_variants import CausalFeatureVariant
from approaches.tabular_models import fit_tabular_model
from approaches.training_common import software_versions
from cmapss_data import add_training_rul, read_cmapss_file
from run_fd001_grouped_audit import (
    ARTIFACTS, CANDIDATES, DATA, REPORTS, SEED, candidate_feature_variant,
    draw_error_analysis, write_markdown_reports,
)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    current_metrics = REPORTS / "metrics_summary.csv"
    prior_metrics = REPORTS / "metrics_summary_prior_fixed_split.csv"
    if current_metrics.exists() and not prior_metrics.exists():
        current_metrics.replace(prior_metrics)
    summary = pd.read_csv(REPORTS / "metrics_summary_grouped_audit.csv").sort_values(["phm08_mean", "rmse"], ignore_index=True)
    selected = str(summary.iloc[0]["candidate"])
    summary["selected_by_validation"] = summary["candidate"].eq(selected)
    summary.to_csv(REPORTS / "metrics_summary.csv", index=False)
    summary.to_csv(REPORTS / "feature_target_ablation.csv", index=False)

    prediction_paths = sorted(REPORTS.glob("predictions_validation_*.csv"))
    frames = [pd.read_csv(path) for path in prediction_paths if "hgb_cap125_no_absolute_cycle" not in path.name]
    all_predictions = pd.concat(frames, ignore_index=True)
    all_predictions.to_csv(REPORTS / "predictions_oof.csv", index=False)
    selected_predictions = all_predictions.loc[all_predictions["candidate"] == selected].copy()
    selected_predictions.to_csv(REPORTS / "predictions_oof_selected.csv", index=False)
    draw_error_analysis(all_predictions, selected, REPORTS)

    train = read_cmapss_file(DATA / "train_FD001.txt").sort_values(["unit", "cycle"], kind="stable")
    write_markdown_reports(train, summary, selected, REPORTS)
    candidate = next(item for item in CANDIDATES if item.name == selected)
    baseline = summary.loc[summary["candidate"] == "hgb_control_cap125_w10_30"].iloc[0]
    (REPORTS / "baseline_reproduction.md").write_text(
        "# Baseline reproduction\n\n"
        "The previous selected HGB configuration, capped at 125 with causal 10/30 features, was rerun on a stronger five-fold engine-grouped validation protocol with fixed 10/25/50/75/100-cycle pseudo-censoring horizons. "
        f"It achieved pooled RMSE {baseline.rmse:.3f}, MAE {baseline.mae:.3f}, R2 {baseline.r2:.3f}, and PHM08 mean {baseline.phm08_mean:.3f}. "
        "These metrics differ from the older fixed 60-case split because the splits and endpoint schedule are deliberately different.\n",
        encoding="utf-8",
    )
    feature_rows = [
        "# Feature and target ablation\n",
        "All results were generated on the same five engine folds and predeclared horizons. Lower RMSE, MAE, and PHM08 are better.\n",
        "| Candidate | Cap | Windows | RMSE | MAE | R2 | PHM08 mean | Result |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary.itertuples(index=False):
        outcome = "Selected by validation" if row.candidate == selected else "Controlled comparison"
        feature_rows.append(f"| {row.candidate} | {row.rul_cap} | {row.windows} | {row.rmse:.3f} | {row.mae:.3f} | {row.r2:.3f} | {row.phm08_mean:.3f} | {outcome} |")
    (REPORTS / "feature_target_ablation.md").write_text("\n".join(feature_rows) + "\n", encoding="utf-8")
    (REPORTS / "experiment_plan.md").write_text(
        "# Audit-improve experiment plan\n\n"
        "Completed validation-only, five-fold engine-grouped suite: HGB caps 100/125/150/uncapped; causal 5/20 versus 10/30 features; initial-reading deltas; corrected time-free ablation; development-only near-constant sensor removal; two HGB parameter/loss choices; random forest; ExtraTrees; and a predeclared equal OOF blend. The official test files were not read for these experiments.\n\n"
        "Selection ranks pooled validation PHM08 mean and then RMSE. The cap-100 candidate is locked for a separately labelled historical benchmark comparison; no official result will feed back into model selection.\n",
        encoding="utf-8",
    )

    full = add_training_rul(train, cap=candidate.rul_cap)
    variant = candidate_feature_variant(candidate).fit(full)
    features = variant.transform(full)
    bundle = fit_tabular_model(candidate.family, features, full["rul"].to_numpy(), full["unit"].to_numpy(), seed=SEED, model_params=candidate.model_params)
    selected_dir = ARTIFACTS / "selected_validation_candidate"
    selected_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": bundle, "feature_variant": variant, "candidate": candidate}, selected_dir / "model.joblib")
    (selected_dir / "manifest.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate.__dict__,
        "software_versions": software_versions(candidate.family),
        "selection_protocol": "five-fold engine-grouped validation only; official test not accessed during selection",
        "validation_summary": summary.loc[summary["candidate"] == selected].iloc[0].to_dict(),
    }, indent=2, default=str), encoding="utf-8")
    print(f"Selected: {selected}")
    print(selected_dir)


if __name__ == "__main__":
    main()
