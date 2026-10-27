"""Materialize comparison and diagnostic deliverables from the completed experiment."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from approaches.next_experiment_utils import add_rul_buckets, enhanced_regression_metrics
from approaches.training_common import regression_metrics
from cmapss_data import SUBSETS, load_subset
from run_next_experiment import DATA, REPORTS, combined_training_data, group_metrics


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "reports" / "domain_aware_next" / "global_train_all_test_all" / "official_test_predictions_all_subsets.csv"


def scoped_metrics(frame: pd.DataFrame, scope: str) -> dict:
    return {"scope": scope, **enhanced_regression_metrics(frame.actual_rul.to_numpy(), frame.predicted_rul.to_numpy(), regression_metrics)}


def capped_view(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    result = frame.copy()
    result["actual_rul"] = result["actual_rul"].clip(upper=cap)
    result["predicted_rul"] = result["predicted_rul"].clip(upper=cap)
    return result


def comparison_table(baseline: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, old, new in [("pooled", baseline, selected), *[
        (subset, baseline.loc[baseline.subset == subset], selected.loc[selected.subset == subset])
        for subset in SUBSETS
    ]]:
        old_metrics = enhanced_regression_metrics(old.actual_rul.to_numpy(), old.predicted_rul.to_numpy(), regression_metrics)
        new_metrics = enhanced_regression_metrics(new.actual_rul.to_numpy(), new.predicted_rul.to_numpy(), regression_metrics)
        row = {"scope": scope, "engines": len(new)}
        for key in ("rmse", "mae", "r2", "phm08_mean", "phm08_sum", "bias_predicted_minus_actual", "within_5_cycles_fraction", "within_10_cycles_fraction", "within_15_cycles_fraction"):
            row[f"baseline_{key}"] = old_metrics[key]
            row[f"selected_{key}"] = new_metrics[key]
            row[f"absolute_change_{key}"] = new_metrics[key] - old_metrics[key]
            if old_metrics[key] != 0:
                row[f"percent_change_{key}"] = 100 * (new_metrics[key] - old_metrics[key]) / abs(old_metrics[key])
        rows.append(row)
    return pd.DataFrame(rows)


def condition_distribution() -> pd.DataFrame:
    payload = joblib.load(ROOT / "artifacts" / "next_experiment" / "selected_validation_candidate" / "model.joblib")
    normalizer = payload["condition_normalizer"]
    if normalizer is None:
        return pd.DataFrame()
    rows = []
    train = combined_training_data()
    _, assigned = normalizer.transform(train)
    counts = pd.Series(assigned).value_counts().sort_index()
    for condition, count in counts.items():
        rows.append({"split": "combined_train", "subset": "ALL", "condition": int(condition), "rows": int(count), "fraction": float(count / len(train))})
    for subset in SUBSETS:
        test = load_subset(DATA, subset)["test"]
        _, assigned = normalizer.transform(test)
        counts = pd.Series(assigned).value_counts().sort_index()
        for condition, count in counts.items():
            rows.append({"split": "official_test", "subset": subset, "condition": int(condition), "rows": int(count), "fraction": float(count / len(test))})
    return pd.DataFrame(rows)


def main() -> None:
    baseline = add_rul_buckets(pd.read_csv(BASELINE))
    selected = add_rul_buckets(pd.read_csv(REPORTS / "official_test_predictions_all_subsets.csv"))
    comparison = comparison_table(baseline, selected)
    comparison.to_csv(REPORTS / "baseline_vs_selected_official_test_comparison.csv", index=False)

    selected_cap = 200
    target_views = []
    for target_view, data in (("raw_official", selected), (f"capped_at_{selected_cap}_diagnostic", capped_view(selected, selected_cap))):
        for subset, group in data.groupby("subset", sort=True):
            target_views.append({"target_view": target_view, "subset": subset, **enhanced_regression_metrics(group.actual_rul.to_numpy(), group.predicted_rul.to_numpy(), regression_metrics)})
        target_views.append({"target_view": target_view, "subset": "POOLED", **enhanced_regression_metrics(data.actual_rul.to_numpy(), data.predicted_rul.to_numpy(), regression_metrics)})
    pd.DataFrame(target_views).to_csv(REPORTS / "selected_raw_and_capped_metrics.csv", index=False)
    group_metrics(selected, "rul_bucket").to_csv(REPORTS / "selected_error_by_rul_bucket.csv", index=False)
    condition_distribution().to_csv(REPORTS / "condition_distribution_train_and_test.csv", index=False)

    rows = comparison.set_index("scope")
    pooled = rows.loc["pooled"]
    f2 = rows.loc["FD002"]
    f4 = rows.loc["FD004"]
    f1 = rows.loc["FD001"]
    f3 = rows.loc["FD003"]
    selection = pd.read_csv(REPORTS / "validation_candidate_comparison.csv")
    selected_row = selection.loc[selection.selected_by_validation].iloc[0]
    tail = pd.read_csv(REPORTS / "baseline_test_target_tail_counts.csv")
    baseline_buckets = pd.read_csv(REPORTS / "baseline_error_by_rul_bucket.csv")
    selected_buckets = pd.read_csv(REPORTS / "selected_error_by_rul_bucket.csv")
    conditions = pd.read_csv(REPORTS / "condition_distribution_train_and_test.csv")
    report = f"""# Targeted global C-MAPSS RUL experiment

## What was run

This study preserves the prior outputs and writes only under `reports/next_experiment/` and `artifacts/next_experiment/`. It uses all FD001--FD004 training trajectories, without a subset-name feature. Every candidate uses the same three subset-stratified engine-grouped folds. Held-out engines contribute causal endpoints at 15%, 30%, 45%, 60%, 75%, and 90% of observed lifetime, so the validation targets cover the high-RUL region that was absent from the earlier 10--100-cycle FD001 protocol.

The selection objective was declared before scoring: lowest macro-average subset PHM08 mean, then lowest pooled RMSE. For `d = predicted RUL - actual RUL`, PHM08 is `exp(-d/13)-1` for `d < 0` and `exp(d/10)-1` otherwise. PHM08 totals and means are both saved. Official test scores had already been inspected in earlier work, so the final test stage is a historical benchmark audit and did not select a model.

## Baseline audit

The existing cap-100 baseline underestimates high-RUL engines. The official test label tail is:

```text
{tail.round(3).to_string(index=False)}
```

For example, 35.5% of FD002 and 40.3% of FD004 labels are above 100 cycles, and 13.9% and 16.9%, respectively, are above 150 cycles. The raw baseline error by subset and RUL bucket is saved in `baseline_error_by_rul_bucket.csv`; it shows very large underprediction in the `>125` bucket.

## Validation result

The selected candidate is `{selected_row.candidate}`: condition-normalized HGB with RUL cap {selected_row.rul_cap}, causal 10/30-cycle features, and raw operating settings. Its full-range validation metrics are RMSE {selected_row.rmse:.3f}, MAE {selected_row.mae:.3f}, macro subset RMSE {selected_row.macro_subset_rmse:.3f}, and macro subset PHM08 mean {selected_row.macro_subset_phm08_mean:.3f}. Full candidate results, fold statistics, fit times, per-subset results, and per-RUL-bucket results are in the validation CSV files.

The cap-100 and cap-125 candidates incurred very large PHM08 penalties at early, high-RUL validation endpoints. A 200-cycle cap materially reduced that high-tail underprediction, though the uncapped candidate produced costly optimistic errors. This is evidence that the prior cap-100 objective mismatched the broader deployment-like endpoint distribution; it does not prove a cap alone explains every subset error.

## Locked historical official-test audit: selected model versus previous baseline

| Scope | Baseline RMSE | Selected RMSE | RMSE change | Baseline PHM08 mean | Selected PHM08 mean | PHM08 change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FD001 | {f1.baseline_rmse:.3f} | {f1.selected_rmse:.3f} | {f1.absolute_change_rmse:+.3f} | {f1.baseline_phm08_mean:.3f} | {f1.selected_phm08_mean:.3f} | {f1.absolute_change_phm08_mean:+.3f} |
| FD002 | {f2.baseline_rmse:.3f} | {f2.selected_rmse:.3f} | {f2.absolute_change_rmse:+.3f} | {f2.baseline_phm08_mean:.3f} | {f2.selected_phm08_mean:.3f} | {f2.absolute_change_phm08_mean:+.3f} |
| FD003 | {f3.baseline_rmse:.3f} | {f3.selected_rmse:.3f} | {f3.absolute_change_rmse:+.3f} | {f3.baseline_phm08_mean:.3f} | {f3.selected_phm08_mean:.3f} | {f3.absolute_change_phm08_mean:+.3f} |
| FD004 | {f4.baseline_rmse:.3f} | {f4.selected_rmse:.3f} | {f4.absolute_change_rmse:+.3f} | {f4.baseline_phm08_mean:.3f} | {f4.selected_phm08_mean:.3f} | {f4.absolute_change_phm08_mean:+.3f} |
| Pooled | {pooled.baseline_rmse:.3f} | {pooled.selected_rmse:.3f} | {pooled.absolute_change_rmse:+.3f} | {pooled.baseline_phm08_mean:.3f} | {pooled.selected_phm08_mean:.3f} | {pooled.absolute_change_phm08_mean:+.3f} |

The selected model genuinely improves the historically inspected aggregate: pooled RMSE changes by {pooled.percent_change_rmse:+.1f}% and mean PHM08 changes by {pooled.percent_change_phm08_mean:+.1f}%. It improves both targeted multi-condition subsets: FD002 RMSE changes by {f2.percent_change_rmse:+.1f}% and FD004 by {f4.percent_change_rmse:+.1f}%. It regresses on FD001 and FD003, where its higher cap changes the bias from conservative to optimistic. This is a real tradeoff, not a universal per-subset win.

## Raw versus capped target views

`selected_raw_and_capped_metrics.csv` keeps raw official RUL as the operational view and contains the separately labelled cap-200 diagnostic view. `baseline_raw_and_capped_metrics.csv` provides the same split for the older cap-100 baseline. Neither capped view replaces the raw official results.

## Operating-condition and error findings

`condition_distribution_train_and_test.csv` records cluster frequencies from the full-training fitted six-condition normalizer. `baseline_worst_10_errors_by_subset.csv`, `selected_error_by_rul_bucket.csv`, and the per-subset actual-versus-predicted and residual plots provide the traceable error analysis. FD002 and FD004 remain difficult because their distribution includes more high-RUL cases and multiple operating regimes; FD004 also has multiple fault modes. The selected cap-200 model reduces their severe conservative high-tail errors but increases optimistic estimates in FD001 and FD003.

## Recommendation

For the Tata Elxsi case study, retain the cap-200 condition-normalized global HGB as the current global candidate because it was selected by internal grouped validation and improves the pooled, FD002, and FD004 historical audits. Present the FD001 and FD003 regressions explicitly. Do not represent the historical audit as a blind test. A next production decision should define whether avoiding early-conservative RUL estimates in multi-condition assets outweighs the increased risk of optimistic estimates in simpler regimes. A regime router or known-domain specialists can be a later diagnostic, but only if their routing inputs are available in deployment.

The anomaly score and deterministic rule-based work-order workflow remain separate from these RUL results. C-MAPSS has no per-cycle anomaly labels, so no anomaly-classification accuracy is claimed.
"""
    (REPORTS / "NEXT_EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")
    (REPORTS / "reproducibility.md").write_text(
        "# Reproducibility\n\n"
        "Run `& '.\\.venv\\Scripts\\python.exe' run_next_experiment.py`, then `& '.\\.venv\\Scripts\\python.exe' finalize_next_experiment.py`. "
        "The experiment uses seed 20261021, three subset-stratified engine-grouped folds, six full-range causal cutoffs, and no official test labels during selection. "
        "The test audit is historically inspected because prior work read it.\n",
        encoding="utf-8",
    )
    print(comparison[["scope", "baseline_rmse", "selected_rmse", "percent_change_rmse", "baseline_phm08_mean", "selected_phm08_mean", "percent_change_phm08_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
