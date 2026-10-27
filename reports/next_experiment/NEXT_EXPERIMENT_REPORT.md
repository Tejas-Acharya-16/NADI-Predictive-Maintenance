# Targeted global C-MAPSS RUL experiment

## What was run

This study preserves the prior outputs and writes only under `reports/next_experiment/` and `artifacts/next_experiment/`. It uses all FD001--FD004 training trajectories, without a subset-name feature. Every candidate uses the same three subset-stratified engine-grouped folds. Held-out engines contribute causal endpoints at 15%, 30%, 45%, 60%, 75%, and 90% of observed lifetime, so the validation targets cover the high-RUL region that was absent from the earlier 10--100-cycle FD001 protocol.

The selection objective was declared before scoring: lowest macro-average subset PHM08 mean, then lowest pooled RMSE. For `d = predicted RUL - actual RUL`, PHM08 is `exp(-d/13)-1` for `d < 0` and `exp(d/10)-1` otherwise. PHM08 totals and means are both saved. Official test scores had already been inspected in earlier work, so the final test stage is a historical benchmark audit and did not select a model.

## Baseline audit

The existing cap-100 baseline underestimates high-RUL engines. The official test label tail is:

```text
subset  engines  above_100_count  above_100_percent  above_125_count  above_125_percent  above_150_count  above_150_percent
 FD001      100               33             33.000               11             11.000                0              0.000
 FD002      259               92             35.521               57             22.008               36             13.900
 FD003      100               32             32.000               15             15.000                0              0.000
 FD004      248              100             40.323               67             27.016               42             16.935
```

For example, 35.5% of FD002 and 40.3% of FD004 labels are above 100 cycles, and 13.9% and 16.9%, respectively, are above 150 cycles. The raw baseline error by subset and RUL bucket is saved in `baseline_error_by_rul_bucket.csv`; it shows very large underprediction in the `>125` bucket.

## Validation result

The selected candidate is `hgb_cap200_normalized`: condition-normalized HGB with RUL cap 200, causal 10/30-cycle features, and raw operating settings. Its full-range validation metrics are RMSE 34.323, MAE 21.747, macro subset RMSE 34.049, and macro subset PHM08 mean 232277.881. Full candidate results, fold statistics, fit times, per-subset results, and per-RUL-bucket results are in the validation CSV files.

The cap-100 and cap-125 candidates incurred very large PHM08 penalties at early, high-RUL validation endpoints. A 200-cycle cap materially reduced that high-tail underprediction, though the uncapped candidate produced costly optimistic errors. This is evidence that the prior cap-100 objective mismatched the broader deployment-like endpoint distribution; it does not prove a cap alone explains every subset error.

## Locked historical official-test audit: selected model versus previous baseline

| Scope | Baseline RMSE | Selected RMSE | RMSE change | Baseline PHM08 mean | Selected PHM08 mean | PHM08 change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FD001 | 16.292 | 18.037 | +1.744 | 3.164 | 17.330 | +14.166 |
| FD002 | 31.128 | 21.738 | -9.390 | 61.477 | 21.065 | -40.412 |
| FD003 | 17.273 | 20.863 | +3.589 | 3.785 | 27.136 | +23.351 |
| FD004 | 33.391 | 22.630 | -10.761 | 62.574 | 49.772 | -12.802 |
| Pooled | 28.737 | 21.455 | -7.282 | 45.454 | 31.465 | -13.989 |

The selected model genuinely improves the historically inspected aggregate: pooled RMSE changes by -25.3% and mean PHM08 changes by -30.8%. It improves both targeted multi-condition subsets: FD002 RMSE changes by -30.2% and FD004 by -32.2%. It regresses on FD001 and FD003, where its higher cap changes the bias from conservative to optimistic. This is a real tradeoff, not a universal per-subset win.

## Raw versus capped target views

`selected_raw_and_capped_metrics.csv` keeps raw official RUL as the operational view and contains the separately labelled cap-200 diagnostic view. `baseline_raw_and_capped_metrics.csv` provides the same split for the older cap-100 baseline. Neither capped view replaces the raw official results.

## Operating-condition and error findings

`condition_distribution_train_and_test.csv` records cluster frequencies from the full-training fitted six-condition normalizer. `baseline_worst_10_errors_by_subset.csv`, `selected_error_by_rul_bucket.csv`, and the per-subset actual-versus-predicted and residual plots provide the traceable error analysis. FD002 and FD004 remain difficult because their distribution includes more high-RUL cases and multiple operating regimes; FD004 also has multiple fault modes. The selected cap-200 model reduces their severe conservative high-tail errors but increases optimistic estimates in FD001 and FD003.

## Recommendation

For the Tata Elxsi case study, retain the cap-200 condition-normalized global HGB as the current global candidate because it was selected by internal grouped validation and improves the pooled, FD002, and FD004 historical audits. Present the FD001 and FD003 regressions explicitly. Do not represent the historical audit as a blind test. A next production decision should define whether avoiding early-conservative RUL estimates in multi-condition assets outweighs the increased risk of optimistic estimates in simpler regimes. A regime router or known-domain specialists can be a later diagnostic, but only if their routing inputs are available in deployment.

The anomaly score and deterministic rule-based work-order workflow remain separate from these RUL results. C-MAPSS has no per-cycle anomaly labels, so no anomaly-classification accuracy is claimed.
