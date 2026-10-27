# Metric and target audit

## Target convention

For a training engine at cycle `t`, uncapped RUL is `final_observed_training_cycle - t`. The study keeps uncapped and capped-at-125 labels as separately named experiments. Validation and official test metrics always compare predictions against uncapped truth.

## PHM08 implementation

For `d = prediction - true_RUL`, `phm08_contributions()` applies:

```text
d < 0 : exp(-d / 13) - 1
d >= 0: exp( d / 10) - 1
```

It reports both `phm08_sum` and `phm08_mean`; `nasa_score` and `nasa_score_mean` remain compatibility aliases. The historical table's displayed “NASA mean” is exactly PHM08 sum divided by 100 official-test engines. It is not a per-cycle score, a sign error, or an array mismatch. Ranking can differ from RMSE because overprediction is penalized more sharply.

Unit tests check `d=-13`, `d=10`, and `d=0` against `exp(1)-1`, `exp(1)-1`, and zero. The test suite also checks endpoint alignment, engine isolation, causal features, artifact identity, and work-order idempotency.

## Current clean selected-model result

The selected HGB model was chosen from validation only, then refit on all 100 training engines and scored once on official FD001 endpoints:

| RMSE | MAE | R² | PHM08 sum | PHM08 mean | Bias | P90 absolute error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 13.532 | 9.799 | 0.894 | 285.222 | 2.852 | 0.157 | 24.135 |

See `selected_official_test_metrics.json` and `predictions_selected_hgb_full_training.csv`.
