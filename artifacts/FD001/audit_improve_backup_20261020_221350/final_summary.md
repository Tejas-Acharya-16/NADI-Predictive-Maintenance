# FD001 implementation summary

## Selected model

The selected model is a HistGradientBoostingRegressor trained on capped-at-125 RUL labels with causal 10- and 30-cycle features. It was selected from held-out engine prefixes, then retrained on all 100 FD001 training engines.

| Stage | RMSE | MAE | R² | PHM08 mean |
| --- | ---: | ---: | ---: | ---: |
| Selection split, seed 42 | 12.773 | 8.609 | 0.877 | 2.832 |
| Three-split repeat mean ± SD | 13.623 ± 1.724 | 8.802 ± 0.561 | 0.866 ± 0.025 | 3.471 ± 1.411 |
| One official-test evaluation | 13.532 | 9.799 | 0.894 | 2.852 |

## What improved

- Capping early-life labels reduced the validation PHM08 mean from 19.728 (uncapped control) to 6.286 with otherwise matching 5/20 features.
- The 10/30-cycle variant improved it further to 2.832, with selection-split RMSE 12.773.
- It outperformed the best compact neural candidate, CNN-LSTM (validation RMSE 16.199, PHM08 mean 4.010).
- The final official-test model has nearly zero bias (0.157 cycles) and balanced 50/50 over- and under-predictions.

## What did not improve

- Removing near-constant sensors, initial-reading differences, 20/50 windows, and the bounded HGB variation did not beat the 10/30 setting on the selection split.
- CNN-LSTM was the best neural model, but it did not beat the selected tree model.
- CNN-LSTM with initial-reading differences stopped after eight epochs and performed worse.
- The historical TinyML and hybrid models remain deployment comparisons, not the selected RUL model.

## Operations demo

An early-life autoencoder derives a 99th-percentile reconstruction threshold from development windows. It is used only as an unlabeled anomaly signal; C-MAPSS has no cycle-level anomaly ground truth. Deterministic rules create idempotent mock work orders using RUL, anomaly persistence, data quality, and model version. The demo produced 43 deduplicated recommendations from validation replay, saved under `triage_demo_work_orders.json`.

## Limitations and next steps

The official test set had already been used for historical candidate scores before this study. The new study itself selected its model without loading it, and its selected-model test score is a single final evaluation. FD001 is a simple single-condition simulated dataset. Next work should establish a deployable prefix-only change-point policy, validate intervals, run FD002/FD004 condition-aware experiments, and calibrate triage thresholds with domain owners.
