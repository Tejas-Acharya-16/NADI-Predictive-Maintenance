# Domain Aware C MAPSS Predictive Maintenance Final Report

## Executive conclusion
The recommended benchmark model is the global condition-normalized Histogram Gradient Boosting model trained with a 200-cycle RUL cap. It was selected from training-only, engine-grouped validation in the prior cap study and then audited against historically inspected official endpoints. It reduces pooled historical official RMSE from 28.737 to 21.455 cycles. It is a measured improvement for multi-condition FD002 and FD004, while sacrificing FD001 and FD003 performance. The recommendation is therefore a global fleet choice with a clearly recorded operating-domain tradeoff.

## Audit and metric definitions
The audit checked the expected 26-column schema, engine/cycle uniqueness, numeric validity, unit/cycle ordering and test-RUL alignment. All eight train/test files had zero missing cells, nonfinite values, duplicate unit-cycle records, cycle gaps and ordering faults. Test RUL labels are aligned to sorted engine IDs by the loader.

**Raw-target metric** means the supplied official RUL target and unmodified prediction. This is the operational result. **Capped-target diagnostic** clips both target and prediction to the stated training cap; it answers how the model performs on its capped representation and must not replace the raw target result. PHM08 uses `d = prediction - actual`: `exp(-d/13)-1` when d < 0 and `exp(d/10)-1` otherwise. Overprediction is therefore costlier per cycle.

## Leakage controls
Engines are kept intact in all validation folds. Endpoint features only use telemetry at or before each pseudo-censored cutoff. KMeans settings clustering, condition-specific sensor moments, sensor screening, standardization and all fitted models are learned from development engines only. No true test RUL, hidden fault label or subset filename is used to choose a deployment route.

## Result summary
| Model | Scope | RMSE | MAE | R2 | PHM08 mean | Decision |
|---|---:|---:|---:|---:|---:|---|
| global_hgb_cap100 | historical official audit | 28.737 | 18.006 | 0.683 | 45.454 | cap200 was validation-selected; cap100 benchmark |
| global_hgb_cap200 | historical official audit | 21.455 | 14.622 | 0.824 | 31.465 | cap200 was validation-selected; cap100 benchmark |
| per_subset_specialists | historical official audit | 22.206 | 15.009 | 0.811 | 37.539 | rejected worse pooled result |
| observable_router | historical official audit | 20.797 | 14.275 | 0.834 | 26.159 | diagnostic only: failed validation gate |
| tuned_hgb_cap200 | grouped validation | 33.401 | 20.419 | 0.758 | 292864.996 | RMSE challenger; PHM08 gate failed |

## Why cap 200 helps selectively
FD002 and FD004 contain more high-RUL endpoints: their cap-100 models have strongly negative raw-target bias. Raising the cap lets the model represent those values and improves their RMSE. FD001 and FD003 do not receive the same benefit; their residual bias becomes optimistic. This is why a cap-200 improvement cannot be called universal. The output `cap_effect_target_tail_and_bias.csv` contains the supporting tail and bias counts.

## Specialist and research experiments
Per-subset specialists were genuinely trained and assessed. They selected cap 200 in the full-range grouped protocol but did not beat the global model in pooled historical audit. The source-pretrain plus target-residual transfer model was trained as a computationally bounded tabular transfer baseline; it did not validate. The compact TCN contains dilated residual blocks at rates 1, 2 and 4 plus sensor and temporal attention. It was deliberately kept small and its grouped validation result was much worse, so it is retained for reproducibility and not promoted. A final tuned HGB improved validation RMSE, but its higher PHM08 mean prevented promotion. The dual-scale Transformer was not run because the simpler attention TCN did not clear the validation gate.

## Competition system
Sensor data feeds training-fitted preprocessing and operating-condition identification, then the recommended RUL predictor. A normal-reference reconstruction score runs separately. Deterministic triage converts prediction, score, threshold, data quality and persistence into a severity and recommended action. Each work order records asset, predicted RUL, anomaly score, severity and triggering rule. C-MAPSS has no supervised anomaly labels, so the score is not claimed as anomaly classification. Local MLflow captures the demo run.

## Final recommendation
Deploy the global cap-200 HGB pipeline as the current benchmark candidate, with interval calibration and condition-drift monitoring before any live trial. Keep the cap-100 scorecard for FD001/FD003 comparison. Do not promote specialists, transfer adaptation, the compact TCN or the observable router because their validation evidence does not support it.
