# OPP Summary Domain Aware C MAPSS Predictive Maintenance

## One line proposal
Use a globally trained, operating-condition-normalized HGB RUL predictor with causal sensor features and explainable triage, then deploy it with model monitoring and a normal-reference anomaly score.

## Business problem
The system estimates Remaining Useful Life from multivariate turbofan telemetry so maintenance teams can prioritise engines before risk becomes critical. NASA C-MAPSS is a simulation benchmark. Results here are not certification evidence for aircraft maintenance.

## What was implemented
1. Engine-aligned data audit across FD001–FD004.
2. KMeans operating-condition identification from three observable settings, fitted only on development training rows.
3. Per-condition sensor normalization, causal 10/30-cycle means, standard deviations and trends, plus early-life baseline differences.
4. Global cap-100 and cap-200 HGB comparison, per-subset specialists, a cross-condition pretrain/residual-adaptation transfer baseline, a compact attention TCN, and an observable condition-diversity router.
5. Sensor stream to RUL prediction to normal-reference anomaly score to deterministic triage to idempotent mock work order, logged in local MLflow.

## Measured recommendation
Use **global condition-normalized HGB with cap 200** for this benchmark. Its historical official pooled RMSE is **21.455 cycles**, MAE **14.622**, R2 **0.824**, and PHM08 mean **31.465**. It improves the cap-100 pooled RMSE from **28.737** by **25.3%**. It specifically improves FD002 from 31.128 to 21.738 and FD004 from 33.391 to 22.630.

## USP
- Handles multi-condition fleets without needing a subset filename at deployment.
- Uses training-fitted operating regimes and causal features, avoiding future leakage.
- Carries prediction into an explainable operational workflow, including severity, trigger reason and recommended action.
- Separates RUL prediction from unsupervised reconstruction-error anomaly monitoring because C-MAPSS has no anomaly labels.

## Important tradeoff
Cap 200 is not universally better: historical FD001 RMSE moves from 16.292 to 18.037 and FD003 from 17.273 to 20.863. The reason is target-range mismatch: cap 100 creates conservative high-RUL predictions, especially on FD002/FD004, while cap 200 reduces that underprediction but adds positive bias on the single-condition subsets. Raw RUL remains the business metric; cap-matched values are diagnostics only.

## Research results retained as evidence
- Specialists: pooled official RMSE 22.206, rejected.
- Transfer pretrain plus target residual adaptation: validation RMSE 29.342 on FD002 and 39.319 on FD004, rejected.
- Compact dilated TCN with sensor/time attention: validation pooled RMSE 43.615, rejected.
- Observable router: historical official RMSE 20.797, but validation RMSE 42.781, not promoted.
- Final tuned HGB: validation RMSE 33.401 versus 34.323 for the prior cap-200 HGB, but PHM08 mean rose from 254901 to 292865; retained as a challenger only.

## Operational demo
The demo processed 2,109 tail telemetry windows using 12,000 early-life normal-reference windows and generated 228 idempotent mock work orders. Local MLflow artifacts record parameters, metrics and work-order outputs. In this Windows environment the optional PyTorch DLL failed to initialise for the autoencoder; the persisted demo records a PCA reconstruction fallback rather than hiding it.
