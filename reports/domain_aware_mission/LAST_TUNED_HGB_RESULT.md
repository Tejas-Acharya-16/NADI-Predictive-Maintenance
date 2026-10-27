# Final tuned HGB experiment

This final validation-gated candidate used a 200-cycle target cap, six training-fitted operating-condition clusters, causal 10/30/50-cycle features, early-life sensor baselines, screening of seven low-information sensors, and a slower 350-tree HGB configuration.

## Result

| Candidate | Pooled validation RMSE | MAE | PHM08 mean |
|---|---:|---:|---:|
| Prior global cap-200 HGB | 34.323 | 21.747 | 254901.147 |
| Final tuned HGB | 33.401 | 20.419 | 292864.996 |

The tuned candidate improves RMSE by 0.922 cycles (2.7%) and MAE by 1.328 cycles, but worsens the asymmetric PHM08 mean by 14.9%. It is therefore retained as the **lowest-RMSE validation challenger**, not promoted over the existing cap-200 HGB recommendation because the project selection rule prioritizes PHM08 safety cost.

Its full per-subset metrics and every validation prediction are saved in `last_tuned_hgb_validation_metrics.csv` and `last_tuned_hgb_validation_predictions.csv`.
