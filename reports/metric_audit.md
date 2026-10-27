# Metric and target audit

For a training engine at cycle t, uncapped RUL is final observed training cycle minus t. Target caps 100, 125, 150, and uncapped are named separately. Validation and official metrics always use uncapped truth.

For PHM08 with d = prediction minus actual: d < 0 uses exp(-d/13)-1 and d >= 0 uses exp(d/10)-1. Both PHM08 sum and PHM08 mean are saved from exactly the same one-prediction-per-endpoint arrays as RMSE and MAE. Unit tests verify d=-13, d=10, and d=0. Overprediction is penalized more sharply.

The grouped audit reports 100 distinct held-out engines and 500 correlated pseudo-censored endpoint cases. Its 300-resample engine-block bootstrap keeps all horizons from a sampled engine together. The separately recorded official benchmark has exactly 100 endpoint predictions.
