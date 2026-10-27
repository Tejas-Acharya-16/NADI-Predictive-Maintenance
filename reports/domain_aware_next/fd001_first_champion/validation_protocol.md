# Validation protocol

The grouped audit uses five `GroupKFold` folds over engine ID. Every held-out engine produces one causal endpoint for each predeclared remaining-life horizon 10, 25, 50, 75, and 100 cycles when supported. FD001 training engines all support these horizons, producing 500 correlated endpoint cases from 100 distinct engines.

Transforms, sensor screening, feature construction choices, and model fitting use development engines only. Endpoint metadata and predictions are saved with fold IDs under `reports/splits/` and `reports/predictions_oof.csv`. Metrics are pooled over endpoints and also summarized across folds; uncertainty is a 300-resample engine-block bootstrap. The official test set is not read by this runner.
