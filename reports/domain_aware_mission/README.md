# Domain Aware C MAPSS Experiment Results

## Protocol
All model selection uses engine-grouped validation with causal, pseudo-censored endpoints at 15, 30, 45, 60, 75 and 90 percent observed life. Operating-condition clusters, normalizers, features and models are fit within each development fold. Official test labels are historical audit data because they were inspected in earlier work.

## Audit finding
The cap 200 model is not a universal improvement. It reduces high-RUL underprediction in FD002/FD004 but may increase optimistic error in FD001/FD003. `audit_raw_and_capped_metrics.csv` separates raw RUL (business metric) from a cap-matched diagnostic.

## Implemented models
- Per-subset HGB specialists: internal comparison of cap 100 and 200; condition normalization is used only for the multi-condition FD002/FD004 subsets.
- Transfer learning proxy: source-domain HGB pretraining followed by a target-domain residual adaptor on FD002 and FD004. This is an implementable tabular transfer baseline rather than a claimed reproduction of the linked neural paper.
- Compact TCN appears in the companion sequence run; it is deliberately CPU-bounded and only retained if grouped validation supports it.

## Files
- `audit_data_integrity.csv`: schema, missingness and engine alignment checks.
- `audit_raw_and_capped_metrics.csv`: historical baseline and cap-200 audit.
- `specialist_validation_comparison.csv` and `specialist_official_metrics.csv`: specialist results.
- `transfer_validation_metrics.csv`: source to target transfer validation.
- `figures/`: actual-versus-predicted and residual plots.