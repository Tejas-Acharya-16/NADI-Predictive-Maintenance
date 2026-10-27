# Baseline reproduction

The prior fixed-split HGB control used a capped-125 target and causal 10/30 features. This audit reproduces the same named configuration on a stronger five-fold engine-grouped, five-horizon protocol. Its pooled validation results are RMSE 10.063, MAE 6.799, R2 0.905, PHM08 mean 2.032. These numbers are not expected to equal the older fixed 60-case split because the held-out engine folds and endpoint horizons differ.
