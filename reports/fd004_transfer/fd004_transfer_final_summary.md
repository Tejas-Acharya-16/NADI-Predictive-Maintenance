# FD004 cross-subset transfer result

The FD004 validation winner is `fd004_hgb_cap100_condition_normalized` with RMSE 15.323, MAE 10.739, R2 0.856, and PHM08 mean 3.460 on 1,494 grouped pseudo-censored endpoint cases. It trains from FD004 only using six FD004-fitted operating-condition clusters and per-cluster sensor normalization.

## Locked cross-subset endpoint results

| Target | Engines | RMSE | MAE | R2 | PHM08 mean | Bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FD001 | 100 | 17.619 | 12.866 | 0.820 | 3.824 | -8.066 |
| FD002 | 259 | 31.074 | 19.194 | 0.666 | 62.550 | -14.266 |
| FD003 | 100 | 18.162 | 13.019 | 0.808 | 4.528 | -5.613 |

FD001 and FD003 transfer with moderate error despite their simpler single-condition domains. FD002 is materially harder and has high PHM08 because its target operating-condition mix differs from FD004. These are transfer measurements, not target-subset tuned models.
