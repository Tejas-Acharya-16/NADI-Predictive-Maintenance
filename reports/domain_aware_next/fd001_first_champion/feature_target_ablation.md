# Feature and target ablation

All candidates use identical engine folds and horizons. Lower PHM08 and RMSE are better.

| Candidate | Change | RMSE | MAE | PHM08 mean | Conclusion |
| --- | --- | ---: | ---: | ---: | --- |
| hgb_cap100_w10_30 | cap=100; windows=10/30; initial_delta=False; cycle=True | 7.576 | 5.390 | 0.985 | Selected by validation |
| hgb_control_cap125_w10_30 | cap=125; windows=10/30; initial_delta=False; cycle=True | 10.063 | 6.799 | 2.032 | Kept as controlled comparison |

The selected validation candidate is `hgb_cap100_w10_30` with PHM08 mean 0.985 and RMSE 7.576. This is a validation-only conclusion.
