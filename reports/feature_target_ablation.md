# Feature and target ablation

All results were generated on the same five engine folds and predeclared horizons. Lower RMSE, MAE, and PHM08 are better.

| Candidate | Cap | Windows | RMSE | MAE | R2 | PHM08 mean | Result |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| hgb_cap100_w10_30 | 100 | 10/30 | 7.576 | 5.390 | 0.946 | 0.985 | Selected by validation |
| hgb_cap125_near_constant_removed | 125 | 10/30 | 9.964 | 6.820 | 0.907 | 1.891 | Controlled comparison |
| hgb_cap125_initial_delta | 125 | 10/30 | 10.037 | 6.922 | 0.905 | 1.897 | Controlled comparison |
| equal_blend_hgb_extra_trees | 125 | 10/30 | 9.904 | 6.768 | 0.908 | 1.941 | Controlled comparison |
| hgb_tuned_small_leaves | 125 | 10/30 | 10.101 | 6.884 | 0.904 | 2.031 | Controlled comparison |
| hgb_control_cap125_w10_30 | 125 | 10/30 | 10.063 | 6.799 | 0.905 | 2.032 | Controlled comparison |
| extra_trees_cap125 | 125 | 10/30 | 10.249 | 7.105 | 0.901 | 2.048 | Controlled comparison |
| hgb_absolute_error_loss | 125 | 10/30 | 10.883 | 7.272 | 0.889 | 2.404 | Controlled comparison |
| hgb_cap125_w5_20 | 125 | 5/20 | 11.540 | 8.052 | 0.875 | 2.596 | Controlled comparison |
| random_forest_cap125 | 125 | 10/30 | 11.380 | 7.861 | 0.879 | 2.746 | Controlled comparison |
| hgb_cap125_without_cycle_or_cycles_seen | 125 | 10/30 | 11.618 | 8.168 | 0.873 | 2.959 | Controlled comparison |
| hgb_cap150_w10_30 | 150 | 10/30 | 13.216 | 8.425 | 0.836 | 4.856 | Controlled comparison |
| hgb_uncapped_w10_30 | uncapped | 10/30 | 16.484 | 9.596 | 0.745 | 94.983 | Controlled comparison |
