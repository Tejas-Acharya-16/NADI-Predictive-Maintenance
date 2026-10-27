# FD004 to FD001 FD002 FD003 transfer study

The source model was selected with FD004 training engines only. Three HGB candidates were compared by 3-fold engine-grouped FD004 validation with predeclared 10/25/50/75/100/125-cycle pseudo-censored endpoints. FD001, FD002, and FD003 official endpoint labels were not read until after source selection.

## FD004 validation selection

```text
                            candidate  rul_cap  condition_normalized  distinct_validation_engines  validation_cases  samples    rmse     mae  median_absolute_error     r2  mape_percent  smape_percent  bias_predicted_minus_actual  overprediction_count  underprediction_count  overprediction_fraction  absolute_error_p90  absolute_error_max  phm08_sum  phm08_mean  nasa_score  nasa_score_mean  rmse_fold_mean  rmse_fold_std  phm08_mean_fold_mean  phm08_mean_fold_std  selected_by_fd004_validation
fd004_hgb_cap100_condition_normalized      100                  True                          249              1494     1494 15.3230 10.7392                 6.0491 0.8558       20.2185        18.9850                      -3.8809                   650                    844                   0.4351             26.5671             64.6263  5169.4473      3.4601   5169.4473           3.4601         15.3155         0.5874                3.4601               0.7820                          True
fd004_hgb_cap125_condition_normalized      125                  True                          249              1494     1494 13.7731  9.5076                 5.9212 0.8835       20.2482        18.2009                       1.7547                   804                    690                   0.5382             23.6194             69.7925  6561.7504      4.3921   6561.7504           4.3921         13.7526         0.9204                4.3921               2.1784                         False
  fd004_hgb_cap125_raw_sensor_control      125                 False                          249              1494     1494 19.2776 14.4832                10.9020 0.7718       34.4451        28.7776                       3.9415                   872                    622                   0.5837             34.2376             67.9586 14208.3065      9.5102  14208.3065           9.5102         19.2775         0.0481                9.5102               1.0625                         False
```

## Cross-subset official endpoint evaluation

```text
source_subset target_subset                             candidate  source_condition_normalized  target_engines  samples    rmse     mae  median_absolute_error     r2  mape_percent  smape_percent  bias_predicted_minus_actual  overprediction_count  underprediction_count  overprediction_fraction  absolute_error_p90  absolute_error_max  phm08_sum  phm08_mean  nasa_score  nasa_score_mean
        FD004         FD001 fd004_hgb_cap100_condition_normalized                         True             100      100 17.6185 12.8658                 8.7581 0.8202       17.5482        17.9145                      -8.0657                    33                     67                   0.3300             32.4219             46.1079   382.3528      3.8235    382.3528           3.8235
        FD004         FD002 fd004_hgb_cap100_condition_normalized                         True             259      259 31.0740 19.1942                 7.4258 0.6661       20.4332        22.0971                     -14.2656                   102                    157                   0.3938             59.1875             96.7861 16200.4862     62.5501  16200.4862          62.5501
        FD004         FD003 fd004_hgb_cap100_condition_normalized                         True             100      100 18.1619 13.0187                 9.1757 0.8075       21.1599        18.7067                      -5.6134                    47                     53                   0.4700             31.2581             53.2533   452.8001      4.5280    452.8001           4.5280
```

These are domain-transfer evaluations, not within-subset tuned benchmarks. FD004 has six operating conditions and two fault modes; target subsets differ in condition and fault coverage. The model uses FD004-fitted condition clusters and sensor standardization. No target-subset rows or labels are used to fit the transformer or model.
