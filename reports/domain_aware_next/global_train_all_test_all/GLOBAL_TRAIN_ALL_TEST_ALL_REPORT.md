# Global FD001--FD004 training and official-test evaluation

## Protocol

All four C-MAPSS training files were concatenated for model development. Engine IDs were remapped before concatenation, so rolling features never cross an engine or subset boundary. Three candidates were compared with 3-fold subset-stratified, engine-grouped validation. Each held-out engine contributes fixed pseudo-censored endpoints at 10, 25, 50, 75, 100, and 125 cycles before its training failure. The selection rule was lowest mean PHM08 score, then lowest RMSE. No official test file or RUL label was loaded until after that choice.

## Global validation comparison

```text
                             candidate  rul_cap  condition_normalized    rmse     mae     r2  phm08_mean  selected_by_global_validation
global_hgb_cap100_condition_normalized      100                  True 14.1009  9.7298 0.8779      2.5320                           True
global_hgb_cap125_condition_normalized      125                  True 11.8001  8.1783 0.9145      2.5608                          False
  global_hgb_cap125_raw_sensor_control      125                 False 17.1690 12.5256 0.8190      7.5457                          False
```

## Validation detail by source subset

```text
                             candidate subset  cases  engines    rmse     mae     r2  phm08_mean
global_hgb_cap100_condition_normalized  FD001    600      100 13.0583  8.8801 0.8953      1.9331
global_hgb_cap100_condition_normalized  FD002   1560      260 14.4457 10.1194 0.8719      2.6696
global_hgb_cap100_condition_normalized  FD003    600      100 12.9514  8.6838 0.8970      1.9016
global_hgb_cap100_condition_normalized  FD004   1494      249 14.5747 10.0843 0.8696      2.8819
global_hgb_cap125_condition_normalized  FD001    600      100 10.3863  7.1311 0.9338      1.8585
global_hgb_cap125_condition_normalized  FD002   1560      260 12.3572  8.6408 0.9062      2.7052
global_hgb_cap125_condition_normalized  FD003    600      100  9.7112  6.8565 0.9421      1.4471
global_hgb_cap125_condition_normalized  FD004   1494      249 12.4833  8.6466 0.9043      3.1393
  global_hgb_cap125_raw_sensor_control  FD001    600      100 11.4275  8.4237 0.9198      2.1512
  global_hgb_cap125_raw_sensor_control  FD002   1560      260 18.8788 14.0455 0.7811      9.6876
  global_hgb_cap125_raw_sensor_control  FD003    600      100 10.9824  8.0923 0.9259      1.8519
  global_hgb_cap125_raw_sensor_control  FD004   1494      249 19.1390 14.3663 0.7751      9.7623
```

## Locked official-test results

```text
subset  cases  engines    rmse     mae     r2  phm08_mean  bias_predicted_minus_actual
 FD001    100      100 16.2923 11.6084 0.8463      3.1639                      -7.6884
 FD002    259      259 31.1283 19.1831 0.6650     61.4772                     -14.7669
 FD003    100      100 17.2734 11.9821 0.8259      3.7854                      -6.6857
 FD004    248      248 33.3912 21.7846 0.6249     62.5738                     -17.4511
```

## Pooled official-test result

```text
 samples    rmse     mae  median_absolute_error     r2  mape_percent  smape_percent  bias_predicted_minus_actual  overprediction_count  underprediction_count  overprediction_fraction  absolute_error_p90  absolute_error_max  phm08_sum  phm08_mean  nasa_score  nasa_score_mean
     707 28.7371 18.0057                 8.4439 0.6834       18.9586        20.5446                     -13.5642                   251                    456                    0.355             53.4778             98.0149 32135.8102     45.4538  32135.8102          45.4538
```

The per-subset rows are the primary result because the four files have different operating-condition and fault-mode coverage. The pooled score is a descriptive total across all 707 official test engines.
