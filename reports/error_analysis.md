# Validation error analysis

The validation-selected candidate is `hgb_cap100_w10_30`. Across 500 OOF endpoint cases from 100 engines, it has RMSE 7.576, MAE 5.390, PHM08 mean 0.985, bias 0.444 cycles, overprediction fraction 0.496, P90 absolute error 13.335, and maximum absolute error 41.751.

## Error by predeclared horizon

```text
         cases    rmse    mae   bias  phm08_mean
horizon                                         
10         100   2.730  2.094  0.104       0.225
25         100   4.031  3.201  1.208       0.390
50         100   7.950  5.859  2.723       1.465
75         100  11.307  9.414  4.511       2.032
100        100   8.499  6.383 -6.329       0.814
```

Generated figures show OOF actual-versus-predicted values, residuals versus true RUL, residual histogram, horizon-band error, highest per-engine absolute error, and largest PHM08 contributions. These plots are based on held-out validation predictions only.
