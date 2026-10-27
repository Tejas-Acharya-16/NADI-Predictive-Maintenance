# FD004 transfer source and target audit

FD004 is the source subset because it covers six operating conditions and two degradation modes. FD001, FD002, and FD003 are explicitly held out as target-subset endpoint evaluations. All source and target raw files passed schema, non-finite, duplicate, cycle-order, gap, and ID checks.

```text
subset  train_rows  train_engines  test_rows  test_engines  test_rul_min  test_rul_max quality_checks
 FD001       20631            100      13096           100             7           145         passed
 FD002       53759            260      33991           259             6           194         passed
 FD003       24720            100      16596           100             6           145         passed
 FD004       61249            249      41214           248             6           195         passed
```

The operating-condition normalizer is fit on FD004 settings and sensors only. Target rows are assigned to their nearest FD004-derived setting cluster at inference. No target rows or labels are used to refit KMeans, scalers, feature screens, or the RUL model.
