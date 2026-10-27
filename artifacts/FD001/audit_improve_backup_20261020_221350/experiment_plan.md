# Experiment protocol and stop criteria

## Hardware budget

Experiments target a 16 GB Windows laptop with 12 logical CPU threads and integrated graphics. Tree runs are serial CPU jobs. Sequence training is restricted to four PyTorch CPU threads, hidden width 16, 20 maximum epochs, batch size 256, and patience 4. No GPU, large transformer, exhaustive sweep, XGBoost, or LightGBM dependency was added.

## Completed controlled study

1. Control: uncapped labels with causal windows 5/20.
2. Capped labels: `min(RUL, 125)`.
3. Remove development-only near-constant sensors.
4. Causal windows 10/30 and 20/50.
5. Difference-to-initial-reading features.
6. A bounded HGB parameter variation.
7. CNN, LSTM, CNN-LSTM, and CNN-LSTM with difference-to-initial features.

All tree candidates share seed 42 and 60 fixed validation prefixes. The candidate with the lowest PHM08 mean, then lowest RMSE, is repeated over splits seeded 17 and 73. The selected model is only then trained on all training engines and evaluated on official test data once.

## Selection and stop criteria

- Select by validation PHM08 mean, then validation RMSE.
- Require repeat results to be plausible across three engine splits; report mean and standard deviation rather than a single lucky split.
- Stop the neural sweep when the compact CNN-LSTM does not improve the selected tree candidate. Do not add attention or the paper's change-point target until it has a separately justified causal inference policy.
- Do not tune on the official test set.

## Change-point labeling status

The 2023 paper's change-point labeling is intentionally not claimed as reproduced. Engine-specific change points found using a complete run-to-failure trajectory are unavailable for a truncated test engine. A deployable version needs a training-learned, prefix-observable rule; that requires a separate design and validation study.
