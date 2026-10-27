# FD001 data audit

## Source and scope

The project reads `CMAPSSData/train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`. The raw files are whitespace-delimited and have 26 columns: unit ID, cycle, three settings, and 21 sensors. FD001 contains one operating condition and one fault mode. It is simulated turbofan telemetry, not the Siemens pump vibration dataset.

## Observed quality

| Split | Rows | Engines | Life length / observed cycles | Quality result |
| --- | ---: | ---: | --- | --- |
| Train | 20,631 | 100 | min 128, median 199, max 362 | No missing/non-finite values, duplicate unit-cycle rows, cycle gaps, invalid IDs, or ordering faults. |
| Test | 13,096 | 100 | min 31, median 133.5, max 303 | Same checks passed. |
| Test RUL vector | 100 | 100 | min 7, median 86, max 145 | Aligned to sorted test unit IDs. |

The validation code verifies the final observed test row and its RUL label have matching sorted engine IDs before scoring.

## Sensor and setting decisions

`sensor_1`, `sensor_5`, `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, and `sensor_19` have standard deviation below 0.01 on FD001 training data. The ablation fit that screening threshold on development engines only, retained 14 sensors, and did not improve the selected validation score. Settings are retained in the causal features, but FD001 does not justify operating-condition clustering.

## Leakage controls

- Development and validation data are split by engine ID: 80 development engines and 20 validation engines.
- Training RUL is `last_training_cycle - current_cycle`; the cap experiment applies `min(RUL, 125)` only to development labels.
- Features use current or earlier cycles within the same engine. Rolling state resets at every engine boundary.
- Difference-to-initial-reading uses only that engine's first observed record.
- Validation uses fixed 50%, 70%, and 90% prefixes of held-out full trajectories: 60 test-like cases. Hidden suffixes are not provided as features.
- The validation study does not open test data. Official FD001 scoring occurs separately in `evaluate_selected_fd001.py` after selection.

## Reproduction

Run `python run_fd001_study.py` from the project virtual environment. Exact split IDs and endpoint outputs are saved beside each run under `artifacts/FD001/validation_study/`.
