# FD001 data audit

FD001 is simulated turbofan telemetry. It must not be represented as a Siemens pump, truck, compressor, or motor-vibration dataset. Raw train and test files have 26 fields: engine ID, cycle, three operating settings, and 21 sensors.

| Split | Rows | Engines | Cycle range | Integrity checks |
| --- | ---: | ---: | --- | --- |
| Train | 20,631 | 100 | 1 to 362 | no missing/nonfinite values, duplicate unit-cycle records, gaps, invalid IDs, or ordering faults |
| Test | 13,096 | 100 | 1 to 303 | same checks passed |
| Supplied endpoint RUL | 100 | 100 | 7 to 145 cycles | aligned to sorted test engine IDs |

Seven training sensors have standard deviation below 0.01: sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, and sensor_19. Their removal was fitted only on development folds and did not beat the cap-100 validation champion. Engine ID is never a model feature. Current cycle and cycles_seen are deliberately available-at-inference features; their removal was tested as a separate causal ablation.
