# FD001 audit and improvement summary

## What was executed

A five-fold, engine-grouped, validation-only audit was run on FD001 training data. Each held-out engine was pseudo-censored at 10, 25, 50, 75, and 100 remaining cycles, yielding 500 endpoint cases across 100 engines. Twelve bounded tree-model and feature/target candidates were compared with the same folds. The official test files were not read during selection.

## Validation results

| Candidate | RMSE | MAE | R2 | PHM08 mean | 5-fold PHM08 SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| HGB cap 100, causal 10/30 - validation champion | 7.576 | 5.390 | 0.946 | 0.985 | 0.311 |
| HGB cap 125, causal 10/30 control | 10.063 | 6.799 | 0.905 | 2.032 | 0.742 |

The cap-100 result is strong within this deliberately bounded 10-to-100-cycle validation schedule. It should not be described as a general deployment replacement for the cap-125 model because the protocol does not assess RUL above 100 cycles. The corrected time-free ablation was worse than the cap-125 control, so current-cycle information is useful and causally available.

## Historical official benchmark comparison

After validation was locked, cap-100 was compared once with the previously inspected official benchmark: RMSE 16.755, MAE 11.959, R2 0.837, PHM08 mean 3.393, and bias -7.330 cycles. The comparison is secondary only and did not influence selection. Its negative bias is consistent with the cap-100 training target.

The earlier cap-125 historical official benchmark remains RMSE 13.532, MAE 9.799, R2 0.894, and PHM08 mean 2.852. It was produced under the older fixed-split study. Do not rank the two official scores as a clean selection comparison.

## End-to-end demo

The repository includes causal feature construction, a compact neural comparison, early-life autoencoder anomaly scoring, deterministic triage with persistence and work-order idempotency, a Streamlit dashboard, data validation, descriptive drift indicators, and a versioned training-only retraining simulation. FD001 has no cycle-level anomaly truth, so anomaly outputs are not reported as classification accuracy.

## Limits and next work

Use an expanded, predeclared validation horizon schedule before making a production cap choice. Validate FD002 and FD004 with operating-condition-aware preprocessing. Calibrate prediction intervals and triage thresholds with domain owners. Measure latency, RAM, quantization loss, and update cost on the actual edge target before making an on-device deployment claim.
