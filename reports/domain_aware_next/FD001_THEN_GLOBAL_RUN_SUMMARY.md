# FD001-first, then global C-MAPSS rerun

## Execution order

1. FD001-only model selection on the FD001 training file, using five engine-grouped folds and 10/25/50/75/100-cycle pseudo-censored endpoints.
2. One frozen FD001 endpoint evaluation on the official FD001 test set.
3. Global model selection on the combined FD001--FD004 training files, using three subset-stratified engine-grouped folds and 10/25/50/75/100/125-cycle endpoints.
4. One frozen official endpoint evaluation for each FD001--FD004 test set.

The official scores were already known before this rerun and are reported as historical benchmark evaluations. They were not used to choose a candidate within either run.

## FD001-only rerun

| Candidate | Validation RMSE | Validation MAE | Validation R2 | PHM08 mean |
| --- | ---: | ---: | ---: | ---: |
| HGB cap 100, causal 10/30 | 7.576 | 5.390 | 0.946 | 0.985 |
| HGB cap 125, causal 10/30 control | 10.063 | 6.799 | 0.905 | 2.032 |

The cap-100 HGB was selected by validation. Its separately evaluated official FD001 result was RMSE 16.755, MAE 11.959, R2 0.837, PHM08 mean 3.393, and bias -7.330 cycles.

## Global rerun

The global run trained on 160,359 telemetry rows from 709 engines, without using the source subset as a model feature. It compared three HGB candidates. The condition-normalized cap-100 candidate was selected by PHM08 mean (2.532); the condition-normalized cap-125 candidate had the lower validation RMSE (11.800) but PHM08 mean 2.561.

| Official test subset | Engines | RMSE | MAE | R2 | PHM08 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| FD001 | 100 | 16.292 | 11.608 | 0.846 | 3.164 |
| FD002 | 259 | 31.128 | 19.183 | 0.665 | 61.477 |
| FD003 | 100 | 17.273 | 11.982 | 0.826 | 3.785 |
| FD004 | 248 | 33.391 | 21.785 | 0.625 | 62.574 |
| Pooled | 707 | 28.737 | 18.006 | 0.683 | 45.454 |

The FD002 and FD004 results are weaker and are associated with the multi-condition subsets. The next experiment should select an alternative cap or specialist architecture using only grouped training validation, then record the already-observed official scores as historical comparisons.
