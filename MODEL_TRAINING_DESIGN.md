# C-MAPSS Model Training Design

## Understanding summary

- Build a reusable, modular training package for the NASA C-MAPSS RUL project.
- Implement the candidate model families discussed, starting with FD001 and the existing engine-wise development/validation split.
- Compare regression models using validation only; leave official test files and labels untouched until a later test stage.
- Persist trained models and the exact preprocessing/configuration needed to reuse them without retraining.
- Keep anomaly scoring distinct from RUL prediction. The autoencoder is an anomaly/health-signal experiment, not an RUL regressor.
- Keep the TinyML/quantization and streaming-update idea as a constrained-deployment experiment; do not claim learning from unlabeled data.
- Keep the workflow local, reproducible, CPU-first, and callable from a notebook or command line.

## Assumptions and constraints

- The first experiment uses FD001 and the project’s deterministic engine-wise development/validation split.
- The present deliverable is model and training code; training is not launched until explicitly requested.
- Scikit-learn is required for tabular models. PyTorch is an optional dependency for neural and sequence models and will not be installed as part of this design.
- No cloud service, multi-user service, production availability target, or external data transfer is in scope.
- Training and validation data fit in local memory.
- A model is reusable only when its recorded configuration, data identity, and split identity match the requested run.
- Hyperparameter search is intentionally limited for the initial comparison.

## Candidate model families

| Family | Candidates | Purpose |
| --- | --- | --- |
| Tabular RUL | Mean/linear reference, Random Forest, HistGradientBoosting | Primary research-backed baseline using raw or causal engineered features |
| Compact neural RUL | MLP on engineered features | Small neural comparison; optional PyTorch |
| Sequence RUL | 1D CNN and LSTM on trailing sensor windows | Compare temporal representations; optional PyTorch |
| Anomaly/health | Early-life reconstruction autoencoder | Reconstruction-error signal; no anomaly-classification claim |
| Deployment experiment | Small-model size/latency and optional quantization | Assess feasibility separately from predictive quality |

RUL target capping, raw-versus-engineered inputs, and AFICv selection are experiment settings, not separate model families. Feature selection and preprocessing are fitted using development engines only. AFICv importance aggregation, when requested, uses engine-grouped folds within development data.

## Architecture and data flow

Shared training utilities load the selected training subset, reproduce the fixed engine-wise split, construct training labels, calculate validation metrics, and manage artifact manifests. Model-specific modules build the inputs and fit/predict through a common runner. The training path must not call the official test loader.

- Tabular models use raw telemetry or past-only causal rolling features.
- The MLP uses the same tabular features with scaling fitted on development rows only.
- CNN/LSTM models use trailing sensor windows with endpoint labels; windows cannot cross engine boundaries.
- The autoencoder fits early-life windows from development engines only and returns reconstruction scores.
- TinyML measurement runs only for an eligible trained small model. Streaming updates require delayed trusted labels; unlabeled observations do not supply RUL errors.

The runner accepts an explicit candidate list. It trains only missing or incompatible artifacts. Scikit-learn models and their preprocessing are saved with joblib; PyTorch models use state dictionaries and a serialized architecture/configuration. Each artifact directory contains the model, manifest, and validation report. The manifest records model and feature configuration, seed, training-data fingerprint, development/validation engine IDs, split fingerprint, dependency versions, and validation metrics. Incompatible artifacts are rejected rather than silently reused.

## Validation and metrics

Report cycle-level RMSE, MAE, and NASA asymmetric score on held-out validation engines. Also report simulated test-time endpoint scores separately: truncate each validation engine at predeclared fractions of its observed training life (50%, 70%, and 90%), predict at the last available cycle, and compare with the known remaining cycles. The truncation rules are fixed before comparing candidates. This avoids the misleading practice of scoring the failure row (RUL 0) as if it were a right-censored official test endpoint.

For the autoencoder, report reconstruction-error distributions and their association with known RUL/engine stage; do not report anomaly classification accuracy because the dataset has no per-cycle anomaly labels. Validation cutoffs, metrics, and caveats are recorded with each artifact.

Official test files and labels are reserved for a separate later testing workflow after model selection. They must not be loaded by training, feature selection, cutoff selection, or validation reporting.

## Proposed project files

- `approaches/tabular_models.py` — tabular estimators and experiment variants.
- `approaches/neural_models.py` — optional compact MLP.
- `approaches/sequence_models.py` — optional 1D CNN and LSTM.
- `approaches/autoencoder_model.py` — optional reconstruction model and scoring.
- `approaches/tinyml.py` — model-size, latency, and optional quantization measurements.
- `approaches/training_common.py` — shared metrics, fingerprints, manifests, and artifact IO.
- `train_models.py` — command-line and notebook-callable runner.
- `requirements-models.txt` — required tabular dependency and documented optional neural dependency.
- Focused tests for data/split isolation, causal windows, metric formulas, censoring, artifact reuse/mismatch, and optional-dependency errors.

## Reliability, risks, and non-goals

Training fails clearly for invalid schemas, non-finite inputs, overlapping splits, empty sequence inputs, missing optional dependencies, or incompatible artifacts. A deterministic synthetic smoke test precedes any full FD001 training run.

Key limitations are the modest number of validation engines and the assumption that early-life windows are a reasonable healthy reference for the autoencoder. These are reported rather than hidden. Exhaustive hyperparameter optimization, official-test evaluation, deployment integration, work-order automation, and unlabeled online learning are out of scope for this implementation stage.

## Decision log

| Decision | Alternatives considered | Rationale |
| --- | --- | --- |
| Use modular model modules with shared training/artifact utilities and one runner. | Standalone duplicated scripts; one monolithic trainer. | Keeps candidate code separate while making leakage controls, metrics, and persistence consistent. |
| Implement all agreed model families, but begin experiments with FD001 tabular baselines. | Train every model and subset at once. | Enables controlled comparison and review before adding dataset complexity or optional neural dependencies. |
| Keep RUL regression, autoencoder anomaly scoring, and deployment measurement as separate tasks. | Treat reconstruction error as RUL or use unlabeled streaming errors. | Avoids unsupported targets and claims; C-MAPSS has no cycle-level anomaly labels. |
| Validate test-like endpoints using predeclared right-censor cutoffs on held-out engines. | Score each complete validation engine’s failure row as its endpoint. | Official test trajectories are right-censored and do not end at RUL 0. |
| Preserve data/split/config fingerprints with each saved artifact and reject mismatches. | Load the newest artifact regardless of configuration. | Prevents accidental reuse of a model trained on different features, data, or split. |
| Require scikit-learn and keep PyTorch optional; do not install PyTorch now. | Make all neural dependencies mandatory or omit neural code. | Supports complete candidate coverage without a large unconditional dependency install. |
| Defer official test loading/scoring to a later, explicit evaluation stage. | Use official test feedback during model selection. | Preserves an unbiased final evaluation. |

## Implementation handoff

After this design is accepted, implement the modules, tests, dependency declaration, and runner. Do not launch FD001 model training until the user explicitly requests the training run.
