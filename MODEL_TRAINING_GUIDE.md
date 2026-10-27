# C-MAPSS Model Training Guide

The model code is implemented, but **no model has been trained by this setup step**. The runner uses only a selected subset's `train_FDxxx.txt`; it does not open official test files or test RUL labels.

## Install tabular-model dependency

From the project root, install the required dependency into the project environment:

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-models.txt
```

PyTorch is not installed by that file. Install a compatible PyTorch build separately only when choosing MLP, CNN/LSTM, or autoencoder candidates.

## First FD001 comparison

Run from the project root:

```powershell
& '.\.venv\Scripts\python.exe' train_models.py `
  --data-dir '.\CMAPSSData' `
  --artifact-dir '.\artifacts' `
  --subset FD001 `
  --models mean ridge random_forest hist_gradient_boosting `
  --feature-mode causal
```

To compare raw features, use `--feature-mode raw`. To try an RUL target cap, pass `--rul-cap 125`; the reported validation metrics remain against uncapped RUL. AFICv is available only for the Random Forest candidate, for example `--models random_forest --aficv-threshold 0.7`; the importance folds stay inside development engines.

After PyTorch is installed, select `mlp`, `cnn`, `lstm`, or `autoencoder`. Examples:

```powershell
& '.\.venv\Scripts\python.exe' train_models.py --subset FD001 --models mlp
& '.\.venv\Scripts\python.exe' train_models.py --subset FD001 --models cnn lstm
& '.\.venv\Scripts\python.exe' train_models.py --subset FD001 --models autoencoder
```

Neural training defaults can be changed with `--epochs`, `--batch-size`, `--hidden-size`, and `--learning-rate`. Sequence and autoencoder window lengths are configurable with `--sequence-window-size` and `--autoencoder-window-size`.

## Validation and saved artifacts

The runner reports cycle-level RMSE, MAE, and NASA asymmetric score. It also reports endpoint scores at fixed 50%, 70%, and 90% life cutoffs on held-out validation engines. These right-censored validation points are more representative of official test endpoints than failure rows with RUL zero. The validation engine count and sequence-window coverage are recorded.

Each configuration is saved under a fingerprinted folder inside `artifacts/<subset>/<model>/`. It contains a model file, `manifest.json`, and validation metrics; endpoint predictions or autoencoder scores are also saved. A repeated matching command validates and loads the saved artifact instead of fitting again. Changed configuration, data, or split produces a different fingerprint. Incomplete or mismatched artifacts fail explicitly; they are never silently overwritten. Do not load joblib artifacts from untrusted sources.

The TinyML utility `approaches.tinyml.measure_tinyml` compares local PyTorch CPU state-dict size and inference latency, optionally with dynamic int8 quantization. These are workstation measurements, not proof of target-device RAM, power, or real-time performance. C-MAPSS has no per-cycle anomaly labels, so autoencoder reconstruction scores must not be reported as anomaly-classification accuracy. Online updates require delayed trusted labels.

## Notebook use

```python
from train_models import train_models

results = train_models(
    data_dir="CMAPSSData",
    output_dir="artifacts",
    subset="FD001",
    models=("mean", "ridge", "random_forest", "hist_gradient_boosting"),
    feature_mode="causal",
)
results
```

After training, load a selected artifact for later evaluation without refitting:

```python
from train_models import load_saved_model

artifact_dir = results.loc[results["model"] == "random_forest", "artifact_dir"].iloc[0]
model, manifest = load_saved_model(artifact_dir)
print(manifest["validation_metrics"])
```

For the later test stage, reconstruct the same causal feature columns from the manifest and then predict:

```python
from approaches.training_common import build_feature_frame
from cmapss_data import read_cmapss_file

model_config = manifest["identity"]["config"]
test_rows = read_cmapss_file("CMAPSSData/test_FD001.txt")
test_features = build_feature_frame(
    test_rows,
    feature_mode=model_config["feature_mode"],
    rolling_windows=tuple(model_config["rolling_windows"]),
)
last_rows = test_rows.loc[test_rows.groupby("unit")["cycle"].idxmax()].sort_values("unit")
last_features = test_features.loc[last_rows.index]
test_rul_predictions = model.predict(last_features)
```

This example obtains predictions only. Keep official `RUL_FD001.txt` untouched until the separate, agreed final evaluation step.

For example, a loaded MLP can be benchmarked on already standardized batches with:

```python
import numpy as np
from approaches.neural_models import MLPRegressor
from approaches.tinyml import measure_tinyml

mlp = MLPRegressor.load("artifacts/FD001/mlp/<fingerprint>/model.pt")
sample = np.zeros((32, len(mlp.mean_)), dtype=np.float32)
print(measure_tinyml(mlp.network, sample))
```

The example uses a shape-only zero batch to measure inference cost; use real, development-fitted standardized inputs for a meaningful deployment benchmark.
