"""CPU-limited staged CNN, LSTM, and CNN-LSTM FD001 validation study.

The script uses only training trajectories. Held-out engines are exposed only
through fixed 50/70/90 percent prefixes for early stopping and final validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from approaches.neural_models import require_torch
from approaches.sequence_models import SequenceRegressor
from approaches.sequence_windows import build_sequence_windows
from approaches.training_common import phm08_contributions, regression_metrics, right_censored_endpoint_metadata
from cmapss_data import SENSOR_COLUMNS, add_training_rul, read_cmapss_file, split_train_by_engine


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "CMAPSSData"
REPORT_DIR = PROJECT_ROOT / "reports"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "FD001" / "sequence_study"
WINDOW = 30
RUL_CAP = 125


def _with_initial_differences(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.sort_values(["unit", "cycle"], kind="stable").copy()
    initial = result.groupby("unit", sort=False)[SENSOR_COLUMNS].transform("first")
    added: list[str] = []
    for sensor in SENSOR_COLUMNS:
        name = f"{sensor}_minus_initial"
        result[name] = result[sensor] - initial[sensor]
        added.append(name)
    return result, [*SENSOR_COLUMNS, *added]


def _endpoint_indices(frame: pd.DataFrame, metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    endpoint = right_censored_endpoint_metadata(frame, (0.5, 0.7, 0.9))
    lookup = {(int(row.unit), int(row.end_cycle)): index for index, row in enumerate(metadata.itertuples(index=False))}
    positions: list[int] = []
    for row in endpoint.itertuples(index=False):
        key = (int(row.unit), int(row.cutoff_cycle))
        if key not in lookup:
            raise RuntimeError(f"Window {key} is unavailable; reduce window length.")
        positions.append(lookup[key])
    return np.asarray(positions, dtype=int), endpoint


def main() -> None:
    torch, _ = require_torch()
    torch.set_num_threads(4)
    raw = read_cmapss_file(DATA_DIR / "train_FD001.txt")
    development_raw, validation_raw, split_summary = split_train_by_engine(raw, validation_fraction=0.2, random_state=42)
    development = add_training_rul(development_raw, cap=RUL_CAP)
    validation = add_training_rul(validation_raw, cap=None)
    candidates = [
        ("cnn_w30_cap125", "cnn", False),
        ("lstm_w30_cap125", "lstm", False),
        ("cnn_lstm_w30_cap125", "cnn_lstm", False),
        ("cnn_lstm_w30_cap125_initial_difference", "cnn_lstm", True),
    ]
    records: list[dict[str, object]] = []
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, architecture, use_difference in candidates:
        if use_difference:
            dev_frame, input_columns = _with_initial_differences(development)
            val_frame, _ = _with_initial_differences(validation)
        else:
            dev_frame, val_frame, input_columns = development, validation, SENSOR_COLUMNS
        x_train, _, y_train = build_sequence_windows(dev_frame, window_size=WINDOW, sensors=input_columns, label_column="rul", stride=1)
        x_validation, metadata, y_validation = build_sequence_windows(val_frame, window_size=WINDOW, sensors=input_columns, label_column="rul", stride=1)
        if y_train is None or y_validation is None:
            raise RuntimeError("Sequence labels were not generated.")
        endpoint_positions, endpoint_metadata = _endpoint_indices(validation, metadata)
        endpoint_x = x_validation[endpoint_positions]
        endpoint_y = y_validation[endpoint_positions]
        model = SequenceRegressor(
            architecture=architecture, hidden_size=16, epochs=20, batch_size=256,
            learning_rate=1e-3, seed=42, patience=4,
        ).fit(x_train, y_train, endpoint_x, np.minimum(endpoint_y, RUL_CAP))
        prediction = model.predict(endpoint_x)
        endpoint_report = endpoint_metadata.copy()
        endpoint_report["predicted_rul"] = prediction
        endpoint_report["error_predicted_minus_actual"] = prediction - endpoint_y
        endpoint_report["absolute_error"] = np.abs(endpoint_report["error_predicted_minus_actual"])
        endpoint_report["phm08_contribution"] = phm08_contributions(endpoint_y, prediction)
        metrics = regression_metrics(endpoint_y, prediction)
        destination = ARTIFACT_DIR / name
        destination.mkdir(parents=True, exist_ok=True)
        model.save(destination / "model.pt")
        endpoint_report.to_csv(destination / "validation_endpoint_predictions.csv", index=False)
        split_summary.to_csv(destination / "split_engines.csv", index=False)
        record = {
            "experiment": name,
            "architecture": architecture,
            "initial_difference_features": use_difference,
            "window_size": WINDOW,
            "rul_cap": RUL_CAP,
            "input_features": len(input_columns),
            "epochs_completed": model.trained_epochs_,
            "validation_cases": len(endpoint_report),
            "protocol": "grouped held-out engines; fixed 50/70/90 percent prefixes; early stopping on capped prefix targets",
            **metrics,
        }
        (destination / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        records.append(record)
        print(name, json.dumps({key: record[key] for key in ("rmse", "mae", "phm08_mean", "epochs_completed")}, sort_keys=True))
    pd.DataFrame(records).sort_values(["phm08_mean", "rmse"]).to_csv(REPORT_DIR / "sequence_metrics_summary.csv", index=False)


if __name__ == "__main__":
    main()
