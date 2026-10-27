"""Evaluate saved C-MAPSS models on official test endpoints and RUL labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from approaches.training_common import build_feature_frame, regression_metrics
from cmapss_data import load_subset
from train_models import load_saved_model


def evaluate_saved_models_on_official_test(
    artifact_dirs: list[str | Path],
    *,
    data_dir: str | Path,
    subset: str = "FD001",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score saved models once against supplied RUL values at each test endpoint.

    Test files and labels are loaded only here, after model fitting and validation.
    The function does not fit, tune, or overwrite model artifacts.
    """
    if not artifact_dirs:
        raise ValueError("Provide at least one saved model artifact directory.")

    dataset = load_subset(data_dir, subset)
    test_rows = dataset["test"].sort_values(["unit", "cycle"], kind="stable")
    endpoint_indices = test_rows.groupby("unit", sort=True)["cycle"].idxmax()
    endpoints = test_rows.loc[endpoint_indices].sort_values("unit").reset_index(drop=True)
    actual = dataset["test_rul"].sort_values("unit").reset_index(drop=True)
    if endpoints["unit"].tolist() != actual["unit"].tolist():
        raise RuntimeError("Test endpoint engine IDs do not align with supplied RUL labels.")

    report_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for artifact_dir in artifact_dirs:
        model, manifest = load_saved_model(artifact_dir)
        identity = manifest.get("identity", {})
        config = identity.get("config", {})
        if config.get("subset") != subset:
            raise ValueError(
                f"Artifact {artifact_dir} was trained for {config.get('subset')}, "
                f"not requested subset {subset}."
            )
        feature_frame = build_feature_frame(
            test_rows,
            feature_mode=config["feature_mode"],
            rolling_windows=tuple(config["rolling_windows"]),
        )
        endpoint_features = feature_frame.loc[endpoint_indices.tolist()]
        endpoint_features = endpoint_features.reset_index(drop=True)
        predicted = model.predict(endpoint_features)
        metrics = regression_metrics(
            actual["rul"].to_numpy(),
            predicted,
        )
        model_name = str(manifest["model_name"])
        report_rows.append(
            {
                "subset": subset,
                "model": model_name,
                "artifact_dir": str(artifact_dir),
                **metrics,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "subset": subset,
                    "model": model_name,
                    "unit": actual["unit"],
                    "endpoint_cycle": endpoints["cycle"],
                    "actual_rul": actual["rul"],
                    "predicted_rul": predicted,
                    "error_predicted_minus_actual": predicted - actual["rul"].to_numpy(),
                }
            )
        )

    return pd.DataFrame(report_rows).sort_values("rmse").reset_index(drop=True), pd.concat(
        prediction_rows, ignore_index=True
    )
