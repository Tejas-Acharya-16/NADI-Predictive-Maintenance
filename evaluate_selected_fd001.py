"""One explicit official-test evaluation for the validation-selected FD001 model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from approaches.training_common import phm08_contributions, regression_metrics
from cmapss_data import load_subset


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "FD001" / "selected_hgb_full_training"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def main() -> None:
    saved = joblib.load(ARTIFACT_DIR / "model.joblib")
    model, variant = saved["model"], saved["variant"]
    dataset = load_subset(PROJECT_ROOT / "CMAPSSData", "FD001")
    test = dataset["test"].sort_values(["unit", "cycle"], kind="stable")
    endpoints = test.loc[test.groupby("unit", sort=True)["cycle"].idxmax()].sort_values("unit")
    truth = dataset["test_rul"].sort_values("unit")
    if endpoints["unit"].tolist() != truth["unit"].tolist():
        raise RuntimeError("Final test endpoints and supplied RUL vector are misaligned.")
    features = variant.transform(test)
    prediction = np.maximum(model.predict(features.loc[endpoints.index]), 0.0)
    report = pd.DataFrame(
        {
            "unit": truth["unit"].to_numpy(),
            "endpoint_cycle": endpoints["cycle"].to_numpy(),
            "actual_rul": truth["rul"].to_numpy(),
            "predicted_rul": prediction,
        }
    )
    report["residual_predicted_minus_actual"] = report["predicted_rul"] - report["actual_rul"]
    report["absolute_error"] = report["residual_predicted_minus_actual"].abs()
    report["phm08_contribution"] = phm08_contributions(report["actual_rul"], report["predicted_rul"])
    metrics = regression_metrics(report["actual_rul"], report["predicted_rul"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_DIR / "predictions_selected_hgb_full_training.csv", index=False)
    report.nlargest(10, "absolute_error").to_csv(REPORT_DIR / "worst_10_engines.csv", index=False)
    (REPORT_DIR / "selected_official_test_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(report["actual_rul"], report["predicted_rul"], alpha=0.75)
    upper = float(max(report["actual_rul"].max(), report["predicted_rul"].max()))
    ax.plot([0, upper], [0, upper], "k--", linewidth=1)
    ax.set(xlabel="Actual RUL (cycles)", ylabel="Predicted RUL (cycles)", title="FD001 selected model: actual vs predicted")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "actual_vs_predicted.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(report["residual_predicted_minus_actual"], bins=20, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--")
    ax.set(xlabel="Prediction minus actual RUL (cycles)", ylabel="Engines", title="FD001 residual distribution")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "residual_distribution.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(report["unit"], report["absolute_error"], color="#4C78A8")
    ax.set(xlabel="Engine", ylabel="Absolute RUL error (cycles)", title="FD001 per-engine absolute error")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "per_engine_absolute_error.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(report["unit"], report["phm08_contribution"], color="#E45756")
    ax.set(xlabel="Engine", ylabel="PHM08 contribution", title="FD001 PHM08 contribution by engine")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "phm08_contribution.png", dpi=150); plt.close(fig)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
