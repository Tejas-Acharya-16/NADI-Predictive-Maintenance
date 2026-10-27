"""One explicitly labelled, previously-inspected official FD001 benchmark comparison."""

from __future__ import annotations

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from approaches.training_common import phm08_contributions, regression_metrics
from cmapss_data import load_subset


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "CMAPSSData"
REPORTS = ROOT / "reports"
ARTIFACT = ROOT / "artifacts" / "FD001" / "grouped_audit" / "selected_validation_candidate" / "model.joblib"


def run(data_dir: Path = DATA, report_dir: Path = REPORTS, artifact: Path = ARTIFACT) -> dict:
    """Evaluate an already validation-selected FD001 artifact once on its endpoint test set."""
    payload = joblib.load(artifact)
    model = payload["model"]
    variant = payload["feature_variant"]
    candidate = payload["candidate"]
    candidate_name = candidate["name"] if isinstance(candidate, dict) else candidate.name
    subset = load_subset(data_dir, "FD001")
    test = subset["test"].sort_values(["unit", "cycle"], kind="stable")
    features = variant.transform(test)
    endpoints = test.groupby("unit", sort=True).tail(1)
    predicted = model.predict(features.loc[endpoints.index])
    truth = subset["test_rul"].sort_values("unit")
    if endpoints["unit"].tolist() != truth["unit"].tolist():
        raise ValueError("Official endpoint and RUL unit IDs are not aligned.")
    output = pd.DataFrame({"unit": endpoints["unit"].to_numpy(), "actual_rul": truth["rul"].to_numpy(), "predicted_rul": predicted})
    output["signed_error"] = output["predicted_rul"] - output["actual_rul"]
    output["absolute_error"] = output["signed_error"].abs()
    output["phm08_contribution"] = phm08_contributions(output["actual_rul"].to_numpy(), output["predicted_rul"].to_numpy())
    metrics = regression_metrics(output["actual_rul"].to_numpy(), output["predicted_rul"].to_numpy())
    metrics.update({
        "candidate": candidate_name,
        "benchmark_status": "historical official test previously inspected; this comparison did not influence selection",
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
    })
    report_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(report_dir / "predictions_grouped_audit_historical_official_test.csv", index=False)
    (report_dir / "grouped_audit_historical_official_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig = report_dir / "figures"; fig.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.2, 4.6)); plt.scatter(output.actual_rul, output.predicted_rul, s=20, alpha=.7)
    limit = max(output.actual_rul.max(), output.predicted_rul.max()) + 5; plt.plot([0, limit], [0, limit], "k--", linewidth=1)
    plt.xlabel("Actual RUL cycles"); plt.ylabel("Predicted RUL cycles"); plt.title("Historical official-test comparison")
    plt.tight_layout(); plt.savefig(fig / "grouped_audit_historical_official_actual_vs_predicted.png", dpi=150); plt.close()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--report-dir", type=Path, default=REPORTS)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.data_dir, arguments.report_dir, arguments.artifact), indent=2))
