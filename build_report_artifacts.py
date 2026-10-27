"""Assemble report CSVs from saved, reproducible experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"
LEGACY_DIR = PROJECT_ROOT / "artifacts" / "FD001"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tree = pd.read_csv(REPORT_DIR / "metrics_summary.csv")
    if "model_family" in tree.columns:
        tree = tree.loc[tree["model_family"].eq("hist_gradient_boosting")].copy()
    tree["model_family"] = "hist_gradient_boosting"
    sequence = pd.read_csv(REPORT_DIR / "sequence_metrics_summary.csv")
    sequence["model_family"] = sequence["architecture"]
    combined = pd.concat([tree, sequence], ignore_index=True, sort=False)
    combined.to_csv(REPORT_DIR / "metrics_summary.csv", index=False)
    combined.to_csv(REPORT_DIR / "metrics_summary_all_models.csv", index=False)

    legacy_metrics = pd.read_csv(LEGACY_DIR / "advanced_model_comparison.csv")
    legacy_metrics.to_csv(REPORT_DIR / "historical_official_test_comparison.csv", index=False)
    legacy_predictions = pd.read_csv(LEGACY_DIR / "advanced_model_predictions.csv")
    for model, rows in legacy_predictions.groupby("model", sort=True):
        safe_name = str(model).replace("/", "_").replace(" ", "_")
        rows.to_csv(REPORT_DIR / f"predictions_historical_{safe_name}.csv", index=False)
    print(f"Wrote {len(combined)} validation experiment rows and {legacy_predictions['model'].nunique()} historical prediction files.")


if __name__ == "__main__":
    main()
