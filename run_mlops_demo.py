"""Create training-only data-validation, drift, and retraining-demo artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from approaches.feature_variants import CausalFeatureVariant
from approaches.tabular_models import fit_tabular_model
from cmapss_data import SENSOR_COLUMNS, add_training_rul, quality_report, read_cmapss_file
from run_fd001_grouped_audit import DATA, HGB_BASE, REPORTS, SEED


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "FD001" / "mlops_demo"


def psi(reference: np.ndarray, observed: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    expected, _ = np.histogram(reference, bins=edges)
    actual, _ = np.histogram(observed, bins=edges)
    expected = np.maximum(expected / max(expected.sum(), 1), 1e-6)
    actual = np.maximum(actual / max(actual.sum(), 1), 1e-6)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def fit_version(train: pd.DataFrame, units: list[int], label: str) -> dict:
    frame = add_training_rul(train.loc[train["unit"].isin(units)].copy(), cap=100)
    variant = CausalFeatureVariant(windows=(10, 30)).fit(frame)
    model = fit_tabular_model("hist_gradient_boosting", variant.transform(frame), frame["rul"].to_numpy(), frame["unit"].to_numpy(), seed=SEED, model_params=HGB_BASE)
    directory = OUT / label
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_variant": variant, "training_units": units}, directory / "model.joblib")
    return {"version": label, "training_engines": len(units), "training_rows": len(frame), "artifact": str(directory.relative_to(ROOT)), "status": "simulated retraining complete"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    train = read_cmapss_file(DATA / "train_FD001.txt").sort_values(["unit", "cycle"], kind="stable")
    quality = quality_report(train, "FD001", "train")
    early = train.groupby("unit", sort=False).head(30)
    late = train.groupby("unit", sort=False).tail(30)
    drift = pd.DataFrame({"sensor": SENSOR_COLUMNS, "population_stability_index": [psi(early[name].to_numpy(), late[name].to_numpy()) for name in SENSOR_COLUMNS], "early_mean": [float(early[name].mean()) for name in SENSOR_COLUMNS], "late_mean": [float(late[name].mean()) for name in SENSOR_COLUMNS]}).sort_values("population_stability_index", ascending=False)
    drift.to_csv(REPORTS / "drift_indicators_training_only.csv", index=False)
    units = sorted(train["unit"].unique().tolist())
    history = [
        fit_version(train, units[:-5], "v1_train_95_engines"),
        fit_version(train, units, "v2_train_100_engines"),
    ]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "training-only demonstration; official test truth was not used for retraining",
        "quality_report": quality,
        "drift_reference": "descriptive comparison of first 30 versus final 30 observed training cycles per engine; degradation is expected in this simulation and PSI is not a production alarm threshold",
        "top_drift_features": drift.head(10).to_dict(orient="records"),
        "retraining_history": history,
    }
    (REPORTS / "mlops_demo.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
