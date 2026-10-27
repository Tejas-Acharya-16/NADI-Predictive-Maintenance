"""Run a small training-only anomaly reference and deterministic triage replay."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from approaches.anomaly_detection import EarlyLifeAnomalyScorer
from approaches.autoencoder_inputs import build_early_life_windows
from approaches.autoencoder_model import EarlyLifeAutoencoder
from approaches.sequence_windows import build_sequence_windows
from cmapss_data import SENSOR_COLUMNS, add_training_rul, read_cmapss_file, split_train_by_engine
from triage import RuleBasedTriage


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"
SELECTED_MODEL = PROJECT_ROOT / "artifacts" / "FD001" / "selected_hgb_full_training" / "model.joblib"


def main() -> None:
    raw = read_cmapss_file(PROJECT_ROOT / "CMAPSSData" / "train_FD001.txt")
    development_raw, validation_raw, _ = split_train_by_engine(raw, validation_fraction=0.2, random_state=42)
    development = add_training_rul(development_raw, cap=125)
    validation = add_training_rul(validation_raw, cap=None)
    healthy_windows, _ = build_early_life_windows(development, early_fraction=0.2, window_size=20, sensors=SENSOR_COLUMNS)
    scorer = EarlyLifeAnomalyScorer(
        EarlyLifeAutoencoder(window_size=20, hidden_size=32, bottleneck_size=8, epochs=20, batch_size=256, seed=42),
        threshold_quantile=0.99,
    ).fit(healthy_windows)
    windows, metadata, _ = build_sequence_windows(validation, window_size=20, sensors=SENSOR_COLUMNS, label_column=None)
    scores = scorer.score(windows)
    saved = joblib.load(SELECTED_MODEL)
    feature_frame = saved["variant"].transform(validation)
    # Metadata rows align to window end cycles, so retrieve matching feature rows by unit/cycle.
    prediction_lookup = pd.DataFrame({"unit": validation["unit"], "cycle": validation["cycle"], "predicted_rul": saved["model"].predict(feature_frame)})
    streamed = metadata.copy()
    streamed["anomaly_score"] = scores
    streamed["anomaly"] = scores > float(scorer.threshold)
    streamed = streamed.merge(prediction_lookup, left_on=["unit", "end_cycle"], right_on=["unit", "cycle"], how="left", validate="one_to_one")
    triage = RuleBasedTriage()
    decisions = []
    for row in streamed.sort_values(["unit", "end_cycle"], kind="stable").itertuples(index=False):
        decisions.append(triage.assess(
            engine_id=int(row.unit), cycle=int(row.end_cycle), predicted_rul=float(row.predicted_rul),
            anomaly=bool(row.anomaly), anomaly_score=float(row.anomaly_score), anomaly_threshold=float(scorer.threshold),
            data_quality="good", model_version="selected_hgb_full_training", interval_width=None,
        ))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(decisions).to_csv(REPORT_DIR / "triage_demo_decisions.csv", index=False)
    (REPORT_DIR / "triage_demo_work_orders.json").write_text(json.dumps(triage.work_orders, indent=2), encoding="utf-8")
    (REPORT_DIR / "anomaly_demo_metrics.json").write_text(json.dumps({
        "healthy_reference_windows": int(len(healthy_windows)),
        "validation_scored_windows": int(len(windows)),
        "threshold_quantile": scorer.threshold_quantile,
        "reconstruction_error_threshold": scorer.threshold,
        "anomaly_rate_validation": float(np.mean(scores > float(scorer.threshold))),
        "note": "C-MAPSS has no cycle-level anomaly labels; this is not classification accuracy.",
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(triage.work_orders)} idempotent mock work orders; threshold={scorer.threshold:.6f}")


if __name__ == "__main__":
    main()
