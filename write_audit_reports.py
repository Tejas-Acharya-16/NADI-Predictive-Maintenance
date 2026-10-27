"""Write the narrative reports after the completed audit-improve experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cmapss_data import load_subset, quality_report


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
DATA = ROOT / "CMAPSSData"


def main() -> None:
    subset = load_subset(DATA, "FD001")
    train_q = quality_report(subset["train"], "FD001", "train")
    test_q = quality_report(subset["test"], "FD001", "test")
    summary = pd.read_csv(REPORTS / "metrics_summary.csv").sort_values(["phm08_mean", "rmse"])
    cap100 = summary.loc[summary["candidate"] == "hgb_cap100_w10_30"].iloc[0]
    cap125 = summary.loc[summary["candidate"] == "hgb_control_cap125_w10_30"].iloc[0]
    official = json.loads((REPORTS / "grouped_audit_historical_official_test_metrics.json").read_text(encoding="utf-8"))

    (REPORTS / "data_audit.md").write_text(
        "# FD001 data audit\n\n"
        "FD001 is simulated turbofan telemetry. It must not be represented as a Siemens pump, truck, compressor, or motor-vibration dataset. Raw train and test files have 26 fields: engine ID, cycle, three operating settings, and 21 sensors.\n\n"
        "| Split | Rows | Engines | Cycle range | Integrity checks |\n| --- | ---: | ---: | --- | --- |\n"
        f"| Train | {train_q['rows']:,} | {train_q['engines']} | {train_q['cycle_min']} to {train_q['cycle_max']} | no missing/nonfinite values, duplicate unit-cycle records, gaps, invalid IDs, or ordering faults |\n"
        f"| Test | {test_q['rows']:,} | {test_q['engines']} | {test_q['cycle_min']} to {test_q['cycle_max']} | same checks passed |\n"
        f"| Supplied endpoint RUL | {len(subset['test_rul'])} | {subset['test_rul']['unit'].nunique()} | {subset['test_rul']['rul'].min()} to {subset['test_rul']['rul'].max()} cycles | aligned to sorted test engine IDs |\n\n"
        "Seven training sensors have standard deviation below 0.01: sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, and sensor_19. Their removal was fitted only on development folds and did not beat the cap-100 validation champion. Engine ID is never a model feature. Current cycle and cycles_seen are deliberately available-at-inference features; their removal was tested as a separate causal ablation.\n",
        encoding="utf-8",
    )
    (REPORTS / "metric_audit.md").write_text(
        "# Metric and target audit\n\n"
        "For a training engine at cycle t, uncapped RUL is final observed training cycle minus t. Target caps 100, 125, 150, and uncapped are named separately. Validation and official metrics always use uncapped truth.\n\n"
        "For PHM08 with d = prediction minus actual: d < 0 uses exp(-d/13)-1 and d >= 0 uses exp(d/10)-1. Both PHM08 sum and PHM08 mean are saved from exactly the same one-prediction-per-endpoint arrays as RMSE and MAE. Unit tests verify d=-13, d=10, and d=0. Overprediction is penalized more sharply.\n\n"
        "The grouped audit reports 100 distinct held-out engines and 500 correlated pseudo-censored endpoint cases. Its 300-resample engine-block bootstrap keeps all horizons from a sampled engine together. The separately recorded official benchmark has exactly 100 endpoint predictions.\n",
        encoding="utf-8",
    )
    (REPORTS / "literature_review.md").write_text(
        "# Literature review and comparison limits\n\n"
        "NASA C-MAPSS is a simulated multivariate turbofan benchmark with censored test trajectories and one RUL value per test engine. The project uses FD001 only for the completed experiments.\n\n"
        "Ensarioglu, Inkaya, and Emel (2023), Applied Sciences 13(21), 11893, motivates the compact CNN-LSTM, sensor-difference, and change-point discussion. The project implements compact CNN/LSTM/CNN-LSTM and initial-reading differences. It does not claim to reproduce the paper's change-point labels because a complete run-to-failure change point would be unavailable for a censored deployment trajectory.\n\n"
        "Ramasso and Saxena (2014) informs the grouped validation and protocol caution. The 2021 Symmetry deep-learning paper is retained as an architecture reference. Published scores are not treated as directly comparable without matching labels, preprocessing, censoring, split, and PHM08 aggregation.\n",
        encoding="utf-8",
    )
    (REPORTS / "final_summary.md").write_text(
        "# FD001 audit and improvement summary\n\n"
        "## What was executed\n\n"
        "A five-fold, engine-grouped, validation-only audit was run on FD001 training data. Each held-out engine was pseudo-censored at 10, 25, 50, 75, and 100 remaining cycles, yielding 500 endpoint cases across 100 engines. Twelve bounded tree-model and feature/target candidates were compared with the same folds. The official test files were not read during selection.\n\n"
        "## Validation results\n\n"
        "| Candidate | RMSE | MAE | R2 | PHM08 mean | 5-fold PHM08 SD |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
        f"| HGB cap 100, causal 10/30 - validation champion | {cap100.rmse:.3f} | {cap100.mae:.3f} | {cap100.r2:.3f} | {cap100.phm08_mean:.3f} | {cap100.phm08_mean_fold_std:.3f} |\n"
        f"| HGB cap 125, causal 10/30 control | {cap125.rmse:.3f} | {cap125.mae:.3f} | {cap125.r2:.3f} | {cap125.phm08_mean:.3f} | {cap125.phm08_mean_fold_std:.3f} |\n\n"
        "The cap-100 result is strong within this deliberately bounded 10-to-100-cycle validation schedule. It should not be described as a general deployment replacement for the cap-125 model because the protocol does not assess RUL above 100 cycles. The corrected time-free ablation was worse than the cap-125 control, so current-cycle information is useful and causally available.\n\n"
        "## Historical official benchmark comparison\n\n"
        f"After validation was locked, cap-100 was compared once with the previously inspected official benchmark: RMSE {official['rmse']:.3f}, MAE {official['mae']:.3f}, R2 {official['r2']:.3f}, PHM08 mean {official['phm08_mean']:.3f}, and bias {official['bias_predicted_minus_actual']:.3f} cycles. The comparison is secondary only and did not influence selection. Its negative bias is consistent with the cap-100 training target.\n\n"
        "The earlier cap-125 historical official benchmark remains RMSE 13.532, MAE 9.799, R2 0.894, and PHM08 mean 2.852. It was produced under the older fixed-split study. Do not rank the two official scores as a clean selection comparison.\n\n"
        "## End-to-end demo\n\n"
        "The repository includes causal feature construction, a compact neural comparison, early-life autoencoder anomaly scoring, deterministic triage with persistence and work-order idempotency, a Streamlit dashboard, data validation, descriptive drift indicators, and a versioned training-only retraining simulation. FD001 has no cycle-level anomaly truth, so anomaly outputs are not reported as classification accuracy.\n\n"
        "## Limits and next work\n\n"
        "Use an expanded, predeclared validation horizon schedule before making a production cap choice. Validate FD002 and FD004 with operating-condition-aware preprocessing. Calibrate prediction intervals and triage thresholds with domain owners. Measure latency, RAM, quantization loss, and update cost on the actual edge target before making an on-device deployment claim.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
