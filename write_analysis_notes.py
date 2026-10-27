"""Generate tuning and validation-error notes from saved OOF outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


def main() -> None:
    summary = pd.read_csv(REPORTS / "metrics_summary.csv").sort_values(["phm08_mean", "rmse"])
    selected = str(summary.iloc[0]["candidate"])
    predictions = pd.read_csv(REPORTS / "predictions_oof_selected.csv")
    by_horizon = predictions.groupby("horizon", sort=True).agg(cases=("unit", "size"), rmse=("signed_error", lambda x: float((x.pow(2).mean()) ** .5)), mae=("absolute_error", "mean"), bias=("signed_error", "mean"), phm08_mean=("phm08_contribution", "mean")).round(3)
    metrics = summary.iloc[0]
    (REPORTS / "hgb_tuning.md").write_text(
        "# HGB tuning record\n\n"
        "The bounded search used the same five engine folds and horizons for every candidate. The HGB family covered target caps 100, 125, and 150 plus uncapped labels; 5/20 and 10/30 trailing windows; initial-reading differences; training-only near-constant sensor removal; time-feature removal; a small-leaf setting (`max_iter=240`, `learning_rate=0.05`, `max_leaf_nodes=15`, `min_samples_leaf=15`, `l2_regularization=2.0`); and `absolute_error` loss.\n\n"
        "The validation winner uses the standard bounded setting: `max_iter=180`, `learning_rate=0.08`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=False`, and cap 100. It was selected by lower pooled PHM08 mean, then RMSE. This was a limited, documented search rather than an exhaustive hyperparameter sweep.\n",
        encoding="utf-8",
    )
    (REPORTS / "error_analysis.md").write_text(
        "# Validation error analysis\n\n"
        f"The validation-selected candidate is `{selected}`. Across 500 OOF endpoint cases from 100 engines, it has RMSE {metrics.rmse:.3f}, MAE {metrics.mae:.3f}, PHM08 mean {metrics.phm08_mean:.3f}, bias {metrics.bias_predicted_minus_actual:.3f} cycles, overprediction fraction {metrics.overprediction_fraction:.3f}, P90 absolute error {metrics.absolute_error_p90:.3f}, and maximum absolute error {metrics.absolute_error_max:.3f}.\n\n"
        "## Error by predeclared horizon\n\n```text\n" + by_horizon.to_string() + "\n```\n\n"
        "Generated figures show OOF actual-versus-predicted values, residuals versus true RUL, residual histogram, horizon-band error, highest per-engine absolute error, and largest PHM08 contributions. These plots are based on held-out validation predictions only.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
