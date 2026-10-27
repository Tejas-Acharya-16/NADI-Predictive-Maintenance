"""Create comparison figures from saved FD001 experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def main() -> None:
    metrics = pd.read_csv(REPORT_DIR / "metrics_summary_all_models.csv")
    # Use only seed-42 tree rows for one-change-per-row ablation readability.
    tree = metrics.loc[(metrics["model_family"] == "hist_gradient_boosting") & (metrics["seed"] == 42)].copy()
    sequence = metrics.loc[metrics["model_family"].isin(["cnn", "lstm", "cnn_lstm"])].copy()
    plot = pd.concat([tree, sequence], ignore_index=True).sort_values("phm08_mean")
    labels = plot["experiment"].str.replace("cap125_", "", regex=False).str.replace("_", " ", regex=False)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].barh(labels, plot["rmse"], color="#4C78A8")
    axes[0].set(xlabel="Validation RMSE (cycles)", title="FD001 ablation RMSE")
    axes[1].barh(labels, plot["phm08_mean"], color="#E45756")
    axes[1].set(xlabel="Validation PHM08 mean", title="FD001 ablation PHM08")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ablation_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
