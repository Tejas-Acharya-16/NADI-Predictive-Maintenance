"""AFICv-style aggregation of feature importances across validation folds."""

from __future__ import annotations

import pandas as pd


def select_aficv_features(
    fold_importances: pd.DataFrame,
    cumulative_threshold: float = 0.70,
) -> pd.DataFrame:
    """Rank features by mean fold importance and retain the cumulative threshold.

    ``fold_importances`` must contain one row per engine-wise validation fold
    and one column per feature. This helper aggregates fitted-model outputs;
    it does not fit models or select features using validation/test outcomes.
    """
    if fold_importances.empty or fold_importances.shape[1] == 0:
        raise ValueError("fold_importances must contain at least one fold and feature.")
    if not 0 < cumulative_threshold <= 1:
        raise ValueError("cumulative_threshold must be in (0, 1].")
    if fold_importances.isna().any().any():
        raise ValueError("fold_importances contains missing values.")
    if (fold_importances < 0).any().any():
        raise ValueError("Feature importances must be non-negative.")

    totals = fold_importances.sum(axis=1)
    normalized = fold_importances.div(totals.replace(0, 1), axis=0)
    summary = pd.DataFrame(
        {
            "mean_importance": normalized.mean(axis=0),
            "std_importance": normalized.std(axis=0, ddof=0),
        }
    ).sort_values("mean_importance", ascending=False)
    summary["cumulative_importance"] = summary["mean_importance"].cumsum()
    summary["selected"] = (
        summary["cumulative_importance"].shift(fill_value=0) < cumulative_threshold
    )
    return summary
