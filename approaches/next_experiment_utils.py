"""Leakage-safe helpers for the targeted global C-MAPSS experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd


FULL_RANGE_FRACTIONS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)


def full_range_endpoint_metadata(
    frame: pd.DataFrame,
    fractions: tuple[float, ...] = FULL_RANGE_FRACTIONS,
) -> pd.DataFrame:
    """Build causal cutoffs spanning each held-out engine's observed life.

    Fractions refer to elapsed lifetime, from 15% through 90%.  This creates
    both early high-RUL and late low-RUL endpoints without looking at features
    after the chosen cutoff.  The frame must already contain only a validation
    engine population.
    """
    if not fractions or any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("fractions must be unique values strictly between zero and one.")
    if len(set(fractions)) != len(fractions):
        raise ValueError("fractions must not contain duplicates.")
    required = {"unit", "cycle", "subset", "source_unit"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing endpoint metadata columns: {sorted(missing)}")
    rows: list[dict[str, int | float | str]] = []
    for unit, group in frame.groupby("unit", sort=True):
        first = int(group["cycle"].min())
        last = int(group["cycle"].max())
        span = last - first
        if span < 2:
            continue
        seen: set[int] = set()
        for fraction in fractions:
            cutoff = first + int(np.floor(span * fraction))
            cutoff = min(max(cutoff, first), last - 1)
            if cutoff in seen:
                continue
            seen.add(cutoff)
            rows.append(
                {
                    "unit": int(unit),
                    "source_unit": int(group["source_unit"].iloc[0]),
                    "subset": str(group["subset"].iloc[0]),
                    "cutoff_cycle": cutoff,
                    "actual_rul": last - cutoff,
                    "elapsed_fraction": float(fraction),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No full-range validation endpoints were generated.")
    return result


def add_rul_buckets(frame: pd.DataFrame, column: str = "actual_rul") -> pd.DataFrame:
    """Attach the predeclared raw-RUL buckets used in all comparisons."""
    result = frame.copy()
    result["rul_bucket"] = pd.cut(
        result[column],
        bins=[-np.inf, 100, 125, np.inf],
        labels=["<=100", "100-125", ">125"],
    ).astype(str)
    return result


def endpoint_feature_matrix(
    frame: pd.DataFrame, features: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Select feature rows for the stated engine/cycle cutoffs."""
    ordered = frame.sort_values(["unit", "cycle"], kind="stable")
    lookup = features.loc[ordered.index].copy()
    lookup["_unit"] = ordered["unit"].to_numpy()
    lookup["_cycle"] = ordered["cycle"].to_numpy()
    positions = pd.MultiIndex.from_arrays(
        [metadata["unit"].to_numpy(), metadata["cutoff_cycle"].to_numpy()]
    )
    selected = lookup.set_index(["_unit", "_cycle"]).loc[positions].copy()
    selected.index = metadata.index
    return selected


def enhanced_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, base_metrics) -> dict:
    """Add operational coverage rates to the common aligned RUL metrics."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    result = dict(base_metrics(actual, predicted))
    absolute = np.abs(predicted - actual)
    for threshold in (5, 10, 15):
        result[f"within_{threshold}_cycles_fraction"] = float(np.mean(absolute <= threshold))
    result["late_optimistic_fraction"] = float(np.mean(predicted > actual))
    result["early_conservative_fraction"] = float(np.mean(predicted < actual))
    return result
