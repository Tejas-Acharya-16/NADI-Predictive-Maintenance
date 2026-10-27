"""Data loading, validation, RUL labeling, and engine-wise splits for C-MAPSS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SETTING_COLUMNS = [f"setting_{index}" for index in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]


def read_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Read one whitespace-delimited C-MAPSS telemetry file."""
    path = Path(path)
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(COLUMNS)} columns, found {frame.shape[1]}"
        )
    frame.columns = COLUMNS
    return frame


def quality_report(frame: pd.DataFrame, subset: str, split: str) -> dict:
    """Summarize schema, numeric integrity, keys, and cycle order without repair."""
    missing = sorted(set(COLUMNS) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(COLUMNS))
    if missing or unexpected:
        raise ValueError(
            f"{subset} {split}: schema mismatch; missing={missing}, unexpected={unexpected}"
        )

    ordered = frame.sort_values(["unit", "cycle"], kind="stable")
    cycle_diff = ordered.groupby("unit", sort=False)["cycle"].diff()
    file_order_diff = frame.groupby("unit", sort=False)["cycle"].diff()
    numeric = frame.select_dtypes(include=np.number)
    units = np.sort(frame["unit"].unique())
    missing_unit_ids = (
        int(len(set(range(int(units[0]), int(units[-1]) + 1)) - set(units.tolist())))
        if len(units)
        else 0
    )
    unit_cycle = frame[["unit", "cycle"]].to_numpy(dtype=float)
    invalid_unit_cycle = (
        int(
            (
                (~np.isfinite(unit_cycle)).any(axis=1)
                | (unit_cycle <= 0).any(axis=1)
                | (unit_cycle != np.floor(unit_cycle)).any(axis=1)
            ).sum()
        )
        if len(frame)
        else 0
    )
    return {
        "subset": subset,
        "split": split,
        "rows": len(frame),
        "engines": int(frame["unit"].nunique()),
        "missing_cells": int(frame.isna().sum().sum()),
        "nonfinite_cells": int((~np.isfinite(numeric)).sum().sum()),
        "duplicate_unit_cycle": int(frame.duplicated(["unit", "cycle"]).sum()),
        "non_increasing_cycles_sorted": int((cycle_diff.dropna() <= 0).sum()),
        "out_of_order_cycles_in_file": int((file_order_diff.dropna() <= 0).sum()),
        "cycle_gaps": int((cycle_diff.dropna() > 1).sum()),
        "invalid_unit_or_cycle": invalid_unit_cycle,
        "missing_unit_ids_in_range": missing_unit_ids,
        "cycle_min": int(frame["cycle"].min()),
        "cycle_max": int(frame["cycle"].max()),
        "unit_min": int(frame["unit"].min()),
        "unit_max": int(frame["unit"].max()),
    }


def add_training_rul(train: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Attach cycle-level training RUL, optionally applying an explicit upper cap."""
    if cap is not None and cap <= 0:
        raise ValueError("RUL cap must be a positive integer or None.")
    result = train.copy()
    result["rul_uncapped"] = (
        result.groupby("unit", sort=False)["cycle"].transform("max") - result["cycle"]
    )
    result["rul"] = (
        result["rul_uncapped"].clip(upper=cap) if cap is not None else result["rul_uncapped"]
    )
    return result


def load_subset(data_dir: str | Path, subset: str) -> dict[str, pd.DataFrame]:
    """Load one subset and align the supplied test RUL values to sorted test units."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {SUBSETS}.")
    data_dir = Path(data_dir)
    train = read_cmapss_file(data_dir / f"train_{subset}.txt")
    test = read_cmapss_file(data_dir / f"test_{subset}.txt")
    rul_path = data_dir / f"RUL_{subset}.txt"
    rul_raw = pd.read_csv(rul_path, sep=r"\s+", header=None)
    if rul_raw.shape[1] != 1:
        raise ValueError(f"{rul_path.name}: expected one RUL value per row.")
    test_units = np.sort(test["unit"].unique())
    if len(test_units) != len(rul_raw):
        raise ValueError(
            f"{subset}: {len(test_units)} test engines but {len(rul_raw)} RUL labels."
        )
    if not np.isfinite(rul_raw.iloc[:, 0]).all():
        raise ValueError(f"{rul_path.name}: RUL labels contain non-finite values.")
    rul_values = rul_raw.iloc[:, 0].to_numpy(dtype=float)
    if (rul_values < 0).any() or (rul_values != np.floor(rul_values)).any():
        raise ValueError(f"{rul_path.name}: RUL labels must be non-negative whole cycles.")
    test_rul = pd.DataFrame({"unit": test_units, "rul": rul_values.astype(int)})
    return {
        "train": train,
        "test": test,
        "test_rul": test_rul,
        "quality": pd.DataFrame(
            [
                quality_report(train, subset, "train"),
                quality_report(test, subset, "test"),
            ]
        ),
    }


def split_train_by_engine(
    train: pd.DataFrame,
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a reproducible lifetime-stratified split with engine IDs kept intact."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    engine_ids = np.sort(train["unit"].unique())
    if len(engine_ids) < 2:
        raise ValueError("At least two engines are required for an engine-wise split.")

    validation_count = min(
        len(engine_ids) - 1,
        max(1, int(round(len(engine_ids) * validation_fraction))),
    )
    rng = np.random.default_rng(random_state)
    last_cycles = train.groupby("unit", sort=True)["cycle"].max()
    bins = pd.qcut(
        last_cycles,
        q=min(5, int(last_cycles.nunique())),
        labels=False,
        duplicates="drop",
    )
    strata = pd.DataFrame({"unit": last_cycles.index, "life_bin": bins.to_numpy()})
    groups = [
        group["unit"].to_numpy()
        for _, group in strata.groupby("life_bin", sort=True)
    ]
    ideal = np.asarray([len(group) * validation_fraction for group in groups])
    allocations = np.floor(ideal).astype(int)
    remainder = validation_count - int(allocations.sum())
    for index in np.argsort(-(ideal - allocations))[:remainder]:
        allocations[index] += 1
    for index, group in enumerate(groups):
        if len(group) >= 2:
            allocations[index] = min(len(group) - 1, max(1, allocations[index]))
    while allocations.sum() < validation_count:
        candidates = [i for i, group in enumerate(groups) if allocations[i] < len(group) - 1]
        if not candidates:
            break
        index = max(candidates, key=lambda i: ideal[i] - allocations[i])
        allocations[index] += 1
    while allocations.sum() > validation_count:
        candidates = [i for i in range(len(groups)) if allocations[i] > (1 if len(groups[i]) > 1 else 0)]
        if not candidates:
            break
        index = min(candidates, key=lambda i: ideal[i] - allocations[i])
        allocations[index] -= 1
    validation_parts = [
        rng.choice(group, size=count, replace=False)
        for group, count in zip(groups, allocations)
        if count
    ]
    validation_ids = np.sort(np.concatenate(validation_parts))
    validation_set = set(validation_ids.tolist())
    development_ids = np.array([unit for unit in engine_ids if unit not in validation_set])

    development = train.loc[train["unit"].isin(development_ids)].copy()
    validation = train.loc[train["unit"].isin(validation_ids)].copy()
    engine_summary = (
        train.groupby("unit", sort=True)
        .agg(first_cycle=("cycle", "min"), last_cycle=("cycle", "max"), rows=("cycle", "size"))
        .reset_index()
    )
    engine_summary["split"] = np.where(
        engine_summary["unit"].isin(validation_set), "validation", "development"
    )
    return development, validation, engine_summary
