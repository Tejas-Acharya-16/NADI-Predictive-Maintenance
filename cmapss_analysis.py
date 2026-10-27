"""Leakage-aware, pre-model diagnostics for the NASA C-MAPSS subsets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmapss_data import SENSOR_COLUMNS, SETTING_COLUMNS, add_training_rul, split_train_by_engine


def _rounded_setting_keys(frame: pd.DataFrame, decimals: int) -> pd.Series:
    values = frame[SETTING_COLUMNS].round(decimals)
    values = values.mask(values.abs() < 0.5 * (10 ** -decimals), 0.0)
    return values.astype(str).agg("|".join, axis=1)


def build_development_splits(
    datasets: dict[str, dict[str, pd.DataFrame]],
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    """Add training RUL labels and split each subset by engine, never by row."""
    prepared: dict[str, dict[str, pd.DataFrame]] = {}
    profile_rows: list[dict] = []
    for offset, (subset, parts) in enumerate(datasets.items()):
        labeled = add_training_rul(parts["train"])
        development, validation, engine_summary = split_train_by_engine(
            labeled,
            validation_fraction=validation_fraction,
            random_state=random_state + offset,
        )
        dev_units = set(development["unit"].unique())
        val_units = set(validation["unit"].unique())
        if dev_units & val_units:
            raise AssertionError(f"{subset}: an engine appears in both development and validation.")
        if len(development) + len(validation) != len(labeled):
            raise AssertionError(f"{subset}: split did not preserve every training row.")

        prepared[subset] = {
            **parts,
            "train_labeled": labeled,
            "development": development,
            "validation": validation,
            "engine_split": engine_summary,
        }
        for split_name, frame in [("development", development), ("validation", validation)]:
            life = frame.groupby("unit")["cycle"].max()
            profile_rows.append(
                {
                    "subset": subset,
                    "split": split_name,
                    "engines": int(frame["unit"].nunique()),
                    "rows": len(frame),
                    "rows_pct": 100 * len(frame) / len(labeled),
                    "median_engine_cycles": float(life.median()),
                    "min_engine_cycles": int(life.min()),
                    "max_engine_cycles": int(life.max()),
                    "median_cycle_level_rul": float(frame["rul"].median()),
                }
            )
    return prepared, pd.DataFrame(profile_rows)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    """Compute Spearman rho from ranks without requiring SciPy."""
    ranked_left = left.rank(method="average")
    ranked_right = right.rank(method="average")
    if ranked_left.nunique() < 2 or ranked_right.nunique() < 2:
        return np.nan
    return float(ranked_left.corr(ranked_right))


def sensor_trend_evidence(
    frame: pd.DataFrame,
    sensors: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize within-engine sensor-vs-life trend strength and consistency."""
    sensor_columns = SENSOR_COLUMNS if sensors is None else sensors
    ordered = frame.sort_values(["unit", "cycle"], kind="stable")
    engine_groups = list(ordered.groupby("unit", sort=True))
    rows: list[dict] = []
    for sensor in sensor_columns:
        correlations = []
        early_late_effects = []
        for _, engine in engine_groups:
            rho = _spearman(engine[sensor], engine["cycle"])
            if np.isfinite(rho):
                correlations.append(rho)
            count = len(engine)
            segment = max(1, int(np.ceil(count * 0.2)))
            early = engine[sensor].iloc[:segment]
            late = engine[sensor].iloc[-segment:]
            scale = float(engine[sensor].std())
            if np.isfinite(scale) and scale > 0:
                early_late_effects.append(float((late.median() - early.median()) / scale))
        correlations = np.asarray(correlations, dtype=float)
        if len(correlations):
            positive_fraction = float(np.mean(correlations > 0))
            negative_fraction = float(np.mean(correlations < 0))
            direction_consistency = max(positive_fraction, negative_fraction)
            median_rho = float(np.median(correlations))
            median_abs_rho = float(np.median(np.abs(correlations)))
            strong_fraction = float(np.mean(np.abs(correlations) >= 0.5))
        else:
            positive_fraction = negative_fraction = direction_consistency = np.nan
            median_rho = median_abs_rho = strong_fraction = np.nan
        rows.append(
            {
                "sensor": sensor,
                "engines_with_trend": len(correlations),
                "median_spearman_sensor_vs_cycle": median_rho,
                "median_abs_engine_spearman": median_abs_rho,
                "direction_consistency": direction_consistency,
                "positive_trend_fraction": positive_fraction,
                "negative_trend_fraction": negative_fraction,
                "engines_abs_rho_ge_0_5": strong_fraction,
                "median_late_minus_early_in_engine_sd": (
                    float(np.median(early_late_effects)) if early_late_effects else np.nan
                ),
                "std": float(frame[sensor].std()),
                "unique_values": int(frame[sensor].nunique()),
            }
        )
    result = pd.DataFrame(rows)
    result["trend_screen_score"] = (
        result["median_abs_engine_spearman"] * result["direction_consistency"]
    )
    return result.sort_values("trend_screen_score", ascending=False, na_position="last")


def setting_effect_evidence(
    frame: pd.DataFrame,
    sensors: list[str] | None = None,
    decimals: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Estimate variance associated with rounded settings groups (descriptive only)."""
    sensor_columns = SENSOR_COLUMNS if sensors is None else sensors
    setting_keys = _rounded_setting_keys(frame, decimals).rename("setting_group")
    grouped = frame[sensor_columns].groupby(setting_keys, observed=True)
    group_counts = grouped.size().sort_values(ascending=False)
    group_means = grouped.mean()
    grand_means = frame[sensor_columns].mean()
    total = ((frame[sensor_columns] - grand_means) ** 2).sum()
    between = (
        (group_means - grand_means).pow(2)
        .mul(group_counts, axis=0)
        .sum(axis=0)
    )
    shares = between.div(total.where(total > 0, 1.0))
    result = pd.DataFrame(
        {
            "sensor": sensor_columns,
            "setting_group_variance_share": [
                (
                    float(np.clip(shares[sensor], 0.0, 1.0))
                    if group_counts.size > 1
                    and total[sensor] > np.finfo(float).eps
                    else 0.0
                )
                for sensor in sensor_columns
            ],
            "overall_std": [float(frame[sensor].std()) for sensor in sensor_columns],
        }
    ).sort_values("setting_group_variance_share", ascending=False)
    return result, group_counts


def engine_mean_shift(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    sensors: list[str] | None = None,
) -> pd.DataFrame:
    """Compare engine-level means to reduce row-count weighting in split checks."""
    sensor_columns = SENSOR_COLUMNS if sensors is None else sensors
    dev_means = development.groupby("unit")[sensor_columns].mean()
    val_means = validation.groupby("unit")[sensor_columns].mean()
    rows = []
    for sensor in sensor_columns:
        pooled_scale = np.sqrt(
            (dev_means[sensor].var(ddof=1) + val_means[sensor].var(ddof=1)) / 2
        )
        difference = float(val_means[sensor].mean() - dev_means[sensor].mean())
        rows.append(
            {
                "sensor": sensor,
                "development_engine_mean": float(dev_means[sensor].mean()),
                "validation_engine_mean": float(val_means[sensor].mean()),
                "standardized_mean_difference": (
                    difference / pooled_scale if pooled_scale > 0 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "standardized_mean_difference",
        key=lambda values: values.abs(),
        ascending=False,
    )


def setting_distribution_shift(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    decimals: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare rounded setting-group coverage across development, validation, test."""
    def proportions(frame: pd.DataFrame, label: str) -> pd.Series:
        result = _rounded_setting_keys(frame, decimals).value_counts(normalize=True)
        result.name = label
        return result

    distributions = pd.concat(
        [
            proportions(development, "development"),
            proportions(validation, "validation"),
            proportions(test, "official_test"),
        ],
        axis=1,
    ).fillna(0)
    shifts = pd.DataFrame(
        [
            {
                "comparison": "development vs validation",
                "total_variation_distance": float(
                    0.5
                    * np.abs(distributions["development"] - distributions["validation"]).sum()
                ),
            },
            {
                "comparison": "development vs official test",
                "total_variation_distance": float(
                    0.5
                    * np.abs(distributions["development"] - distributions["official_test"]).sum()
                ),
            },
        ]
    )
    distributions.index.name = "rounded_setting_group"
    return distributions.sort_index(), shifts


def condition_adjusted_sensor_correlations(
    frame: pd.DataFrame,
    sensors: list[str] | None = None,
    decimals: int = 1,
) -> pd.DataFrame:
    """Correlate sensor residuals after removing rounded-setting group means."""
    sensor_columns = SENSOR_COLUMNS if sensors is None else sensors
    keys = _rounded_setting_keys(frame, decimals)
    residuals = frame[sensor_columns] - frame[sensor_columns].groupby(
        keys, observed=True
    ).transform("mean")
    return residuals.corr(method="spearman")


def high_correlation_pairs(
    correlation: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """List highly correlated sensor pairs from a correlation matrix."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")
    rows = []
    for left_index, left in enumerate(correlation.columns):
        for right in correlation.columns[left_index + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append(
                    {"sensor_a": left, "sensor_b": right, "spearman_rho": float(value)}
                )
    return pd.DataFrame(rows, columns=["sensor_a", "sensor_b", "spearman_rho"]).sort_values(
        "spearman_rho", key=lambda values: values.abs(), ascending=False
    ).reset_index(drop=True)
