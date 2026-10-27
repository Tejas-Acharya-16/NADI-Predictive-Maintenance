"""Tests for training-fitted features and deterministic work-order behavior."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from approaches.feature_variants import CausalFeatureVariant
from approaches.condition_normalization import ConditionAwareNormalizer
from cmapss_data import SENSOR_COLUMNS, SETTING_COLUMNS
from triage import RuleBasedTriage


class FeatureVariantTests(unittest.TestCase):
    def test_initial_difference_features_do_not_use_future_values(self) -> None:
        rows = []
        for cycle in range(1, 5):
            row = {"unit": 1, "cycle": cycle, **{name: 0.0 for name in SETTING_COLUMNS}}
            row.update({sensor: float(cycle) for sensor in SENSOR_COLUMNS})
            rows.append(row)
        original = pd.DataFrame(rows)
        changed = original.copy()
        changed.loc[changed["cycle"] >= 3, "sensor_2"] = 999.0
        variant = CausalFeatureVariant(windows=(2,), include_initial_differences=True).fit(original)
        left = variant.transform(original).loc[original["cycle"] <= 2]
        right = variant.transform(changed).loc[changed["cycle"] <= 2]
        pd.testing.assert_frame_equal(left, right)

    def test_low_variance_screen_is_fit_on_training_only(self) -> None:
        frame = pd.DataFrame({
            "unit": [1, 1, 2, 2], "cycle": [1, 2, 1, 2],
            **{name: np.zeros(4) for name in SETTING_COLUMNS},
            **{sensor: np.arange(4, dtype=float) for sensor in SENSOR_COLUMNS},
        })
        frame["sensor_1"] = 1.0
        variant = CausalFeatureVariant(low_variance_threshold=0.01).fit(frame)
        self.assertNotIn("sensor_1", variant.kept_sensors or [])

    def test_cycle_free_variant_drops_both_time_features(self) -> None:
        frame = pd.DataFrame({
            "unit": [1, 1, 1], "cycle": [1, 2, 3],
            **{name: np.zeros(3) for name in SETTING_COLUMNS},
            **{sensor: np.arange(3, dtype=float) for sensor in SENSOR_COLUMNS},
        })
        features = CausalFeatureVariant(windows=(2,), include_absolute_cycle=False).fit_transform(frame)
        self.assertNotIn("cycle", features.columns)
        self.assertNotIn("cycles_seen", features.columns)

    def test_condition_normalizer_is_fitted_on_source_rows(self) -> None:
        rows = []
        for condition in range(6):
            for cycle in (1, 2):
                row = {"unit": condition + 1, "cycle": cycle,
                       "setting_1": float(condition), "setting_2": float(condition * 2), "setting_3": float(condition % 2)}
                row.update({sensor: float(condition * 10 + cycle) for sensor in SENSOR_COLUMNS})
                rows.append(row)
        source = pd.DataFrame(rows)
        target = source.iloc[:2].copy()
        target.loc[:, SENSOR_COLUMNS] = target[SENSOR_COLUMNS] + 1000.0
        normalizer = ConditionAwareNormalizer(n_conditions=6, random_state=7).fit(source)
        transformed, assignments = normalizer.transform(target)
        self.assertEqual(len(assignments), len(target))
        self.assertTrue(np.isfinite(transformed[SENSOR_COLUMNS].to_numpy()).all())
        self.assertGreater(float(transformed["sensor_1"].abs().mean()), 10.0)


class TriageTests(unittest.TestCase):
    def test_replay_does_not_duplicate_a_work_order(self) -> None:
        triage = RuleBasedTriage()
        kwargs = dict(
            engine_id=7, cycle=42, predicted_rul=10.0, anomaly=True,
            anomaly_score=1.0, anomaly_threshold=0.5, data_quality="good",
            model_version="unit-test",
        )
        triage.assess(**kwargs)
        triage.assess(**kwargs)
        self.assertEqual(len(triage.work_orders), 1)

    def test_bad_quality_requires_human_review(self) -> None:
        decision = RuleBasedTriage().assess(
            engine_id=7, cycle=42, predicted_rul=10.0, anomaly=True,
            anomaly_score=1.0, anomaly_threshold=0.5, data_quality="stale",
            model_version="unit-test",
        )
        self.assertEqual(decision["severity"], "EXCEPTION")


if __name__ == "__main__":
    unittest.main()
