"""Focused tests for leakage controls and C-MAPSS model training utilities."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from approaches.neural_models import require_torch
from approaches.rolling_features import build_causal_features
from approaches.sequence_windows import build_sequence_windows
from approaches.training_common import (
    TrainingConfig,
    endpoint_metric_rows,
    prepare_training_data,
    regression_metrics,
    phm08_contributions,
    right_censored_endpoint_metadata,
    validate_manifest,
    write_manifest,
)
from approaches.tabular_models import fit_tabular_model
from evaluate_test import evaluate_saved_models_on_official_test
from train_models import load_saved_model, train_models


class TrainingCommonTests(unittest.TestCase):
    def test_causal_features_do_not_use_future_readings(self) -> None:
        frame = pd.DataFrame(
            {
                "unit": [1, 1, 1, 1],
                "cycle": [1, 2, 3, 4],
                "setting_1": [0.0] * 4,
                "setting_2": [0.0] * 4,
                "setting_3": [0.0] * 4,
                **{f"sensor_{number}": [float(number)] * 4 for number in range(1, 22)},
            }
        )
        original = build_causal_features(frame, windows=(2,))
        changed = frame.copy()
        changed.loc[changed["cycle"] == 4, "sensor_11"] = 9999.0
        updated = build_causal_features(changed, windows=(2,))
        earlier = frame["cycle"] < 4
        np.testing.assert_allclose(
            original.loc[earlier, "sensor_11_mean_2"],
            updated.loc[earlier, "sensor_11_mean_2"],
        )
        np.testing.assert_allclose(
            original.loc[earlier, "sensor_11_slope_2"],
            updated.loc[earlier, "sensor_11_slope_2"],
        )

    def test_sequence_windows_keep_engines_separate_and_labels_at_endpoint(self) -> None:
        frame = pd.DataFrame(
            {
                "unit": [1, 1, 1, 2, 2, 2],
                "cycle": [1, 2, 3, 1, 2, 3],
                "sensor_1": [11, 12, 13, 21, 22, 23],
                "rul": [2, 1, 0, 4, 3, 2],
            }
        )
        X, metadata, y = build_sequence_windows(
            frame,
            window_size=2,
            sensors=["sensor_1"],
            label_column="rul",
        )
        self.assertEqual(X.shape, (4, 2, 1))
        self.assertEqual(metadata["unit"].tolist(), [1, 1, 2, 2])
        np.testing.assert_array_equal(y, np.array([1, 0, 3, 2], dtype=np.float32))
        self.assertTrue(np.all(X[:2, :, 0] < 20))
        self.assertTrue(np.all(X[2:, :, 0] >= 20))

    def test_nasa_score_penalizes_overprediction_more(self) -> None:
        under = regression_metrics(np.array([10.0]), np.array([9.0]))
        over = regression_metrics(np.array([10.0]), np.array([11.0]))
        self.assertAlmostEqual(under["rmse"], 1.0)
        self.assertAlmostEqual(over["mae"], 1.0)
        self.assertGreater(over["nasa_score"], under["nasa_score"])

    def test_phm08_manual_reference_values(self) -> None:
        actual = np.array([100.0, 100.0, 100.0])
        predicted = np.array([87.0, 110.0, 100.0])
        scores = phm08_contributions(actual, predicted)
        expected = np.array([np.expm1(1.0), np.expm1(1.0), 0.0])
        np.testing.assert_allclose(scores, expected)
        metrics = regression_metrics(actual, predicted)
        self.assertAlmostEqual(metrics["phm08_sum"], float(expected.sum()))
        self.assertAlmostEqual(metrics["phm08_mean"], float(expected.mean()))

    def test_right_censored_endpoint_targets_are_known_remaining_cycles(self) -> None:
        frame = pd.DataFrame(
            {
                "unit": [1] * 10,
                "cycle": list(range(1, 11)),
                "rul": list(range(9, -1, -1)),
            }
        )
        endpoints = right_censored_endpoint_metadata(frame, (0.5, 0.9))
        self.assertEqual(endpoints["cutoff_cycle"].tolist(), [5, 9])
        self.assertEqual(endpoints["actual_rul"].tolist(), [5, 1])
        scores = endpoint_metric_rows(np.array([5.0, 1.0]), endpoints)
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0]["rmse"], 0.0)
        self.assertEqual(scores[1]["rmse"], 0.0)

    def test_manifest_rejects_incompatible_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            identity = {"model": "ridge", "seed": 42}
            write_manifest(manifest_path, {"identity": identity})
            loaded = validate_manifest(manifest_path, identity)
            self.assertEqual(loaded["identity"], identity)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_manifest(manifest_path, {"model": "ridge", "seed": 99})

    def test_tabular_candidates_fit_and_predict(self) -> None:
        rng = np.random.default_rng(7)
        features = pd.DataFrame(
            rng.normal(size=(30, 4)),
            columns=["a", "b", "c", "d"],
        )
        targets = 3.0 * features["a"].to_numpy() - features["b"].to_numpy()
        groups = np.repeat(np.arange(6), 5)
        cases = (
            ("mean", {}),
            ("ridge", {}),
            ("random_forest", {"n_estimators": 8}),
            ("extra_trees", {"n_estimators": 8}),
            ("hist_gradient_boosting", {"max_iter": 8}),
        )
        for name, params in cases:
            with self.subTest(model=name):
                fitted = fit_tabular_model(
                    name,
                    features,
                    targets,
                    groups,
                    seed=13,
                    model_params=params,
                )
                prediction = fitted.predict(features)
                self.assertEqual(prediction.shape, targets.shape)
                self.assertTrue(np.isfinite(prediction).all())
                self.assertTrue((prediction >= 0).all())

    def test_aficv_uses_grouped_development_folds(self) -> None:
        rng = np.random.default_rng(9)
        features = pd.DataFrame(
            rng.normal(size=(40, 5)),
            columns=[f"feature_{index}" for index in range(5)],
        )
        targets = features["feature_0"].to_numpy() * 4 + rng.normal(size=40) * 0.01
        groups = np.repeat(np.arange(8), 5)
        fitted = fit_tabular_model(
            "random_forest",
            features,
            targets,
            groups,
            seed=17,
            model_params={"n_estimators": 10},
            aficv_threshold=0.7,
        )
        self.assertIn("feature_0", fitted.feature_names)
        self.assertLessEqual(len(fitted.feature_names), features.shape[1])

    def test_optional_torch_error_is_actionable(self) -> None:
        if importlib.util.find_spec("torch") is None:
            with self.assertRaisesRegex(ImportError, "PyTorch is required"):
                require_torch()
        else:
            # A present package can still be unusable when its native Windows
            # DLL cannot initialise.  Both a usable backend and an actionable
            # optional-dependency error are acceptable outcomes.
            try:
                self.assertIsNotNone(require_torch())
            except ImportError as error:
                self.assertIn("PyTorch is required", str(error))


class TrainingRunnerTests(unittest.TestCase):
    @staticmethod
    def _write_synthetic_train_file(directory: Path) -> None:
        rows: list[list[float]] = []
        for unit in range(1, 21):
            life = 12 + unit % 4
            for cycle in range(1, life + 1):
                settings = [0.1 * (unit % 3), 0.2, 0.3]
                sensors = [
                    float(sensor * 0.1 + cycle * 0.02 + unit * 0.001)
                    for sensor in range(1, 22)
                ]
                rows.append([float(unit), float(cycle), *settings, *sensors])
        np.savetxt(directory / "train_FD001.txt", np.asarray(rows), fmt="%.6f")

    @staticmethod
    def _write_synthetic_test_files(directory: Path) -> None:
        rows: list[list[float]] = []
        for unit in range(1, 5):
            for cycle in range(1, 8):
                settings = [0.1 * (unit % 3), 0.2, 0.3]
                sensors = [
                    float(sensor * 0.1 + cycle * 0.02 + unit * 0.001)
                    for sensor in range(1, 22)
                ]
                rows.append([float(unit), float(cycle), *settings, *sensors])
        np.savetxt(directory / "test_FD001.txt", np.asarray(rows), fmt="%.6f")
        np.savetxt(directory / "RUL_FD001.txt", [5, 7, 3, 4], fmt="%d")

    def test_fd001_runner_persists_and_reuses_without_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            self._write_synthetic_train_file(data_dir)
            output_dir = root / "artifacts"
            config = TrainingConfig(
                subset="FD001",
                feature_mode="raw",
                rolling_windows=(2,),
                model_name="mean",
                endpoint_fractions=(0.5, 0.7),
            )
            first_prepared = prepare_training_data(data_dir, config)
            second_prepared = prepare_training_data(data_dir, config)
            self.assertEqual(
                first_prepared.split_sha256,
                second_prepared.split_sha256,
            )
            first = train_models(
                data_dir=data_dir,
                output_dir=output_dir,
                subset="FD001",
                models=("mean",),
                feature_mode="raw",
                rolling_windows=(2,),
                endpoint_fractions=(0.5, 0.7),
            )
            second = train_models(
                data_dir=data_dir,
                output_dir=output_dir,
                subset="FD001",
                models=("mean",),
                feature_mode="raw",
                rolling_windows=(2,),
                endpoint_fractions=(0.5, 0.7),
            )
            self.assertEqual(first.loc[0, "artifact_dir"], second.loc[0, "artifact_dir"])
            artifact_dir = Path(first.loc[0, "artifact_dir"])
            self.assertTrue((artifact_dir / "model.joblib").is_file())
            self.assertTrue((artifact_dir / "manifest.json").is_file())
            self.assertTrue((artifact_dir / "validation_metrics.json").is_file())
            loaded_model, loaded_manifest = load_saved_model(artifact_dir)
            self.assertTrue(hasattr(loaded_model, "predict"))
            self.assertEqual(loaded_manifest["model_name"], "mean")
            manifest = json.loads(
                (artifact_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["model_name"], "mean")
            self.assertGreater(manifest["validation_metrics"]["cycle_level"]["samples"], 0)
            model_path = artifact_dir / "model.joblib"
            model_path.write_bytes(model_path.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                train_models(
                    data_dir=data_dir,
                    output_dir=output_dir,
                    subset="FD001",
                    models=("mean",),
                    feature_mode="raw",
                    rolling_windows=(2,),
                    endpoint_fractions=(0.5, 0.7),
                )

    def test_official_test_evaluation_uses_only_endpoint_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            self._write_synthetic_train_file(data_dir)
            self._write_synthetic_test_files(data_dir)
            results = train_models(
                data_dir=data_dir,
                output_dir=root / "artifacts",
                subset="FD001",
                models=("mean",),
                feature_mode="raw",
                rolling_windows=(2,),
            )
            metrics, predictions = evaluate_saved_models_on_official_test(
                [results.loc[0, "artifact_dir"]],
                data_dir=data_dir,
                subset="FD001",
            )
            self.assertEqual(metrics.loc[0, "samples"], 4)
            self.assertEqual(len(predictions), 4)
            self.assertEqual(predictions["unit"].tolist(), [1, 2, 3, 4])
            self.assertEqual(predictions["actual_rul"].tolist(), [5, 7, 3, 4])

    def test_missing_torch_is_detected_before_training_any_selected_model(self) -> None:
        if importlib.util.find_spec("torch") is None:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data_dir = root / "data"
                data_dir.mkdir()
                self._write_synthetic_train_file(data_dir)
                artifact_dir = root / "artifacts"
                with self.assertRaisesRegex(ImportError, "PyTorch is required"):
                    train_models(
                        data_dir=data_dir,
                        output_dir=artifact_dir,
                        subset="FD001",
                        models=("mean", "mlp"),
                    )
                self.assertFalse(artifact_dir.exists())
        else:
            self.skipTest("PyTorch is installed; optional-dependency preflight is not applicable.")


if __name__ == "__main__":
    unittest.main()
