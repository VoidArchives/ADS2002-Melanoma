"""Training-label isolation, fixed-duration refits and bounded calibration tests."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from test_image_model import SyntheticImages, stochastic_tiny_factory
from melanoma_detection import image_nested as raw_stage
from melanoma_detection import image_nested_calibration as stage


class CalibrationFixture(SyntheticImages):
    def setUp(self):
        super().setUp()
        self.config = stage.load_config(ROOT / "configs/image_nested_calibration.yaml")
        self.raw_config = raw_stage.load_config(ROOT / "configs/image_nested.yaml")
        self.raw_config["model"] = self.model_spec
        self.raw_config["training"] = self.training
        self.frame = self.make_frame(patients_per_fold=3, images_per_patient=2)
        audit = raw_stage.pilot._audit_images(self.frame.assign(target=0), decode=False)
        folds = {fold: raw_stage.prepare_outer(self.frame, self.raw_config, fold, audit) for fold in range(5)}
        output = self.root / "results/image_d1_runs"
        self.raw_dir = output / self.config["run"]["reference_run_id"]
        self.run_dir = output / self.config["run"]["run_id"]
        self.prepared = {"root": self.root, "config": self.config, "raw_config": self.raw_config,
                         "raw_dir": self.raw_dir, "run_dir": self.run_dir, "session_path": self.raw_dir / "session.json",
                         "source_sha256": {"synthetic_source": "fixed_source_v1"},
                         "manifest": {"qualification": "synthetic independent outer patients"},
                         "raw_prepared": {"frame": self.frame, "folds": folds, "image_audit": {"status": "complete"}}}
        self.recipe = self.raw_config["selection"]["recipes"][0]

    def selection(self, fold=0):
        return {"fold": fold, "recipe": self.recipe, "refit_epochs": 2,
                "confirmation": [{"inner_fold": n, "best_epoch": 2, "best_metrics": {}} for n in range(3)],
                "selection_uses_outer_labels": False}

    def training_oof(self, fold=0):
        pool = self.prepared["raw_prepared"]["folds"][fold]["pool"].drop(columns="image_path").copy()
        pool["base_probability"] = np.where(pool.target.eq(1), .8, .02)
        return pool

    def write_raw_outer(self):
        for fold, data in self.prepared["raw_prepared"]["folds"].items():
            frame = data["outer"].drop(columns="image_path").copy()
            frame["base_probability"] = np.where(frame.target.eq(1), .7, .08)
            frame["model_name"] = "efficientnet_b0"
            stage._atomic_csv(self.raw_dir / f"folds/{fold}/outer_predictions.csv", frame)

    def mock_inner(self, prepared, fold, inner_fold, selection, **kwargs):
        predictions = self.training_oof(fold)
        predictions = predictions.loc[predictions.inner_fold.eq(inner_fold)].copy().reset_index(drop=True)
        return predictions, {"inner_fold": inner_fold}

    def runner_patches(self):
        return (patch.object(stage, "persist_preparation"),
                patch.object(stage, "shared_deadline", return_value=datetime.now(timezone.utc) + timedelta(minutes=5)),
                patch.object(stage, "_selection", side_effect=lambda prepared, fold: (self.selection(fold), f"selection_{fold}")),
                patch.object(stage, "_inner_fit_predict", side_effect=self.mock_inner))


class CalibrationMethodTests(CalibrationFixture):
    def test_sigmoid_and_thresholds_are_fitted_only_on_training_oof(self):
        oof = self.training_oof()
        stage.validate_oof_coverage(oof, self.prepared["raw_prepared"]["folds"][0]["pool"])
        result = stage.fit_calibration(oof, self.config)
        self.assertEqual(result["status"], "complete")
        self.assertGreater(result["slope"], 0)
        self.assertEqual(result["C"], 1.0)
        self.assertFalse(result["outer_labels_used"])
        calibrated = stage.calibrated_scores(result, oof.base_probability)
        self.assertGreaterEqual(np.mean(calibrated[oof.target.eq(1)] >= result["calibrated_threshold"]), .8)
        self.assertGreaterEqual(np.mean(oof.loc[oof.target.eq(1), "base_probability"] >= result["raw_threshold"]), .8)

    def test_reversed_training_scores_fail_without_a_calibrated_fallback(self):
        oof = self.training_oof()
        oof["base_probability"] = 1 - oof.base_probability
        result = stage.fit_calibration(oof, self.config)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["slope"])
        self.assertIsNone(result["calibrated_threshold"])
        with self.assertRaises(ValueError):
            stage.apply_calibration(oof, result)

    def test_outer_outcome_changes_cannot_change_calibrated_scores_or_decisions(self):
        calibrator = stage.fit_calibration(self.training_oof(), self.config)
        outer = self.prepared["raw_prepared"]["folds"][0]["outer"].copy()
        outer["base_probability"] = np.linspace(.01, .9, len(outer))
        changed = outer.copy()
        changed["target"] = 1 - changed.target
        original = stage.apply_calibration(outer, calibrator)
        perturbed = stage.apply_calibration(changed, calibrator)
        for column in ("calibrated_probability", "raw_threshold", "calibrated_threshold", "decision_raw", "decision_calibrated"):
            pd.testing.assert_series_equal(original[column], perturbed[column])

    def test_duplicate_or_missing_training_predictions_are_rejected(self):
        pool = self.prepared["raw_prepared"]["folds"][0]["pool"]
        oof = self.training_oof()
        for invalid in (oof.iloc[:-1], pd.concat([oof.iloc[:-1], oof.iloc[[0]]], ignore_index=True)):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                stage.validate_oof_coverage(invalid, pool)
        changed = oof.copy()
        changed.loc[0, "inner_fold"] = (changed.loc[0, "inner_fold"] + 1) % 3
        with self.assertRaises(ValueError):
            stage.validate_oof_coverage(changed, pool)

    def test_reloaded_integer_targets_match_canonical_int8_targets(self):
        pool = self.prepared["raw_prepared"]["folds"][0]["pool"].copy()
        pool["target"] = pool.target.astype(np.int8)
        oof = self.training_oof()
        oof["target"] = oof.target.astype(np.int64)
        stage.validate_oof_coverage(oof, pool)
        self.prepared["raw_prepared"]["folds"][0]["outer"]["target"] = self.prepared["raw_prepared"]["folds"][0]["outer"].target.astype(np.int8)
        raw = self.prepared["raw_prepared"]["folds"][0]["outer"].copy()
        raw["target"] = raw.target.astype(np.int64)
        raw["base_probability"] = .1
        stage._check_raw_outer(self.prepared, 0, raw)

    def test_three_inner_folds_cover_training_patients_and_exclude_outer_patients(self):
        for data in self.prepared["raw_prepared"]["folds"].values():
            pool, outer = data["pool"], data["outer"]
            observed = []
            for inner in range(3):
                fitting = pool.loc[pool.inner_fold.ne(inner)]
                validation = pool.loc[pool.inner_fold.eq(inner)]
                self.assertFalse(set(fitting.patient_id) & set(validation.patient_id))
                self.assertFalse((set(fitting.patient_id) | set(validation.patient_id)) & set(outer.patient_id))
                observed.extend(validation.image_id)
            self.assertEqual(set(observed), set(pool.image_id))
            self.assertEqual(len(observed), len(pool))

    def test_job_identity_ignores_outer_labels_and_full_raw_summary(self):
        original = stage.inner_job_manifest(self.prepared, 0, 1, self.selection())
        changed = copy.deepcopy(self.prepared)
        changed["raw_prepared"]["folds"][0]["outer"]["target"] = 1 - changed["raw_prepared"]["folds"][0]["outer"].target
        changed["manifest"]["reference_outer_oof_sha256"] = "different_outer_outcomes"
        self.assertEqual(original, stage.inner_job_manifest(changed, 0, 1, self.selection()))
        self.assertNotIn("reference_outer_oof_sha256", original)

    def test_fixed_protocol_changes_are_rejected(self):
        for key, value in (("C", 2.0), ("target_recall", .9), ("inner_folds", 2), ("reject_nonpositive_slope", False)):
            changed = copy.deepcopy(self.config)
            changed["calibration"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                stage.validate_config(changed)


class CalibrationRefitTests(CalibrationFixture):
    def test_real_cpu_inner_refit_uses_no_validation_labels_and_zero_label_inference(self):
        train_original, load_original, predict_original = stage.core.train_recipe, stage.core.load_model_checkpoint, stage.core.predict_frame
        fitting_calls, prediction_targets = [], []

        def train_tiny(fitting, validation, *args, **kwargs):
            fitting_calls.append((fitting.copy(), validation, kwargs["refit"]))
            kwargs["model_factory"] = stochastic_tiny_factory
            return train_original(fitting, validation, *args, **kwargs)

        def load_tiny(*args, **kwargs):
            kwargs["model_factory"] = stochastic_tiny_factory
            return load_original(*args, **kwargs)

        def predict_spy(model, frame, *args, **kwargs):
            prediction_targets.extend(frame.target.tolist())
            return predict_original(model, frame, *args, **kwargs)

        with patch.object(stage.core, "train_recipe", side_effect=train_tiny), \
             patch.object(stage.core, "load_model_checkpoint", side_effect=load_tiny), \
             patch.object(stage.core, "predict_frame", side_effect=predict_spy):
            predictions, receipt = stage._inner_fit_predict(self.prepared, 0, 1, self.selection(), device="cpu", deadline=time.monotonic() + 300)
        self.assertEqual(len(fitting_calls), 1)
        fitting, validation, refit = fitting_calls[0]
        self.assertIsNone(validation)
        self.assertTrue(refit)
        self.assertTrue(prediction_targets)
        self.assertEqual(set(prediction_targets), {0})
        expected = self.prepared["raw_prepared"]["folds"][0]["pool"]
        expected = expected.loc[expected.inner_fold.eq(1)].reset_index(drop=True)
        self.assertEqual(predictions.target.tolist(), expected.target.tolist())
        self.assertEqual(receipt["fixed_epochs"], 2)
        self.assertFalse(set(fitting.patient_id) & set(expected.patient_id))

    def test_expired_inner_fit_preserves_checkpoint_and_resumes_fixed_duration(self):
        original = stage.core.train_recipe

        def train_tiny(*args, **kwargs):
            kwargs["model_factory"] = stochastic_tiny_factory
            return original(*args, **kwargs)

        load_original = stage.core.load_model_checkpoint

        def load_tiny(*args, **kwargs):
            kwargs["model_factory"] = stochastic_tiny_factory
            return load_original(*args, **kwargs)

        with patch.object(stage.core, "train_recipe", side_effect=train_tiny), patch.object(stage.core, "load_model_checkpoint", side_effect=load_tiny):
            predictions, paused = stage._inner_fit_predict(self.prepared, 0, 0, self.selection(), device="cpu", deadline=time.monotonic() - 1)
            self.assertIsNone(predictions)
            self.assertEqual(paused["status"], "budget_exhausted")
            self.assertIsNone(paused["final_checkpoint"])
            resumed, receipt = stage._inner_fit_predict(self.prepared, 0, 0, self.selection(), device="cpu", deadline=time.monotonic() + 300)
        self.assertIsNotNone(resumed)
        self.assertEqual(receipt["fixed_epochs"], 2)


class CalibrationBudgetAndRunnerTests(CalibrationFixture):
    def test_shared_clock_requires_existing_session_and_cannot_be_extended(self):
        with self.assertRaisesRegex(ValueError, "original session"):
            stage.shared_deadline(self.prepared)
        self.assertFalse(self.prepared["session_path"].exists())
        original = raw_stage.claim_training_window(self.prepared["session_path"], self.raw_config)
        claimed = stage.shared_deadline(self.prepared, (original + timedelta(hours=4)).isoformat())
        self.assertEqual(claimed, original)
        receipt = stage._read_json(self.prepared["session_path"])
        receipt["training_cutoff_utc"] = (original + timedelta(minutes=1)).isoformat()
        stage._atomic_json(self.prepared["session_path"], receipt)
        with self.assertRaises(ValueError):
            stage.shared_deadline(self.prepared)

    def test_all_five_completed_folds_are_required_for_combined_calibrated_oof(self):
        self.write_raw_outer()
        patches = self.runner_patches()
        with patches[0], patches[1], patches[2], patches[3] as fit:
            summary = stage.run_calibration(self.root, self.config, prepared=self.prepared, device="cpu")
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["completed_outer_folds"], list(range(5)))
        self.assertEqual(fit.call_count, 15)
        combined = stage._read_predictions(self.run_dir / "outer_oof_predictions.csv")
        self.assertEqual(len(combined), len(self.frame))
        self.assertTrue(combined.image_id.is_unique)
        self.assertTrue({"base_probability", "calibrated_probability", "raw_threshold", "calibrated_threshold", "decision_raw", "decision_calibrated"}.issubset(combined))

    def test_partial_refit_never_promotes_combined_calibration(self):
        self.write_raw_outer()
        patches = self.runner_patches()
        calls = 0

        def paused_inner(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                return None, {"status": "budget_exhausted"}
            return self.mock_inner(*args, **kwargs)

        with patches[0], patches[1], patches[2], patch.object(stage, "_inner_fit_predict", side_effect=paused_inner):
            summary = stage.run_calibration(self.root, self.config, prepared=self.prepared, device="cpu")
        self.assertEqual(summary["status"], "incomplete")
        self.assertFalse(summary["calibration_fitted"])
        self.assertFalse((self.run_dir / "outer_oof_predictions.csv").exists())

    def test_failed_sigmoid_does_not_silently_use_raw_scores_as_calibrated(self):
        self.write_raw_outer()
        patches = self.runner_patches()
        with patches[0], patches[1], patches[2], patches[3], \
             patch.object(stage, "fit_calibration", return_value={"status": "failed", "failure_reason": "nonpositive slope"}):
            summary = stage.run_calibration(self.root, self.config, prepared=self.prepared, device="cpu")
        self.assertEqual(summary["stopping_reason"], "calibration_failed_no_fallback")
        self.assertFalse((self.run_dir / "outer_oof_predictions.csv").exists())
        self.assertFalse((self.run_dir / "folds/0/outer_predictions.csv").exists())

    def test_five_fold_progress_is_incomplete_until_aggregate_is_written(self):
        records = [{"fold": fold, "status": "complete"} for fold in range(5)]
        result = stage._progress(self.prepared, records)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["calibration_fitted"])


if __name__ == "__main__":
    unittest.main()
