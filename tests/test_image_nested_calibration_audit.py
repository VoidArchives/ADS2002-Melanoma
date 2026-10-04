"""Independent CPU audit of training-only calibration and the shared deadline.

Training and inference are mocked where needed. Calibration itself is fitted
on synthetic OOF scores, so no GPU, real image, or reserved test is accessed.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melanoma_detection import image_nested as raw_stage
from melanoma_detection import image_nested_calibration as stage


class SyntheticCalibration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.now = datetime(2026, 10, 4, 3, 0, tzinfo=timezone.utc)
        self.raw_config = raw_stage.load_config(ROOT / "configs/image_nested.yaml")
        self.raw_config["run"]["run_id"] = "raw_audit"
        self.config = stage.load_config(ROOT / "configs/image_nested_calibration.yaml")
        self.config["run"].update(run_id="calibration_audit", reference_run_id="raw_audit",
                                  reference_session="results/image_d1_runs/raw_audit/session.json")
        configs = self.root / "configs"
        configs.mkdir()
        (configs / "image_nested.yaml").write_text(yaml.safe_dump(self.raw_config), encoding="utf-8")
        (configs / "image_nested_calibration.yaml").write_text(yaml.safe_dump(self.config), encoding="utf-8")
        csv = self.root / "data/subset/processed/train_folds.csv"
        csv.parent.mkdir(parents=True)
        (self.root / "data/subset/train").mkdir(parents=True)
        rows = [{"image_name": f"image_{fold}_{patient}_{lesion}",
                 "patient_id": f"patient_{fold}_{patient}", "fold": fold,
                 "target": int(patient < 4 and lesion == 0)}
                for fold in range(5) for patient in range(12) for lesion in range(2)]
        self.frame = pd.DataFrame(rows)
        self.frame.to_csv(csv, index=False)
        for name in self.frame.image_name:
            (self.root / "data/subset/train" / f"{name}.jpg").touch()
        self.raw_prepared = raw_stage.prepare_nested(self.root, self.raw_config, audit_images=False)
        self.raw_dir = self.raw_prepared["run_dir"]
        stage._atomic_json(self.raw_dir / "run_manifest.json", self.raw_prepared["manifest"])
        with patch.object(raw_stage, "_utc_now", return_value=self.now), patch.object(stage, "_utc_now", return_value=self.now):
            raw_stage.claim_training_window(self.raw_dir / "session.json", self.raw_config)
            stage.declare_protocol(self.root, self.config)
        self.prepared = stage.prepare_calibration(self.root, self.config, raw_prepared=self.raw_prepared, audit_images=False)
        # Explicit synthetic seam; production prepare still requires decoded images.
        self.prepared["raw_prepared"]["image_audit"]["status"] = "complete"
        for relative in self.prepared["source_sha256"]:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / relative).read_bytes())
        self.inner_calls = []

    def tearDown(self):
        self.temporary.cleanup()

    def selection(self, fold):
        recipe = self.raw_config["selection"]["recipes"][fold % 4]
        return {"fold": fold, "recipe": recipe, "refit_epochs": 4,
                "confirmation": [{"inner_fold": inner, "best_epoch": epoch,
                                  "best_metrics": {"average_precision": 0.3}}
                                 for inner, epoch in enumerate([2, 4, 8])],
                "selection_uses_outer_labels": False}, f"training_selection_{fold}"

    def write_raw_outer(self):
        for fold, data in self.prepared["raw_prepared"]["folds"].items():
            predictions = data["outer"].drop(columns="image_path").rename(columns={"image_name": "image_id"})
            predictions["base_probability"] = np.where(predictions.target.eq(1), 0.65, 0.15)
            stage._atomic_csv(self.raw_dir / f"folds/{fold}/outer_predictions.csv", predictions)

    def fake_inner(self, prepared, fold, inner_fold, selection, *, device, deadline):
        data = prepared["raw_prepared"]["folds"][fold]
        validation = data["pool"].loc[data["pool"].inner_fold == inner_fold]
        fitting = data["pool"].loc[data["pool"].inner_fold != inner_fold]
        self.inner_calls.append((fold, inner_fold, fitting.copy(), validation.copy(), selection["refit_epochs"]))
        predictions = validation.drop(columns="image_path").rename(columns={"image_name": "image_id"}).copy()
        predictions["base_probability"] = np.where(predictions.target.eq(1), 0.7, 0.1)
        folder = prepared["run_dir"] / f"folds/{fold}/inner/{inner_fold}"
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / "latest.pt"
        checkpoint.write_bytes(b"synthetic fixed-duration checkpoint")
        stage._atomic_json(folder / "job_manifest.json", stage.inner_job_manifest(prepared, fold, inner_fold, selection))
        (folder / "history.csv").write_text("epoch,train_loss\n1,0.1\n2,0.1\n3,0.1\n4,0.1\n", encoding="utf-8")
        stage._atomic_csv(folder / "predictions.csv", predictions)
        receipt = {"inner_fold": inner_fold, "fixed_epochs": selection["refit_epochs"],
                   "checkpoint_sha256": stage.file_sha256(checkpoint),
                   "job_manifest_sha256": stage.file_sha256(folder / "job_manifest.json"),
                   "history_sha256": stage.file_sha256(folder / "history.csv"),
                   "predictions_sha256": stage.file_sha256(folder / "predictions.csv")}
        return predictions, receipt

    def run_fake(self, inner=None, *, now=None):
        moment = self.now if now is None else now
        with patch.object(stage, "_utc_now", return_value=moment), \
             patch.object(raw_stage, "_utc_now", return_value=moment), \
             patch.object(stage, "_selection", side_effect=lambda prepared, fold: self.selection(fold)), \
             patch.object(stage, "_inner_fit_predict", side_effect=self.fake_inner if inner is None else inner):
            return stage.run_calibration(self.root, self.config, prepared=self.prepared, device="cpu")


class CalibrationMathematicalAudit(unittest.TestCase):
    def setUp(self):
        self.config = stage.load_config(ROOT / "configs/image_nested_calibration.yaml")
        self.training = pd.DataFrame({"patient_id": [f"patient_{n}" for n in range(10)],
                                      "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                      "base_probability": [.01, .07, .15, .2, .25, .9, .8, .7, .6, .1]})

    def test_coefficients_match_independent_unweighted_c1_logistic_fit_and_exact_recall_rule(self):
        actual = stage.fit_calibration(self.training, self.config)
        p = self.training.base_probability.to_numpy()
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        independent = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(logits, self.training.target)
        self.assertEqual(actual["status"], "complete")
        self.assertAlmostEqual(actual["slope"], independent.coef_[0, 0], places=12)
        self.assertAlmostEqual(actual["intercept"], independent.intercept_[0], places=12)
        self.assertEqual(actual["raw_threshold"], .6)
        self.assertAlmostEqual(actual["calibrated_threshold"], expit(actual["slope"] * np.log(.6 / .4) + actual["intercept"]), places=12)
        flagged = self.training.base_probability >= actual["raw_threshold"]
        self.assertEqual(np.mean(flagged[self.training.target.eq(1)]), .8)
        self.assertFalse(actual["outer_labels_used"])

    def test_outer_target_changes_do_not_change_transformed_scores_or_frozen_decisions(self):
        calibrator = stage.fit_calibration(self.training, self.config)
        outer = pd.DataFrame({"target": [0, 1, 0, 1], "base_probability": [0, .25, .8, 1]})
        first = stage.apply_calibration(outer, calibrator)
        second = stage.apply_calibration(outer.assign(target=1 - outer.target), calibrator)
        for column in ("calibrated_probability", "raw_threshold", "calibrated_threshold", "decision_raw", "decision_calibrated"):
            np.testing.assert_array_equal(first[column], second[column])
        self.assertTrue(np.isfinite(first.calibrated_probability).all())

    def test_zero_or_negative_slope_fails_without_raw_fallback_promotion(self):
        for scores in (np.full(10, .2), np.where(self.training.target.eq(1), .01, .9)):
            with self.subTest(scores=scores):
                actual = stage.fit_calibration(self.training.assign(base_probability=scores), self.config)
                self.assertEqual(actual["status"], "failed")
                self.assertIsNone(actual["raw_threshold"])
                self.assertIsNone(actual["calibrated_threshold"])
                with self.assertRaises(ValueError):
                    stage.apply_calibration(self.training, actual)


class CalibrationDeadlineAudit(SyntheticCalibration):
    def test_later_requested_deadline_cannot_extend_original_session_cutoff(self):
        before = self.prepared["session_path"].read_bytes()
        with patch.object(raw_stage, "_utc_now", return_value=self.now):
            deadline = stage.shared_deadline(self.prepared, (self.now + timedelta(hours=8)).isoformat())
            earlier = stage.shared_deadline(self.prepared, (self.now + timedelta(minutes=20)).isoformat())
        self.assertEqual(deadline, self.now + timedelta(minutes=280))
        self.assertEqual(earlier, self.now + timedelta(minutes=20))
        self.assertEqual(self.prepared["session_path"].read_bytes(), before)

    def test_missing_original_session_is_rejected_without_creating_replacement(self):
        self.prepared["session_path"].unlink()
        with self.assertRaises(ValueError):
            stage.shared_deadline(self.prepared)
        self.assertFalse(self.prepared["session_path"].exists())

    def test_fifth_fold_progress_is_incomplete_until_aggregate_is_written(self):
        records = [{"fold": fold, "status": "complete"} for fold in range(5)]
        progress = stage._progress(self.prepared, records)
        self.assertEqual(progress["status"], "incomplete")
        self.assertFalse(progress["calibration_fitted"])
        self.assertFalse(progress["threshold_fitted"])


class CalibrationRunnerAudit(SyntheticCalibration):
    def test_all_folds_use_complete_training_oof_and_preserve_raw_reference_bytes(self):
        self.write_raw_outer()
        raw_files = {path: path.read_bytes() for path in self.raw_dir.rglob("*") if path.is_file()}
        summary = self.run_fake()
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["completed_outer_folds"], list(range(5)))
        self.assertEqual(len(self.inner_calls), 15)
        for fold, inner, fitting, validation, epochs in self.inner_calls:
            outer_patients = set(self.raw_prepared["folds"][fold]["outer"].patient_id)
            self.assertFalse((set(fitting.patient_id) | set(validation.patient_id)) & outer_patients)
            self.assertFalse(set(fitting.patient_id) & set(validation.patient_id))
            self.assertEqual(epochs, 4)
        combined = pd.read_csv(self.prepared["run_dir"] / "outer_oof_predictions.csv")
        self.assertEqual(set(combined.image_id), set(self.frame.image_name))
        self.assertTrue(combined.image_id.is_unique)
        for fold in range(5):
            folder = self.prepared["run_dir"] / f"folds/{fold}"
            training_oof = stage._read_predictions(folder / "training_oof_predictions.csv")
            stage.validate_oof_coverage(training_oof, self.raw_prepared["folds"][fold]["pool"])
            actual = stage._read_json(folder / "calibrator.json")
            independently_recomputed = stage.fit_calibration(training_oof, self.config)
            for key in ("slope", "intercept", "raw_threshold", "calibrated_threshold"):
                self.assertAlmostEqual(actual[key], independently_recomputed[key], places=12)
        self.assertTrue(summary["calibration_fitted"])
        self.assertTrue(summary["threshold_fitted"])
        self.assertFalse(summary["test_set_loaded"])
        for path, before in raw_files.items():
            self.assertEqual(path.read_bytes(), before)

    def test_partial_inner_fit_cannot_generate_calibrator_or_aggregate_predictions(self):
        self.write_raw_outer()
        summary = self.run_fake(inner=lambda *args, **kwargs: (None, {"status": "budget_exhausted"}))
        self.assertEqual(summary["status"], "incomplete")
        self.assertFalse(summary["calibration_fitted"])
        self.assertFalse(summary["threshold_fitted"])
        self.assertFalse((self.prepared["run_dir"] / "folds/0/calibrator.json").exists())
        self.assertFalse((self.prepared["run_dir"] / "outer_oof_predictions.csv").exists())

    def test_expired_session_only_revalidates_saved_folds_without_new_training(self):
        self.write_raw_outer()
        first = self.run_fake()
        before = self.prepared["session_path"].read_bytes()
        def forbidden(*args, **kwargs):
            self.fail("Expired original session launched a new inner fit.")
        resumed = self.run_fake(inner=forbidden, now=self.now + timedelta(minutes=281))
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(first["outer_oof_predictions_sha256"], resumed["outer_oof_predictions_sha256"])
        self.assertEqual(self.prepared["session_path"].read_bytes(), before)

    def test_changed_prepared_configuration_is_rejected_before_any_training(self):
        changed = copy.deepcopy(self.config)
        changed["evaluation"]["bootstrap_replicates"] = 3000
        with patch.object(stage, "_inner_fit_predict") as fit, self.assertRaises(ValueError):
            stage.run_calibration(self.root, changed, prepared=self.prepared, device="cpu")
        fit.assert_not_called()


class CalibrationInnerInferenceAudit(SyntheticCalibration):
    def test_fixed_refit_has_no_validation_and_hidden_outer_labels_cannot_affect_inner_input_or_manifest(self):
        selection, _ = self.selection(0)
        train_calls, inference_inputs = [], []
        def train(fitting, validation, spec, training, recipe, folder, manifest, **kwargs):
            train_calls.append((fitting.copy(), validation, copy.deepcopy(manifest), dict(kwargs)))
            self.assertIsNone(validation)
            self.assertTrue(kwargs["refit"])
            self.assertEqual(kwargs["epochs"], 4)
            folder.mkdir(parents=True, exist_ok=True)
            checkpoint = folder / "latest.pt"
            checkpoint.write_bytes(b"fixed checkpoint")
            (folder / "history.csv").write_text("epoch,train_loss\n1,0.1\n", encoding="utf-8")
            return {"status": "complete", "final_epoch": 4, "best_checkpoint": None,
                    "final_checkpoint": str(checkpoint), "microbatch": 64}
        def predict(model, frame, spec, training, device):
            inference_inputs.append(frame.copy())
            self.assertTrue(frame.target.eq(0).all())
            result = frame.drop(columns="image_path").rename(columns={"image_name": "image_id"}).copy()
            result["base_probability"] = .2
            return result
        checkpoint = {"epoch": 4, "spec": self.raw_config["model"], "training": self.raw_config["training"]}
        with patch.object(stage.core, "train_recipe", side_effect=train), \
             patch.object(stage.core, "predict_frame", side_effect=predict), \
             patch.object(stage.core, "load_model_checkpoint", return_value=(object(), checkpoint)), \
             patch.object(stage.core, "seed_everything"), \
             patch.object(stage.core, "recover_interrupted_checkpoint"):
            first, _ = stage._inner_fit_predict(self.prepared, 0, 0, selection, device="cpu", deadline=100)
            changed = copy.deepcopy(self.prepared)
            changed["raw_prepared"]["folds"][0]["outer"]["target"] = 1 - changed["raw_prepared"]["folds"][0]["outer"].target
            second, _ = stage._inner_fit_predict(changed, 0, 0, selection, device="cpu", deadline=100)
        pd.testing.assert_frame_equal(train_calls[0][0], train_calls[1][0])
        self.assertEqual(train_calls[0][2], train_calls[1][2])
        pd.testing.assert_frame_equal(inference_inputs[0], inference_inputs[1])
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first.patient_id), set(self.raw_prepared["folds"][0]["pool"].query("inner_fold == 0").patient_id))


class CalibrationWatchdogAudit(SyntheticCalibration):
    def test_supervisor_passes_original_absolute_cutoff_and_marks_timeout_incomplete(self):
        spec = importlib.util.spec_from_file_location("calibration_cli_audit", ROOT / "scripts/run_image_nested_calibration.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        class Child:
            def __init__(self):
                self.waits = 0
                self.terminated = False
            def wait(self, timeout):
                self.waits += 1
                if self.waits == 1:
                    raise subprocess.TimeoutExpired("synthetic calibration worker", timeout)
                return 2
            def terminate(self):
                self.terminated = True
            def kill(self):
                self.terminated = True
        child = Child()
        session_before = self.prepared["session_path"].read_bytes()
        config_path = self.root / "configs/image_nested_calibration.yaml"
        arguments = ["runner", "--config", str(config_path), "--device", "cpu"]
        with patch.object(cli, "ROOT", self.root), \
             patch.object(sys, "argv", arguments), \
             patch.object(stage, "prepare_calibration", return_value=self.prepared), \
             patch.object(stage, "_utc_now", return_value=self.now), \
             patch.object(raw_stage, "_utc_now", return_value=self.now), \
             patch.object(cli.subprocess, "Popen", return_value=child) as popen:
            self.assertEqual(cli.main(), 2)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--deadline-utc") + 1], (self.now + timedelta(minutes=280)).isoformat())
        self.assertTrue(child.terminated)
        summary = stage._read_json(self.prepared["run_dir"] / "summary.json")
        self.assertEqual(summary["status"], "incomplete")
        self.assertFalse(summary["calibration_fitted"])
        self.assertFalse(summary["threshold_fitted"])
        self.assertEqual(summary["stopping_reason"], "shared_watchdog_deadline")
        self.assertEqual(self.prepared["session_path"].read_bytes(), session_before)


if __name__ == "__main__":
    unittest.main()
