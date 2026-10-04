"""Patient-safe nested evaluation and bounded resume contracts, without a GPU.

Fixtures are small synthetic development-fold CSVs.  The runner test replaces
training and prediction with observable CPU mocks so its independent inner
selection and full outer-training refit can be checked across all five folds.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melanoma_detection import image_nested as stage


class SyntheticNested(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.csv = self.root / "data/subset/processed/train_folds.csv"
        self.csv.parent.mkdir(parents=True)
        (self.root / "data/subset/train").mkdir(parents=True)
        self.config = stage.load_config(ROOT / "configs/image_nested.yaml")
        self.config["run"]["run_id"] = "synthetic_nested"
        rows = []
        for outer in range(5):
            for patient in range(12):
                for image in range(2):
                    rows.append({"image_name": f"image_{outer}_{patient}_{image}",
                                 "patient_id": f"patient_{outer}_{patient}",
                                 "target": int(patient < 4 and image == 0),
                                 "fold": outer})
        self.frame = pd.DataFrame(rows)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, frame=None):
        (self.frame if frame is None else frame).to_csv(self.csv, index=False)
        return stage.prepare_nested(self.root, self.config, audit_images=False)

    def copy_sources(self, prepared):
        for relative in prepared["folds"][0]["fold_manifest"]["compatibility"]["source_sha256"]:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / relative).read_bytes())


class NestedPreparationTests(SyntheticNested):
    def test_all_five_outer_partitions_have_patient_disjoint_complete_inner_folds(self):
        prepared = self.prepare()
        frame, audit = prepared["frame"], prepared["image_audit"]
        covered = []
        for outer_fold in range(5):
            partition = stage.prepare_outer(frame, self.config, outer_fold, audit)
            pool, outer, manifest = (partition[name] for name in ("pool", "outer", "fold_manifest"))
            self.assertFalse(set(pool.patient_id) & set(outer.patient_id))
            self.assertEqual(set(pool.image_name) | set(outer.image_name), set(frame.image_name))
            self.assertEqual(len(pool) + len(outer), len(frame))
            self.assertEqual(set(pool.inner_fold), {0, 1, 2})
            self.assertTrue(pool.groupby("patient_id").inner_fold.nunique().eq(1).all())
            covered.extend(outer.image_name.tolist())
            for inner_fold in range(3):
                fitting = pool[pool.inner_fold != inner_fold]
                validation = pool[pool.inner_fold == inner_fold]
                self.assertFalse(set(fitting.patient_id) & set(validation.patient_id))
                self.assertEqual(fitting.target.nunique(), 2)
                self.assertEqual(validation.target.nunique(), 2)
            self.assertIsInstance(manifest["compatibility_sha256"], str)
        self.assertEqual(len(covered), len(set(covered)))
        self.assertEqual(set(covered), set(frame.image_name))

    def test_each_outer_label_perturbation_cannot_change_its_inner_split_or_manifest(self):
        prepared = self.prepare()
        baseline, baseline_audit = prepared["frame"], prepared["image_audit"]
        for outer_fold in range(5):
            with self.subTest(outer_fold=outer_fold):
                partition = stage.prepare_outer(baseline, self.config, outer_fold, baseline_audit)
                pool, outer, manifest = (partition[name] for name in ("pool", "outer", "fold_manifest"))
                changed = self.frame.copy()
                changed.loc[changed.fold.eq(outer_fold), "target"] = 1 - changed.loc[changed.fold.eq(outer_fold), "target"]
                perturbed = self.prepare(changed)
                changed_partition = stage.prepare_outer(perturbed["frame"], self.config, outer_fold, perturbed["image_audit"])
                changed_pool, changed_outer, changed_manifest = (changed_partition[name] for name in ("pool", "outer", "fold_manifest"))
                pd.testing.assert_frame_equal(pool, changed_pool)
                self.assertEqual(manifest["compatibility_sha256"], changed_manifest["compatibility_sha256"])
                self.assertEqual(set(outer.image_name), set(changed_outer.image_name))

    def test_patient_crossing_outer_folds_is_rejected_before_training(self):
        changed = self.frame.copy()
        changed.loc[changed.fold.eq(1), "patient_id"] = changed.loc[changed.fold.eq(1), "patient_id"].str.replace("patient_1_", "patient_0_", regex=False)
        with self.assertRaisesRegex(ValueError, "patient"):
            self.prepare(changed)

    def test_unsafe_input_path_and_fractional_target_are_rejected(self):
        changed = self.frame.copy()
        changed["target"] = changed.target.astype(str)
        changed.loc[0, "target"] = "1.000000000000000001"
        with self.assertRaises(ValueError):
            self.prepare(changed)
        self.frame.to_csv(self.csv, index=False)
        self.config["run"]["folds_csv"] = "data/subset/processed/../../../outside.csv"
        with self.assertRaises(ValueError):
            stage.prepare_nested(self.root, self.config, audit_images=False)

    def test_incompatible_resume_cannot_overwrite_original_manifest(self):
        prepared = self.prepare()
        self.copy_sources(prepared)
        run_dir = stage.persist_preparation(prepared)
        original = (run_dir / "run_manifest.json").read_bytes()
        changed = copy.deepcopy(prepared)
        changed["manifest"]["compatibility_sha256"] = "different_training_identity"
        with self.assertRaises(ValueError):
            stage.persist_preparation(changed)
        self.assertEqual((run_dir / "run_manifest.json").read_bytes(), original)

    def test_resume_rejects_tampered_membership_fold_identity_audit_and_source_archive(self):
        prepared = self.prepare()
        self.copy_sources(prepared)
        run_dir = stage.persist_preparation(prepared)
        relative = next(iter(prepared["folds"][0]["fold_manifest"]["compatibility"]["source_sha256"]))
        paths = [run_dir / "folds/0/inner_memberships.csv", run_dir / "folds/0/fold_manifest.json",
                 run_dir / "image_audit.json", run_dir / "source_at_training" / relative]
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                if path.suffix == ".json":
                    changed = json.loads(original)
                    changed["tampered"] = True
                    stage._atomic_json(path, changed)
                else:
                    path.write_bytes(original + b"\nchanged after preparation\n")
                with self.assertRaises(ValueError):
                    stage.persist_preparation(prepared)
                path.write_bytes(original)
        self.assertEqual(stage.persist_preparation(prepared), run_dir)


class NestedSelectionTests(SyntheticNested):
    def records(self, scores=None):
        scores = [0.3] * 4 if scores is None else scores
        return [{"recipe": copy.deepcopy(recipe), "status": "complete", "completed_epochs": 6,
                 "best_epoch": 4, "best_metrics": {"average_precision": score}}
                for recipe, score in zip(self.config["selection"]["recipes"], scores)]

    def test_ties_prefer_unweighted_then_lower_learning_rate_independent_of_order(self):
        records = self.records([0.3, 0.30001, 0.30002, 0.30003])
        expected = self.config["selection"]["recipes"][0]
        self.assertEqual(stage.select_recipe(records, 0.0001), expected)
        self.assertEqual(stage.select_recipe(list(reversed(records)), 0.0001), expected)
        records[0]["best_metrics"]["average_precision"] = 0.1
        self.assertEqual(stage.select_recipe(records, 0.0001), self.config["selection"]["recipes"][1])

    def test_recipe_outside_ap_tolerance_wins_without_tie_preferences(self):
        records = self.records([0.1, 0.2, 0.4, 0.3])
        self.assertEqual(stage.select_recipe(records, 0.0001), self.config["selection"]["recipes"][2])

    def test_missing_partial_or_nonfinite_recipe_cannot_produce_a_winner(self):
        records = self.records()
        for changed in (records[:-1], [{**records[0], "status": "budget_exhausted"}, *records[1:]],
                        [{**records[0], "completed_epochs": 5}, *records[1:]],
                        [{**records[0], "best_metrics": {"average_precision": np.nan}}, *records[1:]]):
            with self.subTest(records=changed), self.assertRaises(ValueError):
                stage.select_recipe(changed, 0.0001)

    def test_unequal_or_fractional_screening_lengths_are_rejected(self):
        for epochs in (7, 6.5, True):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                stage.select_recipe([{**self.records()[0], "completed_epochs": epochs}, *self.records()[1:]])

    def test_duration_is_median_of_three_complete_positive_integer_inner_epochs(self):
        self.assertEqual(stage.select_duration([8, 2, 4]), 4)
        self.assertEqual(stage.select_duration([12, 12, 12]), 12)
        for epochs in ([2, 4], [0, 2, 4], [True, 2, 4], [2.5, 4, 8], [13, 4, 8]):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                stage.select_duration(epochs)


class NestedBudgetTests(SyntheticNested):
    def setUp(self):
        super().setUp()
        self.session_path = self.root / "results/image_d1_runs/synthetic_nested/session.json"
        self.now = datetime(2026, 10, 4, 1, 0, tzinfo=timezone.utc)

    def test_resume_cannot_reset_initial_start_or_reporting_reserve(self):
        first = stage.claim_training_window(self.session_path, self.config, now=self.now)
        expected = self.now + timedelta(minutes=280)
        self.assertEqual(first, expected)
        original = json.loads(self.session_path.read_text(encoding="utf-8"))
        resumed = stage.claim_training_window(self.session_path, self.config, now=self.now + timedelta(minutes=60),
                                               started_utc=(self.now + timedelta(minutes=60)).isoformat())
        self.assertEqual(resumed, first)
        self.assertEqual(json.loads(self.session_path.read_text(encoding="utf-8")), original)
        with self.assertRaises(TimeoutError):
            stage.claim_training_window(self.session_path, self.config, now=expected)

    def test_implementation_time_counts_when_explicit_start_is_earlier(self):
        start = self.now - timedelta(minutes=30)
        deadline = stage.claim_training_window(self.session_path, self.config, now=self.now, started_utc=start.isoformat())
        self.assertEqual(deadline, self.now + timedelta(minutes=250))

    def test_budget_change_and_future_start_are_rejected(self):
        with self.assertRaises(ValueError):
            stage.claim_training_window(self.session_path, self.config, now=self.now,
                                         started_utc=(self.now + timedelta(seconds=1)).isoformat())
        self.assertFalse(self.session_path.exists())
        stage.claim_training_window(self.session_path, self.config, now=self.now)
        changed = copy.deepcopy(self.config)
        changed["budget"]["reporting_reserve_minutes"] = 10
        with self.assertRaises(ValueError):
            stage.claim_training_window(self.session_path, changed, now=self.now + timedelta(minutes=1))


class NestedRunnerTests(SyntheticNested):
    def setUp(self):
        super().setUp()
        self.prepared = self.prepare()
        self.prepared["image_audit"]["status"] = "complete"
        self.copy_sources(self.prepared)
        self.calls = []
        self.predictions = []
        self.now = datetime(2026, 10, 4, 1, 0, tzinfo=timezone.utc)

    def fake_fit(self, prepared, fitting, validation, recipe, folder, manifest, epochs, device, deadline, *, refit=False):
        self.calls.append({"fitting": fitting.copy(), "validation": None if validation is None else validation.copy(),
                           "recipe": copy.deepcopy(recipe), "folder": folder, "manifest": copy.deepcopy(manifest),
                           "epochs": epochs, "refit": refit})
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / "latest.pt"
        checkpoint.write_bytes(b"mock durable checkpoint")
        if refit:
            history = [{"epoch": epoch + 1, "epoch_seconds": 1.0, "train_loss": 0.1} for epoch in range(epochs)]
            best_epoch, best_metrics = None, None
        else:
            fold = manifest["outer_fold"]
            selected = self.config["selection"]["recipes"][fold % 4]["id"]
            inner = manifest["inner_fold"]
            best_epoch = 4 if epochs == 6 else [2, 4, 8][inner]
            history = [{"epoch": epoch + 1, "epoch_seconds": 1.0,
                        "average_precision": (0.5 if recipe["id"] == selected else 0.1) - 0.001 * abs(epoch + 1 - best_epoch),
                        "roc_auc": 0.8, "log_loss": 0.1, "brier": 0.02} for epoch in range(epochs)]
            best_metrics = {key: history[best_epoch - 1][key] for key in ("average_precision", "roc_auc", "log_loss", "brier")}
        return {"status": "complete", "history": history, "best_epoch": best_epoch, "best_metrics": best_metrics,
                "microbatch": 64, "latest_checkpoint": str(checkpoint), "best_checkpoint": None if refit else str(checkpoint)}

    def fake_prediction(self, prepared, fold, recipe, duration, result, manifest, device):
        self.predictions.append({"fold": fold, "recipe": recipe, "duration": duration})
        frame = prepared["folds"][fold]["outer"].drop(columns="image_path").rename(columns={"image_name": "image_id"})
        frame["base_probability"] = np.where(frame.target.eq(1), 0.7, 0.1)
        return frame

    def run_fake(self, fit=None):
        with patch.object(stage, "_utc_now", return_value=self.now), \
             patch.object(stage, "_fit", side_effect=self.fake_fit if fit is None else fit), \
             patch.object(stage, "_fixed_prediction", side_effect=self.fake_prediction):
            return stage.run_nested(self.root, self.config, prepared=self.prepared, device="cpu")

    def test_five_folds_independently_screen_confirm_and_refit_all_outer_training_patients(self):
        summary = self.run_fake()
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["completed_outer_folds"], list(range(5)))
        self.assertEqual(len(self.calls), 40)
        self.assertEqual(len(self.predictions), 5)
        for fold in range(5):
            with self.subTest(fold=fold):
                calls = [call for call in self.calls if call["manifest"]["outer_fold"] == fold]
                pool = self.prepared["folds"][fold]["pool"]
                excluded = set(self.prepared["folds"][fold]["outer"].patient_id)
                self.assertEqual([call["epochs"] for call in calls], [6] * 4 + [12] * 3 + [4])
                winner = self.config["selection"]["recipes"][fold % 4]["id"]
                self.assertEqual({call["recipe"]["id"] for call in calls[4:]}, {winner})
                for call in calls:
                    self.assertFalse(set(call["fitting"].patient_id) & excluded)
                    if call["validation"] is not None:
                        self.assertFalse(set(call["validation"].patient_id) & excluded)
                        self.assertFalse(set(call["fitting"].patient_id) & set(call["validation"].patient_id))
                refit = calls[-1]
                self.assertTrue(refit["refit"])
                self.assertIsNone(refit["validation"])
                pd.testing.assert_frame_equal(refit["fitting"], pool)
                screened = next(call for call in calls[:4] if call["recipe"]["id"] == winner)
                self.assertEqual(screened["folder"], calls[4]["folder"])
                self.assertEqual(screened["manifest"], calls[4]["manifest"])
        oof = pd.read_csv(self.prepared["run_dir"] / "outer_oof_predictions.csv")
        self.assertTrue(oof.image_id.is_unique)
        self.assertEqual(set(oof.image_id), set(self.frame.image_name))
        expected = self.frame.rename(columns={"image_name": "image_id"}).sort_values("image_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(oof[expected.columns], expected)
        self.assertFalse(summary["test_set_loaded"])
        self.assertFalse(summary["calibration_fitted"])
        self.assertFalse(summary["threshold_fitted"])

    def test_partial_screening_stops_without_selection_refit_or_outer_predictions(self):
        def partial(*args, **kwargs):
            result = self.fake_fit(*args, **kwargs)
            return {**result, "status": "budget_exhausted", "history": result["history"][:3], "best_epoch": 3}
        summary = self.run_fake(fit=partial)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["completed_outer_folds"], [])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.predictions, [])
        self.assertFalse((self.prepared["run_dir"] / "folds/0/selection.json").exists())
        self.assertFalse((self.prepared["run_dir"] / "outer_oof_predictions.csv").exists())

    def test_unaudited_fixture_and_changed_preparation_config_cannot_launch_training(self):
        self.prepared["image_audit"]["status"] = "skipped_for_synthetic_preparation"
        with self.assertRaises(ValueError):
            self.run_fake()
        self.assertEqual(self.calls, [])
        self.prepared["image_audit"]["status"] = "complete"
        changed = copy.deepcopy(self.config)
        changed["training"]["head_lr"] *= 2
        with patch.object(stage, "_fit") as fit, self.assertRaises(ValueError):
            stage.run_nested(self.root, changed, prepared=self.prepared, device="cpu")
        fit.assert_not_called()

    def test_resume_rejects_screening_score_edit_against_saved_first_six_epochs(self):
        checkpoints = {}
        def stop_confirmation(*args, **kwargs):
            result = self.fake_fit(*args, **kwargs)
            folder, manifest, epochs = args[4], args[5], args[6]
            if epochs == 6:
                checkpoints[folder / "latest.pt"] = {"manifest": manifest, "history": result["history"]}
                return result
            return {**result, "status": "budget_exhausted", "history": result["history"][:6]}
        first = self.run_fake(fit=stop_confirmation)
        self.assertEqual(first["folds"][0]["stage"], "confirmation")
        screen_path = self.prepared["run_dir"] / "folds/0/screening.json"
        screens = stage._read_json(screen_path)
        screens[0]["best_metrics"]["average_precision"] += 0.01
        stage._atomic_json(screen_path, screens)
        with patch.object(stage, "_utc_now", return_value=self.now), \
             patch.object(stage, "_fit") as fit, \
             patch("melanoma_detection.image_model.recover_interrupted_checkpoint"), \
             patch("melanoma_detection.image_model._read_checkpoint", side_effect=lambda path: checkpoints[Path(path)]), \
             self.assertRaisesRegex(ValueError, "screening scores"):
            stage.run_nested(self.root, self.config, prepared=self.prepared, device="cpu")
        fit.assert_not_called()
        self.assertEqual(self.predictions, [])


class NestedPredictionTests(SyntheticNested):
    def setUp(self):
        super().setUp()
        self.prepared = self.prepare()
        self.result = {"status": "complete", "history": [{}] * 4, "microbatch": 64,
                       "latest_checkpoint": str(self.root / "synthetic_refit.pt")}
        self.recipe = self.config["selection"]["recipes"][0]
        self.refit_manifest = {"purpose": "synthetic_fixed_refit"}
        self.checkpoint = {"epoch": 4, "spec": self.config["model"], "training": self.config["training"]}

    def test_real_prediction_wrapper_hides_outer_labels_and_restores_them_after_inference(self):
        observed = []
        def predict(model, frame, spec, training, device):
            observed.append(frame.copy())
            self.assertTrue(frame.target.eq(0).all())
            predictions = frame.drop(columns="image_path").rename(columns={"image_name": "image_id"})
            predictions["base_probability"] = 0.02
            return predictions
        with patch("melanoma_detection.image_model.load_model_checkpoint", return_value=(object(), self.checkpoint)), \
             patch("melanoma_detection.image_model.predict_frame", side_effect=predict), \
             patch("melanoma_detection.image_model.seed_everything"):
            first = stage._fixed_prediction(self.prepared, 0, self.recipe, 4, self.result, self.refit_manifest, "cpu")
            perturbed = copy.deepcopy(self.prepared)
            perturbed["folds"][0]["outer"]["target"] = 1 - perturbed["folds"][0]["outer"]["target"]
            second = stage._fixed_prediction(perturbed, 0, self.recipe, 4, self.result, self.refit_manifest, "cpu")
        pd.testing.assert_frame_equal(observed[0], observed[1])
        np.testing.assert_array_equal(first.base_probability, second.base_probability)
        np.testing.assert_array_equal(first.target, self.prepared["folds"][0]["outer"].target)
        np.testing.assert_array_equal(second.target, perturbed["folds"][0]["outer"].target)

    def test_partial_refit_or_wrong_checkpoint_epoch_cannot_predict_outer_fold(self):
        with patch("melanoma_detection.image_model.load_model_checkpoint", return_value=(object(), {**self.checkpoint, "epoch": 5})), \
             patch("melanoma_detection.image_model.predict_frame") as predict, \
             patch("melanoma_detection.image_model.seed_everything"):
            for result in ({**self.result, "status": "budget_exhausted"},
                           {**self.result, "history": [{}] * 3}, self.result):
                with self.subTest(result=result), self.assertRaises(ValueError):
                    stage._fixed_prediction(self.prepared, 0, self.recipe, 4, result, self.refit_manifest, "cpu")
            predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
