"""Numerical and patient-integrity checks for the first-stage image pipeline.

The training tests use a tiny CPU model and synthetic JPEGs, never B0 weights
or development images.  They test contracts needed before a real pilot runs.
"""

from __future__ import annotations

import copy
import hashlib
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melanoma_detection import image_model as core


MODEL = {
    "architecture": "efficientnet_b0",
    "weights": "IMAGENET1K_V1",
    "image_size": 32,
    "dropout": 0.2,
    "augmentation": "weak",
    "preprocessing": "none",
}

TRAINING = {
    "microbatch": 4,
    "effective_batch": 4,
    "min_microbatch": 4,
    "head_lr": 0.001,
    "weight_decay": 0.0001,
    "warmup_epochs": 1,
    "schedule_epochs": 30,
    "min_lr_factor": 0.1,
    "gradient_clip": 1.0,
    "amp": False,
    "num_workers": 0,
    "seed": 42,
}

RECIPE = {"id": "test_unweighted", "backbone_lr": 0.0001, "loss_weight": "none"}


class TinyImageModel(nn.Module):
    """A three-channel classifier with the same feature/head contract as B0."""

    def __init__(self, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(3, 1))

    def forward(self, images):
        return self.classifier(self.features(images)).flatten()


def tiny_factory(spec, pretrained=True, device="cpu"):
    return TinyImageModel().to(device)


def stochastic_tiny_factory(spec, pretrained=True, device="cpu"):
    return TinyImageModel(dropout=0.2).to(device)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SyntheticImages(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "data" / "subset" / "train"
        self.images.mkdir(parents=True)
        self.csv = self.root / "data" / "subset" / "processed" / "train_folds.csv"
        self.csv.parent.mkdir(parents=True)
        self.training = copy.deepcopy(TRAINING)
        self.model_spec = copy.deepcopy(MODEL)
        # Tiny models have no large convolutions; avoid thread startup overhead.
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)
        self.temporary.cleanup()

    def make_frame(self, patients_per_fold=12, images_per_patient=2):
        rows = []
        for fold in range(5):
            for person in range(patients_per_fold):
                for lesion in range(images_per_patient):
                    index = len(rows)
                    name = f"image_{fold}_{person}_{lesion}"
                    path = self.images / f"{name}.jpg"
                    rng = np.random.default_rng(10_000 + index)
                    pixels = rng.integers(0, 256, size=(28, 32, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(path, quality=94)
                    rows.append({
                        "image_name": name,
                        "image_id": name,
                        "image_path": str(path),
                        "patient_id": f"patient_{fold}_{person}",
                        "fold": fold,
                        "target": int(person < 4 and lesion == 0),
                    })
        return pd.DataFrame(rows)

    def config(self):
        from melanoma_detection import image_pilot as pilot

        config = pilot.load_config(ROOT / "configs" / "image_model.yaml")
        config["run"]["folds_csv"] = "data/subset/processed/train_folds.csv"
        config["run"]["image_dir"] = "data/subset/train"
        config["run"]["output_dir"] = "results/image_d1_runs"
        config["run"]["run_id"] = "synthetic_pilot"
        return config

    def prepare(self, frame, audit=False):
        from melanoma_detection import image_pilot as pilot

        frame[["image_name", "patient_id", "fold", "target"]].to_csv(self.csv, index=False)
        return pilot.prepare_pilot(self.root, self.config(), audit_images=audit)

    def training_parts(self):
        frame = self.make_frame(patients_per_fold=3, images_per_patient=2)
        fitting = frame.iloc[:7].copy().reset_index(drop=True)
        validation = frame.iloc[12:18].copy().reset_index(drop=True)
        fitting.loc[:, "target"] = [0, 1, 0, 1, 0, 0, 1]
        validation.loc[:, "target"] = [0, 1, 0, 1, 0, 1]
        return fitting, validation

    def train(self, output, epochs=2, training=None, manifest=None,
              factory=tiny_factory, deadline=None):
        fitting, validation = self.training_parts()
        return core.train_recipe(
            fitting, validation, self.model_spec,
            self.training if training is None else training, RECIPE,
            self.root / output,
            {"compatibility_sha256": "synthetic_partition_v1"} if manifest is None else manifest,
            epochs=epochs, device="cpu", deadline=deadline,
            resume=True, model_factory=factory,
        )


class ImagePartitionTests(SyntheticImages):
    def test_outer_label_perturbation_cannot_change_fitting_partition_or_manifest(self):
        frame = self.make_frame()
        first = self.prepare(frame)
        changed = frame.copy()
        changed.loc[changed.fold.eq(0), "target"] = 1 - changed.loc[changed.fold.eq(0), "target"]
        second = self.prepare(changed)
        for name in ("fitting", "validation", "inner_memberships"):
            pd.testing.assert_frame_equal(first[name], second[name])
        self.assertEqual(first["manifest"]["compatibility_sha256"],
                         second["manifest"]["compatibility_sha256"])
        fitting_patients = set(first["fitting"].patient_id)
        inner_patients = set(first["validation"].patient_id)
        outer_patients = set(first["outer_excluded"].patient_id)
        self.assertFalse(fitting_patients & inner_patients)
        self.assertFalse((fitting_patients | inner_patients) & outer_patients)

    def test_invalid_target_identifier_and_cross_fold_patient_are_rejected(self):
        frame = self.make_frame()
        for column, value in (("target", 0.5), ("patient_id", ""), ("image_name", "")):
            bad = frame.copy()
            if column == "target":
                bad[column] = bad[column].astype(float)
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises((ValueError, AssertionError)):
                self.prepare(bad)
        bad = frame.copy()
        bad.loc[bad.fold.eq(1).idxmax(), "patient_id"] = frame.iloc[0].patient_id
        with self.assertRaises((ValueError, AssertionError)):
            self.prepare(bad)

    def test_duplicate_image_identifier_is_rejected(self):
        frame = self.make_frame()
        frame.loc[1, "image_name"] = frame.loc[0, "image_name"]
        with self.assertRaises((ValueError, AssertionError)):
            self.prepare(frame)

    def test_fractional_csv_target_is_rejected_before_float_rounding(self):
        frame = self.make_frame()
        frame["target"] = frame.target.astype(str)
        frame.loc[0, "target"] = "1.000000000000000001"
        with self.assertRaises((ValueError, AssertionError)):
            self.prepare(frame)

    def test_audit_rejects_missing_corrupt_and_crosspatient_duplicate_images(self):
        frame = self.make_frame()
        missing = Path(frame.iloc[0].image_path)
        original = missing.read_bytes()
        missing.unlink()
        with self.subTest(case="missing"), self.assertRaises((ValueError, FileNotFoundError, AssertionError)):
            self.prepare(frame, audit=True)
        missing.write_bytes(b"not a JPEG")
        with self.subTest(case="corrupt"), self.assertRaises((ValueError, OSError, AssertionError)):
            self.prepare(frame, audit=True)
        missing.write_bytes(original)
        shutil.copyfile(missing, Path(frame.iloc[2].image_path))
        with self.subTest(case="content_duplicate"), self.assertRaises((ValueError, AssertionError)):
            self.prepare(frame, audit=True)

    def test_train_recipe_rejects_overlapping_patient_groups(self):
        fitting, validation = self.training_parts()
        validation.loc[0, "patient_id"] = fitting.iloc[0].patient_id
        with self.assertRaises((ValueError, AssertionError)):
            core.train_recipe(fitting, validation, self.model_spec, self.training,
                              RECIPE, self.root / "overlap", {}, epochs=1,
                              device="cpu", model_factory=tiny_factory)


class ImageTransformTests(SyntheticImages):
    def test_validation_preparation_is_deterministic_and_training_can_be_reseeded(self):
        frame = self.make_frame(patients_per_fold=1)
        with Image.open(frame.iloc[0].image_path) as source:
            image = source.convert("RGB")
        deterministic = core.make_transform(self.model_spec, training=False)
        first = deterministic(image)
        core.seed_everything(17)
        second = deterministic(image)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(tuple(first.shape), (3, 32, 32))
        stochastic = core.make_transform(self.model_spec, training=True)
        core.seed_everything(42)
        first = stochastic(image)
        core.seed_everything(42)
        second = stochastic(image)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_unimplemented_variants_fail_explicitly(self):
        for key, value in (("architecture", "swin_t"),
                           ("augmentation", "unsafe_crop"),
                           ("preprocessing", "hair_removal"),
                           ("weights", "DEFAULT")):
            spec = {**self.model_spec, key: value}
            with self.subTest(key=key), self.assertRaises((ValueError, NotImplementedError)):
                core.make_transform(spec, training=False)

    def test_batchnorm_affines_and_statistics_stay_frozen_in_both_stages(self):
        model = TinyImageModel()
        model.features = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3),
                                       nn.AdaptiveAvgPool2d(1), nn.Flatten())
        bn = model.features[1]
        before = {name: value.clone() for name, value in bn.named_buffers()}
        for warmup in (True, False):
            core.configure_training_stage(model, warmup=warmup)
            core.freeze_batchnorm(model)
            self.assertFalse(bn.training)
            self.assertFalse(bn.weight.requires_grad)
            self.assertFalse(bn.bias.requires_grad)
            output = model(torch.randn(3, 3, 8, 8))
            self.assertTrue(torch.isfinite(output).all())
            for name, value in bn.named_buffers():
                torch.testing.assert_close(before[name], value, rtol=0, atol=0)
        self.assertTrue(model.features[0].weight.requires_grad)


class ImageTrainingTests(SyntheticImages):
    def test_expired_budget_preserves_epoch_zero_checkpoint_for_resume(self):
        result = self.train("budget", epochs=2, deadline=time.monotonic() - 1)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["history"], [])
        self.assertIsNone(result["best_checkpoint"])
        state = torch.load(result["latest_checkpoint"], map_location="cpu", weights_only=True)
        self.assertEqual(state["epoch"], 0)
        continued = self.train("budget", epochs=2)
        self.assertEqual(continued["status"], "complete")
        self.assertEqual(len(continued["history"]), 2)

    def test_accumulation_matches_effective_batches_including_partial_window(self):
        direct = self.train("direct", epochs=2)
        accumulated_settings = {**self.training, "microbatch": 2, "min_microbatch": 2}
        accumulated = self.train("accumulated", epochs=2, training=accumulated_settings)
        a = torch.load(direct["latest_checkpoint"], map_location="cpu", weights_only=False)
        b = torch.load(accumulated["latest_checkpoint"], map_location="cpu", weights_only=False)
        for key in a["model_state"]:
            torch.testing.assert_close(a["model_state"][key], b["model_state"][key],
                                       rtol=1e-5, atol=1e-7)

    def test_epoch_boundary_resume_matches_uninterrupted_stochastic_training(self):
        full = self.train("full", epochs=3, factory=stochastic_tiny_factory)
        self.train("resumed", epochs=1, factory=stochastic_tiny_factory)
        # Disturb all RNGs to prove that continuation uses stored RNG state.
        core.seed_everything(901)
        _ = torch.rand(7)
        continued = self.train("resumed", epochs=3, factory=stochastic_tiny_factory)
        a = torch.load(full["latest_checkpoint"], map_location="cpu", weights_only=False)
        b = torch.load(continued["latest_checkpoint"], map_location="cpu", weights_only=False)
        for key in a["model_state"]:
            torch.testing.assert_close(a["model_state"][key], b["model_state"][key],
                                       rtol=0, atol=0)
        pd.testing.assert_frame_equal(pd.DataFrame(full["history"])[[
            "epoch", "train_loss", "average_precision", "log_loss"]],
            pd.DataFrame(continued["history"])[[
                "epoch", "train_loss", "average_precision", "log_loss"]])

    def test_short_run_uses_fixed_schedule_and_compatible_resume_only(self):
        first = self.train("fixed_schedule", epochs=1)
        second = self.train("fixed_schedule", epochs=3)
        self.assertEqual(first["history"][0]["head_lr"], second["history"][0]["head_lr"])
        self.assertEqual(len(second["history"]), 3)
        changed = {"compatibility_sha256": "different_partition"}
        with self.assertRaises((ValueError, AssertionError)):
            self.train("fixed_schedule", epochs=4, manifest=changed)
        with self.assertRaises((ValueError, AssertionError)):
            self.train("fixed_schedule", epochs=4,
                       training={**self.training, "schedule_epochs": 12})

    def test_short_training_cap_does_not_compress_the_thirty_epoch_schedule(self):
        result = self.train("schedule_values", epochs=6)
        rates = [row["head_lr"] for row in result["history"]]
        self.assertEqual(rates[0], self.training["head_lr"])
        self.assertEqual(rates[1], self.training["head_lr"])
        self.assertGreater(rates[-1], self.training["head_lr"] * 0.9)
        state = torch.load(result["latest_checkpoint"], map_location="cpu", weights_only=True)
        self.assertEqual(state["scheduler_state"]["epochs"], 30)

    def test_oom_retries_roll_back_partial_updates_before_reducing_microbatch(self):
        reference_settings = {**self.training, "microbatch": 2, "min_microbatch": 2}
        reference = self.train("oom_reference", epochs=2, training=reference_settings)
        original = core.train_epoch
        calls = 0

        def partial_update_then_fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                # Simulate allocation failure after updates have already occurred.
                original(*args, **kwargs)
                raise torch.cuda.OutOfMemoryError("injected partial-epoch failure")
            return original(*args, **kwargs)

        with patch.object(core, "train_epoch", side_effect=partial_update_then_fail):
            retried = self.train("oom_retry", epochs=2,
                                 training={**self.training, "min_microbatch": 2})
        self.assertEqual(retried["status"], "complete")
        self.assertEqual(retried["microbatch"], 2)
        a = torch.load(reference["latest_checkpoint"], map_location="cpu", weights_only=True)
        b = torch.load(retried["latest_checkpoint"], map_location="cpu", weights_only=True)
        for key in a["model_state"]:
            torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)


    def test_checkpoint_prediction_preserves_history_and_patient_export(self):
        result = self.train("prediction", epochs=2)
        checkpoint = Path(result["best_checkpoint"])
        history = checkpoint.parent / "history.csv"
        previous = digest(history)
        loaded = core.load_model_checkpoint(checkpoint, device="cpu",
                                            model_factory=tiny_factory)
        model = loaded[0] if isinstance(loaded, tuple) else loaded["model"]
        _, validation = self.training_parts()
        predictions = core.predict_frame(model, validation, self.model_spec,
                                         self.training, device="cpu")
        self.assertEqual(digest(history), previous)
        self.assertEqual(len(predictions), len(validation))
        required = {"image_id", "patient_id", "fold", "target", "model_name", "base_probability"}
        self.assertTrue(required.issubset(predictions.columns))
        self.assertEqual(predictions.patient_id.tolist(), validation.patient_id.tolist())
        self.assertTrue(predictions.base_probability.between(0, 1).all())
        repeat = core.predict_frame(model, validation, self.model_spec,
                                    self.training, device="cpu")
        pd.testing.assert_frame_equal(predictions, repeat)
        with self.assertRaises((ValueError, AssertionError)):
            core.load_model_checkpoint(checkpoint, device="cpu",
                                       expected_manifest={"compatibility_sha256": "different"},
                                       model_factory=tiny_factory)


class ImageFixedDurationRefitTests(SyntheticImages):
    def refit(self, output, epochs=3, *, fitting=None, training=None,
              factory=stochastic_tiny_factory, deadline=None):
        if fitting is None:
            fitting, _ = self.training_parts()
        return core.train_recipe(
            fitting, None, self.model_spec, self.training if training is None else training,
            RECIPE, self.root / output, {"training_partition": "synthetic_refit_v1"},
            epochs=epochs, refit=True, device="cpu", resume=True,
            model_factory=factory, deadline=deadline,
        )

    def test_refit_never_predicts_or_selects_an_epoch_from_validation_labels(self):
        with patch.object(core, "predict_frame", side_effect=AssertionError("Refit must not evaluate")) as predict:
            result = self.refit("fixed", epochs=3)
        predict.assert_not_called()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["completed_epochs"], 3)
        self.assertEqual(result["final_epoch"], 3)
        self.assertEqual(result["final_checkpoint"], result["latest_checkpoint"])
        self.assertIsNone(result["best_checkpoint"])
        self.assertIsNone(result["best_epoch"])
        self.assertIsNone(result["best_metrics"])
        self.assertFalse((self.root / "fixed" / "best.pt").exists())
        self.assertEqual([row["epoch"] for row in result["history"]], [1, 2, 3])
        for row in result["history"]:
            self.assertTrue(np.isfinite(row["train_loss"]))
            self.assertTrue({"average_precision", "roc_auc", "log_loss", "brier"}.isdisjoint(row))
        state = torch.load(result["final_checkpoint"], map_location="cpu", weights_only=True)
        self.assertEqual(state["epoch"], 3)
        self.assertIsNone(state["identity"]["validation"])
        self.assertEqual(state["identity"]["training_purpose"], "fixed_duration_refit")
        core.load_model_checkpoint(result["final_checkpoint"], device="cpu", model_factory=stochastic_tiny_factory)

    def test_refit_requires_explicit_mode_and_rejects_any_validation_frame(self):
        fitting, validation = self.training_parts()
        arguments = (fitting, None, self.model_spec, self.training, RECIPE, self.root / "invalid", {})
        with self.assertRaisesRegex(ValueError, "use refit=True"):
            core.train_recipe(*arguments, epochs=1, device="cpu", model_factory=tiny_factory)
        with self.assertRaisesRegex(ValueError, "validation=None"):
            core.train_recipe(fitting, validation, self.model_spec, self.training, RECIPE,
                              self.root / "invalid", {}, epochs=1, refit=True,
                              device="cpu", model_factory=tiny_factory)

    def test_held_out_label_changes_do_not_change_refit_identity_or_parameters(self):
        frame = self.make_frame(patients_per_fold=3, images_per_patient=2)
        changed = frame.copy()
        changed.loc[changed.fold.eq(0), "target"] = 1 - changed.loc[changed.fold.eq(0), "target"]
        original = self.refit("original_labels", epochs=2,
                              fitting=frame.loc[frame.fold.ne(0)].reset_index(drop=True))
        perturbed = self.refit("perturbed_labels", epochs=2,
                               fitting=changed.loc[changed.fold.ne(0)].reset_index(drop=True))
        first = torch.load(original["final_checkpoint"], map_location="cpu", weights_only=True)
        second = torch.load(perturbed["final_checkpoint"], map_location="cpu", weights_only=True)
        self.assertEqual(first["compatibility_hash"], second["compatibility_hash"])
        for key in first["model_state"]:
            torch.testing.assert_close(first["model_state"][key], second["model_state"][key], rtol=0, atol=0)

    def test_stochastic_refit_resume_matches_uninterrupted_fixed_duration_fit(self):
        uninterrupted = self.refit("uninterrupted_refit", epochs=3)
        self.refit("resumed_refit", epochs=1)
        core.seed_everything(901)
        _ = torch.rand(7)
        resumed = self.refit("resumed_refit", epochs=3)
        first = torch.load(uninterrupted["final_checkpoint"], map_location="cpu", weights_only=True)
        second = torch.load(resumed["final_checkpoint"], map_location="cpu", weights_only=True)
        for key in first["model_state"]:
            torch.testing.assert_close(first["model_state"][key], second["model_state"][key], rtol=0, atol=0)
        columns = ["epoch", "train_loss", "head_lr", "backbone_lr"]
        pd.testing.assert_frame_equal(pd.DataFrame(uninterrupted["history"])[columns],
                                      pd.DataFrame(resumed["history"])[columns])

    def test_partial_refit_has_no_final_checkpoint_and_rejects_shorter_duration(self):
        initial = self.refit("paused_refit", epochs=1)
        paused = self.refit("paused_refit", epochs=3, deadline=time.monotonic() - 1)
        self.assertEqual(paused["status"], "budget_exhausted")
        self.assertEqual(paused["completed_epochs"], 1)
        self.assertIsNone(paused["final_checkpoint"])
        self.assertIsNone(paused["final_epoch"])
        resumed = self.refit("paused_refit", epochs=3)
        self.assertEqual(resumed["final_epoch"], 3)
        previous = digest(resumed["final_checkpoint"])
        with self.assertRaisesRegex(ValueError, "exceeds the requested fixed duration"):
            self.refit("paused_refit", epochs=2)
        self.assertEqual(digest(resumed["final_checkpoint"]), previous)
        self.assertEqual(initial["final_epoch"], 1)

    def test_refit_oom_rolls_back_before_reduced_microbatch_retry(self):
        settings = {**self.training, "microbatch": 2, "min_microbatch": 2}
        reference = self.refit("refit_oom_reference", epochs=2, training=settings, factory=tiny_factory)
        original = core.train_epoch
        calls = 0

        def partial_update_then_fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                original(*args, **kwargs)
                raise torch.cuda.OutOfMemoryError("injected refit partial-epoch failure")
            return original(*args, **kwargs)

        with patch.object(core, "train_epoch", side_effect=partial_update_then_fail):
            retried = self.refit("refit_oom_retry", epochs=2,
                                 training={**self.training, "min_microbatch": 2}, factory=tiny_factory)
        self.assertEqual(retried["status"], "complete")
        self.assertEqual(retried["microbatch"], 2)
        first = torch.load(reference["final_checkpoint"], map_location="cpu", weights_only=True)
        second = torch.load(retried["final_checkpoint"], map_location="cpu", weights_only=True)
        for key in first["model_state"]:
            torch.testing.assert_close(first["model_state"][key], second["model_state"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
