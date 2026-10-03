"""Transformation identity, isolation and matched-initialisation contracts."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection import image_model as core
from melanoma_detection import image_transforms as profiles


def example_image(size=224):
    pixels = np.random.default_rng(727).integers(15, 240, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


class FakeCLAHE:
    def apply(self, lightness):
        return np.minimum(lightness.astype(np.int16) + 7, 255).astype(np.uint8)


class FakeOpenCV:
    __version__ = "test_fixed"
    COLOR_RGB2LAB, COLOR_LAB2RGB = 1, 2

    def createCLAHE(self, *, clipLimit, tileGridSize):
        if clipLimit != 2.0 or tileGridSize != (8, 8):
            raise AssertionError("CLAHE parameters changed")
        return FakeCLAHE()

    def cvtColor(self, pixels, operation):
        return pixels.copy()


class TransformProfileTests(unittest.TestCase):
    def setUp(self):
        self.image = example_image()
        self.weak = dict(core.DEFAULT_SPEC)
        self.strong = {**self.weak, "augmentation": "stronger"}

    def test_weak_is_bitwise_identical_to_executed_legacy_chain(self):
        # Independently construct the preserved pre-extension public chain.
        legacy = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20), transforms.ToTensor(), transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        for seed in (7, 42, 2026):
            with self.subTest(seed=seed):
                core.seed_everything(seed)
                expected = legacy(self.image)
                core.seed_everything(seed)
                actual = core.make_transform(self.weak, training=True)(self.image)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weak_operations = core.stochastic_augmentation(self.weak).transforms
        self.assertEqual(weak_operations[-1].interpolation, InterpolationMode.NEAREST)
        self.assertEqual(weak_operations[-1].fill, 0)

    def test_weak_golden_pre_normalisation_fixture(self):
        core.seed_everything(42)
        actual = np.asarray(core.make_preview_transform(self.weak)(self.image))
        self.assertEqual(hashlib.sha256(actual.tobytes()).hexdigest(),
                         "4adc70246df985ebd1e2bf46a14fdda8bee381a5591ca6657ac776288a46805f")

    def test_stronger_is_reproducible_and_has_declared_colour_and_geometry(self):
        core.seed_everything(42)
        first = core.make_transform(self.strong, training=True)(self.image)
        core.seed_everything(42)
        second = core.make_transform(self.strong, training=True)(self.image)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(tuple(first.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(first).all())
        candidate = core.transformation_profile(self.strong)["augmentation"]
        self.assertEqual(candidate["version"], "stronger_v1")
        self.assertEqual(candidate["parameters"]["hue"], [0.0, 0.0])
        affine = profiles.ReflectAffine()
        for _ in range(100):
            angle, translation, scale = affine.parameters(self.image)
            self.assertLessEqual(abs(angle), 45)
            self.assertTrue(all(abs(value) <= 11 for value in translation))
            self.assertTrue(0.9 <= scale <= 1.05)

    def test_extreme_affine_does_not_introduce_black_fill(self):
        image = Image.fromarray(np.full((224, 224, 3), [97, 137, 201], dtype=np.uint8))
        affine = profiles.ReflectAffine()
        for angle in (-45, 0, 45):
            for scale in (0.9, 1.05):
                for translation in ((-11, -11), (11, 11), (-11, 11), (11, -11)):
                    output = np.asarray(affine.apply(image, angle=angle,
                                                    scale=scale, translate=translation))
                    self.assertEqual(output.shape, (224, 224, 3))
                    np.testing.assert_array_equal(output, np.asarray(image))

    def test_ordinary_validation_is_identical_for_weak_and_stronger(self):
        first = core.make_transform(self.weak)(self.image)
        core.seed_everything(222)
        second = core.make_transform(self.strong)(self.image)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        repeated = core.make_transform(self.strong)(self.image)
        torch.testing.assert_close(second, repeated, rtol=0, atol=0)

    def test_unknown_profile_version_or_changed_parameters_fail(self):
        cases = ({"augmentation": "unsafe_crop"}, {"preprocessing": "hair_removal"},
                 {"augmentation_version": "weak_v2"},
                 {"preprocessing": "clahe", "preprocessing_version": "clahe_v9"},
                 {"augmentation_parameters": {"rotation_degrees": [-90, 90]}})
        for update in cases:
            with self.subTest(update=update), self.assertRaises(ValueError):
                core.make_transform({**self.weak, **update})

    def test_profile_metadata_cannot_modify_registered_profiles(self):
        identity = core.transformation_profile(self.strong)
        identity["augmentation"]["parameters"]["scale"][0] = 0.1
        self.assertEqual(core.transformation_profile(self.strong)["augmentation"]["parameters"]["scale"],
                         [0.9, 1.05])

    def test_clahe_is_lightness_only_and_never_modifies_the_source(self):
        spec = {**self.weak, "preprocessing": "clahe"}
        before = np.array(self.image)
        with patch.object(profiles, "require_opencv", return_value=FakeOpenCV()):
            correction = core.deterministic_prepare(spec)
            result = np.array(correction(self.image))
            np.testing.assert_array_equal(result[..., 1:], before[..., 1:])
            np.testing.assert_array_equal(result[..., 0], np.minimum(before[..., 0].astype(np.int16) + 7, 255))
            np.testing.assert_array_equal(np.array(self.image), before)
            # The same deterministic preparation is executed by train and eval.
            train_transform = core.make_preview_transform(spec, training=True)
            eval_transform = core.make_preview_transform(spec, training=False)
            prepared_for_training = np.array(train_transform.transforms[0](self.image))
            prepared_for_evaluation = np.array(eval_transform(self.image))
            np.testing.assert_array_equal(prepared_for_training, prepared_for_evaluation)
            first = core.make_transform(spec)(self.image)
            second = core.make_transform(spec)(self.image)
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_missing_clahe_dependency_fails_without_substitution(self):
        with patch.object(profiles.importlib, "import_module", side_effect=ImportError("absent")):
            core.make_transform(self.weak)(self.image)
            with self.assertRaisesRegex(ImportError, "never substituted"):
                core.make_transform({**self.weak, "preprocessing": "clahe"})

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "optional OpenCV is absent")
    def test_installed_clahe_matches_fixed_opencv_operation(self):
        import cv2
        resized = core.deterministic_prepare(self.weak)(self.image)
        lab = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2LAB)
        expected_lab = lab.copy()
        expected_lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        expected = cv2.cvtColor(expected_lab, cv2.COLOR_LAB2RGB)
        actual = np.array(core.deterministic_prepare({**self.weak, "preprocessing": "clahe"})(self.image))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(expected_lab[..., 1:], lab[..., 1:])

    def test_loader_shuffle_does_not_depend_on_augmentation_rng_draws(self):
        frame = pd.DataFrame({"image_name": [str(i) for i in range(20)],
                              "patient_id": [str(i) for i in range(20)],
                              "target": [i % 2 for i in range(20)],
                              "image_path": ["not_read.jpg"] * 20})
        first = core._loader(frame, self.weak, core.DEFAULT_TRAINING, training_mode=True, seed=43)
        for _ in range(10):
            core.make_preview_transform(self.strong)(self.image)
        second = core._loader(frame, self.strong, core.DEFAULT_TRAINING, training_mode=True, seed=43)
        self.assertEqual(list(first.sampler), list(second.sampler))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(3, 1))

    def forward(self, images):
        return self.classifier(self.features(images)).flatten()


def tiny_factory(spec, pretrained=True, device="cpu"):
    return TinyModel().to(device)


class InitialisationContractTests(unittest.TestCase):
    def test_epoch_guard_applies_margin_to_observed_duration_plus_checkpoint_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.jpg"
            example_image(32).save(image)
            fitting = pd.DataFrame({"image_name": ["fit0", "fit1"], "patient_id": ["p0", "p1"],
                                    "target": [0, 1], "image_path": [str(image)] * 2})
            validation = fitting.assign(image_name=["val0", "val1"], patient_id=["p2", "p3"])
            training = {**core.DEFAULT_TRAINING, "microbatch": 4, "effective_batch": 4, "amp": False}
            recipe = {"id": "test", "backbone_lr": 3e-5, "loss_weight": "none"}
            spec = {**core.DEFAULT_SPEC, "image_size": 32}
            original_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                core.train_recipe(fitting, validation, spec, training, recipe,
                                  root / "guard", {}, epochs=1, device="cpu", model_factory=tiny_factory)
                latest = root / "guard" / "latest.pt"
                checkpoint = core._read_checkpoint(latest)
                checkpoint["history"][0]["epoch_seconds"] = 80.0
                core._save_checkpoint(latest, checkpoint)
                # 1.25 * 80 + 30 = 130 seconds, exceeding the remaining 125.
                result = core.train_recipe(fitting, validation, spec, training, recipe,
                                           root / "guard", {}, epochs=2, device="cpu", model_factory=tiny_factory,
                                           deadline=time.monotonic() + 125,
                                           epoch_admission_seconds=1,
                                           epoch_admission_safety_factor=1.25)
                self.assertEqual(result["status"], "budget_exhausted")
                self.assertEqual(len(result["history"]), 1)
                state = core._read_checkpoint(latest)
                event = state["events"][-1]
                self.assertEqual(event["required_seconds"], 130.0)
                self.assertEqual(event["epoch"], 2)
                for bad in (0.9, float("nan"), float("inf")):
                    with self.subTest(factor=bad), self.assertRaises(ValueError):
                        core.train_recipe(fitting, validation, spec, training, recipe,
                                          root / "invalid", {}, epochs=1, device="cpu", model_factory=tiny_factory,
                                          epoch_admission_safety_factor=bad)
            finally:
                torch.set_num_threads(original_threads)

    def test_initial_hash_supports_scalar_batchnorm_and_bfloat_buffers(self):
        model = TinyModel()
        model.features.add_module("bn", nn.BatchNorm1d(3))
        model.register_buffer("low_precision", torch.ones(3, dtype=torch.bfloat16))
        first = core.model_state_sha256(model)
        self.assertEqual(first, core.model_state_sha256(model))
        model.low_precision[0] = 2
        self.assertNotEqual(first, core.model_state_sha256(model))

    def test_initial_hash_is_matched_before_training_and_persists_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.jpg"
            example_image(32).save(image)
            fitting = pd.DataFrame({"image_name": ["fit0", "fit1"], "patient_id": ["p0", "p1"],
                                    "target": [0, 1], "image_path": [str(image)] * 2})
            validation = pd.DataFrame({"image_name": ["val0", "val1"], "patient_id": ["p2", "p3"],
                                       "target": [0, 1], "image_path": [str(image)] * 2})
            training = {**core.DEFAULT_TRAINING, "microbatch": 4, "effective_batch": 4,
                        "amp": False}
            recipe = {"id": "test", "backbone_lr": 3e-5, "loss_weight": "none"}
            common = dict(epochs=1, device="cpu", deadline=time.monotonic() - 1,
                          model_factory=tiny_factory)
            first = core.train_recipe(fitting, validation, core.DEFAULT_SPEC, training, recipe,
                                      root / "A", {"condition": "A"}, **common)
            expected = first["initial_model_state_sha256"]
            second = core.train_recipe(fitting, validation, {**core.DEFAULT_SPEC, "augmentation": "stronger"},
                                       training, recipe, root / "B", {"condition": "B"},
                                       expected_initial_model_state_sha256=expected, **common)
            self.assertEqual(expected, second["initial_model_state_sha256"])
            resumed = core.train_recipe(fitting, validation, core.DEFAULT_SPEC, training, recipe,
                                        root / "A", {"condition": "A"},
                                        expected_initial_model_state_sha256=expected, **common)
            self.assertEqual(resumed["initial_model_state_sha256"], expected)
            state = torch.load(root / "A" / "latest.pt", weights_only=True)
            self.assertEqual(state["initial_model_state_sha256"], expected)
            self.assertIn("transformation_source_sha256", state["identity"])
            self.assertEqual(state["identity"]["preprocessing_environment"], {"opencv": "not_used"})
            with patch.dict(profiles.AUGMENTATION_PROFILES["weak"], {"version": "weak_v2"}):
                with self.assertRaisesRegex(ValueError, "transformation profile differs"):
                    core.load_model_checkpoint(root / "A" / "latest.pt", model_factory=tiny_factory)
            with self.assertRaisesRegex(ValueError, "Initial model parameters differ"):
                core.train_recipe(fitting, validation, core.DEFAULT_SPEC, training, recipe,
                                  root / "mismatch", {},
                                  expected_initial_model_state_sha256="0" * 64, **common)
            self.assertFalse((root / "mismatch" / "latest.pt").exists())

    def test_clahe_checkpoint_rejects_missing_or_changed_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.jpg"
            example_image(32).save(image)
            fitting = pd.DataFrame({"image_name": ["fit0", "fit1"], "patient_id": ["p0", "p1"],
                                    "target": [0, 1], "image_path": [str(image)] * 2})
            validation = fitting.assign(image_name=["val0", "val1"], patient_id=["p2", "p3"])
            training = {**core.DEFAULT_TRAINING, "microbatch": 4, "effective_batch": 4, "amp": False}
            with patch.object(profiles, "require_opencv", return_value=FakeOpenCV()):
                core.train_recipe(fitting, validation, {**core.DEFAULT_SPEC, "preprocessing": "clahe"},
                                  training, {"id": "test", "backbone_lr": 3e-5, "loss_weight": "none"},
                                  root / "C", {}, epochs=1, device="cpu", deadline=time.monotonic() - 1,
                                  model_factory=tiny_factory)
                core.load_model_checkpoint(root / "C" / "latest.pt", model_factory=tiny_factory)
            changed = FakeOpenCV()
            changed.__version__ = "changed_version"
            with patch.object(profiles, "require_opencv", return_value=changed):
                with self.assertRaisesRegex(ValueError, "CLAHE dependency version differs"):
                    core.load_model_checkpoint(root / "C" / "latest.pt", model_factory=tiny_factory)
            with patch.object(profiles, "require_opencv", side_effect=ImportError("optional dependency absent")):
                with self.assertRaises(ImportError):
                    core.load_model_checkpoint(root / "C" / "latest.pt", model_factory=tiny_factory)


class SupervisorCheckpointRecoveryTests(unittest.TestCase):
    @staticmethod
    def state(epoch, best_epoch):
        identity = {"pipeline": "test_verified_backup"}
        return {"checkpoint_version": core.CHECKPOINT_VERSION,
                "identity": identity, "compatibility_hash": core.canonical_hash(identity),
                "manifest": {"run": "synthetic_recovery"}, "epoch": epoch,
                "best_epoch": best_epoch, "history": [{"epoch": index} for index in range(1, epoch + 1)],
                "model_state": {"weight": torch.tensor([float(epoch)])}}

    def test_explicit_recovery_restores_previous_pair_and_matching_best_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest, best = root / "latest.pt", root / "best.pt"
            core._save_checkpoint(best, self.state(1, 1))
            core._save_checkpoint(latest, self.state(1, 1))
            core._save_checkpoint(best, self.state(2, 2))
            core._save_checkpoint(latest, self.state(2, 2))
            self.assertEqual(core._read_checkpoint(root / "latest.previous.pt")["epoch"], 1)
            self.assertEqual(core._read_checkpoint(root / "best.previous.pt")["epoch"], 1)
            latest.write_bytes(b"interrupted_tensor_replace")
            with self.assertRaises(ValueError):
                core._read_checkpoint(latest)
            with self.assertRaises(ValueError):
                core.recover_interrupted_checkpoint(latest, expected_manifest={"run": "wrong"})
            result = core.recover_interrupted_checkpoint(latest, expected_manifest={"run": "synthetic_recovery"})
            self.assertEqual(result["epoch"], 1)
            self.assertEqual(core._read_checkpoint(latest)["epoch"], 1)
            self.assertEqual(core._read_checkpoint(best)["epoch"], 1)
            self.assertTrue(list(root.glob("latest.interrupted.*.pt")))
            self.assertTrue(list(root.glob("best.interrupted.*.pt")))
            self.assertEqual(len([event for event in result["recovery_events"]
                                  if event["event"] == "checkpoint_recovered"]), 2)

    def test_best_repaired_from_latest_only_when_latest_is_selected_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest, best = root / "latest.pt", root / "best.pt"
            core._save_checkpoint(latest, self.state(2, 2))
            core._save_checkpoint(best, self.state(1, 1))
            result = core.recover_interrupted_checkpoint(latest)
            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(core._read_checkpoint(best)["epoch"], 2)
            core._save_checkpoint(latest, self.state(3, 2))
            best.write_bytes(b"bad_best")
            # Its backup has epoch 1; epoch 2 cannot be reconstructed from 3.
            with self.assertRaisesRegex(ValueError, "selected best epoch"):
                core.recover_interrupted_checkpoint(latest)

    def test_epoch_zero_retains_and_clears_uncommitted_best_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest, best = root / "latest.pt", root / "best.pt"
            core._save_checkpoint(latest, self.state(0, None))
            core._save_checkpoint(best, self.state(1, 1))
            result = core.recover_interrupted_checkpoint(latest)
            self.assertIsNone(result["best_checkpoint"])
            self.assertFalse(best.exists())
            self.assertTrue(list(root.glob("best.interrupted.*.pt")))

    def test_existing_invalid_pair_cannot_be_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory) / "latest.pt"
            core._save_checkpoint(latest, self.state(1, 1))
            latest.with_suffix(".pt.sha256.json").write_text('{"sha256":"wrong"}')
            with self.assertRaisesRegex(ValueError, "explicit supervisor recovery"):
                core._save_checkpoint(latest, self.state(2, 2))


if __name__ == "__main__":
    unittest.main()
