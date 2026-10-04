"""Contracts for read-only uncertainty inference, using tiny CPU fixtures."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision.ops import StochasticDepth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melanoma_detection import image_model as core
from melanoma_detection import image_uncertainty as inference
from melanoma_detection.image_uncertainty_evaluation import validate_raw


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.depth = StochasticDepth(0.8, "row")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(3, 1))
        with torch.no_grad():
            self.classifier[-1].weight.fill_(0.7)
            self.classifier[-1].bias.fill_(0.1)

    def forward(self, images):
        return self.classifier(self.pool(self.depth(self.bn(images))).flatten(1))


class InferenceContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        images = self.root / "data/subset/train"
        images.mkdir(parents=True)
        rows = []
        for fold in range(5):
            for target in (0, 1):
                name = f"image_{fold}_{target}"
                pixels = np.random.default_rng(fold * 2 + target).integers(0, 256, (23, 27, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(images / f"{name}.jpg")
                rows.append({"image_name": name, "patient_id": f"patient_{fold}_{target}",
                             "fold": fold, "target": target})
        inventory = pd.DataFrame(rows)
        csv_path = self.root / "data/subset/processed/train_folds.csv"
        csv_path.parent.mkdir(parents=True)
        inventory.to_csv(csv_path, index=False)
        inventory = inventory[sorted(inventory.columns)].sort_values("image_name").reset_index(drop=True)
        inventory["image_path"] = inventory.image_name.map(lambda name: images / f"{name}.jpg")
        pool = inventory[inventory.fold != 0].copy().reset_index(drop=True)
        pool["inner_fold"] = pool.fold.map({1: 0, 2: 1, 3: 2, 4: 1})
        self.fitting = pool[pool.inner_fold != 0].reset_index(drop=True)
        self.validation = pool[pool.inner_fold == 0].reset_index(drop=True)
        self.excluded = inventory[inventory.fold == 0].drop(columns="target").reset_index(drop=True)
        self.memberships = {
            "training_pool": pool.rename(columns={"image_name": "image_id"})[
                ["image_id", "patient_id", "fold", "inner_fold", "target"]].to_dict("records"),
            "outer_excluded": self.excluded.rename(columns={"image_name": "image_id"})[
                ["image_id", "patient_id", "fold"]].to_dict("records")}
        self.manifest = {"provenance": {"full_csv_sha256": core.file_sha256(csv_path)}, "compatibility": {
            "training_pool_memberships": self.memberships["training_pool"],
            "training_pool_images": [{"image_id": row.image_name, "source_sha256": core.file_sha256(row.image_path)}
                                     for row in pool.itertuples()]}}
        self.checkpoint = {"identity": {"fitting": core._frame_identity(self.fitting),
                                         "validation": core._frame_identity(self.validation)}}
        self.spec = {**core.DEFAULT_SPEC, "image_size": 32}
        self.training = {**core.DEFAULT_TRAINING, "microbatch": 2, "effective_batch": 4, "min_microbatch": 1,
                         "amp": False}
        self.model = TinyModel().eval()
        self.saved = core.predict_frame(self.model, self.validation, self.spec, self.training, "cpu")
        self.recipe_dir = self.root / "recipe"
        self.recipe_dir.mkdir()
        (self.recipe_dir / "history.csv").write_text("epoch,loss\n1,0.4\n", encoding="utf-8")
        (self.recipe_dir / "best.pt").write_bytes(b"immutable synthetic checkpoint fixture")
        self.context = {"model": self.model, "frame": self.validation, "spec": self.spec,
                        "training": self.training, "device": "cpu", "saved_predictions": self.saved,
                        "recipe_dir": self.recipe_dir, "provenance": {}}
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)
        self.temporary.cleanup()

    def test_exact_membership_and_image_hash_reconstruction(self):
        fitting, validation, excluded, receipt = inference.reconstruct_partitions(
            self.root, self.manifest, self.memberships, self.checkpoint)
        self.assertEqual(core._frame_identity(fitting), core._frame_identity(self.fitting))
        self.assertEqual(core._frame_identity(validation), core._frame_identity(self.validation))
        self.assertNotIn("target", excluded)
        self.assertEqual(receipt["patient_overlap"], 0)
        self.assertEqual(receipt["evaluation_image_hashes_verified"], 2)

    def test_training_or_excluded_patient_leakage_is_rejected(self):
        for changed_partition in (self.fitting, self.excluded):
            altered = changed_partition.copy()
            altered.loc[0, "patient_id"] = self.validation.iloc[0].patient_id
            with self.assertRaisesRegex(ValueError, "patients must be separate"):
                inference.verify_patient_separation(
                    altered if changed_partition is self.fitting else self.fitting,
                    self.validation, altered if changed_partition is self.excluded else self.excluded)

    def test_changed_memberships_and_labels_are_rejected(self):
        membership = copy.deepcopy(self.memberships)
        membership["training_pool"][0]["target"] = 1
        with self.assertRaisesRegex(ValueError, "identities/labels"):
            inference.reconstruct_partitions(self.root, self.manifest, membership, self.checkpoint)
        membership = copy.deepcopy(self.memberships)
        membership["training_pool"][0]["inner_fold"] = 1
        with self.assertRaisesRegex(ValueError, "bound into the run manifest"):
            inference.reconstruct_partitions(self.root, self.manifest, membership, self.checkpoint)

    def test_checkpoint_frame_identity_and_changed_image_bytes_are_rejected(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["identity"]["validation"] = "wrong"
        with self.assertRaisesRegex(ValueError, "checkpoint frame identity"):
            inference.reconstruct_partitions(self.root, self.manifest, self.memberships, checkpoint)
        self.validation.iloc[0].image_path.write_bytes(b"changed evaluation image")
        with self.assertRaisesRegex(ValueError, "image bytes differ"):
            inference.reconstruct_partitions(self.root, self.manifest, self.memberships, self.checkpoint)

    def _compatibility_fixture(self):
        identity = {"spec": self.spec, "training": self.training,
                    "recipe": {"id": "test_recipe"}, "manifest": {"run_id": "fixture"},
                    "architecture_policy": core.architecture_policy(self.spec)}
        checkpoint = {**copy.deepcopy(identity), "identity": copy.deepcopy(identity),
                      "epoch": 2, "best_epoch": 2, "best_metrics": {"average_precision": 0.2}}
        result = {"best_epoch": 2, "best_metrics": {"average_precision": 0.2},
                  "history": [{"epoch": 1, "average_precision": 0.1}, {"epoch": 2, "average_precision": 0.2}]}
        return checkpoint, copy.deepcopy(identity), result

    def test_checkpoint_compatibility_rejects_detached_spec_and_wrong_best_epoch(self):
        checkpoint, manifest, result = self._compatibility_fixture()
        self.assertTrue(inference.verify_checkpoint_compatibility(checkpoint, manifest, result))
        checkpoint["spec"]["dropout"] = 0.5
        with self.assertRaisesRegex(ValueError, "spec differs"):
            inference.verify_checkpoint_compatibility(checkpoint, manifest, result)
        checkpoint, manifest, result = self._compatibility_fixture()
        result["history"][0]["average_precision"] = 0.3
        with self.assertRaisesRegex(ValueError, "training-history selection"):
            inference.verify_checkpoint_compatibility(checkpoint, manifest, result)

    def test_earlier_architecture_policy_without_profile_is_supported_exactly(self):
        checkpoint, manifest, result = self._compatibility_fixture()
        checkpoint["identity"]["architecture_policy"].pop("transformation_profile")
        self.assertTrue(inference.verify_checkpoint_compatibility(checkpoint, manifest, result))
        checkpoint["identity"]["architecture_policy"]["centre_crop"] = True
        with self.assertRaisesRegex(ValueError, "preprocessing policy"):
            inference.verify_checkpoint_compatibility(checkpoint, manifest, result)

    def test_selection_is_explicit_and_never_chooses_pilot_winner(self):
        for args in ({"run_id": "unknown"}, {"architecture": "convnext_tiny"},
                     {"run_id": "architecture_pilot_3h_v1", "architecture": "densenet121", "recipe_id": "auto"}):
            with self.assertRaises(ValueError):
                inference.load_selected_checkpoint(self.root, device="cpu", **args)

    def test_only_dropout_active_preserving_modes_rate_and_batchnorm(self):
        self.model.train()
        modes = [module.training for module in self.model.modules()]
        bn = inference.batchnorm_buffers(self.model)
        rate = self.model.classifier[0].p
        tensors = inference.prepare_tensor_cache(self.context)
        with inference.explicit_dropout_only(self.model):
            self.assertTrue(self.model.classifier[0].training)
            self.assertFalse(self.model.bn.training)
            self.assertFalse(self.model.depth.training)
            self.assertFalse(self.model.training)
            core.seed_everything(101)
            first = inference.predict_cached(self.model, tensors, batch_size=2)
            core.seed_everything(101)
            repeat = inference.predict_cached(self.model, tensors, batch_size=2)
            core.seed_everything(102)
            different = inference.predict_cached(self.model, tensors, batch_size=2)
        np.testing.assert_array_equal(first, repeat)
        self.assertFalse(np.array_equal(first, different))
        self.assertEqual(rate, self.model.classifier[0].p)
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        inference.assert_unchanged_batchnorm(self.model, bn)

    def test_mode_restoration_also_occurs_on_inference_error(self):
        self.model.train()
        modes = [module.training for module in self.model.modules()]
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with inference.explicit_dropout_only(self.model):
                raise RuntimeError("fixture")
        self.assertEqual(modes, [module.training for module in self.model.modules()])

    def test_identity_tta_and_full_image_flips_preserve_all_pixels(self):
        images = torch.arange(36).reshape(1, 3, 3, 4).float()
        self.assertIs(images, inference.tta_transform(images, "identity"))
        for pass_id in inference.TTA_PASSES:
            transformed = inference.tta_transform(images, pass_id)
            self.assertEqual(transformed.shape, images.shape)
            torch.testing.assert_close(torch.sort(transformed.flatten()).values, torch.sort(images.flatten()).values)
            torch.testing.assert_close(inference.tta_transform(transformed, pass_id), images)
        with self.assertRaises(ValueError):
            inference.tta_transform(images, "rotate_reflect")

    def test_invalid_raw_logits_or_wrong_batch_shape_fail_before_aggregation(self):
        class BadModel(nn.Module):
            def __init__(self, value, wrong_shape=False):
                super().__init__()
                self.value, self.wrong_shape = value, wrong_shape
            def forward(self, images):
                return torch.full((1 if self.wrong_shape else len(images),), self.value)
        tensors = inference.prepare_tensor_cache(self.context)
        for model in (BadModel(float("nan")), BadModel(float("inf")), BadModel(0.1, True)):
            with self.assertRaisesRegex(ValueError, "Raw prediction logits"):
                inference.predict_cached(model, tensors, batch_size=2)

    def test_short_smoke_and_full_raw_inference_preserve_checkpoint_and_history(self):
        before = {str(path): core.file_sha256(path) for path in self.recipe_dir.iterdir()}
        smoke = inference.run_smoke(self.context, n_images=2, seed=10)
        self.assertTrue(smoke["passed"])
        self.assertTrue(smoke["batchnorm_buffers_unchanged"])
        output = inference.run_inference(self.context, mc_passes=5, seed=10)
        expected = self.validation.assign(image_id=self.validation.image_name)
        validate_raw(output["mc_raw"], expected, range(1, 6))
        validate_raw(output["tta_raw"], expected, inference.TTA_PASSES)
        self.assertEqual(len(output["mc_raw"]), 10)
        identity = output["tta_raw"].query("pass_id == 'identity'").score.to_numpy()
        np.testing.assert_array_equal(output["deterministic"].base_score.to_numpy(), identity)
        self.assertTrue(output["checks"]["model_state_unchanged"])
        self.assertEqual(before, {str(path): core.file_sha256(path) for path in self.recipe_dir.iterdir()})

    def test_snapshot_includes_historical_pth_and_detects_mutation(self):
        model_dir = self.root / "models"
        model_dir.mkdir()
        historical = model_dir / "image_model_prototype.pth"
        historical.write_bytes(b"historic")
        before = inference.snapshot_preserved_artifacts(self.root)
        self.assertIn(str(historical), before)
        inference.assert_preserved_artifacts(before)
        historical.write_bytes(b"changed")
        with self.assertRaisesRegex(AssertionError, "modified protected artifacts"):
            inference.assert_preserved_artifacts(before)

    def test_saved_predictions_reject_duplicate_invalid_or_missing_scores(self):
        for changed in (pd.concat([self.saved, self.saved.iloc[[0]]], ignore_index=True),
                        self.saved.assign(base_probability=np.inf), self.saved.head(1)):
            with self.assertRaises(ValueError):
                inference.compare_saved_predictions(self.validation, changed)


if __name__ == "__main__":
    unittest.main()
