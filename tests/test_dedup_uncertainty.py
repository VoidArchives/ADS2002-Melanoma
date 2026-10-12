"""Clean outer-checkpoint and patient-bootstrap contracts; synthetic CPU only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection import dedup_uncertainty as clean
from melanoma_detection import image_model as core
from melanoma_detection import image_pilot as pilot
from melanoma_detection import image_uncertainty as inference
from melanoma_detection.development_dataset import stable_sha256


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(3, 1))
        with torch.no_grad():
            self.classifier[-1].weight.fill_(.7)
            self.classifier[-1].bias.fill_(.1)

    def forward(self, images):
        return self.classifier(self.bn(images).mean(dim=(-2, -1)))


class CleanUncertaintyContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "results/image_d1_runs/b0_nested_dedup_v1"
        self.source.mkdir(parents=True)
        self.config = clean.load_config(ROOT, "configs/image_uncertainty_dedup.yaml")
        self.identity = {"variant": "dedup_v1", "dataset_manifest_sha256": "frozen_manifest", "dataset_identity": "clean_identity"}
        images = self.root / "data/subset/train"
        images.mkdir(parents=True)
        rows = []
        for fold in range(5):
            for target in (0, 1):
                name = f"image_{fold}_{target}"
                pixels = np.random.default_rng(fold * 2 + target).integers(0, 256, (13, 17, 3), dtype=np.uint8)
                path = images / f"{name}.jpg"
                Image.fromarray(pixels).save(path)
                rows.append({"image_name": name, "patient_id": f"patient_{fold}_{target}", "fold": fold,
                             "target": target, "image_path": path})
        self.frame = pd.DataFrame(rows)
        self.counts = {"images": 10, "patients": 10, "malignant_images": 5, "malignant_patients": 5}
        self.spec = {**core.DEFAULT_SPEC, "architecture": "efficientnet_b0", "weights": "IMAGENET1K_V1",
                     "image_size": 224, "dropout": .2, "augmentation": "weak", "preprocessing": "none"}
        self.training = {**core.DEFAULT_TRAINING, "microbatch": 64, "effective_batch": 64, "amp": False}
        self.recipe = {"id": "lr1e4_none", "backbone_lr": .0001, "loss_weight": "none"}
        self.checkpoints, self.fold_manifests, self.selections, self.results, self.records = {}, {}, {}, {}, {}
        fold_identities, predictions = {}, []
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        for fold in range(5):
            folder = self.source / f"folds/{fold}"
            (folder / "refit").mkdir(parents=True)
            pool = self.frame.loc[self.frame.fold.ne(fold)].reset_index(drop=True)
            held = self.frame.loc[self.frame.fold.eq(fold)].reset_index(drop=True)
            members = pool.assign(inner_fold=pool.fold.mod(3))
            compatibility = {"dataset_identity": self.identity, "training_pool_memberships": pilot._memberships(members), "source_sha256": {}}
            fold_manifest = {"run_id": "b0_nested_dedup_v1", "outer_fold": fold, "compatibility": compatibility,
                             "compatibility_sha256": core.canonical_hash(compatibility), "held_out_memberships": pilot._memberships(held, targets=False)}
            manifest = {"run_id": "b0_nested_dedup_v1", "outer_fold": fold, "purpose": "outer_training_fixed_duration_refit",
                        "fold_compatibility_sha256": fold_manifest["compatibility_sha256"], "refit_epochs": 1}
            checkpoint = {"manifest": manifest, "spec": self.spec, "training": self.training, "recipe": self.recipe,
                          "identity": {"manifest": manifest, "spec": self.spec, "training": self.training, "recipe": self.recipe,
                                       "training_purpose": "fixed_duration_refit", "validation": None,
                                       "fitting": core._frame_identity(pool), "architecture_policy": core.architecture_policy(self.spec)},
                          "epoch": 1, "history": [{"epoch": 1}], "microbatch": 64,
                          "pretrained_identity": {"kind": "torchvision", "weight_enum": "EfficientNet_B0_Weights.IMAGENET1K_V1"}}
            selection = {"refit_epochs": 1, "selection_uses_outer_labels": False, "recipe": self.recipe}
            result = {"status": "complete", "final_epoch": 1, "history": [{"epoch": 1}]}
            path = folder / "refit/latest.pt"
            path.write_bytes(f"synthetic outer {fold}".encode())
            self._json(path.with_suffix(".pt.sha256.json"), {"sha256": core.file_sha256(path)})
            model = TinyModel().eval()
            scores = core.predict_frame(model, held, self.spec, self.training, device="cpu")
            scores.to_csv(folder / "outer_predictions.csv", index=False)
            record = {"fold": fold, "status": "complete", "refit_epochs": 1, "selected_recipe": self.recipe["id"],
                      "checkpoint": str(path), "checkpoint_sha256": core.file_sha256(path),
                      "predictions_sha256": core.file_sha256(folder / "outer_predictions.csv")}
            for name, value in (("fold_manifest.json", fold_manifest), ("selection.json", selection), ("refit/fit_result.json", result)):
                self._json(folder / name, value)
            self.checkpoints[str(path)] = checkpoint
            self.fold_manifests[fold], self.selections[fold], self.results[fold], self.records[fold] = fold_manifest, selection, result, record
            fold_identities[str(fold)] = fold_manifest["compatibility_sha256"]
            predictions.append(scores)
        combined = pd.concat(predictions).sort_values("image_id").reset_index(drop=True)
        combined.to_csv(self.source / "outer_oof_predictions.csv", index=False)
        compat = {"fold_identities": fold_identities}
        self.manifest = {"run_id": "b0_nested_dedup_v1", "dataset_identity": self.identity,
                         "compatibility": compat, "compatibility_sha256": core.canonical_hash(compat), "test_set_loaded": False}
        self.summary = {"status": "complete", "folds": list(self.records.values()), "test_set_loaded": False,
                        "outer_oof_predictions_sha256": core.file_sha256(self.source / "outer_oof_predictions.csv")}
        self._json(self.source / "run_manifest.json", self.manifest)
        self._json(self.source / "summary.json", self.summary)
        ledger_path = self.root / "data/subset/processed/dedup_v1/retention_ledger.csv"
        ledger_path.parent.mkdir(parents=True)
        pd.DataFrame({"image_name": self.frame.image_name,
                      "source_sha256": [core.file_sha256(path) for path in self.frame.image_path]}).to_csv(ledger_path, index=False)

    def _json(self, path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def tearDown(self):
        torch.set_num_threads(self.old_threads)
        self.temporary.cleanup()

    def _prepare(self, smoke=False):
        with patch.object(clean, "EXPECTED_COUNTS", self.counts), patch.object(clean, "load_development_dataset", return_value=(self.frame.drop(columns="image_path"), self.identity)), patch.object(core, "_read_checkpoint", side_effect=lambda path: self.checkpoints[str(path)]):
            return clean.prepare_uncertainty(self.root, self.config, smoke=smoke)

    def _verify(self, checkpoint=None, fold_manifest=None, frame=None):
        return clean.verify_fold_checkpoint(self.frame if frame is None else frame, 0, self.identity,
                    self.fold_manifests[0] if fold_manifest is None else fold_manifest,
                    self.checkpoints[str(self.source / "folds/0/refit/latest.pt")] if checkpoint is None else checkpoint,
                    self.selections[0], self.results[0], self.records[0])

    def test_preflight_complete_and_exact_retained_coverage(self):
        prepared = self._prepare()
        self.assertEqual(list(prepared["folds"]), list(range(5)))
        self.assertEqual(prepared["counts"], self.counts)
        predictions = pd.read_csv(self.source / "outer_oof_predictions.csv")
        with self.assertRaisesRegex(ValueError, "exactly the retained"):
            clean.verify_coverage(self.frame, predictions.iloc[:-1])
        predictions.loc[0, "fold"] = 4
        with self.assertRaisesRegex(ValueError, "outer fold"):
            clean.verify_coverage(self.frame, predictions)

    def test_full_preflight_waits_for_all_folds_and_writes_nothing(self):
        summary = {**self.summary, "status": "incomplete", "folds": [self.records[0]]}
        self._json(self.source / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "all five completed"):
            self._prepare()
        self.assertFalse((self.root / "results/dedup_v1/uncertainty").exists())
        self.assertEqual(list(self._prepare(smoke=True)["folds"]), [0])

    def test_wrong_fold_and_historical_dataset_are_rejected(self):
        altered = copy.deepcopy(self.fold_manifests[0])
        altered["outer_fold"] = 1
        with self.assertRaisesRegex(ValueError, "different fold"):
            self._verify(fold_manifest=altered)
        altered = copy.deepcopy(self.fold_manifests[0])
        altered["compatibility"]["dataset_identity"] = {"variant": "original"}
        with self.assertRaisesRegex(ValueError, "dataset identity"):
            self._verify(fold_manifest=altered)

    def test_inner_checkpoint_and_early_epoch_are_rejected(self):
        original = self.checkpoints[str(self.source / "folds/0/refit/latest.pt")]
        for key, value in (("outer_fold", 1), ("purpose", "inner_screen_and_confirmation")):
            altered = copy.deepcopy(original)
            altered["manifest"][key] = value
            with self.assertRaisesRegex(ValueError, "outer-training refit"):
                self._verify(checkpoint=altered)
        altered = copy.deepcopy(original)
        altered["epoch"] = 0
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self._verify(checkpoint=altered)

    def test_patient_overlap_and_fitting_identity_are_rejected(self):
        altered = self.frame.copy()
        altered.loc[2, "patient_id"] = altered.loc[0, "patient_id"]
        with self.assertRaisesRegex(ValueError, "patients overlap"):
            self._verify(frame=altered)
        altered = self.frame.copy()
        altered.loc[2, "target"] = 1
        with self.assertRaisesRegex(ValueError, "fitting identities"):
            self._verify(frame=altered)

    def test_changed_checkpoint_or_image_bytes_are_rejected(self):
        path = self.source / "folds/0/refit/latest.pt"
        path.write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            self._prepare()
        path.write_bytes(b"synthetic outer 0")
        self.frame.iloc[0].image_path.write_bytes(b"changed source JPEG")
        with self.assertRaisesRegex(ValueError, "image bytes changed"):
            self._prepare()

    def test_training_source_integrity_is_verified(self):
        source_path = self.root / "src/fixture_source.py"
        source_path.parent.mkdir()
        source_path.write_text("old source", encoding="utf-8")
        self.fold_manifests[0]["compatibility"]["source_sha256"] = {"src/fixture_source.py": core.file_sha256(source_path)}
        self.fold_manifests[0]["compatibility_sha256"] = core.canonical_hash(self.fold_manifests[0]["compatibility"])
        digest = self.fold_manifests[0]["compatibility_sha256"]
        checkpoint = self.checkpoints[str(self.source / "folds/0/refit/latest.pt")]
        checkpoint["manifest"]["fold_compatibility_sha256"] = digest
        self.manifest["compatibility"]["fold_identities"]["0"] = digest
        self.manifest["compatibility_sha256"] = core.canonical_hash(self.manifest["compatibility"])
        self._json(self.source / "run_manifest.json", self.manifest)
        self._json(self.source / "folds/0/fold_manifest.json", self.fold_manifests[0])
        source_path.write_text("changed source", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self._prepare()

    def test_outer_labels_are_placeholder_only_and_model_state_stays_fixed(self):
        prepared = self._prepare(smoke=True)
        checkpoint = self.checkpoints[str(self.source / "folds/0/refit/latest.pt")]
        model = TinyModel().eval()
        with patch.object(core, "load_model_checkpoint", return_value=(model, checkpoint)):
            context = clean.load_fold_context(prepared, 0, device="cpu")
        self.assertTrue(context["frame"].target.eq(0).all())
        self.assertTrue(context["saved_predictions"].target.eq(0).all())
        modes = [module.training for module in model.modules()]
        state = core.model_state_sha256(model)
        result = inference.run_inference(context, mc_passes=3, seed=20261004, batch_size=64)
        self.assertEqual(state, core.model_state_sha256(model))
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertTrue(result["checks"]["batchnorm_buffers_unchanged"])
        self.assertTrue(result["checks"]["only_explicit_dropout_active"])
        clean._join_targets(result, prepared["folds"][0]["frame"])
        self.assertEqual(result["deterministic"].target.tolist(), [0, 1])
        self.assertEqual(result["mc_raw"].target.sum(), 3)
        context["saved_predictions"] = context["saved_predictions"].assign(base_probability=.01)
        with self.assertRaisesRegex(ValueError, "disagree"):
            inference.run_inference(context, mc_passes=3, seed=20261004, batch_size=64)

    def _referral(self):
        frame = self.frame.rename(columns={"image_name": "image_id"}).drop(columns="image_path")
        frame["base_score"] = [.1, .9] * 5
        frame["mc_mean_score"] = frame.base_score
        frame["tta_mean_score"] = frame.base_score
        frame["mc_score_sd"] = np.linspace(.1, .2, len(frame))
        frame["tta_score_sd"] = frame.mc_score_sd
        return frame

    def test_bootstrap_preserves_fold_patient_counts_and_shared_draws(self):
        frame = self._referral()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        frame.loc[len(frame) - 1, "image_id"] = "second_same_patient_image"
        indices = clean.stratified_patient_indices(frame, np.random.default_rng(42))
        sample = frame.iloc[indices]
        for fold in range(5):
            original_counts = frame.loc[frame.fold.eq(fold)].patient_id.value_counts()
            selected_counts = sample.loc[sample.fold.eq(fold)].patient_id.value_counts().reindex(original_counts.index, fill_value=0)
            multiplicities = selected_counts / original_counts
            self.assertTrue(multiplicities.mod(1).eq(0).all())
            self.assertEqual(multiplicities.sum(), len(original_counts))
        # Identical MC/TTA rankings use the exact same patients and random baseline.
        result, identities = clean.referral_bootstrap(self._referral(), n_bootstrap=12, n_random=5)
        mc = result[result.method.eq("mc")].drop(columns="method").reset_index(drop=True)
        tta = result[result.method.eq("tta")].drop(columns="method").reset_index(drop=True)
        pd.testing.assert_frame_equal(mc, tta)
        repeated, repeated_identities = clean.referral_bootstrap(self._referral(), n_bootstrap=12, n_random=5)
        pd.testing.assert_frame_equal(result, repeated)
        pd.testing.assert_frame_equal(identities, repeated_identities)

    def test_undefined_malignant_bootstrap_metrics_are_missing(self):
        frame = self._referral().assign(target=0)
        result, draws = clean.referral_bootstrap(frame, n_bootstrap=4, n_random=2)
        missing = result[result.metric.eq("malignant_referred_fraction")]
        self.assertTrue(missing.estimate.isna().all())
        self.assertTrue(missing.ci_lower.isna().all())
        self.assertTrue(missing.n_undefined.eq(4).all())
        self.assertTrue(draws.single_class.all())

    def test_end_to_end_artifacts_and_resume_reuse_all_folds(self):
        real_bootstrap = clean.referral_bootstrap
        def fake_context(prepared, fold, device):
            return {"fold": fold, "frame": prepared["folds"][fold]["frame"].assign(target=0),
                    "saved": prepared["folds"][fold]["saved"]}
        def fake_inference(context, **kwargs):
            self.assertTrue(context["frame"].target.eq(0).all())
            expected = context["frame"].rename(columns={"image_name": "image_id"}).drop(columns="image_path")
            scores = context["saved"].base_probability.to_numpy()
            base = expected.assign(base_score=scores)
            raw = expected.assign(score=scores)
            mc = pd.concat([raw.assign(pass_id=i, seed=20261003 + i, score=scores + (i % 2) * .001) for i in range(1, 31)], ignore_index=True)
            tta = pd.concat([raw.assign(pass_id=view, seed=20261004) for view in inference.TTA_PASSES], ignore_index=True)
            return {"deterministic": base, "mc_raw": mc, "tta_raw": tta,
                    "checks": {"batchnorm_buffers_unchanged": True, "only_explicit_dropout_active": True},
                    "provenance": {"real_labels_supplied_to_inference": False}}
        with patch.object(clean, "EXPECTED_COUNTS", self.counts), patch.object(clean, "load_development_dataset", return_value=(self.frame.drop(columns="image_path"), self.identity)), patch.object(core, "_read_checkpoint", side_effect=lambda path: self.checkpoints[str(path)]), patch.object(clean, "load_fold_context", side_effect=fake_context), patch.object(inference, "run_inference", side_effect=fake_inference) as calls, patch.object(clean, "referral_bootstrap", side_effect=lambda frame: real_bootstrap(frame, n_bootstrap=8, n_random=3)):
            result = clean.run_uncertainty(self.root, self.config, device="cpu")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(calls.call_count, 5)
            calls.reset_mock()
            repeated = clean.run_uncertainty(self.root, self.config, device="cpu", resume=True)
            self.assertEqual(repeated["status"], "complete")
            calls.assert_not_called()
        folder = self.root / "results/dedup_v1/uncertainty/b0_uncertainty_dedup_v1"
        receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["training_fits"], 0)
        self.assertEqual(receipt["completed_outer_folds"], list(range(5)))
        self.assertFalse(receipt["test_set_loaded"])
        self.assertEqual(receipt["dataset_manifest_sha256"], "frozen_manifest")
        self.assertEqual(len(pd.read_csv(folder / "mc_raw_predictions.csv")), 300)
        self.assertEqual(len(pd.read_csv(folder / "tta_raw_predictions.csv")), 40)
        for relative, digest in receipt["artifacts"].items():
            self.assertEqual(core.file_sha256(folder / relative), digest)


if __name__ == "__main__":
    unittest.main()
