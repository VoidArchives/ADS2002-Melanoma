"""Actual architecture contracts without pretrained downloads or real training."""

from __future__ import annotations

import gc
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melanoma_detection import image_model as core


class DeadlineTinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(3, 1))

    def forward(self, images):
        return self.classifier(self.features(images))


def deadline_tiny_factory(spec, pretrained=True, device="cpu"):
    return DeadlineTinyModel().to(device)


class ImageArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def spec(self, architecture):
        return {**core.DEFAULT_SPEC, "architecture": architecture}

    def test_pinned_adapter_identity_is_available_without_downloads(self):
        expected = {
            "efficientnet_b0": "EfficientNet_B0_Weights.IMAGENET1K_V1",
            "densenet121": "DenseNet121_Weights.IMAGENET1K_V1",
            "convnext_tiny": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        }
        for architecture, enum in expected.items():
            with self.subTest(architecture=architecture):
                identity = core.pretrained_weight_identity(self.spec(architecture), require_cached=False)
                self.assertEqual(identity["weight_enum"], enum)
                self.assertTrue(identity["url"].startswith("https://download.pytorch.org/"))
                self.assertEqual(len(identity["expected_sha256_prefix"]), 8)
                policy = core.architecture_policy(self.spec(architecture))
                self.assertFalse(policy["centre_crop"])
                self.assertEqual(policy["image_size"], 224)
                self.assertEqual(policy["head_lr_parameters"], "new_output_linear_only")

    def test_all_real_heads_return_one_finite_logit_per_image(self):
        for architecture in core.ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = core.build_model(self.spec(architecture), pretrained=False)
                model.eval()
                with torch.inference_mode():
                    output = model(torch.randn(2, 3, 224, 224))
                self.assertEqual(tuple(output.shape), (2, 1))
                self.assertTrue(torch.isfinite(output).all())
                head, backbone = core.training_parameter_groups(model)
                self.assertEqual(len(head), 2)
                self.assertEqual({id(p) for p in head} | {id(p) for p in backbone},
                                 {id(p) for p in model.parameters()})
                self.assertFalse({id(p) for p in head} & {id(p) for p in backbone})
                del model
                gc.collect()

    def test_convnext_preserves_classifier_layernorm_as_backbone(self):
        model = core.build_model(self.spec("convnext_tiny"), pretrained=False)
        layernorm = model.classifier[0]
        self.assertIsInstance(layernorm, nn.LayerNorm)
        head, backbone = core.training_parameter_groups(model)
        self.assertTrue({id(p) for p in layernorm.parameters()} <= {id(p) for p in backbone})
        core.configure_training_stage(model, warmup=True)
        self.assertFalse(layernorm.training)
        self.assertTrue(all(not p.requires_grad for p in layernorm.parameters()))
        self.assertTrue(all(p.requires_grad for p in head))
        self.assertTrue(all(not p.requires_grad for p in backbone))
        self.assertTrue(model.classifier[2].training)
        core.configure_training_stage(model, warmup=False)
        self.assertTrue(all(p.requires_grad for p in layernorm.parameters()))
        self.assertTrue(layernorm.training)

    def test_b0_and_densenet_batchnorm_remain_frozen(self):
        for architecture in ("efficientnet_b0", "densenet121"):
            with self.subTest(architecture=architecture):
                model = core.build_model(self.spec(architecture), pretrained=False)
                batchnorms = [module for module in model.modules()
                              if isinstance(module, nn.modules.batchnorm._BatchNorm)]
                self.assertTrue(batchnorms)
                for warmup in (True, False):
                    core.configure_training_stage(model, warmup)
                    self.assertTrue(all(not module.training for module in batchnorms))
                    self.assertTrue(all(not p.requires_grad for module in batchnorms
                                        for p in module.parameters()))
                del model
                gc.collect()

    def test_controlled_preparation_is_identical_for_every_family(self):
        source = Image.fromarray(np.random.default_rng(42).integers(
            0, 256, (257, 281, 3), dtype=np.uint8))
        for training in (False, True):
            outputs = []
            for architecture in core.ARCHITECTURES:
                core.seed_everything(42)
                outputs.append(core.make_transform(self.spec(architecture), training)(source))
            for output in outputs[1:]:
                torch.testing.assert_close(outputs[0], output, rtol=0, atol=0)

    def test_production_checkpoint_roundtrip_reconstructs_each_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = torch.randn(1, 3, 224, 224)
            for architecture in core.ARCHITECTURES:
                with self.subTest(architecture=architecture):
                    spec = self.spec(architecture)
                    model = core.build_model(spec, pretrained=False)
                    model.eval()
                    with torch.inference_mode():
                        expected = model(inputs)
                    identity = {"architecture": architecture, "test": "real_roundtrip"}
                    path = Path(temporary) / f"{architecture}.pt"
                    core._save_checkpoint(path, {
                        "checkpoint_version": core.CHECKPOINT_VERSION,
                        "spec": spec, "manifest": {"test": "no_training"},
                        "identity": identity, "compatibility_hash": core.canonical_hash(identity),
                        "model_state": model.state_dict(),
                    })
                    del model
                    restored, checkpoint = core.load_model_checkpoint(path)
                    with torch.inference_mode():
                        actual = restored(inputs)
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                    self.assertEqual(checkpoint["spec"]["architecture"], architecture)
                    del restored
                    gc.collect()

    def test_historical_b0_head_keys_are_preserved(self):
        model = core.build_model(self.spec("efficientnet_b0"), pretrained=False)
        keys = model.state_dict()
        self.assertIn("classifier.1.weight", keys)
        self.assertIn("classifier.1.bias", keys)
        self.assertEqual(tuple(keys["classifier.1.weight"].shape), (1, 1280))

    def test_unregistered_architecture_fails_instead_of_substitution(self):
        with self.assertRaises(ValueError):
            core.build_model(self.spec("swin_t"), pretrained=False)

    def test_epoch_admission_pauses_without_starting_an_unaffordable_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = []
            for index in range(4):
                path = Path(temporary) / f"image_{index}.png"
                Image.new("RGB", (32, 32), color=(index * 30, 40, 60)).save(path)
                rows.append({"image_id": f"image_{index}", "patient_id": f"patient_{index}",
                             "target": index % 2, "image_path": str(path)})
            frame = pd.DataFrame(rows)
            training = {**core.DEFAULT_TRAINING, "microbatch": 4, "effective_batch": 4,
                        "min_microbatch": 4, "amp": False}
            with patch.object(core, "train_epoch", side_effect=AssertionError("must not start")):
                result = core.train_recipe(
                    frame.iloc[:2], frame.iloc[2:], {**core.DEFAULT_SPEC, "image_size": 32},
                    training, {"id": "deadline", "backbone_lr": 3e-5, "loss_weight": "none"},
                    Path(temporary) / "run", {"test": "deadline"}, epochs=6, device="cpu",
                    deadline=time.monotonic() + 60, epoch_admission_seconds=180,
                    model_factory=deadline_tiny_factory,
                )
            self.assertEqual(result["status"], "budget_exhausted")
            self.assertEqual(result["history"], [])
            checkpoint = core._read_checkpoint(Path(result["latest_checkpoint"]))
            self.assertEqual(checkpoint["epoch"], 0)
            self.assertEqual(checkpoint["events"][-1]["event"], "epoch_admission_budget_pause")
            self.assertIn("optimizer_state", checkpoint)
            self.assertIn("scheduler_state", checkpoint)
            self.assertIn("scaler_state", checkpoint)
            self.assertIn("rng_state", checkpoint)


if __name__ == "__main__":
    unittest.main()
