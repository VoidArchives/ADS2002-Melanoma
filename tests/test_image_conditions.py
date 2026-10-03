"""Meaningful controls for the bounded fresh B0 condition stage."""
from __future__ import annotations
import copy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from melanoma_detection import image_conditions as conditions
from melanoma_detection import image_pilot

ROOT = Path(__file__).resolve().parents[1]


def synthetic_project(root):
    image_dir = root / "data/subset/train"
    fold_dir = root / "data/subset/processed"
    reference_dir = root / "results/image_d1_runs/b0_pilot_v1"
    for path in (image_dir, fold_dir, reference_dir):
        path.mkdir(parents=True)
    rows = []
    for fold in range(5):
        for label in (0, 1):
            name = f"image_{fold}_{label}"
            rows.append({"image_name": name, "patient_id": f"patient_{fold}_{label}", "target": label, "fold": fold})
            image = np.full((24, 24, 3), 20 + fold * 35 + label * 10, dtype=np.uint8)
            image[0, 0] = (fold, label, 150)
            Image.fromarray(image).save(image_dir / f"{name}.jpg")
    # A legal same-patient duplicate in excluded patients. Its outcome may be
    # perturbed independently without affecting any development decision.
    rows.append({**rows[0], "image_name": "image_0_duplicate"})
    (image_dir / "image_0_duplicate.jpg").write_bytes((image_dir / "image_0_0.jpg").read_bytes())
    frame = pd.DataFrame(rows)
    frame.to_csv(fold_dir / "train_folds.csv", index=False)
    pool = frame[frame.fold != 0].copy()
    pool["inner_fold"] = pool.fold.map({1: 0, 2: 1, 3: 2, 4: 1})
    image_pilot._atomic_json(reference_dir / "split_memberships.json", {
        "training_pool": image_pilot._memberships(pool),
        "outer_excluded": image_pilot._memberships(frame[frame.fold == 0], targets=False)})
    return frame


def reviewed_receipt(condition="B", *, unsafe=False):
    examples = [{"image_id": f"qa_{i}", "patient_id": f"p_{i}", "target": i % 2,
                 "source_sha256": str(i)} for i in range(24)]
    rows = [{"image_id": row["image_id"], "variant": variant, "seed": 4200 + 4 * index + variant,
             "clipping": 0, "boundary": 0, "texture": 0, "colour": 0, "reflection": 0}
            for index, row in enumerate(examples) for variant in range(1 if condition == "C" else 4)]
    if unsafe:
        rows[0]["texture"] = 2
    receipt = {"version": 1, "selected_examples": examples,
               "reviewed_by": "explicit test reviewer", "reviewed_at_utc": "2026-10-03T09:40:00Z",
               "conditions": {condition: {"verdict": "passed", "reviews": rows,
                    "reviewed_image_ids": [row["image_id"] for row in examples],
                    "preview_sha256": {"synthetic_preview.png": "synthetic"}}}}
    receipt["review_sha256"] = conditions._hash(receipt)
    return receipt


class ConditionPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.frame = synthetic_project(self.root)
        self.config = conditions.load_config(ROOT / "configs/image_augmentation.yaml")

    def tearDown(self):
        self.temporary.cleanup()

    def test_saved_memberships_and_patient_isolation(self):
        prepared = conditions.prepare_conditions(self.root, self.config)
        self.assertEqual(set(prepared["validation"].image_name), {"image_1_0", "image_1_1"})
        self.assertFalse(set(prepared["fitting"].patient_id) & set(prepared["validation"].patient_id))
        self.assertNotIn("target", prepared["outer_excluded"])
        self.assertEqual(prepared["image_audit"]["status"], "complete")

    def test_outer_target_conflict_and_invalid_values_are_ignored_with_decoded_audit(self):
        original = conditions.prepare_conditions(self.root, self.config)
        changed = self.frame.copy()
        changed.loc[changed.image_name == "image_0_duplicate", "target"] = 1
        changed["target"] = changed.target.astype(object)
        changed.loc[changed.image_name == "image_0_1", "target"] = "unused-invalid-outcome"
        changed.to_csv(self.root / "data/subset/processed/train_folds.csv", index=False)
        altered = conditions.prepare_conditions(self.root, self.config)
        self.assertEqual(original["manifest"]["compatibility_sha256"], altered["manifest"]["compatibility_sha256"])
        self.assertNotEqual(original["manifest"]["provenance"]["full_csv_sha256"], altered["manifest"]["provenance"]["full_csv_sha256"])
        pd.testing.assert_frame_equal(original["fitting"], altered["fitting"])
        pd.testing.assert_frame_equal(original["validation"], altered["validation"])

    def test_changed_training_outcome_rejected(self):
        changed = self.frame.copy()
        changed.loc[changed.image_name == "image_2_0", "target"] = 1
        changed.to_csv(self.root / "data/subset/processed/train_folds.csv", index=False)
        with self.assertRaises(ValueError):
            conditions.prepare_conditions(self.root, self.config)

    def test_cross_patient_content_duplicate_rejected(self):
        image_dir = self.root / "data/subset/train"
        (image_dir / "image_2_0.jpg").write_bytes((image_dir / "image_1_0.jpg").read_bytes())
        with self.assertRaisesRegex(ValueError, "crosses patients"):
            conditions.prepare_conditions(self.root, self.config)

    def test_training_control_or_horizon_change_rejected(self):
        for mapping, key, value in (("training", "seed", 17), ("recipe", "backbone_lr", 1e-4),
                                    ("experiment", "epochs", 6), ("model", "architecture", "densenet121")):
            config = copy.deepcopy(self.config)
            config[mapping][key] = value
            with self.assertRaises(ValueError):
                conditions.validate_config(config)


class QualityGateTests(unittest.TestCase):
    def test_fully_reviewed_safe_variants_pass(self):
        receipt = reviewed_receipt()
        self.assertEqual(conditions.validate_quality_review(receipt, "B", verify_files=False), receipt["review_sha256"])

    def test_unsigned_or_modified_review_rejected(self):
        receipt = reviewed_receipt()
        receipt["conditions"]["B"]["reviews"][0]["texture"] = 1
        with self.assertRaisesRegex(ValueError, "unsigned|changed"):
            conditions.validate_quality_review(receipt, "B", verify_files=False)

    def test_unsafe_or_unreviewed_variants_block_training(self):
        for value in (None, 2, 3):
            receipt = reviewed_receipt()
            receipt["conditions"]["B"]["reviews"][0]["clipping"] = value
            receipt["review_sha256"] = conditions._hash({k: v for k, v in receipt.items() if k != "review_sha256"})
            with self.assertRaisesRegex(ValueError, "Unsafe/unreviewed"):
                conditions.validate_quality_review(receipt, "B", verify_files=False)

    def test_missing_variant_rejected(self):
        receipt = reviewed_receipt()
        receipt["conditions"]["B"]["reviews"].pop()
        receipt["review_sha256"] = conditions._hash({k: v for k, v in receipt.items() if k != "review_sha256"})
        with self.assertRaisesRegex(ValueError, "every declared"):
            conditions.validate_quality_review(receipt, "B", verify_files=False)


class SessionAndSummaryTests(unittest.TestCase):
    def test_summary_never_compares_incomplete_candidates(self):
        config = conditions.load_config(ROOT / "configs/image_augmentation.yaml")
        prepared = {"config": config}
        complete = {"status": "complete", "completed_epochs": 12, "predictions_file": "best.csv",
                    "epoch12_predictions_file": "last.csv", "best_metrics": {"average_precision": 0.2}}
        partial = {**complete, "status": "budget_exhausted", "completed_epochs": 11,
                   "best_metrics": {"average_precision": 0.3}}
        result = conditions._summary(prepared, {"A": complete, "B": partial, "C": {"status": "deferred"}})
        self.assertFalse(result["comparison_complete"])
        self.assertEqual(result["comparisons"], {})

    def test_session_resume_does_not_reset_clock(self):
        config = conditions.load_config(ROOT / "configs/image_augmentation.yaml")
        with tempfile.TemporaryDirectory() as temp:
            prepared = {"root": Path(temp), "config": config}
            path, first = conditions.ensure_session(prepared)
            _, second = conditions.ensure_session(prepared)
            self.assertEqual(first, second)
            self.assertEqual(conditions._utc(first["training_cutoff_utc"]), conditions._utc(first["started_utc"]) + timedelta(minutes=100))
            self.assertEqual(conditions._utc(first["session_deadline_utc"]), conditions._utc(first["started_utc"]) + timedelta(minutes=120))

    def test_optional_c_requires_completed_ab_before_seventy_minutes(self):
        config = conditions.load_config(ROOT / "configs/image_augmentation.yaml")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = {"root": root, "config": config, "run_dir": root / "results/image_d1_runs/b0_augmentation_12e_v1",
                        "image_audit": {"status": "complete"}}
            prepared["run_dir"].mkdir(parents=True)
            records = {key: {"condition_id": key, "status": "pending", "completed_epochs": 0, "history": []}
                       for key in ("A", "B", "C")}
            conditions._atomic_json(prepared["run_dir"] / "summary.json", {"conditions": records})
            with patch.object(conditions, "persist_preparation", return_value=prepared["run_dir"]), \
                 patch.object(conditions, "_now", return_value=conditions._utc(config["run"]["session_start_utc"]) + timedelta(minutes=30)):
                result = conditions.run_conditions(root, config, prepared=prepared, condition_ids=("C",))
            self.assertEqual(result["conditions"]["C"]["status"], "deferred")
            self.assertIn("not_complete", result["stopping_reason"])


class RunnerContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        synthetic_project(self.root)
        self.config = conditions.load_config(ROOT / "configs/image_augmentation.yaml")
        self.prepared = conditions.prepare_conditions(self.root, self.config)
        self.moment = conditions._utc(self.config["run"]["session_start_utc"]) + timedelta(minutes=30)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def fit(*args, **kwargs):
        output = Path(args[5])
        return {"status": "complete", "best_epoch": 7,
                "best_metrics": {"average_precision": 0.2},
                "history": [{"epoch": epoch, "epoch_seconds": 34} for epoch in range(1, 13)],
                "latest_checkpoint": str(output / "latest.pt"), "best_checkpoint": str(output / "best.pt"),
                "microbatch": 64, "initial_model_state_sha256": "same-initial-parameters"}

    @staticmethod
    def predict(prepared, record, manifest, **kwargs):
        return {**record, "predictions_file": str(Path(record["output_dir"]) / "best.csv"),
                "epoch12_predictions_file": str(Path(record["output_dir"]) / "last.csv"),
                "predictions_file_sha256": "prediction-hash", "epoch12_metrics": {"average_precision": 0.19}}

    def test_matched_conditions_use_shared_trainer_twelve_epochs_and_initial_hash(self):
        with patch.object(conditions, "_now", return_value=self.moment), \
             patch.object(conditions, "validate_quality_review", return_value="review-hash"), \
             patch.object(conditions, "_prediction_export", side_effect=self.predict), \
             patch("melanoma_detection.image_model.pretrained_weight_identity", return_value={}), \
             patch("melanoma_detection.image_model.train_recipe", side_effect=self.fit) as fit:
            result = conditions.run_conditions(self.root, self.config, prepared=self.prepared)
        self.assertTrue(result["comparison_complete"])
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(fit.call_args_list[0].kwargs["epochs"], 12)
        self.assertEqual(fit.call_args_list[1].kwargs["epochs"], 12)
        self.assertIsNone(fit.call_args_list[0].kwargs["expected_initial_model_state_sha256"])
        self.assertEqual(fit.call_args_list[1].kwargs["expected_initial_model_state_sha256"], "same-initial-parameters")
        self.assertEqual(fit.call_args_list[1].kwargs["epoch_admission_safety_factor"], 1.25)
        self.assertEqual(fit.call_args_list[0].args[2]["augmentation"], "weak")
        self.assertEqual(fit.call_args_list[1].args[2]["augmentation"], "stronger")
        self.assertEqual(fit.call_args_list[0].args[3], fit.call_args_list[1].args[3])

    def test_unsafe_stronger_condition_never_trains_or_receives_inferiority_claim(self):
        def gate(path, condition_id, **kwargs):
            if condition_id == "B":
                raise ValueError("Visible lesion texture loss")
            return "review-hash"
        with patch.object(conditions, "_now", return_value=self.moment), \
             patch.object(conditions, "validate_quality_review", side_effect=gate), \
             patch.object(conditions, "_prediction_export", side_effect=self.predict), \
             patch("melanoma_detection.image_model.pretrained_weight_identity", return_value={}), \
             patch("melanoma_detection.image_model.train_recipe", side_effect=self.fit) as fit:
            result = conditions.run_conditions(self.root, self.config, prepared=self.prepared)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(result["conditions"]["B"]["status"], "quality_blocked")
        self.assertFalse(result["comparison_complete"])
        self.assertEqual(result["comparisons"], {})

    def test_prediction_only_preserves_summary_history_and_snapshots(self):
        conditions.persist_preparation(self.prepared)
        summary_path = self.prepared["run_dir"] / "summary.json"
        summary = conditions._read(summary_path)
        for key in ("A", "B"):
            record = summary["conditions"][key]
            record.update(self.fit(None, None, None, None, None, record["output_dir"]))
            record["completed_epochs"] = 12
        conditions._atomic_json(summary_path, summary)
        before = summary_path.read_bytes()
        with patch.object(conditions, "validate_quality_review", return_value="review-hash"), \
             patch.object(conditions, "_prediction_export", side_effect=self.predict), \
             patch("melanoma_detection.image_model.train_recipe") as fit:
            conditions.run_conditions(self.root, self.config, prepared=self.prepared, predict_only=True)
        self.assertFalse(fit.called)
        self.assertEqual(summary_path.read_bytes(), before)
        self.assertTrue((self.prepared["run_dir"] / "prediction_receipt.json").exists())

    def test_altered_source_snapshot_rejected(self):
        conditions.persist_preparation(self.prepared)
        snapshot = self.prepared["run_dir"] / "source_snapshots/src/melanoma_detection/image_conditions.py"
        snapshot.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "snapshot"):
            conditions.persist_preparation(self.prepared)


if __name__ == "__main__":
    unittest.main()
