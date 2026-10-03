"""Controlled reporting and explicit visual-gate integrity tests."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from melanoma_detection.image_condition_reporting import (QUALITY_CATEGORIES,
    condition_comparisons, fitting_membership_hash, load_condition_outputs,
    select_quality_examples, validate_quality_review)
from melanoma_detection.image_evaluation import classification_metrics, sha256
from melanoma_detection.image_model import canonical_hash
from melanoma_detection.metadata_evaluation import probability_metrics


class ConditionReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rows = []
        for index in range(24):
            path = self.root / f"image_{index}.png"
            Image.new("RGB", (24, 24), (index * 8, 100, 140)).save(path)
            rows.append({"image_id": f"image_{index}", "patient_id": f"patient_{index}",
                "target": int(index < 12), "image_path": path})
        self.fitting = pd.DataFrame(rows)

    def tearDown(self):
        self.temporary.cleanup()

    def receipt(self):
        preview = self.root / "preview.png"
        Image.new("RGB", (16, 16), "white").save(preview)
        selected = [{"image_id": row.image_id, "patient_id": row.patient_id,
            "target": row.target, "source_sha256": sha256(row.image_path)}
            for row in self.fitting.itertuples()]
        value = {"fitting_membership_sha256": fitting_membership_hash(self.fitting),
            "selected_examples": selected, "preview_sha256": {str(preview): sha256(preview)},
            "sample_coverage": {"framing": "large and peripheral", "hair": "visible and absent",
                "lighting": "dark and bright"}, "reviewed_by": "test reviewer",
            "reviewed_at_utc": "2026-10-03T00:00:00Z",
            "conditions": {"B": {"verdict": "passed", "reviewed_image_ids": list(self.fitting.image_id),
                "reviews": [{"image_id": row.image_id, "variant": variant,
                    **{category: 0 for category in QUALITY_CATEGORIES}}
                    for row in self.fitting.itertuples() for variant in range(4)]}}}
        value["review_sha256"] = canonical_hash(value)
        return value

    def test_quality_sample_is_balanced_distinct_and_fitting_only(self):
        chosen = select_quality_examples(self.fitting)
        self.assertEqual(chosen.patient_id.nunique(), 24)
        self.assertEqual(chosen.target.sum(), 12)
        pd.testing.assert_frame_equal(chosen, select_quality_examples(self.fitting))
        with self.assertRaisesRegex(ValueError, "fitting images only"):
            select_quality_examples(self.fitting, [*list(self.fitting.image_id)[:-1], "outer_image"])

    def test_completed_visual_review_verifies_all_artifacts(self):
        receipt = self.receipt()
        checked = validate_quality_review(receipt, "B", fitting=self.fitting)
        self.assertEqual(checked["verdict"], "passed")
        self.assertEqual(checked["n_variants"], 96)
        Image.new("RGB", (16, 16), "black").save(self.root / "preview.png")
        with self.assertRaisesRegex(ValueError, "preview file changed"):
            validate_quality_review(receipt, "B", fitting=self.fitting)

    def test_unreviewed_or_unsafe_variant_cannot_pass(self):
        receipt = self.receipt()
        receipt["conditions"]["B"]["reviews"][0]["clipping"] = 2
        receipt["review_sha256"] = canonical_hash({key: value for key, value in receipt.items()
                                                    if key != "review_sha256"})
        with self.assertRaisesRegex(ValueError, "verdict disagrees"):
            validate_quality_review(receipt, "B", fitting=self.fitting)
        receipt["conditions"]["B"]["verdict"] = "quality_blocked"
        receipt["review_sha256"] = canonical_hash({key: value for key, value in receipt.items()
                                                    if key != "review_sha256"})
        self.assertEqual(validate_quality_review(receipt, "B")["verdict"], "quality_blocked")
        pending = self.receipt()
        pending["conditions"]["B"]["reviews"][0]["colour"] = None
        pending["review_sha256"] = canonical_hash({key: value for key, value in pending.items()
                                                    if key != "review_sha256"})
        with self.assertRaisesRegex(ValueError, "explicit severity"):
            validate_quality_review(pending, "B")

    def _complete(self, condition_id, scores):
        frame = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "patient_id": ["p1", "p2", "p3", "p4"],
            "target": [0, 1, 0, 1], "fold": [1] * 4, "base_probability": scores})
        path = self.root / f"{condition_id}.csv"
        frame.to_csv(path, index=False)
        return {"condition_id": condition_id, "pipeline_id": condition_id, "status": "complete",
            "completed_epochs": 12, "best_epoch": 10, "predictions_file": str(path),
            "epoch12_predictions_file": str(path), "best_metrics": probability_metrics(frame.target, scores),
            "epoch12_metrics": probability_metrics(frame.target, scores)}

    def test_comparisons_require_matching_completed_horizons(self):
        a = self._complete("A", [.2, .1, .3, .8])
        b = self._complete("B", [.1, .8, .2, .9])
        bundle = load_condition_outputs(self.root, {"conditions": {"A": a, "B": b}})
        result = condition_comparisons(bundle).set_index("comparison")
        self.assertEqual(result.loc["B - A", "status"], "complete")
        self.assertTrue(result.loc["B - A", "promising_b0_hypothesis"])
        b["status"] = "paused"
        b["completed_epochs"] = 11
        bundle = load_condition_outputs(self.root, {"conditions": {"A": a, "B": b}})
        result = condition_comparisons(bundle).set_index("comparison")
        self.assertEqual(result.loc["B - A", "status"], "incomplete")
        self.assertTrue(np.isnan(result.loc["B - A", "ap_delta"]))
        b["status"] = "complete"
        with self.assertRaisesRegex(ValueError, "twelve epochs"):
            load_condition_outputs(self.root, {"conditions": {"A": a, "B": b}})

    def test_prediction_membership_mismatch_is_rejected(self):
        a = self._complete("A", [.2, .1, .3, .8])
        b = self._complete("B", [.1, .8, .2, .9])
        path = Path(b["predictions_file"])
        frame = pd.read_csv(path)
        frame.loc[0, "patient_id"] = "different_patient"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "membership/labels differ"):
            load_condition_outputs(self.root, {"conditions": {"A": a, "B": b}})

    def test_fixed_threshold_undefined_precision_remains_missing(self):
        metrics = classification_metrics([0, 1], [.1, .2], threshold=.5)
        self.assertTrue(np.isnan(metrics["precision"]))
        self.assertEqual(metrics["threshold"], .5)
        self.assertEqual(metrics["FN"], 1)


if __name__ == "__main__":
    unittest.main()
