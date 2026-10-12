"""Variant image preparation must preserve splits and hide held-out outcomes."""
from __future__ import annotations
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection import image_nested as stage


class PreservedVariantTests(unittest.TestCase):
    def setUp(self):
        self.config = stage.load_config(ROOT / "configs/image_nested_dedup.yaml")
        self.frame = pd.DataFrame([
            {"image_name": f"img_{outer}_{patient}_{image}", "patient_id": f"p_{outer}_{patient}",
             "fold": outer, "target": int(patient % 3 == 0 and image == 0)}
            for outer in range(5) for patient in range(9) for image in range(2)])
        self.members = self.frame[self.frame.fold.ne(0)].copy()
        self.members["inner_fold"] = self.members.patient_id.str.rsplit("_", n=1).str[-1].astype(int) // 3
        self.audit = {"rows": [{"image_id": name, "sha256": name} for name in self.frame.image_name]}
        self.identity = {"variant": "dedup_v1", "dataset_identity": "frozen_memberships"}

    def prepare(self, frame=None, members=None):
        with patch.object(stage, "StratifiedGroupKFold", side_effect=AssertionError("No split regeneration")):
            return stage.prepare_outer(self.frame if frame is None else frame, self.config, 0, self.audit,
                preserved_memberships=self.members if members is None else members, dataset_identity=self.identity)

    def test_uses_preserved_memberships_after_filtering_and_shuffling(self):
        removed = self.frame.image_name.eq("img_1_1_1")
        changed = self.frame[~removed].sample(frac=1, random_state=11).reset_index(drop=True)
        members = self.members[self.members.image_name.ne("img_1_1_1")].sample(frac=1, random_state=12)
        result = self.prepare(changed, members)
        actual = result["pool"].set_index("image_name").inner_fold.sort_index()
        expected = members.set_index("image_name").inner_fold.sort_index()
        pd.testing.assert_series_equal(actual, expected, check_dtype=False)

    def test_outer_labels_do_not_change_training_identity(self):
        first = self.prepare()
        changed = self.frame.copy()
        changed.loc[changed.fold.eq(0), "target"] = 1 - changed.loc[changed.fold.eq(0), "target"]
        second = self.prepare(changed)
        self.assertEqual(first["fold_manifest"], second["fold_manifest"])
        pd.testing.assert_frame_equal(first["pool"], second["pool"])

    def test_missing_wrong_or_cross_patient_membership_is_rejected(self):
        wrong = self.members.copy()
        wrong.loc[wrong.index[0], "patient_id"] = "wrong_patient"
        cross = self.members.copy()
        cross.loc[cross.index[1], "inner_fold"] = 1
        for members in (self.members.iloc[:-1], wrong, cross):
            with self.subTest(rows=len(members)), self.assertRaises((ValueError, AssertionError)):
                self.prepare(members=members)


if __name__ == "__main__":
    unittest.main()
