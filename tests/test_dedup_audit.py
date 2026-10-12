"""Exact pixels alone mask rows; source selection precedes label parsing."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from melanoma_detection.development_dataset import CANONICAL_CSV, sha256_file
from melanoma_detection.dedup_audit import (
    AUDIT_DIRECTORY, HISTORICAL_AUDIT, HISTORICAL_RUN, VARIANT_DIRECTORY,
    audit_images, freeze_development_variant, image_fingerprint, read_development_source,
)


def create_audit_fixture(root):
    rows, image_rows = [], []
    image_directory = root / "data/subset/train"
    image_directory.mkdir(parents=True)
    for index in range(60):
        patient = index // 2
        target = patient % 2
        name = f"ISIC_{index:07d}"
        rows.append({"image_name": name, "patient_id": f"IP_{patient:07d}", "sex": "male", "age_approx": 50,
                     "anatom_site_general_challenge": "torso", "diagnosis": "melanoma" if target else "unknown",
                     "benign_malignant": "malignant" if target else "benign", "target": target, "fold": patient // 6})
        image_path = image_directory / f"{name}.jpg"
        if index == 1:
            image_path.write_bytes((image_directory / "ISIC_0000000.jpg").read_bytes())
        else:
            pixels = np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(image_path, "JPEG", quality=95)
        with Image.open(image_path) as im:
            image_rows.append({"image_id": name, "patient_id": rows[-1]["patient_id"], "fold": rows[-1]["fold"],
                               "size": list(im.size), "source_sha256": sha256_file(image_path), "pixel_sha256": image_fingerprint(im.convert("RGB"))})
    frame = pd.DataFrame(rows)
    path = root / CANONICAL_CSV
    path.parent.mkdir(parents=True)
    frame.to_csv(path, index=False)
    for relative in ["data/raw/train.csv", "data/subset/train_labels.csv", "data/subset/processed/train_clean.csv"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        source = frame.drop(columns="fold")
        # An unrelated malformed label must be discarded before parsing.
        unrelated = source.iloc[[0]].copy()
        unrelated["image_name"] = "NOT_A_DEVELOPMENT_IMAGE"
        unrelated["target"] = "not a label"
        pd.concat([unrelated, source]).to_csv(path, index=False)
    saved = {"rows": image_rows}
    path = root / HISTORICAL_AUDIT
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(saved))
    for outer in range(5):
        inner = frame.loc[frame.fold.ne(outer), ["fold", "image_name", "patient_id", "target"]].copy()
        inner["inner_fold"] = inner.patient_id.str[-7:].astype(int) % 3
        path = root / HISTORICAL_RUN / "folds" / str(outer) / "inner_memberships.csv"
        path.parent.mkdir(parents=True)
        inner.to_csv(path, index=False)
    expected = {"original_images": 60, "duplicate_groups": 1, "excluded_benign": 1, "excluded_malignant": 0,
                "images": 59, "malignant_images": 30, "patients": 30, "malignant_patients": 15}
    return frame, saved, expected


class DedupAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frame, self.saved, self.expected = create_audit_fixture(self.root)

    def test_only_development_rows_returned_before_bad_label_is_parsed(self):
        result = read_development_source(self.root / "data/raw/train.csv", set(self.frame.image_name))
        self.assertEqual(len(result), 60)
        self.assertEqual(set(result.target), {"0", "1"})

    def test_identical_metadata_distinct_photos_remain(self):
        ledger, audit = audit_images(self.root, self.frame, self.saved)
        self.assertEqual(audit["duplicate_groups"], [["ISIC_0000000", "ISIC_0000001"]])
        self.assertEqual(ledger.decision.eq("excluded").sum(), 1)
        self.assertEqual(ledger.set_index("image_name").loc["ISIC_0000001", "retained_image_name"], "ISIC_0000000")
        self.assertEqual(ledger.set_index("image_name").loc["ISIC_0000003", "decision"], "retained")

    def test_duplicate_metadata_conflict_stops_before_freeze(self):
        changed = self.frame.copy()
        changed.loc[1, "age_approx"] = 65
        with self.assertRaisesRegex(ValueError, "conflict"):
            audit_images(self.root, changed, self.saved)
        self.assertFalse((self.root / VARIANT_DIRECTORY).exists())

    def test_changed_original_jpeg_stops_audit(self):
        Image.new("RGB", (32, 32), "black").save(self.root / "data/subset/train/ISIC_0000002.jpg", "JPEG")
        with self.assertRaisesRegex(ValueError, "differs from saved audit"):
            audit_images(self.root, self.frame, self.saved)

    def test_unreadable_image_stops_before_any_exports(self):
        (self.root / "data/subset/train/ISIC_0000002.jpg").write_bytes(b"invalid jpeg")
        with self.assertRaises(Exception):
            freeze_development_variant(self.root, expected_counts=self.expected, progress=None)
        self.assertFalse((self.root / VARIANT_DIRECTORY).exists())

    def test_freeze_is_idempotent_originals_unchanged_and_no_test_access_needed(self):
        original_digest = sha256_file(self.root / CANONICAL_CSV)
        receipt = freeze_development_variant(self.root, expected_counts=self.expected, progress=None)
        self.assertEqual(receipt["retained_counts"]["images"], 59)
        self.assertFalse(receipt["reserved_test_loaded"])
        self.assertFalse(receipt["training_started"])
        before = {path.relative_to(self.root).as_posix(): sha256_file(path) for path in (self.root / VARIANT_DIRECTORY).rglob("*") if path.is_file()}
        second = freeze_development_variant(self.root, expected_counts=self.expected, progress=None)
        self.assertEqual(receipt, second)
        self.assertEqual(original_digest, sha256_file(self.root / CANONICAL_CSV))
        self.assertEqual(before, {path.relative_to(self.root).as_posix(): sha256_file(path) for path in (self.root / VARIANT_DIRECTORY).rglob("*") if path.is_file()})

    def test_divergent_existing_artifact_never_overwritten(self):
        directory = self.root / VARIANT_DIRECTORY
        directory.mkdir()
        path = directory / "retention_ledger.csv"
        path.write_text("do not overwrite")
        with self.assertRaisesRegex(ValueError, "Immutable"):
            freeze_development_variant(self.root, expected_counts=self.expected, progress=None)
        self.assertEqual(path.read_text(), "do not overwrite")
        self.assertFalse((directory / "dataset_manifest.json").exists())
        self.assertFalse((self.root / AUDIT_DIRECTORY).exists())


if __name__ == "__main__":
    unittest.main()
