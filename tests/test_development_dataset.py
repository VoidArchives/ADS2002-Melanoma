"""Variant integrity, deterministic masking and preserved patient memberships."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from melanoma_detection.development_dataset import (
    CANONICAL_CSV, VARIANT_DIRECTORY, load_development_dataset,
    load_preserved_inner_memberships, retained_id_hash, sha256_file,
)


def create_loader_fixture(root):
    rows = []
    for index in range(10):
        rows.append({"image_name": f"ISIC_{index:07d}", "patient_id": f"IP_{index:07d}",
                     "sex": "male", "age_approx": 50, "anatom_site_general_challenge": "torso",
                     "diagnosis": "melanoma" if index % 2 else "unknown",
                     "benign_malignant": "malignant" if index % 2 else "benign",
                     "target": index % 2, "fold": index // 2})
    frame = pd.DataFrame(rows)
    canonical = root / CANONICAL_CSV
    canonical.parent.mkdir(parents=True)
    frame.to_csv(canonical, index=False)
    directory = root / VARIANT_DIRECTORY
    directory.mkdir()
    ledger = frame[["image_name"]].assign(pixel_sha256=[str(i) for i in range(10)],
                    decision="retained", retained_image_name=frame.image_name)
    ledger.to_csv(directory / "retention_ledger.csv", index=False)
    frame.to_csv(directory / "train_folds.csv", index=False)
    artifacts = {f"{VARIANT_DIRECTORY}/{name}": sha256_file(directory / name) for name in ("retention_ledger.csv", "train_folds.csv")}
    paths = {}
    for fold in range(5):
        inner = frame.loc[frame.fold.ne(fold), ["fold", "image_name", "patient_id", "target"]].copy()
        inner["inner_fold"] = [index % 3 for index in range(len(inner))]
        relative = f"{VARIANT_DIRECTORY}/folds/{fold}/inner_memberships.csv"
        path = root / relative
        path.parent.mkdir(parents=True)
        inner.to_csv(path, index=False)
        artifacts[relative] = sha256_file(path)
        paths[str(fold)] = relative
    manifest = {"schema_version": 1, "variant": "dedup_v1", "original_source": {"path": CANONICAL_CSV, "sha256": sha256_file(canonical)},
                "artifacts": artifacts, "retained_ids_sha256": retained_id_hash(frame), "preserved_inner_memberships": paths,
                "counts": {"images": 10, "malignant_images": 5, "patients": 10, "malignant_patients": 5}}
    manifest_path = directory / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return frame, manifest, manifest_path


class DevelopmentDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frame, self.manifest, self.path = create_loader_fixture(self.root)

    def test_original_default_and_variant_preserve_frame(self):
        original, identity = load_development_dataset(self.root)
        self.assertEqual(identity["variant"], "original")
        self.assertIsNone(identity["dataset_manifest"])
        variant, identity = load_development_dataset(self.root, self.path)
        pd.testing.assert_frame_equal(original, variant)
        self.assertEqual(identity["variant"], "dedup_v1")
        inner = load_preserved_inner_memberships(self.root, identity, 0)
        self.assertEqual(set(inner.image_name), set(variant.loc[variant.fold.ne(0), "image_name"]))

    def test_source_change_stops_loading(self):
        with (self.root / CANONICAL_CSV).open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Canonical development source changed"):
            load_development_dataset(self.root, self.path)

    def test_artifact_tampering_stops_loading(self):
        with (self.root / VARIANT_DIRECTORY / "retention_ledger.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            load_development_dataset(self.root, self.path)

    def test_rehashed_noncanonical_export_still_stops_loading(self):
        path = self.root / VARIANT_DIRECTORY / "train_folds.csv"
        changed = pd.read_csv(path)
        changed.loc[0, "age_approx"] = 60
        changed.to_csv(path, index=False)
        self.manifest["artifacts"][path.relative_to(self.root).as_posix()] = sha256_file(path)
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "differs from the canonical"):
            load_development_dataset(self.root, self.path)

    def test_manifest_cannot_escape_variant_directory(self):
        with self.assertRaisesRegex(ValueError, "must stay within"):
            load_development_dataset(self.root, self.root / "outside.json")

    def test_changed_inner_assignment_detected_even_when_rehashed(self):
        relative = self.manifest["preserved_inner_memberships"]["0"]
        path = self.root / relative
        inner = pd.read_csv(path)
        inner.loc[0, "patient_id"] = "IP_9999999"
        inner.to_csv(path, index=False)
        self.manifest["artifacts"][relative] = sha256_file(path)
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "outer-training pool"):
            load_development_dataset(self.root, self.path)


if __name__ == "__main__":
    unittest.main()
