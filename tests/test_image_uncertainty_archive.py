"""Immutable receipt hashes remain enforceable after explicit notebook moves."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from melanoma_detection.image_evaluation import (notebook_rename_registry,
    preserved_artifact_path, sha256, workspace_relative_path)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_image_condition_session",
    ROOT / "scripts/verify_image_condition_session.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class HistoricalUncertaintyArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.original = self.root / verifier.UNCERTAINTY_ORIGINAL
        self.archive = self.root / verifier.UNCERTAINTY_ARCHIVE
        self.original.parent.mkdir(parents=True)
        self.original.write_bytes(b"original historical notebook")
        self.expected = sha256(self.original)
        self.inventory = self.root / "preservation_inventory.json"
        self.inventory.write_text(json.dumps({"files": {
            verifier.UNCERTAINTY_ORIGINAL: {"sha256": self.expected}}}), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def relocate(self):
        self.archive.parent.mkdir(parents=True)
        self.archive.write_bytes(self.original.read_bytes())
        self.original.write_bytes(b"current updated notebook")

    def test_unchanged_original_needs_no_archive(self):
        self.assertEqual(preserved_artifact_path(self.root, verifier.UNCERTAINTY_ORIGINAL,
                                               self.expected), self.original)
        self.assertEqual(verifier.verify(self.root, self.inventory)["unchanged_count"], 1)

    def test_exact_archive_matches_immutable_receipt(self):
        receipt_before = self.inventory.read_bytes()
        self.relocate()
        self.assertEqual(preserved_artifact_path(self.root, verifier.UNCERTAINTY_ORIGINAL,
                                               self.expected), self.archive)
        report = verifier.verify(self.root, self.inventory)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["authorised_historical_archives"][0]["sha256"], self.expected)
        self.assertEqual(self.inventory.read_bytes(), receipt_before)

    def test_missing_or_tampered_archive_is_rejected(self):
        self.original.write_bytes(b"current updated notebook")
        for content in (None, b"tampered historical notebook"):
            with self.subTest(content=content):
                if content is not None:
                    self.archive.parent.mkdir(parents=True, exist_ok=True)
                    self.archive.write_bytes(content)
                with self.assertRaisesRegex(ValueError, "Preserved historical artifact changed"):
                    preserved_artifact_path(self.root, verifier.UNCERTAINTY_ORIGINAL, self.expected)
                report = verifier.verify(self.root, self.inventory)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(len(report["historical_artifact_failures"]), 1)

    def test_archive_cannot_bypass_another_protected_path_or_digest(self):
        self.relocate()
        with self.assertRaises(ValueError):
            preserved_artifact_path(self.root, "notebooks/another_notebook.ipynb", self.expected)
        with self.assertRaises(ValueError):
            preserved_artifact_path(self.root, verifier.UNCERTAINTY_ORIGINAL, "0" * 64)

    def test_supporting_source_change_requires_exact_original_snapshot(self):
        relative = "src/melanoma_detection/image_evaluation.py"
        current = self.root / relative
        current.parent.mkdir(parents=True)
        current.write_bytes(b"original saved-report loader")
        expected = sha256(current)
        self.inventory.write_text(json.dumps({"files": {relative: {"sha256": expected}}}), encoding="utf-8")
        current.write_bytes(b"loader with the exact archive fallback")
        self.assertEqual(verifier.verify(self.root, self.inventory)["status"], "failed")
        snapshot = self.inventory.parent / "baseline_sources" / relative
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(b"original saved-report loader")
        self.assertEqual(verifier.verify(self.root, self.inventory)["status"], "passed")
        snapshot.write_bytes(b"changed original snapshot")
        self.assertEqual(verifier.verify(self.root, self.inventory)["status"], "failed")

    def rename_record(self, old, new, content=b"original historical notebook"):
        snapshot = f"results/notebook_organization/test_sequence/originals/{old}"
        saved = self.root / snapshot
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(content)
        return {"old_path": old, "new_path": new, "snapshot_path": snapshot,
                "original_sha256": sha256(saved)}

    def write_registry(self, records, sources=None):
        path = self.root / "notebooks/filename_map.json"
        path.write_text(json.dumps({"schema_version": 1, "renames": records,
                                  "authorized_source_snapshots": sources or {}}), encoding="utf-8")
        return path

    def test_renamed_unchanged_file_satisfies_exact_old_receipt(self):
        record = self.rename_record(verifier.UNCERTAINTY_ORIGINAL, "notebooks/Subset/13_image_uncertainty.ipynb")
        renamed = self.root / record["new_path"]
        renamed.parent.mkdir(parents=True)
        renamed.write_bytes(self.original.read_bytes())
        self.original.unlink()
        self.write_registry([record])
        self.assertEqual(preserved_artifact_path(self.root, record["old_path"], self.expected), renamed)
        report = verifier.verify(self.root, self.inventory)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["authorised_historical_archives"][0]["archive"], record["new_path"])

    def test_edited_renamed_notebook_uses_only_exact_original_snapshot(self):
        old = "notebooks/05e_image_architecture_comparison_d1.ipynb"
        record = self.rename_record(old, "notebooks/Subset/10_image_architecture_comparison.ipynb")
        current = self.root / record["new_path"]
        current.parent.mkdir(parents=True)
        current.write_bytes(b"same executed outputs, updated headings and links")
        self.write_registry([record])
        self.inventory.write_text(json.dumps({"files": {old: {"sha256": self.expected}}}), encoding="utf-8")
        original = self.root / record["snapshot_path"]
        self.assertEqual(preserved_artifact_path(self.root, old, self.expected), original)
        report = verifier.verify(self.root, self.inventory)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["current_architecture_notebook_preserved"])
        original.write_bytes(b"tampered original snapshot")
        with self.assertRaisesRegex(ValueError, "Preserved historical artifact changed"):
            preserved_artifact_path(self.root, old, self.expected)
        self.assertEqual(verifier.verify(self.root, self.inventory)["status"], "failed")
        original.unlink()
        with self.assertRaises(ValueError): preserved_artifact_path(self.root, old, self.expected)

    def test_original_key_and_original_sha_are_not_interchangeable(self):
        record = self.rename_record("notebooks/different.ipynb", "notebooks/Subset/new.ipynb")
        self.original.write_bytes(b"changed")
        self.write_registry([record])
        with self.assertRaises(ValueError):
            preserved_artifact_path(self.root, verifier.UNCERTAINTY_ORIGINAL, self.expected)
        record["old_path"] = verifier.UNCERTAINTY_ORIGINAL
        record["snapshot_path"] = self.rename_record(verifier.UNCERTAINTY_ORIGINAL, record["new_path"])["snapshot_path"]
        record["original_sha256"] = "0" * 64
        self.write_registry([record])
        with self.assertRaises(ValueError):
            preserved_artifact_path(self.root, verifier.UNCERTAINTY_ORIGINAL, self.expected)

    def test_original_prototype_archive_survives_current_notebook_rename(self):
        current_record = self.rename_record(verifier.UNCERTAINTY_ORIGINAL,
                                            "notebooks/Subset/13_image_uncertainty.ipynb", b"current inference notebook")
        historical_record = self.rename_record(verifier.UNCERTAINTY_ARCHIVE,
                                               "notebooks/archive/image_uncertainty_three_epoch_fold0.ipynb")
        self.original.unlink()
        self.write_registry([current_record, historical_record])
        self.assertEqual(preserved_artifact_path(self.root, verifier.UNCERTAINTY_ORIGINAL, self.expected),
                         self.root / historical_record["snapshot_path"])
        # A historical result path cannot use any notebook-original snapshot.
        with self.assertRaises(ValueError):
            preserved_artifact_path(self.root, "results/image_d1_wip/raw.csv", self.expected)

    def test_registry_and_direct_paths_reject_escaping_or_redirected_snapshots(self):
        record = self.rename_record(verifier.UNCERTAINTY_ORIGINAL, "notebooks/Subset/new.ipynb")
        for path in ["../outside", "C:/outside", "/absolute", "notebooks/../outside", "notebooks\\wrong.ipynb"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                workspace_relative_path(self.root, path)
        for field, path in [("old_path", "../outside.ipynb"), ("new_path", "C:/outside.ipynb"),
                            ("snapshot_path", "results/notebook_organization/test_sequence/originals/notebooks/unrelated.ipynb"),
                            ("snapshot_path", "results/notebook_organization/test_sequence/originals/../outside")]:
            altered = dict(record, **{field: path})
            self.write_registry([altered])
            with self.subTest(field=field, path=path), self.assertRaises(ValueError):
                notebook_rename_registry(self.root)
            self.assertEqual(verifier.verify(self.root, self.inventory)["status"], "failed")
        for records in [[record, record], [record, dict(record, old_path="notebooks/other.ipynb")]]:
            self.write_registry(records)
            with self.assertRaisesRegex(ValueError, "unique"):
                notebook_rename_registry(self.root)

    def test_supporting_source_registry_requires_explicit_path_and_original_digest(self):
        relative = "scripts/build_image_notebook.py"
        current = self.root / relative
        current.parent.mkdir(parents=True)
        current.write_bytes(b"builder before renamed paths")
        expected = sha256(current)
        snapshot_path = f"results/notebook_organization/test_sequence/original_sources/{relative}"
        snapshot = self.root / snapshot_path
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(current.read_bytes())
        current.write_bytes(b"builder after renamed paths")
        self.inventory.write_text(json.dumps({"files": {relative: {"sha256": expected}}}), encoding="utf-8")
        registry = self.write_registry([], {relative: {"snapshot_path": snapshot_path, "original_sha256": expected}})
        before = (self.inventory.read_bytes(), snapshot.read_bytes(), registry.read_bytes())
        report = verifier.verify(self.root, self.inventory)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["authorised_source_changes"][0]["original_snapshot"], snapshot_path)
        self.assertEqual(before, (self.inventory.read_bytes(), snapshot.read_bytes(), registry.read_bytes()))
        self.write_registry([], {relative: {"snapshot_path": snapshot_path, "original_sha256": "0"*64}})
        self.assertEqual(verifier.verify(self.root, self.inventory)["status"], "failed")
        self.write_registry([], {relative: {"snapshot_path": snapshot_path, "original_sha256": expected}})
        snapshot.write_bytes(b"changed original builder")
        self.assertEqual(verifier.verify(self.root, self.inventory)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
