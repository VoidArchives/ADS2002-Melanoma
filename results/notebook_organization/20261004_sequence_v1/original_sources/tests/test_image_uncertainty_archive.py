"""Old receipts permit exactly one byte-identical notebook relocation."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from melanoma_detection.image_evaluation import preserved_artifact_path, sha256

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


if __name__ == "__main__":
    unittest.main()
