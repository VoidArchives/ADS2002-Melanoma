"""Historical notebook bindings never accept changed fitting sources."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.metadata_source_archive import ARCHIVE, historical_notebook_path, verified_source_path


class MetadataArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.relative = "notebooks/Subset/06_metadata_neural_network.ipynb"
        self.current = self.root / self.relative
        self.current.parent.mkdir(parents=True)
        self.current.write_text("updated report", encoding="utf-8")
        self.snapshot = self.root / ARCHIVE / self.relative
        self.snapshot.parent.mkdir(parents=True)
        self.snapshot.write_text("historical fitting source", encoding="utf-8")
        self.digest = hashlib.sha256(self.snapshot.read_bytes()).hexdigest()
        self.receipt = self.root / ARCHIVE / "preservation_receipt.json"
        self.receipt.write_text(json.dumps({"notebook_snapshots": {self.relative: self.digest}}))

    def test_verified_original_source_survives_report_edit(self):
        self.assertEqual(verified_source_path(self.root, self.relative, self.digest), self.snapshot)
        self.assertEqual(historical_notebook_path(self.root, self.relative), self.snapshot)
        with self.assertRaisesRegex(ValueError, "binding differs"):
            verified_source_path(self.root, self.relative, "different experiment")

    def test_changed_or_missing_archive_is_rejected(self):
        self.snapshot.write_text("tampered historical fitting", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            verified_source_path(self.root, self.relative, self.digest)
        with self.assertRaisesRegex(ValueError, "changed"):
            historical_notebook_path(self.root, self.relative)
        self.receipt.unlink()
        with self.assertRaisesRegex(ValueError, "receipt is missing"):
            verified_source_path(self.root, self.relative, self.digest)

    def test_python_sources_and_escaped_paths_cannot_use_notebook_fallback(self):
        relative = "src/example_training.py"
        current = self.root / relative
        current.parent.mkdir()
        current.write_text("changed fitter")
        with self.assertRaisesRegex(ValueError, "source changed"):
            verified_source_path(self.root, relative, self.digest)
        with self.assertRaisesRegex(ValueError, "outside the project"):
            verified_source_path(self.root, "../other/source.py", self.digest)
        unchanged = hashlib.sha256(current.read_bytes()).hexdigest()
        self.assertEqual(verified_source_path(self.root, relative, unchanged), current)


if __name__ == "__main__":
    unittest.main()
