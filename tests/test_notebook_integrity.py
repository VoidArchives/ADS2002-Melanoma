"""Filename relocation must not weaken recorded computational provenance."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.notebook_integrity import code_sha256, verify_recorded_notebook_code


class NotebookIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "notebooks/Subset").mkdir(parents=True)
        self.old = "notebooks/03b_metadata_neural_network_d1.ipynb"
        self.new = "notebooks/Subset/06_metadata_neural_network.ipynb"
        self.snapshot = "results/notebook_organization/rename/originals/" + self.old
        self.document = {"cells": [{"cell_type": "code", "source": [
            "SEED = 42\n", "source_path = '" + self.old + "'\n"]}]}
        self.expected = code_sha256(self.document)
        encoded = json.dumps(self.document).encode()
        (self.root / self.snapshot).parent.mkdir(parents=True)
        (self.root / self.snapshot).write_bytes(encoded)
        self.registry = {"schema_version": 1, "renames": [{
            "old_path": self.old, "new_path": self.new, "snapshot_path": self.snapshot,
            "original_sha256": hashlib.sha256(encoded).hexdigest()}]}
        self.write_registry()
        current = json.loads(json.dumps(self.document))
        current["cells"][0]["source"][1] = "source_path = '" + self.new + "'\n"
        self.current = current
        self.write_current()

    def write_registry(self):
        (self.root / "notebooks/filename_map.json").write_text(json.dumps(self.registry))

    def write_current(self):
        (self.root / self.new).write_text(json.dumps(self.current))

    def verify(self, expected=None):
        return verify_recorded_notebook_code(self.root, self.new, self.old, expected or self.expected)

    def test_accepts_filename_changes_and_exact_new_run_code(self):
        self.assertEqual(self.verify(), "verified_filename_substitutions_only")
        self.assertEqual(self.verify(code_sha256(self.current)), "exact_current_code")

    def test_computational_change_remains_stale(self):
        self.current["cells"][0]["source"][0] = "SEED = 43\n"
        self.write_current()
        with self.assertRaisesRegex(ValueError, "computation changed"):
            self.verify()

    def test_missing_or_tampered_original_is_rejected(self):
        (self.root / self.snapshot).write_text("{}")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            self.verify()
        (self.root / self.snapshot).unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_snapshot_cannot_escape_rename_directory(self):
        self.registry["renames"][0]["snapshot_path"] = "../outside.ipynb"
        self.write_registry()
        with self.assertRaisesRegex(ValueError, "outside the rename"):
            self.verify()

    def test_real_saved_metadata_predictions_keep_original_code_check(self):
        config = json.loads((ROOT / "results/metrics/metadata_nn_run_config_d1.json").read_text())
        self.assertEqual(verify_recorded_notebook_code(ROOT, self.new,
            config["notebook_path"], config["code_sha256"]), "verified_filename_substitutions_only")


if __name__ == "__main__":
    unittest.main()
