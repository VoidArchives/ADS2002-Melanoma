"""Fixed caps, shared patient draws, snapshots and persistent resume budgets."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from melanoma_detection import dedup_image_pilots as pilots
from melanoma_detection import image_model as core
from melanoma_detection.image_nested import claim_training_window

ROOT = Path(__file__).resolve().parents[1]


class DedupPilotTests(unittest.TestCase):
    def setUp(self):
        self.config = pilots.load_config(ROOT / "configs/image_dedup_pilots.yaml")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_predeclared_endpoints_and_recipes_cannot_expand(self):
        for field, value in [("endpoints", [12, 30]), ("schedule_epochs", 24), ("outer_predictions_allowed", True)]:
            changed = copy.deepcopy(self.config)
            changed["protocol"][field] = value
            with self.assertRaises(ValueError):
                pilots.validate_config(changed)
        changed = copy.deepcopy(self.config)
        changed["challenger"]["recipes"][0]["backbone_lr"] = 3e-5
        with self.assertRaises(ValueError):
            pilots.validate_config(changed)

    def test_baseline_prerequisite_requires_both_full_clean_runs(self):
        with patch("melanoma_detection.image_nested_reporting.load_saved_nested_run", return_value={"complete": True}), \
             patch("melanoma_detection.image_nested_calibration_reporting.load_saved_calibration_run", return_value={"complete": False}):
            with self.assertRaisesRegex(ValueError, "Both clean five-fold"):
                pilots.prerequisite_baseline(self.root, self.config, {})

    def test_job_identity_stays_same_at_both_caps_and_is_distinct_from_historical(self):
        prepared = {"manifest": {"run_id": "b0_duration_dedup_v1", "compatibility_sha256": "fresh-clean"}}
        recipe = {"id": "low_none", "backbone_lr": 1e-5, "loss_weight": "none"}
        manifest = pilots._job_manifest(prepared, recipe, 1)
        self.assertEqual(manifest["epoch_cap"], 24)
        self.assertFalse(manifest["outer_labels_used"])
        self.assertEqual(manifest["purpose"], "fresh_clean_inner_trajectory")
        self.assertNotEqual(manifest["run_id"], "b0_nested_5fold_5h_v1")

    def _save_checkpoint(self, path, manifest, epoch):
        identity = {"manifest": manifest, "fresh": True}
        checkpoint = {"checkpoint_version": core.CHECKPOINT_VERSION, "manifest": manifest, "epoch": epoch,
                      "identity": identity, "compatibility_hash": core.canonical_hash(identity)}
        core._save_checkpoint(path, checkpoint)

    def test_endpoint_freezes_checkpoint_pairs_before_continuation(self):
        folder = self.root / "trajectory"
        folder.mkdir()
        manifest = {"run_id": "clean", "inner_fold": 0}
        self._save_checkpoint(folder / "latest.pt", manifest, 12)
        self._save_checkpoint(folder / "best.pt", manifest, 8)
        result = {"status": "complete", "history": [{"epoch": i} for i in range(1, 13)],
                  "latest_checkpoint": str(folder / "latest.pt"), "best_checkpoint": str(folder / "best.pt")}
        paths = pilots.snapshot_endpoint(folder, result, manifest, 12)
        self._save_checkpoint(folder / "latest.pt", manifest, 24)
        self.assertEqual(core._read_checkpoint(Path(paths["endpoint"]["path"]))["epoch"], 12)
        self.assertEqual(core._read_checkpoint(Path(paths["best_in_cap"]["path"]))["epoch"], 8)

    def test_partial_or_later_checkpoint_cannot_be_labelled_epoch12(self):
        folder = self.root / "trajectory"
        folder.mkdir()
        manifest = {"run_id": "clean"}
        self._save_checkpoint(folder / "latest.pt", manifest, 24)
        self._save_checkpoint(folder / "best.pt", manifest, 24)
        result = {"status": "complete", "history": list(range(24)), "latest_checkpoint": str(folder / "latest.pt"), "best_checkpoint": str(folder / "best.pt")}
        with self.assertRaisesRegex(ValueError, "exactly its completed cap"):
            pilots.snapshot_endpoint(folder, result, manifest, 12)

    def test_trajectory_calls_same_fit_directory_at_12_and24(self):
        prepared = {"run_dir": self.root, "manifest": {"run_id": "clean", "compatibility_sha256": "membership"}}
        recipe = {"id": "lr", "backbone_lr": 1e-5, "loss_weight": "none"}
        observed = []
        def fit(prepared, recipe, inner, cap, deadline, device):
            observed.append((recipe["id"], inner, cap))
            return {"status": "complete"}
        with patch.object(pilots, "_fit", side_effect=fit), patch.object(pilots, "snapshot_endpoint"), patch.object(pilots, "_predict_snapshots"):
            result = pilots._trajectory(prepared, recipe, 2, 99999, "cpu")
        self.assertEqual(observed, [("lr", 2, 12), ("lr", 2, 24)])
        self.assertEqual(result["status"], "complete")

    def test_partial_trajectory_returns_incomplete_before_second_cap(self):
        prepared = {"run_dir": self.root, "manifest": {"run_id": "clean", "compatibility_sha256": "membership"}}
        recipe = {"id": "lr"}
        with patch.object(pilots, "_fit", return_value={"status": "budget_exhausted"}) as fit, patch.object(pilots, "snapshot_endpoint") as snap:
            self.assertIsNone(pilots._trajectory(prepared, recipe, 0, 0, "cpu"))
        self.assertEqual(fit.call_count, 1)
        snap.assert_not_called()

    def test_resume_does_not_reset_90minute_budget(self):
        path = self.root / "session.json"
        config = {"run": {"run_id": "b0_duration_dedup_v1"}, "budget": {"session_minutes": 90, "reporting_reserve_minutes": 10}}
        start = datetime(2026, 10, 5, tzinfo=timezone.utc)
        first = claim_training_window(path, config, now=start)
        self.assertEqual(first, claim_training_window(path, config, now=start + timedelta(minutes=60)))
        with self.assertRaises(TimeoutError):
            claim_training_window(path, config, now=start + timedelta(minutes=81))

    def test_paired_bootstrap_is_order_invariant_and_shared_for_identical_scores(self):
        records = []
        for fold in range(3):
            for patient in range(10):
                for image in range(2):
                    target = patient % 2
                    records.append({"image_id": f"f{fold}p{patient}i{image}", "patient_id": f"f{fold}p{patient}", "target": target,
                                    "inner_fold": fold, "base_probability": .7 if target else .3})
        frame = pd.DataFrame(records)
        output = pilots.paired_inner_bootstrap(frame, frame.sample(frac=1, random_state=7), "identical", n_bootstrap=100)
        self.assertTrue(all(row["difference"] == 0 and row["ci_lower"] == 0 and row["ci_upper"] == 0 for row in output))
        changed = frame.copy()
        changed.loc[0, "target"] = 1
        with self.assertRaisesRegex(ValueError, "matching retained records"):
            pilots.paired_inner_bootstrap(frame, changed, "mismatch", n_bootstrap=10)


if __name__ == "__main__":
    unittest.main()
