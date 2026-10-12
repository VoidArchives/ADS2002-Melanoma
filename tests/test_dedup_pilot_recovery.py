"""Independent restart/integrity regression checks; no image model is trained."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection import dedup_image_pilots as pilots
from melanoma_detection import image_model as core
from melanoma_detection.development_dataset import load_development_dataset, load_preserved_inner_memberships
from melanoma_detection.dedup_audit import freeze_development_variant
from melanoma_detection.metadata_evaluation import probability_metrics
from test_dedup_audit import create_audit_fixture


class PilotRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = pilots.load_config(ROOT / "configs/image_dedup_pilots.yaml")

    def save_checkpoint(self, path, manifest, epoch, best_epoch=None):
        identity = {"manifest": manifest, "fresh": True}
        history = [{"epoch": i, "average_precision": .1+.003*i, "roc_auc": .6+.001*i,
                    "log_loss": .2, "brier": .05, "epoch_seconds": .01,
                    "peak_memory_bytes": 0, "peak_reserved_memory_bytes": 0} for i in range(1, epoch+1)]
        checkpoint = {"checkpoint_version": core.CHECKPOINT_VERSION, "manifest": manifest, "epoch": epoch,
                      "identity": identity, "compatibility_hash": core.canonical_hash(identity),
                      "history": history, "best_epoch": epoch if best_epoch is None else best_epoch}
        core._save_checkpoint(path, checkpoint)
        return checkpoint

    def test_truncated_uncommitted_snapshot_is_repaired_from_verified_checkpoint(self):
        folder = self.root / "trajectory"
        folder.mkdir()
        manifest = {"run_id": "fresh-clean", "inner_fold": 0}
        saved = self.save_checkpoint(folder / "latest.pt", manifest, 12)
        self.save_checkpoint(folder / "best.pt", manifest, 8)
        (folder / "endpoint_12.pt").write_bytes(b"interrupted partial copy")
        (folder / "endpoint_12.pt.sha256.json").write_text('{"sha256":"interrupted"}')
        result = {"status": "complete", "history": saved["history"],
                  "latest_checkpoint": str(folder / "latest.pt"), "best_checkpoint": str(folder / "best.pt")}
        receipt = pilots.snapshot_endpoint(folder, result, manifest, 12)
        self.assertEqual(core._read_checkpoint(Path(receipt["endpoint"]["path"]))["epoch"], 12)
        self.assertEqual(core._read_checkpoint(Path(receipt["best_in_cap"]["path"]))["epoch"], 8)
        frozen = Path(receipt["endpoint"]["path"])
        frozen.write_bytes(b"committed snapshot changed")
        with self.assertRaisesRegex(ValueError, "Immutable endpoint"):
            pilots.snapshot_endpoint(folder, result, manifest, 12)
        self.assertEqual(frozen.read_bytes(), b"committed snapshot changed")

    def test_completed_run_is_verified_without_claiming_an_expired_clock(self):
        run_dir = self.root / "run"
        run_dir.mkdir()
        manifest = {"run_id": self.config["duration"]["run_id"], "stage": "duration"}
        saved = {"status": "complete", "stage": "duration", "completed_inner_folds": [0, 1, 2]}
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
        (run_dir / "summary.json").write_text(json.dumps(saved))
        prepared = {"run_dir": run_dir, "manifest": manifest}
        with patch.object(pilots, "prepare_pilot", return_value=prepared), \
             patch.object(pilots, "load_complete_pilot", return_value=saved) as verify, \
             patch("melanoma_detection.image_nested.claim_training_window", side_effect=TimeoutError("expired")) as clock, \
             patch.object(pilots, "_fit", side_effect=AssertionError("No new fit")) as fit:
            self.assertEqual(pilots.run_pilot(self.root, self.config, "duration", device="cpu"), saved)
        verify.assert_called_once()
        clock.assert_not_called()
        fit.assert_not_called()

    def test_empty_artifact_inventory_cannot_claim_complete(self):
        directory = self.root / "duration_dedup_v1"
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps({"status": "complete",
            "completed_inner_folds": [0, 1, 2], "artifact_sha256": {}}))
        with self.assertRaisesRegex(ValueError, "artifact inventory"):
            pilots.load_complete_pilot(directory)

    def create_complete_report(self):
        _, _, counts = create_audit_fixture(self.root)
        freeze_development_variant(self.root, expected_counts=counts, progress=None)
        _, identity = load_development_dataset(self.root, self.config["run"]["dataset_manifest"])
        pool = load_preserved_inner_memberships(self.root, identity, 0)
        recipe = {"id": "lr3e5_none", "backbone_lr": 3e-5, "loss_weight": "none"}
        compatibility = {"stage": "duration", "inner_memberships": pool.to_dict("records"),
                         "selected_clean_b0_recipe": recipe, "settings": {"protocol": self.config["protocol"]}}
        manifest = {"schema_version": 1, "run_id": self.config["duration"]["run_id"], "stage": "duration",
                    "compatibility": compatibility, "compatibility_sha256": pilots.stable_sha256(compatibility),
                    "dataset_identity": identity, "outer_labels_used": False,
                    "outer_predictions_generated": False, "reserved_test_loaded": False}
        run_dir = self.root / "results/image_d1_runs" / manifest["run_id"]
        run_dir.mkdir(parents=True)
        pilots._atomic_json(run_dir / "run_manifest.json", manifest)
        prepared = {"root": self.root, "run_dir": run_dir, "stage": "duration", "manifest": manifest, "config": self.config}
        records = []
        for inner in range(3):
            folder = run_dir / f"inner/{recipe['id']}/{inner}"
            folder.mkdir(parents=True)
            job = pilots._job_manifest(prepared, recipe, inner)
            for cap in (12, 24):
                saved = self.save_checkpoint(folder / "latest.pt", job, cap, cap-4)
                self.save_checkpoint(folder / "best.pt", job, cap-4)
                result = {"status": "complete", "history": saved["history"],
                          "latest_checkpoint": str(folder / "latest.pt"), "best_checkpoint": str(folder / "best.pt")}
                pilots.snapshot_endpoint(folder, result, job, cap)
                path = folder / f"endpoint_receipt_{cap}.json"
                receipt = pilots._read(path)
                receipt["predictions"] = {}
                for kind in ("endpoint", "best_in_cap"):
                    predictions = pool[pool.inner_fold.eq(inner)].copy().rename(columns={"image_name": "image_id"})
                    predictions["base_probability"] = .05+.6*predictions.target+.001*cap
                    predictions["endpoint_cap"], predictions["selection_kind"] = cap, kind
                    predictions["run_id"] = manifest["run_id"]
                    output = folder / f"predictions_{kind}_{cap}.csv"
                    predictions.to_csv(output, index=False)
                    receipt["predictions"][kind] = {"path": str(output), "sha256": pilots.sha256_file(output),
                        "metrics": probability_metrics(predictions.target, predictions.base_probability)}
                pilots._atomic_json(path, receipt)
            records.append({"inner_fold": inner, "recipe": recipe, "status": "complete",
                            "receipts": {str(cap): str(folder / f"endpoint_receipt_{cap}.json") for cap in (12, 24)}})
        summary = pilots._finish_report(prepared, records, [])
        return run_dir, summary

    def test_complete_report_roundtrip_and_changed_endpoint_annotations_rejected(self):
        run_dir, summary = self.create_complete_report()
        verified = pilots.load_complete_pilot(run_dir)
        self.assertEqual(verified["training_epochs"], 72)
        self.assertEqual(verified["fits"], 3)
        self.assertFalse(verified["final_winner_selected"])
        self.assertEqual(verified["artifact_sha256"], summary["artifact_sha256"])
        path = run_dir / "inner/lr3e5_none/1/predictions_endpoint_12.csv"
        changed = pd.read_csv(path)
        changed["endpoint_cap"] = 24
        changed.to_csv(path, index=False)
        digest = pilots.sha256_file(path)
        summary["artifact_sha256"][path.relative_to(run_dir).as_posix()] = digest
        receipt_path = path.parent / "endpoint_receipt_12.json"
        receipt = pilots._read(receipt_path)
        receipt["predictions"]["endpoint"]["sha256"] = digest
        pilots._atomic_json(receipt_path, receipt)
        summary["artifact_sha256"][receipt_path.relative_to(run_dir).as_posix()] = pilots.sha256_file(receipt_path)
        pilots._atomic_json(run_dir / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "annotations"):
            pilots.load_complete_pilot(run_dir)

    def test_combined_scores_and_self_declared_inner_reassignments_are_rejected(self):
        run_dir, summary = self.create_complete_report()
        path = run_dir / "evaluation/predictions_endpoint_24.csv"
        original = path.read_bytes()
        changed = pd.read_csv(path)
        changed.loc[0, "base_probability"] = .4
        changed.to_csv(path, index=False)
        relative = path.relative_to(run_dir).as_posix()
        summary["artifact_sha256"][relative] = pilots.sha256_file(path)
        pilots._atomic_json(run_dir / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "combined|Combined|scores"):
            pilots.load_complete_pilot(run_dir)
        path.write_bytes(original)
        summary["artifact_sha256"][relative] = pilots.sha256_file(path)
        manifest_path = run_dir / "run_manifest.json"
        manifest = pilots._read(manifest_path)
        rows = manifest["compatibility"]["inner_memberships"]
        patient = rows[0]["patient_id"]
        for row in rows:
            if row["patient_id"] == patient:
                row["inner_fold"] = (row["inner_fold"]+1)%3
        manifest["compatibility_sha256"] = pilots.stable_sha256(manifest["compatibility"])
        pilots._atomic_json(manifest_path, manifest)
        summary["artifact_sha256"]["run_manifest.json"] = pilots.sha256_file(manifest_path)
        pilots._atomic_json(run_dir / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "preserved|Preserved"):
            pilots.load_complete_pilot(run_dir)


if __name__ == "__main__":
    unittest.main()
