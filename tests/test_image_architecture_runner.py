"""Bounded work, fair comparisons and deadline contracts without GPU training."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melanoma_detection import image_architecture as stage


class Clock:
    def __init__(self):
        self.seconds = 0.0

    def monotonic(self):
        return self.seconds


class ArchitectureRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = stage.load_config(ROOT / "configs/image_architecture_pilot.yaml")
        self.run_dir = self.root / "results/image_d1_runs" / self.config["run"]["run_id"]
        self.run_dir.mkdir(parents=True)
        self.session_path = self.run_dir / "session.json"
        self.now = datetime(2026, 10, 3, 7, 0, tzinfo=timezone.utc)
        self.session = {"session_id": "synthetic_session", "started_utc": self.now.isoformat(),
                        "latest_training_start_utc": (self.now + timedelta(minutes=115)).isoformat(),
                        "training_cutoff_utc": (self.now + timedelta(minutes=165)).isoformat(),
                        "session_deadline_utc": (self.now + timedelta(minutes=180)).isoformat(),
                        "max_training_minutes": 50, "training_started": False}
        stage._atomic_json(self.session_path, self.session)
        self.clock = Clock()
        pool = pd.DataFrame({"image_name": ["fit_0", "fit_1", "valid_0", "valid_1"],
                             "patient_id": ["patient_0", "patient_1", "patient_2", "patient_3"],
                             "fold": [1, 2, 3, 4], "target": [0, 1, 0, 1], "inner_fold": [1, 2, 0, 0]})
        excluded = pd.DataFrame({"image_name": ["outer_0"], "patient_id": ["outer_patient"],
                                 "fold": [0], "target": [1]})
        self.prepared = {"manifest": {"run_id": self.config["run"]["run_id"],
                                      "compatibility_sha256": "synthetic_identity"},
                         "config": copy.deepcopy(self.config), "run_dir": self.run_dir,
                         "split_audit": {"patient_overlap": 0, "test_set_loaded": False},
                         "image_audit": {"status": "complete"}, "inner_memberships": pool,
                         "outer_excluded": excluded, "fitting": pool.iloc[:2], "validation": pool.iloc[2:],
                         "b0_reference": [{"architecture": "efficientnet_b0", "recipe_slot": slot,
                                           "pipeline_id": f"efficientnet_b0/{slot}", "status": "complete",
                                           "completed_epochs": 6} for slot in stage.SLOTS]}

    def tearDown(self):
        self.temp.cleanup()

    def fake_fit(self, *args, **kwargs):
        self.clock.seconds += self.seconds_per_fit
        epochs = self.completed_epochs
        status = "complete" if epochs == 6 else "budget_exhausted"
        history = [{"epoch": index + 1, "average_precision": 0.1 + index * 0.01,
                    "roc_auc": 0.8, "log_loss": 0.1, "brier": 0.02,
                    "epoch_seconds": 100.0, "peak_memory_bytes": 1000} for index in range(epochs)]
        return {"status": status, "history": history, "best_epoch": epochs,
                "best_metrics": history[-1] if history else None, "microbatch": args[3]["microbatch"],
                "best_checkpoint": str(Path(args[5]) / "best.pt"), "elapsed_seconds": self.seconds_per_fit}

    def fake_predictions(self, prepared, config, trial, record, **kwargs):
        return {**record, "predictions_file": str(Path(record["output_dir"]) / "inner_validation_predictions.csv")}

    def run_fake(self, *, seconds_per_fit=1, completed_epochs=6):
        self.seconds_per_fit, self.completed_epochs = seconds_per_fit, completed_epochs
        with patch.object(stage, "_utc_now", return_value=self.now), \
             patch.object(stage.time, "monotonic", side_effect=self.clock.monotonic), \
             patch("melanoma_detection.image_model.train_recipe", side_effect=self.fake_fit), \
             patch.object(stage, "_prediction_export", side_effect=self.fake_predictions):
            return stage.run_architecture_pilot(self.root, self.config, prepared=self.prepared,
                                                device="cpu", session_path=self.session_path)

    def test_family_specific_lr_library_and_pair_order(self):
        trials = stage.candidate_trials(self.config)
        self.assertEqual(len(trials), 8)
        self.assertEqual([item["recipe_slot"] for item in trials], [slot for slot in stage.SLOTS for _ in range(2)])
        self.assertEqual([item["architecture"] for item in trials], list(stage.FAMILIES) * 4)
        self.assertEqual(trials[0]["backbone_lr"], 3e-5)
        self.assertEqual(trials[1]["backbone_lr"], 1e-5)
        self.assertEqual(trials[6]["backbone_lr"], 1e-4)
        self.assertEqual(trials[7]["backbone_lr"], 3e-5)

    def test_fixed_horizon_and_candidates_reject_undeclared_settings(self):
        for key, value in (("augmentation", "stronger"), ("preprocessing", "hair"), ("image_size", 256)):
            config = copy.deepcopy(self.config)
            config["model"][key] = value
            with self.assertRaises(ValueError):
                stage.validate_config(config)
        config = copy.deepcopy(self.config)
        config["families"]["swin_tiny"] = {"microbatch": 16, "backbone_lr": [1e-5, 3e-5]}
        with self.assertRaises(ValueError):
            stage.validate_config(config)
        config = copy.deepcopy(self.config)
        config["training"]["schedule_epochs"] = 6
        with self.assertRaises(ValueError):
            stage.validate_config(config)

    def test_complete_four_matched_pairs_do_not_select_final_architecture(self):
        summary = self.run_fake()
        self.assertEqual(summary["matched_recipe_slots"], list(stage.SLOTS))
        self.assertTrue(summary["comparison_complete"])
        self.assertIsNone(summary["selected_architecture"])
        self.assertEqual(summary["unmatched_completed_trials"], [])
        self.assertFalse(summary["outer_predictions_generated"])
        self.assertFalse(summary["test_set_loaded"])

    def test_next_pair_not_admitted_if_measured_time_will_not_fit(self):
        summary = self.run_fake(seconds_per_fit=1000)
        self.assertEqual(summary["matched_recipe_slots"], ["low_none"])
        self.assertEqual(sum(item["status"] == "complete" for item in summary["recipes"]), 2)
        self.assertEqual(summary["stopping_reason"], "insufficient_measured_time_for_next_matched_pair")
        self.assertFalse(summary["comparison_complete"])
        self.assertIsNone(summary["selected_architecture"])

    def test_partial_first_trial_cannot_win_or_advance_pair(self):
        summary = self.run_fake(completed_epochs=3)
        self.assertEqual(summary["matched_recipe_slots"], [])
        self.assertEqual(summary["recipes"][0]["completed_epochs"], 3)
        self.assertEqual(summary["recipes"][1]["status"], "pending")
        self.assertFalse(summary["comparison_complete"])

    def test_incompatible_run_rejected_without_overwriting_manifest(self):
        stage.persist_architecture_preparation(self.prepared)
        original = (self.run_dir / "run_manifest.json").read_bytes()
        changed = copy.deepcopy(self.prepared)
        changed["manifest"]["compatibility_sha256"] = "changed_code_or_data"
        with self.assertRaises(ValueError):
            stage.persist_architecture_preparation(changed)
        self.assertEqual((self.run_dir / "run_manifest.json").read_bytes(), original)

    def test_resuming_same_session_cannot_reset_first_start_deadline(self):
        first = stage.claim_training_window(self.session_path, now=self.now)
        resumed = stage.claim_training_window(self.session_path, now=self.now + timedelta(minutes=20))
        self.assertEqual(first, self.now + timedelta(minutes=50))
        self.assertEqual(first, resumed)
        with self.assertRaises(TimeoutError):
            stage.claim_training_window(self.session_path, now=self.now + timedelta(minutes=51))

    def test_latest_start_prevents_new_training(self):
        with self.assertRaises(TimeoutError):
            stage.claim_training_window(self.session_path, now=self.now + timedelta(minutes=116))
        self.assertFalse(stage._read_json(self.session_path)["training_started"])

    def test_prediction_only_preserves_summary_and_history(self):
        self.run_fake()
        summary_before = (self.run_dir / "summary.json").read_bytes()
        folder = self.run_dir / "families/densenet121/recipes/low_none"
        history = folder / "history.csv"
        history.write_text("epoch,loss\n1,0.2\n", encoding="utf-8")
        history_before = history.read_bytes()
        with patch.object(stage, "_prediction_export", side_effect=self.fake_predictions):
            receipt = stage.run_architecture_pilot(self.root, self.config, prepared=self.prepared,
                      predict_only=True, device="cpu", session_path=self.session_path)
        self.assertTrue(receipt["training_history_preserved"])
        self.assertEqual((self.run_dir / "summary.json").read_bytes(), summary_before)
        self.assertEqual(history.read_bytes(), history_before)

    def test_historical_membership_change_rejected_before_reference_comparison(self):
        config_dir = self.root / "configs"
        config_dir.mkdir()
        (config_dir / "image_model.yaml").write_bytes((ROOT / "configs/image_model.yaml").read_bytes())
        reference = self.root / self.config["run"]["reference_run"]
        reference.mkdir()
        old = {"training_pool_memberships": [{"patient_id": "old"}], "training_pool_images": [],
               "config": {"model": {}}}
        stage._atomic_json(reference / "run_manifest.json", {"compatibility": old})
        changed = copy.deepcopy(self.prepared)
        changed["manifest"] = {"compatibility": {**old, "training_pool_memberships": [{"patient_id": "new"}]}}
        with patch.object(stage.image_pilot, "prepare_pilot", return_value=changed), \
             patch.object(stage, "_reference_records") as references:
            with self.assertRaisesRegex(ValueError, "training_pool_memberships"):
                stage.prepare_architecture_pilot(self.root, self.config)
            references.assert_not_called()

    def test_watchdog_pause_recovers_only_checksum_verified_epoch(self):
        import torch
        stage.persist_architecture_preparation(self.prepared)
        summary = stage._read_json(self.run_dir / "summary.json")
        record = summary["recipes"][0]
        record["status"] = "running"
        folder = Path(record["output_dir"])
        folder.mkdir(parents=True)
        saved = {"history": [{"epoch": 1, "average_precision": 0.2}], "epoch": 1,
                 "best_epoch": 1, "best_metrics": {"average_precision": 0.2}, "microbatch": 32}
        torch.save(saved, folder / "latest.pt")
        torch.save(saved, folder / "best.pt")
        for name in ("latest", "best"):
            stage._atomic_json(folder / f"{name}.pt.sha256.json",
                               {"sha256": stage.image_pilot.file_sha256(folder / f"{name}.pt")})
        stage._atomic_json(self.run_dir / "summary.json", summary)
        recovered = stage.record_watchdog_pause(self.run_dir)
        self.assertEqual(recovered["recipes"][0]["completed_epochs"], 1)
        self.assertEqual(recovered["recipes"][0]["status"], "budget_exhausted")
        self.assertTrue(recovered["checkpoint_checks"][0]["latest"])
        self.assertFalse(recovered["comparison_complete"])
        (folder / "latest.pt").write_bytes(b"incomplete bytes")
        recovered["recipes"][0]["status"] = "running"
        stage._atomic_json(self.run_dir / "summary.json", recovered)
        bad = stage.record_watchdog_pause(self.run_dir)
        self.assertEqual(bad["recipes"][0]["status"], "checkpoint_recovery_required")
        self.assertFalse(bad["checkpoint_checks"][0]["latest"])

    def test_cli_watchdog_terminates_child_and_keeps_session_deadline(self):
        spec = importlib.util.spec_from_file_location("architecture_cli_test", ROOT / "scripts/run_image_architecture.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        class Child:
            def __init__(self):
                self.calls = 0
                self.terminated = False
            def wait(self, timeout):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("synthetic child", timeout)
                return 2
            def terminate(self):
                self.terminated = True
            def kill(self):
                self.terminated = True
        child = Child()
        args = ["runner", "--config", str(ROOT / "configs/image_architecture_pilot.yaml"), "--device", "cpu"]
        with patch.object(cli, "PROJECT_ROOT", self.root), \
             patch.object(sys, "argv", args), \
             patch.object(stage, "prepare_architecture_pilot", return_value=self.prepared), \
             patch.object(stage, "_utc_now", return_value=self.now), \
             patch.object(cli.subprocess, "Popen", return_value=child):
            self.assertEqual(cli.main(), 2)
        self.assertTrue(child.terminated)
        session = stage._read_json(self.session_path)
        self.assertEqual(stage._utc(session["training_window_deadline_utc"]), self.now + timedelta(minutes=50))
        self.assertEqual(session["status"], "training_paused")


if __name__ == "__main__":
    unittest.main()
