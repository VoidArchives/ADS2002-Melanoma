"""Verify committed architecture outputs without fitting a model.

The optional prediction check reconstructs one completed checkpoint from each
new family and compares its deterministic scores with the exported CSV.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import subprocess

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melanoma_detection import image_architecture as stage
from melanoma_detection.image_evaluation import load_architecture_outputs, load_historical_pilot, sha256
from melanoma_detection.image_model import _read_checkpoint, load_model_checkpoint, predict_frame, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-roundtrip", action="store_true")
    parser.add_argument("--prediction-only", action="store_true",
                        help="Regenerate executed predictions and verify training artifacts are untouched.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    config = stage.load_config(ROOT / "configs/image_architecture_pilot.yaml")
    run_dir = ROOT / config["run"]["output_dir"] / config["run"]["run_id"]
    summary = stage._read_json(run_dir / "summary.json")
    manifest = stage._read_json(run_dir / "run_manifest.json")
    historical = load_historical_pilot(ROOT)
    bundle = load_architecture_outputs(ROOT, summary)
    seed_everything(config["training"]["seed"])
    prediction_only_checks = []
    preserved_training_files = 0
    if args.prediction_only:
        protected = [run_dir / name for name in ("summary.json", "run_manifest.json", "work_ledger.json")]
        for record in summary["recipes"]:
            folder = Path(record["output_dir"])
            protected.extend(folder / name for name in (
                "history.csv", "history.json", "recipe_result.json", "latest.pt",
                "best.pt", "latest.pt.sha256.json", "best.pt.sha256.json") if (folder / name).exists())
        before_hashes = {path: sha256(path) for path in protected}
        execution = subprocess.run([sys.executable, str(ROOT / "scripts/run_image_architecture.py"),
                                    "--predict-only", "--device", args.device], cwd=ROOT)
        if execution.returncode not in (0, 2) or (summary["comparison_complete"] and execution.returncode != 0):
            raise RuntimeError("Prediction-only CLI did not complete the expected available checkpoints.")
        for path, expected in before_hashes.items():
            if sha256(path) != expected:
                raise ValueError(f"Prediction-only execution changed training artifact: {path}")
        preserved_training_files = len(before_hashes)
        after = load_architecture_outputs(ROOT, summary)
        for identity, before in bundle["predictions"].items():
            if identity.startswith("efficientnet_b0/"):
                continue
            loaded = after["predictions"][identity]
            merged = before.merge(loaded, on="image_id", suffixes=("_before", "_after"), validate="one_to_one")
            difference = float(np.max(np.abs(merged.base_probability_before - merged.base_probability_after)))
            if len(merged) != len(historical["validation"]) or difference > 1e-7:
                raise ValueError(f"Prediction-only scores changed: {identity}")
            prediction_only_checks.append({"pipeline_id": identity, "n_images": len(merged),
                                           "maximum_absolute_score_difference": difference})
    for relative, expected in manifest["compatibility"]["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Numerical source changed: {relative}")
    trials = {trial["pipeline_id"]: trial for trial in stage.candidate_trials(config)}
    checked, roundtrips, predicted_families = [], [], set()
    for record in summary["recipes"]:
        if record["status"] == "complete" and not all(
                record.get(key) for key in ("latest_checkpoint", "best_checkpoint")):
            raise ValueError(f"Completed ledger is missing checkpoints: {record['pipeline_id']}")
        if not record.get("latest_checkpoint"):
            continue
        expected = stage._trial_manifest({"manifest": manifest}, trials[record["pipeline_id"]])
        latest = _read_checkpoint(Path(record["latest_checkpoint"]))
        if latest["manifest"] != expected or latest["epoch"] != record["completed_epochs"]:
            raise ValueError(f"Checkpoint/ledger mismatch: {record['pipeline_id']}")
        required = {"optimizer_state", "scheduler_state", "scaler_state", "rng_state", "history"}
        if not required.issubset(latest):
            raise ValueError("Checkpoint is missing resumable state.")
        if latest["history"] != record["history"]:
            raise ValueError("Checkpoint history differs from the committed ledger.")
        if not all(np.isfinite(row[name]) for row in latest["history"]
                   for name in ("train_loss", "average_precision", "roc_auc", "log_loss", "brier")):
            raise ValueError("Nonfinite committed learning history.")
        if latest["effective_batch"] != 64:
            raise ValueError("Effective batch differs from the declared comparison.")
        if record.get("best_checkpoint"):
            best = _read_checkpoint(Path(record["best_checkpoint"]))
            if (best["manifest"] != expected or best["epoch"] != record["best_epoch"]
                    or best["best_metrics"] != record["best_metrics"]):
                raise ValueError(f"Best-checkpoint/ledger mismatch: {record['pipeline_id']}")
            del best
        checked.append({"pipeline_id": record["pipeline_id"], "epoch": latest["epoch"],
                        "microbatch": latest["microbatch"], "status": record["status"],
                        "latest_sha256": sha256(record["latest_checkpoint"])})
        del latest
        family = record["architecture"]
        if (args.prediction_roundtrip and record["status"] == "complete"
                and family not in predicted_families):
            model, saved = load_model_checkpoint(record["best_checkpoint"], device=args.device,
                                                 expected_manifest=expected)
            frame = predict_frame(model, historical["validation"], saved["spec"],
                                  {**saved["training"], "microbatch": saved["microbatch"]}, args.device)
            exported = pd.read_csv(record["predictions_file"])
            merged = exported.merge(frame, on="image_id", suffixes=("_saved", "_loaded"), validate="one_to_one")
            if len(merged) != len(historical["validation"]):
                raise ValueError("Prediction round-trip has incomplete image coverage.")
            difference = float(np.max(np.abs(merged.base_probability_saved - merged.base_probability_loaded)))
            if difference > 1e-6:
                raise ValueError(f"Prediction round-trip differs by {difference}: {family}")
            roundtrips.append({"architecture": family, "n_images": len(merged),
                               "maximum_absolute_score_difference": difference, "device": args.device})
            predicted_families.add(family)
            del model, saved
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    receipt = {"status": "passed", "historical_artifacts_preserved": True,
               "numerical_source_hashes_match": True, "comparison_complete": summary["comparison_complete"],
               "matched_recipe_slots": summary["matched_recipe_slots"],
               "prediction_sets_checked": len(bundle["predictions"]),
               "checkpoints": checked, "prediction_roundtrips": roundtrips,
               "prediction_only_checks": prediction_only_checks,
               "training_artifacts_preserved_after_prediction_only": preserved_training_files,
               "outer_predictions_generated": False, "reserved_test_loaded": False}
    stage._atomic_json(run_dir / "validation_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
