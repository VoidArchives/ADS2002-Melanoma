"""Finalise a complete, separately saved training-only calibration stage.

Reads saved scores and frozen maps only. Does not train or fit calibration,
alter the raw reference report, extend a deadline or read the reserved test.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.image_evaluation import sha256
from melanoma_detection.image_nested_calibration_reporting import build_calibration_evaluation, load_saved_calibration_run


def finalize(project_root, run_id="b0_nested_calibration_5h_v1"):
    root = Path(project_root).resolve()
    stage = load_saved_calibration_run(root, run_id)
    if not stage.get("ready_for_reporting"):
        raise ValueError(f"Calibration report requires all five complete verified folds; current status: {stage['status']}.")
    run_dir = stage["run_dir"]
    summary_path = run_dir / "summary.json"
    before = sha256(summary_path)
    manifest = stage["manifest"]
    config = manifest["config"]
    reference_id = stage["summary"]["reference_run_id"]
    raw_manifest = json.loads((root / "results/image_d1_runs" / reference_id / "run_manifest.json").read_text(encoding="utf-8"))
    evaluation = build_calibration_evaluation(stage["predictions"], stage["raw_predictions"], stage["expected"], run_dir,
        bootstrap_replicates=config["evaluation"]["bootstrap_replicates"], seed=raw_manifest["config"]["training"]["seed"])
    if sha256(summary_path) != before:
        raise ValueError("Calibration summary changed during reporting; wait for the stage to stop before retrying.")
    summary = stage["summary"]
    summary["evaluation"] = evaluation
    temporary = summary_path.with_name("summary.reporting.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="b0_nested_calibration_5h_v1")
    args = parser.parse_args()
    summary = finalize(ROOT, args.run_id)
    result = summary["evaluation"]
    print(json.dumps({"status": summary["status"], "run_id": args.run_id,
        "mean_fold_metrics": result["mean_fold_metrics"],
        "mean_fold_ap_invariant": result["mean_fold_ap_invariant"],
        "training_selected_decisions_identical": result["training_selected_decisions_identical"],
        "valid_bootstrap_replicates": result["valid_bootstrap_replicates"],
        "reserved_test_loaded": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
