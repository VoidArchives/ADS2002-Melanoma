"""Complete clean reports, shared-clock calibration and the separate bounded pilots.

The already running raw baseline is waited on, never started a second time.
Every downstream stage requires complete verified upstream results. A budget
stop remains incomplete and stops the queue. No clock is reset by this script.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection import image_nested as stage

STATUS = ROOT / "results/dedup_v1/implementation/workflow_status.json"


def update(phase, status, **details):
    previous = json.loads(STATUS.read_text()) if STATUS.exists() else {"completed_stages": []}
    previous.update(phase=phase, status=status, updated_utc=stage._utc_now().isoformat(),
                    test_set_loaded=False, **details)
    if status == "complete" and phase not in previous["completed_stages"]:
        previous["completed_stages"].append(phase)
    stage._atomic_json(STATUS, previous)
    print(f"{phase}: {status}", flush=True)


def command(phase, *args):
    update(phase, "running")
    result = subprocess.run([sys.executable, "-B", *args], cwd=ROOT)
    if result.returncode:
        update(phase, "incomplete", exit_code=result.returncode)
        raise RuntimeError(f"{phase} stopped with code {result.returncode}; queue will not advance.")
    update(phase, "complete")


def wait_raw():
    directory = ROOT / "results/image_d1_runs/b0_nested_dedup_v1"
    update("raw_b0", "running", completed_stages=["development_audit", "saved_sensitivity", "metadata"])
    while True:
        clock = json.loads((directory / "session.json").read_text())
        path = directory / "summary.json"
        summary = json.loads(path.read_text()) if path.exists() else {}
        if clock.get("status") == "training_finished" and summary.get("status") == "complete":
            update("raw_b0", "complete")
            return
        if clock.get("status") == "training_paused" or stage._utc_now() >= stage._utc(clock["session_deadline_utc"]):
            update("raw_b0", "incomplete", stopping_reason=summary.get("stopping_reason", "budget_or_worker_stop"))
            raise RuntimeError("Raw baseline is incomplete; preserve checkpoints and stop downstream stages.")
        time.sleep(10)


def verify_preservation():
    receipt = json.loads((STATUS.parent / "preservation_before.json").read_text())
    checked = 0
    for row in receipt["historical_artifacts"]:
        if stage.file_sha256(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"Historical artifact changed: {row['path']}")
        checked += 1
    changed_sources = []
    for row in receipt["original_sources"]:
        path = ROOT / row["path"]
        if stage.file_sha256(path) != row["sha256"]:
            archived = STATUS.parent / "source_before" / row["path"]
            if row["path"].startswith("notebooks/") or not archived.is_file() or stage.file_sha256(archived) != row["sha256"]:
                raise ValueError(f"Original source was changed without preserving its bytes: {row['path']}")
            changed_sources.append(row["path"])
    manifest = json.loads((ROOT / "data/subset/processed/dedup_v1/dataset_manifest.json").read_text())
    for relative, digest in manifest["audited_input_sha256"].items():
        if stage.file_sha256(ROOT / relative) != digest:
            raise ValueError(f"Audited original development source changed: {relative}")
    record = {"status": "complete", "historical_artifacts_verified": checked,
              "original_sources_archived_before_changes": changed_sources,
              "original_notebooks_unchanged": True, "test_set_loaded": False}
    stage._atomic_json(STATUS.parent / "preservation_after.json", record)
    update("preservation", "complete", preservation=record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_only:
            verify_preservation()
            return 0
        wait_raw()
        command("raw_reporting", "scripts/finalize_image_nested.py", "--run-id", "b0_nested_dedup_v1")
        command("image_calibration", "scripts/run_image_nested_calibration.py", "--config", "configs/image_nested_calibration_dedup.yaml", "--resume")
        command("calibration_reporting", "scripts/finalize_image_nested_calibration.py", "--run-id", "b0_nested_calibration_dedup_v1")
        command("clean_three_condition", "scripts/run_dedup_sensitivity.py", "--run-id", "clean_three_condition_v1",
                "--clean-image-run", "b0_nested_dedup_v1", "--clean-metadata-receipt", "results/dedup_v1/metadata/comparator_receipt.json",
                "--clean-calibration-run", "b0_nested_calibration_dedup_v1")
        command("bounded_pilots", "scripts/run_dedup_image_pilots.py", "--config", "configs/image_dedup_pilots.yaml")
        verify_preservation()
        update("workflow", "complete")
        return 0
    except Exception as error:
        current = json.loads(STATUS.read_text()) if STATUS.exists() else {}
        update(current.get("phase", "workflow"), "incomplete", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
