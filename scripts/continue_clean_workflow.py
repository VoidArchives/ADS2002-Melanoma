"""Finish the additional clean workflow without duplicating image training.

The existing controller owns raw B0, calibration and both bounded pilots. This
controller waits for that queue to terminate, then runs inference and refreshes
reports. Persisted image training deadlines are never changed or restarted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.development_dataset import load_development_dataset, sha256_file
from melanoma_detection.dedup_workflow import DIRECTORY, MANIFEST, build_evaluation, read_json, verify_cleaning, write_json

STATUS = ROOT / DIRECTORY / "status.json"
IMAGE_STATUS = ROOT / "results/dedup_v1/implementation/workflow_status.json"


@contextmanager
def controller_lock(root=ROOT):
    """An OS-held lock prevents two controllers from launching the same work."""
    path = Path(root) / DIRECTORY / "controller.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("The clean workflow already has an active controller.") from error
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def update(phase, status, **details):
    previous = read_json(STATUS) if STATUS.exists() else {"completed_stages": []}
    previous.update(phase=phase, status=status, updated_utc=datetime.now(timezone.utc).isoformat(),
        dataset_manifest=MANIFEST, reserved_test_loaded=False, image_training_restarted=False, **details)
    if status == "complete" and phase not in previous["completed_stages"]:
        previous["completed_stages"].append(phase)
    write_json(STATUS, previous)
    print(f"Clean workflow: {phase} / {status}", flush=True)


def terminal_image_state(state):
    """Intermediate completed stages must never wake the inference queue."""
    return state.get("status") in {"incomplete", "failed", "stopped", "cancelled"} or (
        state.get("status") == "complete" and state.get("phase") == "workflow")


def wait_images(timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    previous = None
    while time.monotonic() < deadline:
        try:
            state = read_json(IMAGE_STATUS)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(10)
            continue
        marker = (state.get("phase"), state.get("status"))
        if marker != previous:
            update("waiting_for_existing_image_queue", "running", image_queue_state=state)
            previous = marker
        if terminal_image_state(state):
            return state
        time.sleep(10)
    raise TimeoutError("Existing image queue did not reach a terminal state; no training was restarted.")


def command(phase, *arguments):
    update(phase, "running")
    subprocess.run([sys.executable, "-B", *arguments], cwd=ROOT, check=True)
    update(phase, "complete")


def verify_preservation(root=ROOT):
    root = Path(root).resolve()
    receipt_path = root / "results/dedup_v1/implementation/preservation_before.json"
    receipt = read_json(receipt_path)
    artifact_count = 0
    for row in receipt["historical_artifacts"]:
        if sha256_file(root / row["path"]) != row["sha256"]:
            raise ValueError(f"Historical artifact changed: {row['path']}")
        artifact_count += 1
    notebook_count = 0
    for row in receipt["original_sources"]:
        if row["path"].startswith("notebooks/"):
            if sha256_file(root / row["path"]) != row["sha256"]:
                raise ValueError(f"Historical notebook changed: {row['path']}")
            notebook_count += 1
    _, identity = load_development_dataset(root, dataset_manifest=MANIFEST)
    result = {"status": "complete", "historical_artifacts_verified": artifact_count,
        "historical_notebooks_verified": notebook_count, "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "reserved_test_loaded": False}
    write_json(root / DIRECTORY / "preservation_receipt.json", result)
    return result


def reports():
    summary = build_evaluation(ROOT)
    # Read the current notebook definitions when refreshing a long-lived queue.
    # A notebook revision must not be overwritten by definitions imported hours
    # earlier while the image worker was still training.
    subprocess.run([sys.executable, "-B", "scripts/build_dedup_workflow_notebooks.py", "--execute"],
                   cwd=ROOT, check=True)
    update("notebook_reports", "complete", evaluation_status=summary["status"],
           evaluation_output=summary["output_directory"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--wait-timeout", type=int, default=43200)
    args = parser.parse_args()
    if args.wait_timeout < 1:
        raise ValueError("Wait timeout must be positive.")
    with controller_lock():
        try:
            if args.report_only:
                reports()
                verify_preservation()
                update("report_refresh", "complete")
                return 0
            update("cleaning", "running")
            # Full current-byte/pixel verification is reused only when recorded
            # in this execution; report readers still verify manifest artifacts.
            receipt_path = ROOT / DIRECTORY / "cleaning/receipt.json"
            receipt = read_json(receipt_path) if receipt_path.exists() else {}
            if receipt.get("pixels_verified") != 6864:
                verify_cleaning(ROOT, verify_pixels=True)
            update("cleaning", "complete")
            command("metadata_suite", "scripts/run_dedup_metadata_suite.py", "--config", "configs/metadata_dedup_suite.yaml")
            reports()
            state = wait_images(args.wait_timeout)
            raw_path = ROOT / "results/image_d1_runs/b0_nested_dedup_v1/summary.json"
            raw = read_json(raw_path) if raw_path.exists() else {}
            uncertainty_complete = False
            if raw.get("status") == "complete" and raw.get("completed_outer_folds") == list(range(5)):
                command("clean_uncertainty", "scripts/run_dedup_uncertainty.py", "--config", "configs/image_uncertainty_dedup.yaml", "--resume")
                uncertainty_complete = True
            else:
                update("clean_uncertainty", "pending", reason="All five clean B0 checkpoints are required.")
            summary = reports()
            preservation = verify_preservation()
            complete = state.get("status") == "complete" and summary["comparison_complete"] and uncertainty_complete
            update("workflow", "complete" if complete else "incomplete", image_queue_state=state,
                preservation=preservation, uncertainty_complete=uncertainty_complete,
                reason=None if complete else "Upstream image work remains incomplete; existing deadlines and checkpoints are preserved.")
            return 0 if complete else 2
        except Exception as error:
            current = read_json(STATUS) if STATUS.exists() else {}
            update(current.get("phase", "workflow"), "incomplete", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
