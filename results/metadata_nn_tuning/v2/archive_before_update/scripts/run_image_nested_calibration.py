"""Run training-only image recalibration inside the original five-hour clock.

  .venv/Scripts/python.exe scripts/run_image_nested_calibration.py --prepare-only
  .venv/Scripts/python.exe scripts/run_image_nested_calibration.py --resume

No new session can be created. Start this stage after the raw GPU worker ends.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection import image_nested_calibration as stage


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/image_nested_calibration.yaml")
    parser.add_argument("--declare-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume verified epoch-boundary state (also default).")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline-utc", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = stage.load_config(args.config)
    if args.declare_only:
        receipt = stage.declare_protocol(ROOT, config, args.config)
        print(f"Calibration protocol declared at {receipt['declared_utc']}.", flush=True)
        return 0
    prepared = stage.prepare_calibration(ROOT, config)
    stage.persist_preparation(prepared)
    if args.prepare_only:
        print(f"Calibration preparation verified: {prepared['run_dir']}; original clock preserved.", flush=True)
        return 0
    if args.worker:
        if not args.deadline_utc:
            raise ValueError("Calibration worker requires its supervisor's absolute shared cutoff.")
        summary = stage.run_calibration(ROOT, config, prepared=prepared, device=args.device, deadline_utc=args.deadline_utc)
        print(f"Calibration {summary['status']}; completed folds {summary['completed_outer_folds']}.", flush=True)
        return 0 if summary["status"] == "complete" else 2
    try:
        deadline = stage.shared_deadline(prepared)
    except TimeoutError:
        summary = stage.run_calibration(ROOT, config, prepared=prepared, device=args.device)
        print(f"Shared allocation exhausted; calibration {summary['status']}.", flush=True)
        return 0 if summary["status"] == "complete" else 2
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--resume", "--config", str(args.config.resolve()),
               "--device", args.device, "--deadline-utc", deadline.isoformat()]
    print(f"Calibration watchdog shares raw cutoff {deadline.isoformat()}; no new allocation.", flush=True)
    child = subprocess.Popen(command, cwd=ROOT)
    try:
        return child.wait(timeout=max(0.1, (deadline - stage._utc_now()).total_seconds()))
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
        path = prepared["run_dir"] / "summary.json"
        summary = stage._read_json(path) if path.exists() else {"run_id": config["run"]["run_id"], "completed_outer_folds": []}
        summary.update({"status": "incomplete", "calibration_fitted": False, "threshold_fitted": False,
                        "stopping_reason": "shared_watchdog_deadline" if isinstance(error, subprocess.TimeoutExpired) else "user_interruption",
                        "checkpoint_recovery_required_on_resume": True})
        stage._atomic_json(path, summary)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
