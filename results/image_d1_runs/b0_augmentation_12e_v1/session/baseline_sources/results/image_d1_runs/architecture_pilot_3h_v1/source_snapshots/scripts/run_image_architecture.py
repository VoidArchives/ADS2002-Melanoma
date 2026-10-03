"""Bounded architecture pilot with a hard process watchdog.

Examples:
  .venv/Scripts/python.exe scripts/run_image_architecture.py --audit-only
  .venv/Scripts/python.exe scripts/run_image_architecture.py --resume
  .venv/Scripts/python.exe scripts/run_image_architecture.py --predict-only

The default session has a fixed three-hour deadline. Resuming training in a
later session requires an explicit new --session-file; the old clock never resets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection import image_architecture as stage


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/image_architecture_pilot.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--predict-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume compatible checkpoints; also the default.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="CPU is explicit and intended for tests.")
    parser.add_argument("--session-file", type=Path, help="Explicit session allocation in results/image_d1_runs.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline-utc", help=argparse.SUPPRESS)
    return parser


def _print_summary(summary, output):
    print(json.dumps({key: summary.get(key) for key in
          ("status", "comparison_complete", "matched_recipe_slots", "unmatched_completed_trials", "stopping_reason")}, indent=2), flush=True)
    print(f"Outputs: {output}", flush=True)


def main():
    args = _parser().parse_args()
    config = stage.load_config(args.config)
    print("Auditing d1 images and the exact historical B0 memberships; no reserved test is read.", flush=True)
    prepared = stage.prepare_architecture_pilot(PROJECT_ROOT, config, audit_images=True)
    print(json.dumps(prepared["split_audit"], indent=2), flush=True)
    output = stage.persist_architecture_preparation(prepared)
    if args.audit_only:
        print(f"Architecture audit complete: {output}", flush=True)
        return 0
    if args.predict_only:
        summary = stage.run_architecture_pilot(PROJECT_ROOT, config, prepared=prepared,
                    predict_only=True, device=args.device, session_path=args.session_file)
        _print_summary(summary, output)
        return 0 if summary["status"] == "complete" else 2
    session_path = stage._session_path(PROJECT_ROOT, config, args.session_file)
    if args.worker:
        if not args.deadline_utc:
            raise ValueError("Worker requires its parent watchdog's deadline.")
        summary = stage.run_architecture_pilot(PROJECT_ROOT, config, prepared=prepared,
                    device=args.device, resume=True, session_path=session_path,
                    deadline_utc=args.deadline_utc)
        _print_summary(summary, output)
        return 0 if summary["status"] == "complete" else 2
    try:
        deadline = stage.claim_training_window(session_path)
    except TimeoutError as error:
        print(str(error), flush=True)
        return 2
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--resume",
               "--config", str(args.config.resolve()), "--device", args.device,
               "--session-file", str(session_path), "--deadline-utc", deadline.isoformat()]
    print(f"Watchdog deadline: {deadline.isoformat()}; the allowance cannot silently reset.", flush=True)
    child = subprocess.Popen(command, cwd=PROJECT_ROOT)
    timed_out = False
    try:
        code = child.wait(timeout=max(0.1, (deadline - stage._utc_now()).total_seconds()))
    except subprocess.TimeoutExpired:
        timed_out = True
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
        summary = stage.record_watchdog_pause(output)
        _print_summary(summary, output)
        code = 2
    except KeyboardInterrupt:
        child.terminate()
        child.wait(timeout=10)
        stage.record_watchdog_pause(output, stopping_reason="user_interruption")
        code = 130
    finally:
        session = stage._read_json(session_path)
        session.update({"status": "training_paused" if timed_out or code != 0 else "training_finished",
                        "last_training_exit_utc": stage._utc_now().isoformat(),
                        "training_wall_seconds_since_first_start": max(0, (stage._utc_now() - stage._utc(session["training_started_utc"])).total_seconds())})
        stage._atomic_json(session_path, session)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
