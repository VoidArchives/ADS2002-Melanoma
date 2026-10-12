"""Run declared clean inner pilots after the completed clean baseline.

  .venv/Scripts/python.exe scripts/run_dedup_image_pilots.py
  .venv/Scripts/python.exe scripts/run_dedup_image_pilots.py --stage duration --audit-only
  .venv/Scripts/python.exe scripts/run_dedup_image_pilots.py --stage duration
  .venv/Scripts/python.exe scripts/run_dedup_image_pilots.py --stage challenger

Each stage has its own persistent budget; resume does not reset its clock.
An explicit new --session-file may allocate a new window to retained checkpoints.
"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection import dedup_image_pilots as pilots
from melanoma_detection.image_nested import claim_training_window


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/image_dedup_pilots.yaml")
    parser.add_argument("--stage", choices=("duration", "challenger", "all"), default="all")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--session-file", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline-utc", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = pilots.load_config(args.config)
    if args.stage == "all":
        if args.worker or args.session_file:
            raise ValueError("All stages use their own workers and session files; choose a stage for a new allocation.")
        if args.audit_only:
            prepared = pilots.prepare_pilot(ROOT, config, "duration")
            print(json.dumps({"status": "prepared", "stage_order": ["duration", "challenger"],
                              "images": len(prepared["pool"]), "training_started": False,
                              "challenger_prerequisite": "completed matching duration pilot",
                              "reserved_test_loaded": False}, indent=2))
            return 0
        for stage in ("duration", "challenger"):
            command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
                       "--config", str(args.config.resolve()), "--device", args.device]
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                return result.returncode
        return 0
    if args.worker:
        result = pilots.run_pilot(ROOT, config, args.stage, device=args.device,
                                  session_file=args.session_file, deadline_utc=args.deadline_utc)
        print(json.dumps({key: result.get(key) for key in ("status", "stage", "completed_inner_folds", "stopping_reason")}, indent=2))
        return 0 if result["status"] == "complete" else 2
    prepared = pilots.prepare_pilot(ROOT, config, args.stage)
    if args.audit_only:
        print(json.dumps({"status": "prepared", "stage": args.stage, "images": len(prepared["pool"]),
                          "patients": int(prepared["pool"].patient_id.nunique()), "training_started": False,
                          "reserved_test_loaded": False, "outer_predictions_generated": False}, indent=2))
        return 0
    run_dir = prepared["run_dir"]
    if (run_dir / "summary.json").exists() and pilots._read(run_dir / "summary.json").get("status") == "complete":
        pilots.load_complete_pilot(run_dir)
        print(json.dumps({"status": "complete", "stage": args.stage, "already_complete": True}, indent=2))
        return 0
    session = args.session_file.resolve() if args.session_file else run_dir / "session.json"
    if not session.is_relative_to(run_dir.resolve()):
        raise ValueError("Pilot session must remain within its dedicated run.")
    budget = {"run": {"run_id": config[args.stage]["run_id"]},
              "budget": {"session_minutes": config[args.stage]["session_minutes"],
                         "reporting_reserve_minutes": config["protocol"]["reporting_reserve_minutes"]}}
    try:
        cutoff = claim_training_window(session, budget)
    except TimeoutError as error:
        print(str(error), flush=True)
        return 2
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--stage", args.stage,
               "--config", str(args.config.resolve()), "--device", args.device,
               "--session-file", str(session), "--deadline-utc", cutoff.isoformat()]
    worker = subprocess.Popen(command, cwd=ROOT)
    try:
        return worker.wait(timeout=max(1, (cutoff - datetime.now(timezone.utc)).total_seconds()))
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as interruption:
        worker.terminate()
        try:
            worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        reason = "user_interruption" if isinstance(interruption, KeyboardInterrupt) else "persistent_training_deadline"
        pilots._atomic_json(run_dir / "supervisor_stop.json", {"status": "incomplete", "reason": reason,
                                                               "training_cutoff_utc": cutoff.isoformat()})
        message = "Pilot interrupted" if reason == "user_interruption" else "Pilot reached its persisted budget"
        print(message + "; verified epoch checkpoints remain resumable.", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
