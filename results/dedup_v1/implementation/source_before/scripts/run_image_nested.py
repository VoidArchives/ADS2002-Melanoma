"""Run the bounded five-fold B0 evaluation with a persistent budget watchdog.

  .venv/Scripts/python.exe scripts/run_image_nested.py --audit-only
  .venv/Scripts/python.exe scripts/run_image_nested.py --resume

An expired allocation can resume only with an explicit new --session-file.
The report notebook reads saved results; it never launches training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection import image_nested as stage


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/image_nested.yaml")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume verified epoch-boundary checkpoints (also default).")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--session-file", type=Path)
    parser.add_argument("--started-utc", help="Include implementation time in the first allocation; cannot alter existing clocks.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline-utc", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = stage.load_config(args.config)
    run_dir = ROOT / config["run"]["output_dir"] / config["run"]["run_id"]
    session_path = args.session_file.resolve() if args.session_file else run_dir / "session.json"
    if args.audit_only:
        prepared = stage.prepare_nested(ROOT, config)
        stage.persist_preparation(prepared)
        print(json.dumps({"images": len(prepared["frame"]), "patients": int(prepared["frame"].patient_id.nunique()),
                          "outer_folds": 5, "inner_folds": 3, "duplicate_crosspatient_groups": 0,
                          "test_set_loaded": False, "run_dir": str(run_dir)}, indent=2), flush=True)
        return 0
    if args.worker:
        if not args.deadline_utc:
            raise ValueError("Worker requires the supervisor's absolute deadline.")
        print("Auditing Subset images, patient groups and fresh outer-contained inner folds.", flush=True)
        prepared = stage.prepare_nested(ROOT, config)
        summary = stage.run_nested(ROOT, config, prepared=prepared, device=args.device,
                                   session_path=session_path, deadline_utc=args.deadline_utc)
        print(json.dumps({key: summary.get(key) for key in ("status", "completed_outer_folds", "mean_fold_average_precision", "stopping_reason")}, indent=2), flush=True)
        return 0 if summary["status"] == "complete" else 2
    if not session_path.is_relative_to(ROOT / "results/image_d1_runs"):
        raise ValueError("Session allocation must remain in image run outputs.")
    try:
        deadline = stage.claim_training_window(session_path, config, started_utc=args.started_utc)
    except TimeoutError as error:
        print(str(error), flush=True)
        return 2
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--resume", "--config", str(args.config.resolve()),
               "--device", args.device, "--session-file", str(session_path), "--deadline-utc", deadline.isoformat()]
    print(f"Watchdog cutoff {deadline.isoformat()}; reporting time is reserved separately.", flush=True)
    child = subprocess.Popen(command, cwd=ROOT)
    try:
        code = child.wait(timeout=max(0.1, (deadline - stage._utc_now()).total_seconds()))
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
        path = run_dir / "summary.json"
        summary = stage._read_json(path) if path.exists() else {"run_id": config["run"]["run_id"], "completed_outer_folds": []}
        summary.update({"status": "incomplete", "stopping_reason": "watchdog_deadline" if isinstance(error, subprocess.TimeoutExpired) else "user_interruption",
                        "checkpoint_recovery_required_on_resume": True})
        stage._atomic_json(path, summary)
        code = 2
    session = stage._read_json(session_path)
    session.update({"status": "training_finished" if code == 0 else "training_paused", "last_exit_utc": stage._utc_now().isoformat()})
    stage._atomic_json(session_path, session)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
