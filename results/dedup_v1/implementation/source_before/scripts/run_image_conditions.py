"""Explicit runner for reviewed fresh d1 B0 conditions; notebooks are report-only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.image_conditions import (
    begin_optional_setup, ensure_session, load_config, persist_preparation, prepare_conditions,
    record_watchdog_pause, run_conditions,
)


def brief(result):
    return {key: result.get(key) for key in ("status", "comparison_complete", "comparisons", "stopping_reason")} | {
        "conditions": {key: {field: record.get(field) for field in
            ("status", "completed_epochs", "best_epoch", "best_metrics", "epoch12_metrics", "stopping_reason")}
            for key, record in result["conditions"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/image_augmentation.yaml")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--condition", choices=("AB", "A", "B", "C"), default="AB")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--session-file", type=Path)
    parser.add_argument("--admit-lighting-setup", action="store_true", help="Claim optional ten-minute dependency/QA window; do not install or train.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.audit_only and args.predict_only:
        parser.error("Choose audit-only or prediction-only.")
    config = load_config(args.config)
    prepared = prepare_conditions(ROOT, config)
    folder = persist_preparation(prepared)
    if args.audit_only:
        print(json.dumps(prepared["split_audit"], indent=2), flush=True)
        return 0
    if args.admit_lighting_setup:
        print(json.dumps(begin_optional_setup(prepared, args.session_file), indent=2), flush=True)
        return 0
    ids = ("A", "B") if args.condition == "AB" else (args.condition,)
    if args.worker or args.predict_only:
        result = run_conditions(ROOT, config, prepared=prepared, condition_ids=ids,
                                device=args.device, predict_only=args.predict_only, session_path=args.session_file)
        print(json.dumps(brief(result), indent=2), flush=True)
        return 0 if result["comparison_complete"] else 2
    # The supervisor owns the hard wall clock. If an epoch exceeds its estimate,
    # killing the worker leaves the previous atomic epoch checkpoint intact.
    _, session = ensure_session(prepared, args.session_file)
    cutoff = datetime.fromisoformat(session["training_cutoff_utc"])
    remaining = max(0, (cutoff - datetime.now(timezone.utc)).total_seconds())
    if remaining <= 0:
        result = run_conditions(ROOT, config, prepared=prepared, condition_ids=ids,
                                device=args.device, session_path=args.session_file)
        print(json.dumps(brief(result), indent=2), flush=True)
        return 2
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str(args.config),
               "--condition", args.condition, "--device", args.device]
    if args.session_file:
        command += ["--session-file", str(args.session_file)]
    process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    try:
        return process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        result = record_watchdog_pause(folder)
        print(json.dumps(brief(result), indent=2), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
