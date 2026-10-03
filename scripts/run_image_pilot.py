"""Run or resume the bounded d1 B0 pilot; later stages are declarations only.

Examples from the project root:
  .venv/Scripts/python.exe scripts/run_image_pilot.py --audit-only
  .venv/Scripts/python.exe scripts/run_image_pilot.py --resume
  .venv/Scripts/python.exe scripts/run_image_pilot.py --predict-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection.image_pilot import load_config, persist_preparation, prepare_pilot, run_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/image_model.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true", help="Decode/hash JPEGs and persist memberships; do not train.")
    modes.add_argument("--predict-only", action="store_true", help="Use compatible best checkpoints; preserve training history.")
    parser.add_argument("--resume", action="store_true", help="Resume compatible checkpoints (default behavior).")
    parser.add_argument("--device", default="cuda", help="cuda (default) or explicit cpu; no implicit CPU fallback.")
    args = parser.parse_args()
    config = load_config(args.config)
    print("Auditing d1 images and patient splits; the reserved test is not read.", flush=True)
    prepared = prepare_pilot(PROJECT_ROOT, config, audit_images=True)
    print(json.dumps(prepared["split_audit"], indent=2), flush=True)
    if args.audit_only:
        run_dir = persist_preparation(prepared)
        print(f"Audit complete: {run_dir}", flush=True)
        return 0
    summary = run_pilot(PROJECT_ROOT, config, resume=True, predict_only=args.predict_only,
                        prepared=prepared, device=args.device)
    print(json.dumps({key: summary.get(key) for key in ("status", "comparison_complete", "selected_recipe", "recommendation", "missing_recipe_checkpoints")}, indent=2), flush=True)
    print(f"Outputs: {prepared['run_dir']}", flush=True)
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
