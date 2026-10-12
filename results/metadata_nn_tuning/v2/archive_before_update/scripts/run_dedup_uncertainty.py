"""Infer uncertainty from completed clean B0 outer refits; never trains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.dedup_uncertainty import run_uncertainty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/image_uncertainty_dedup.yaml")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Check a complete fold 0 only; not full OOF analysis.")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse completed uncertainty folds.")
    args = parser.parse_args()
    result = run_uncertainty(ROOT, args.config, device=args.device, smoke=args.smoke, resume=args.resume)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
