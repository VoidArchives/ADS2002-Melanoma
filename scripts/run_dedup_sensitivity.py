"""Filter verified saved development OOF predictions; never start training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.dedup_sensitivity import run_sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/subset/processed/dedup_v1/dataset_manifest.json")
    parser.add_argument("--run-id", default="saved_oof_v2")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-image-run")
    parser.add_argument("--clean-metadata-receipt")
    parser.add_argument("--clean-calibration-run")
    args = parser.parse_args()
    receipt = run_sensitivity(ROOT, args.manifest, run_id=args.run_id, n_bootstrap=args.bootstrap,
        seed=args.seed, clean_image_run=args.clean_image_run, clean_metadata_receipt=args.clean_metadata_receipt,
        clean_calibration_run=args.clean_calibration_run)
    print(json.dumps({"status": receipt["status"], "run_id": receipt["run_id"],
        "counts": receipt["summary"]["counts"], "valid_bootstrap_draws": receipt["summary"]["bootstrap_valid"],
        "training_started": False, "reserved_test_loaded": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
