"""Verify clean development data and report saved models; never start training."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.dedup_workflow import build_evaluation, verify_cleaning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--skip-cleaning", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if not args.skip_cleaning:
        receipt = verify_cleaning(ROOT, verify_pixels=args.verify_images)
        print(json.dumps({key: receipt[key] for key in ("status", "counts", "pixels_verified")}), flush=True)
    result = build_evaluation(ROOT, replicates=args.bootstrap_replicates)
    print(json.dumps({key: result[key] for key in ("status", "stages", "model_names", "output_directory")}), flush=True)


if __name__ == "__main__":
    main()
