"""Fit the approved remaining metadata models on the frozen clean development set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from melanoma_detection.dedup_metadata_suite import run_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metadata_dedup_suite.yaml")
    arguments = parser.parse_args()
    receipt = run_suite(PROJECT_ROOT, arguments.config)
    print(json.dumps({key: receipt[key] for key in (
        "status", "rows", "patients", "malignant_images", "executed_fits", "reused_fits",
        "completed_fold_seconds", "metrics_path",
    )}, indent=2))


if __name__ == "__main__":
    main()
