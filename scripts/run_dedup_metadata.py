"""Run the fixed, patient-safe CPU metadata replication on the frozen variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection.dedup_metadata import run_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metadata_dedup.yaml")
    args = parser.parse_args()
    receipt = run_metadata(PROJECT_ROOT, args.config)
    print(json.dumps({key: receipt[key] for key in ("status", "rows", "patients", "fit_counts")}, indent=2))


if __name__ == "__main__":
    main()
