"""Run the declared metadata-only NN tuning on original and clean development."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
from threading import Event

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from melanoma_detection.metadata_nn_tuning import run_tuning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metadata_nn_tuning.yaml")
    parser.add_argument("--condition", choices=["original", "dedup_v1"])
    parser.add_argument("--resume", action="store_true",
                        help="Resume verified completed phases; safe reuse is also the default.")
    arguments = parser.parse_args()
    stopped = Event()

    def request_stop(signum, frame):
        stopped.set()
        print("Stop requested; preserving the active fitting phase before exiting.", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    try:
        result = run_tuning(PROJECT_ROOT, arguments.config, condition=arguments.condition,
                            progress=lambda message: print(message, flush=True), stop_event=stopped)
    except InterruptedError as error:
        print(str(error), flush=True)
        return 130
    receipts = result if arguments.condition is None else {arguments.condition: result}
    print(json.dumps({name: {key: receipt[key] for key in (
        "status", "rows", "patients", "positive_images", "executed_fits", "reused_fits", "oof_path",
    )} for name, receipt in receipts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
