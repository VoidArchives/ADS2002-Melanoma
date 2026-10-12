"""Run the explicitly versioned, verified-reuse metadata selection procedure."""
import argparse
import json
from pathlib import Path
import signal
import sys
from threading import Event

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.metadata_tuning_v2 import plan_run, run_tuning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metadata_nn_tuning_roc_auc.yaml")
    parser.add_argument("--condition", choices=("original", "dedup_v1"))
    parser.add_argument("--plan", action="store_true", help="Verify immutable inputs/cache without fitting or creating run outputs")
    parser.add_argument("--resume", action="store_true", help="Confirm safe reuse of compatible sealed v2 phases")
    args = parser.parse_args()
    stop = Event()
    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: stop.set())
    if args.plan:
        result = plan_run(ROOT, args.config)
    else:
        result = run_tuning(ROOT, args.config, condition=args.condition, stop_event=stop)
        if args.condition:
            result = {key: result[key] for key in ("status", "condition", "experiment_version", "selection_metric", "rows",
                                                  "executed_fits", "external_reused_fits", "resumed_fits", "rescoring_operations")}
        else:
            result = {condition: {key: receipt[key] for key in ("status", "rows", "executed_fits", "external_reused_fits")}
                      for condition, receipt in result.items()}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
