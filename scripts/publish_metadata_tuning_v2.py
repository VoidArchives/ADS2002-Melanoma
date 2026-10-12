"""Publish separate v2 latest pointers only after independent five-fold verification."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.metadata_tuning_v2 import load_tuning_outputs
from melanoma_detection.metadata_model import file_sha256


def main():
    pointers = []
    for condition in ["original", "dedup_v1"]:
        bundle = load_tuning_outputs(ROOT, condition, experiment_version=2, run_id="roc_auc_grid36_v1", expected_selection_metric="roc_auc")
        if bundle["status"] != "complete":
            raise ValueError("Both verified five-fold ROC experiments are required")
        verification = bundle["directory"] / "report/independent_verification.json"
        checked = json.loads(verification.read_text(encoding="utf-8"))
        completion_hash = file_sha256(bundle["directory"] / "receipt.json")
        if checked.get("status") != "complete" or checked.get("completion_receipt_sha256") != completion_hash or checked.get("selection_metric") != "roc_auc":
            raise ValueError("Independent result verification is absent or stale")
        pointer = {"experiment_version": 2, "run_id": "roc_auc_grid36_v1", "condition": condition,
                   "selection_metric": "roc_auc", "status": "complete", "output_root": bundle["receipt"]["output_root"],
                   "completion_receipt_sha256": completion_hash, "independent_verification_sha256": file_sha256(verification),
                   "publication_source_sha256": file_sha256(Path(__file__))}
        pointers.append((bundle["directory"].parent / "latest.json", pointer))
    for destination, pointer in pointers:
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    print(json.dumps({"status": "complete", "latest_pointers": [p.relative_to(ROOT).as_posix() for p, _ in pointers]}, indent=2))


if __name__ == "__main__":
    main()
