"""Verify completed v2 resume compatibility with every fitting entrypoint disabled."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection import metadata_tuning_v2 as m
from melanoma_detection.metadata_model import file_sha256


def fail_fit(*args, **kwargs):
    raise AssertionError("A completed compatible resume attempted fitting")


if __name__ == "__main__":
    paths = {c: m.condition_directory(ROOT, c, "roc_auc_grid36_v1", experiment_version=2) / "receipt.json" for c in ["original", "dedup_v1"]}
    before = {c: file_sha256(p) for c, p in paths.items()}
    m.fit_nn_pair = m.fit_network = m._classical_candidates = fail_fit
    receipts = m.run_tuning(ROOT, "configs/metadata_nn_tuning_roc_auc.yaml", resume=True)
    after = {c: file_sha256(p) for c, p in paths.items()}
    if before != after or any(r["status"] != "complete" for r in receipts.values()):
        raise AssertionError("Completed resume changed its immutable receipt")
    receipt = {"status": "complete", "experiment_version": 2, "selection_metric": "roc_auc", "executed_fits_this_check": 0,
               "fitting_entrypoints_disabled": True, "completion_receipts_unchanged": before,
               "source_sha256": file_sha256(Path(__file__)), "test_set_loaded": False, "final_model_fitted": False}
    output = ROOT / "results/metadata_nn_tuning/v2/report_verification/completed_resume_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
