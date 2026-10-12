"""Bind completed human figure reviews to every current report and export."""
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.metadata_model import file_sha256

REPORT = ROOT / "results/metadata_nn_tuning/v2/report_verification"


if __name__ == "__main__":
    execution_path = REPORT / "execution_receipt.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution["status"] != "complete" or len(execution["notebooks"]) != 21:
        raise ValueError("All 21 report executions must be complete")
    inputs = [REPORT / "metadata_paired_visual_review.json", REPORT / "image_visual_inspection.json"]
    reviewed_notebooks, reviewed_figures = {}, {}
    for path in inputs:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete":
            raise ValueError("A visual-review component remains pending")
        for row in receipt["notebooks"]:
            relative = row.get("path", row.get("notebook"))
            if relative in reviewed_notebooks or file_sha256(ROOT / relative) != row["sha256"]:
                raise ValueError("Visual notebook identity is duplicate or stale")
            reviewed_notebooks[relative] = row["sha256"]
        figures = receipt.get("figures", []) or [fig for row in receipt["notebooks"] for fig in row.get("figures", [])]
        for fig in figures:
            relative = fig["path"]
            if relative in reviewed_figures or file_sha256(ROOT / relative) != fig["sha256"]:
                raise ValueError("Visual figure identity is duplicate or stale")
            reviewed_figures[relative] = fig["sha256"]
    expected_notebooks = {row["notebook"]: row["sha256"] for row in execution["notebooks"]}
    expected_figures = {fig for row in execution["notebooks"] for fig in row["figures"]}
    if reviewed_notebooks != expected_notebooks or set(reviewed_figures) != expected_figures:
        raise ValueError("Visual reviews do not cover the exact current reports and figures")
    result = dict(status="complete", schema_version=2, experiment_version=2, selection_metric="roc_auc",
                  reviewed_utc=datetime.now(timezone.utc).isoformat(), notebooks_reviewed=21,
                  figures_reviewed=len(reviewed_figures), notebook_sha256=reviewed_notebooks, figure_sha256=reviewed_figures,
                  source_reviews={p.relative_to(ROOT).as_posix(): file_sha256(p) for p in inputs},
                  execution_receipt_sha256=file_sha256(execution_path), source_sha256=file_sha256(Path(__file__)),
                  training_started=False, reserved_test_loaded=False, final_model_fitted=False)
    (REPORT / "visual_inspection_receipt.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "notebooks": 21, "figures": len(reviewed_figures)}, indent=2))
