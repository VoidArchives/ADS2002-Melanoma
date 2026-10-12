"""Accept all 21 ROC-first reports against execution, archived prose and visual receipts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verify_metadata_nn_tuning_reports import notebook_checks, execution_checks, visual_checks, headings, sha256
from execute_metadata_roc_reports import NOTEBOOKS
from melanoma_detection.metadata_roc_archive import original_snapshot, verify_preservation

REPORT = Path("results/metadata_nn_tuning/v2/report_verification")
DECLARED_CLEAN_ADDITIONS = {
    "04": {"Additional deduplicated-workflow diagnostics", "Added diagnostic: 7. Inspect saved logistic associations and state the limits"},
    "05": {"Additional deduplicated-workflow diagnostics", "Added diagnostic: 7. Inspect tree importances with the proper cautions"},
    "08": {"Additional deduplicated-workflow diagnostics", "Added diagnostic: 3. Compare probability error and equal-patient weighting",
           "Added diagnostic: 5. Assess frozen cutoffs on outer patients", "Added diagnostic: 6. Read detection and review-workload intervals",
           "Added diagnostic: 7. Check age, sex, site and site availability with support"},
}


def verify():
    notebooks, rows, coverage = {}, [], []
    for relative in NOTEBOOKS:
        notebook, row = notebook_checks(ROOT, relative)
        rows.append(row)
        notebooks[relative] = notebook
        source = original_snapshot(ROOT, relative)
        archived = nbformat.read(source, as_version=4)
        before = sum(len(cell.source.split()) for cell in archived.cells if cell.cell_type == "markdown")
        after = sum(len(cell.source.split()) for cell in notebook.cells if cell.cell_type == "markdown")
        errors = []
        if after < .9 * before:
            errors.append("Markdown coverage is below 90% of the pre-ROC notebook")
        # Existing headings may change a training verb to reflect artifact-only reporting.
        oldlevels = [level for level, _ in headings(archived)]
        newlevels = [level for level, _ in headings(notebook)]
        position = 0
        for level in oldlevels:
            try:
                position = newlevels.index(level, position) + 1
            except ValueError:
                errors.append("Archived heading hierarchy was shortened/reordered")
                break
        coverage.append(dict(notebook=relative, archived_sha256=sha256(source), before_words=before,
                             after_words=after, ratio=after / before if before else 1, errors=errors))
    for stage in ["04", "05", "06", "07", "08"]:
        keys = [next(key for key in NOTEBOOKS if key.startswith(f"notebooks/{collection}/{stage}_"))
                for collection in ["Subset", "Subset_dedup"]]
        original = headings(notebooks[keys[0]])[1:]
        clean = headings(notebooks[keys[1]])[1:]
        additions = DECLARED_CLEAN_ADDITIONS.get(stage, set())
        # These precisely named, labelled diagnostics already existed in the clean workflow.
        # Preserve their explanatory depth while requiring an identical shared main hierarchy.
        common = [(level, text) for level, text in clean if text not in additions]
        if original != common:
            rows[NOTEBOOKS.index(keys[0])]["errors"].append(f"Stage {stage} original/clean shared headings differ")
        coverage[NOTEBOOKS.index(keys[1])]["declared_clean_scope_additions"] = [text for _, text in clean if text in additions]
    execution = execution_checks(ROOT, rows, ROOT / REPORT / "execution_receipt.json", NOTEBOOKS)
    visual = visual_checks(ROOT, rows, ROOT / REPORT / "visual_inspection_receipt.json")
    receipt = json.loads((ROOT / REPORT / "execution_receipt.json").read_text(encoding="utf-8"))
    figure_hashes = json.loads((ROOT / REPORT / "visual_inspection_receipt.json").read_text(encoding="utf-8")).get("figure_sha256", {})
    errors = [f"{row['notebook']}: {error}" for row in rows + coverage for error in row["errors"]]
    errors += execution.get("errors", []) + visual.get("errors", [])
    for row in execution["notebooks"]:
        errors += [f"{row['notebook']}: {error}" for error in row["errors"]]
    for row in receipt["notebooks"]:
        for relative in row["figures"]:
            if figure_hashes.get(relative) != sha256(ROOT / relative):
                errors.append(f"Figure visual fingerprint absent/stale: {relative}")
    complete = not errors and execution["status"] == visual["status"] == "complete"
    result = dict(status="complete" if complete else "failed", schema_version=2, experiment_version=2,
                  selection_metric="roc_auc", verified_utc=datetime.now(timezone.utc).isoformat(),
                  verification_source_sha256=sha256(Path(__file__)), notebooks=rows, coverage=coverage,
                  errors=errors, execution=execution, visual_inspection=visual,
                  preservation=verify_preservation(ROOT), training_started=False, reserved_test_loaded=False)
    (ROOT / REPORT / "report_acceptance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    parser.parse_args()
    result = verify()
    print(json.dumps(dict(status=result["status"], notebooks=len(result["notebooks"]), errors=result["errors"],
                          historical_files_verified=result["preservation"]), indent=2))
    raise SystemExit(0 if result["status"] == "complete" else 1)
