"""Refresh read-only robustness/downstream notebooks without touching old results."""
from pathlib import Path
import json
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.metadata_nn_tuning import DEFAULT_RUN_ID, condition_directory
from melanoma_detection.metadata_nn_tuning_reporting import metadata_nn_assessment_sections as assessment_sections

ARCHIVE = ROOT / "results/metadata_nn_tuning/v1/archive_before_update"


def read_only_execution_text(notebook):
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace(
                "Re-executing later cells refits metadata models and replaces their named development outputs.",
                "Executing these cells reads completed artifacts and writes new reports. The separate runners "
                "own model fitting; historical predictions and decisions remain unchanged.")


def append_assessment(notebook, condition):
    read_only_execution_text(notebook)
    for kind, markdown, code in assessment_sections(condition):
        notebook.cells.append(nbformat.v4.new_markdown_cell(markdown) if kind == "__markdown__"
                              else nbformat.v4.new_code_cell(code))
    notebook.metadata["metadata_nn_tuning"] = {
        "condition": condition, "run_id": DEFAULT_RUN_ID, "training_on_execute": False,
        "reserved_test_loaded": False, "historical_source": str(ARCHIVE.relative_to(ROOT)),
    }
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return notebook


def original_robustness():
    relative = "notebooks/Subset/08_metadata_robustness.ipynb"
    notebook = nbformat.read(ARCHIVE / relative, as_version=4)
    notebook.cells[0].source = notebook.cells[0].source.replace(
        "[07_metadata_tuning.ipynb](07_metadata_tuning.ipynb) retains the fixed spline. We read the network predictions "
        "from [06_metadata_neural_network.ipynb](06_metadata_neural_network.ipynb) and assess that choice using metadata, "
        "the information alongside each image.",
        "[07_metadata_tuning.ipynb](07_metadata_tuning.ipynb) now reports tuning for four model families. "
        "The historical fixed-model sections below use saved network predictions from the archived stage 06 workflow. "
        "[06_metadata_neural_network.ipynb](06_metadata_neural_network.ipynb) reports the current fixed NN comparison. "
        "The appended assessment compares the fresh, versioned original fixed and tuned NN runs.")
    notebook.cells[2].source = notebook.cells[2].source.replace(
        "Use the saved three-run network predictions from notebook 06",
        "Use the historical three-run network predictions from the archived stage 06 workflow")
    notebook.cells[0].source += ("\n\n**Updated execution boundary:** the original fixed-model sections below reproduce "
        "saved development predictions, including their historical pooled summaries as secondary diagnostics. "
        "The added NN tuning assessment reports primary mean-fold comparisons separately. Executing this "
        "report starts no model training, creates no final bundle and does not read the reserved test.")
    setup = notebook.cells[3].source
    setup = setup.replace("FEATURES, SPECIFICATION, evaluate_outer_fold, file_sha256, fit_final_bundle,",
                          "FEATURES, SPECIFICATION, file_sha256,")
    setup = setup.replace('OUTPUT_ROOT = PROJECT_ROOT / "results"',
        'from melanoma_detection.metadata_nn_tuning import condition_directory, DEFAULT_RUN_ID\n'
        'OUTPUT_ROOT = condition_directory(PROJECT_ROOT, "original", DEFAULT_RUN_ID) / "report" / "08_metadata_robustness"')
    setup = setup.replace('MODEL_DIR = PROJECT_ROOT / "models/metadata_d1"', 'MODEL_DIR = OUTPUT_ROOT / "boundary_only"')
    notebook.cells[3].source = setup
    old = notebook.cells[7].source
    start = old.index('outer_predictions = pd.concat')
    end = old.index('if not split_audit.patient_overlap')
    loader = '''# Reuse immutable historical fixed-spline calibration and training-selected decisions.
outer_predictions = pd.read_csv(PROJECT_ROOT / "results/predictions/metadata_robustness_oof_d1.csv", float_precision="round_trip")
outer_predictions = outer_predictions.set_index("image_name").loc[train.image_name].reset_index()
split_audit = pd.read_csv(PROJECT_ROOT / "results/tables/metadata_robustness_split_audit_d1.csv")
for column in ["patient_id", "fold", "target"]:
    if not np.array_equal(outer_predictions[column], train[column]):
        raise ValueError("Historical robustness prediction identity changed.")
'''
    notebook.cells[7].source = loader + old[end:]
    notebook.cells[16].source = ("### 7. Record the final-fit boundary\n\n"
        "The original workflow previously created an all-development spline bundle. That historical artifact "
        "is preserved, while this updated report keeps fitting outside notebook execution. A newly tuned NN "
        "has its own preprocessing and predictions; the spline's probability calibration and recall cutoff "
        "cannot be attached to it. Choosing a final procedure requires an explicit decision about its settings, "
        "calibration and threshold before any independent test is opened.\n\n"
        "The cell records this boundary only. The current experiment compares patient-held-out development "
        "predictions and produces no all-development model or reserved-test evaluation.")
    notebook.cells[17].source = 'print("Report only: no final all-development fit or reserved-test evaluation is performed.")'
    notebook.cells[19].source = notebook.cells[19].source.replace(
        'print("Primary model remains the raw spline. The final test is still closed.")',
        'print("Historical fixed-model diagnostics; the added tuned-NN comparison is reported below. The final test remains closed.")')
    for cell in notebook.cells:
        cell.source = cell.source.replace("NN mean of three", "Fixed NN mean of three")
    nbformat.write(append_assessment(notebook, "original"), ROOT / relative)


def clean_report(filename):
    relative = "notebooks/Subset_dedup/" + filename
    notebook = nbformat.read(ARCHIVE / relative, as_version=4)
    notebook.cells[0].source += (
        "\n\n**Scope clarification:** the preserved fixed-model robustness, calibration and decision sections "
        "use their verified historical clean artifacts. The additional NN tuning appendix compares the new "
        "versioned tuning predictions with the same clean fixed NN and spline baselines. Its findings do not "
        "transfer the existing calibration maps or cutoffs to the tuned network.")
    for cell in notebook.cells:
        cell.source = cell.source.replace("NN mean of three", "Fixed NN mean of three")
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace(
                "[07_metadata_tuning.ipynb](07_metadata_tuning.ipynb) retains the fixed spline. We read the network predictions "
                "from [06_metadata_neural_network.ipynb](06_metadata_neural_network.ipynb) and assess that choice using metadata, "
                "the information alongside each image.",
                "[07_metadata_tuning.ipynb](07_metadata_tuning.ipynb) now reports tuning for four model families. "
                "The preserved fixed-model sections below use verified historical clean NN and spline predictions. "
                "[06_metadata_neural_network.ipynb](06_metadata_neural_network.ipynb) reports the current fixed NN comparison, "
                "using the same verified clean fixed-NN artifacts. The appended assessment evaluates the new tuned NN separately.")
            cell.source = cell.source.replace(
                "Use the saved three-run network predictions from notebook 06",
                "Use the verified historical clean three-run network predictions")
            cell.source = cell.source.replace("prescribed NN mean", "prescribed fixed NN mean")
        if cell.cell_type != "code":
            continue
        cell.source = cell.source.replace(
            'FIGURE_DIR = ROOT / DIRECTORY / "figures" / "template_v3" / NOTEBOOK_STAGE',
            'from melanoma_detection.metadata_nn_tuning import condition_directory, DEFAULT_RUN_ID\n'
            'NN_REPORT = condition_directory(ROOT, "dedup_v1", DEFAULT_RUN_ID) / "report" / NOTEBOOK_STAGE\n'
            'FIGURE_DIR = NN_REPORT / "figures"')
        cell.source = cell.source.replace(
            'REPORT_DIR = ROOT / DIRECTORY / "metadata_depth" / NOTEBOOK_STAGE',
            'REPORT_DIR = NN_REPORT / "tables"')
        cell.source = cell.source.replace(
            'REPORT_DIR = ROOT / "results/dedup_v1/clean_workflow/metadata_depth" / NOTEBOOK_STAGE',
            'REPORT_DIR = NN_REPORT / "tables"')
    nbformat.write(append_assessment(notebook, "dedup_v1"), ROOT / relative)


def main():
    original_robustness()
    for filename in ["08_metadata_robustness.ipynb", "12_combined_evaluation.ipynb"]:
        clean_report(filename)
    relative = "notebooks/Subset/14_image_nested_evaluation.ipynb"
    original = nbformat.read(ARCHIVE / relative, as_version=4)
    original.cells[15].source += ("\n\nThe saved image-versus-metadata comparisons in this section use the historical fixed "
        "NN ensemble and fixed spline. They were not fitted or recalculated for the new tuned NN. "
        "The additional assessment below identifies the new versioned fixed/tuned NN experiment separately; "
        "historical image calibration and decision cutoffs remain attached to their original procedures.")
    nbformat.write(append_assessment(original, "original"), ROOT / relative)
    print("Updated original robustness/image comparison and clean robustness/combined evaluation.")


if __name__ == "__main__":
    main()
