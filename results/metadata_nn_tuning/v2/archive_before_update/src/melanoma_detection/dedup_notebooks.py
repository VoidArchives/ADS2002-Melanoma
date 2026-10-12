"""Build report notebooks around one frozen development dataset identity."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
import os
import re

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

from .development_dataset import sha256_file, stable_sha256
from .dedup_workflow import DIRECTORY, MANIFEST, read_json, write_json

ORIGINAL_SOURCES = {
    "01": ["01_data_cleaning.ipynb"], "02": ["02_data_exploration.ipynb"],
    "03": ["03_patient_splitting.ipynb"], "04": ["04_dummy_logistic_baseline.ipynb"],
    "05": ["05_metadata_models.ipynb"], "06": ["06_metadata_neural_network.ipynb"],
    "07": ["07_metadata_tuning.ipynb"], "08": ["08_metadata_robustness.ipynb"],
    "09": ["09_image_b0_pilot.ipynb", "14_image_nested_evaluation.ipynb"],
    "10": ["10_image_architecture_comparison.ipynb", "11_image_augmentation.ipynb", "12_image_lighting.ipynb"],
    "11": ["13_image_uncertainty.ipynb"], "12": ["08_metadata_robustness.ipynb", "14_image_nested_evaluation.ipynb"],
}

NOTEBOOK_DIRECTORY = "notebooks/Subset_dedup"
SETUP = '''from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcdefaults()
from IPython.display import display, Markdown, Image
from melanoma_detection.development_dataset import load_development_dataset, load_preserved_inner_memberships
from melanoma_detection.dedup_workflow import load_context, latest_evaluation, load_clean_scores, load_image_report, load_uncertainty_report, read_json, project_path, DIRECTORY, METADATA_SUITE, UNCERTAINTY
get_ipython().run_line_magic("matplotlib", "inline")
MANIFEST = "data/subset/processed/dedup_v1/dataset_manifest.json"
data, dataset_identity = load_development_dataset(ROOT, dataset_manifest=MANIFEST)
context = load_context(ROOT)
if NOTEBOOK_STAGE.startswith(("06", "07", "08", "12")):
    from melanoma_detection.metadata_nn_tuning import condition_directory, DEFAULT_RUN_ID
    NN_REPORT = condition_directory(ROOT, "dedup_v1", DEFAULT_RUN_ID) / "report" / NOTEBOOK_STAGE
    FIGURE_DIR = NN_REPORT / "figures"
else:
    FIGURE_DIR = ROOT / DIRECTORY / "figures" / "template_v3" / NOTEBOOK_STAGE
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def finish_figure(figure, filename, *, dpi=None):
    if not filename or Path(filename).name != filename:
        raise ValueError("Figure name must be a simple stem.")
    figure.savefig(FIGURE_DIR / (filename + ".png"), dpi=dpi or globals().get("ORIGINAL_EXPORT_DPI", 150), bbox_inches="tight")
    figure.savefig(FIGURE_DIR / (filename + ".pdf"), bbox_inches="tight")
    plt.show()
    plt.close(figure)

_score_bundle = None
def score_bundle():
    global _score_bundle
    if _score_bundle is None:
        _score_bundle = load_clean_scores(ROOT)
    return _score_bundle
display(pd.DataFrame([{"Images": len(data), "Malignant images": int(data.target.sum()),
    "Patients": data.patient_id.nunique(), "Malignant patients": data.loc[data.target.eq(1), "patient_id"].nunique()}]))
display(Markdown("**Dataset:** `dedup_v1` · original patient folds · development records only. This notebook reads saved results and starts no training."))

def show_saved_table(relative, limit=30):
    path = project_path(ROOT, relative)
    if not path.exists():
        display(Markdown("**Pending:** " + relative + ". No results are inferred from unfinished runs."))
        return pd.DataFrame()
    table = pd.read_csv(path)
    display(table.head(limit))
    if len(table) > limit:
        display(Markdown(f"Showing {limit} of {len(table)} rows; the complete CSV is saved."))
    return table

def comparison_table():
    evaluation = latest_evaluation(ROOT)
    if "output_directory" not in evaluation:
        display(Markdown("**Pending:** clean evaluation has not been generated."))
        return pd.DataFrame()
    path = project_path(ROOT, evaluation["output_directory"] + "/model_comparison.csv")
    return pd.read_csv(path)

def model_chart(table, title):
    if table.empty:
        display(Markdown("**Pending:** no verified scores are available for this stage."))
        return
    table = table.sort_values("average_precision")
    fig, ax = plt.subplots(figsize=(9, max(3.2, .38 * len(table) + 1.5)), layout="constrained")
    ax.barh(table.model, table.average_precision, color="#276b91")
    ax.set(title=title, xlabel="Unweighted mean outer-fold average precision", xlim=(0, max(.1, float(table.average_precision.max()) * 1.25)))
    finish_figure(fig, "model_comparison_ap")

def pilot_report(run_id):
    directory = ROOT / "results/image_d1_runs" / run_id
    path = directory / "summary.json"
    summary = read_json(path) if path.exists() else {"status": "pending"}
    if summary.get("status") == "complete":
        manifest = read_json(directory / "run_manifest.json")
        if manifest["dataset_identity"] != dataset_identity:
            raise ValueError("Pilot belongs to another dataset.")
        from melanoma_detection.development_dataset import sha256_file
        for relative, digest in summary["artifact_sha256"].items():
            artifact = (directory / relative).resolve()
            if not artifact.is_relative_to(directory.resolve()) or sha256_file(artifact) != digest:
                raise ValueError("Completed pilot artifact changed.")
    return directory, summary
'''


def _notebook(title, purpose, sections, filename):
    sources = ORIGINAL_SOURCES[filename[:2]]
    templated = any(heading in {"__markdown__", "__code__"} for heading, _, _ in sections)
    if templated:
        cells = []
        for kind, source, code in sections:
            if kind == "__markdown__":
                cells.append(nbformat.v4.new_markdown_cell(source))
            elif kind == "__code__":
                cells.append(nbformat.v4.new_code_cell(code))
            else:
                raise ValueError(f"Mixed template and generic sections in {filename}.")
        from .dedup_notebook_templates import original_template_path
        template_root = Path(__file__).resolve().parents[2]
        original_words = sum(len(re.findall(r"\b[\w]+\b", cell.source))
            for name in sources
            for cell in nbformat.read(original_template_path(template_root,
                "notebooks/Subset/" + name), as_version=4).cells
            if cell.cell_type == "markdown")
        adapted_words = sum(len(re.findall(r"\b[\w]+\b", cell.source))
            for cell in cells if cell.cell_type == "markdown")
        if adapted_words < .9 * original_words:
            raise ValueError(f"{filename}: adapted Markdown ({adapted_words} words) is substantially shorter than its templates ({original_words}).")
    else:
        raise ValueError(f"{filename} must use original Markdown cells as its presentation template.")
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "ADS2002 project Python", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "deduplicated_workflow": {"dataset_manifest": MANIFEST,
            "training_on_execute": False, "depth_revision": 3, "original_method_sources": sources,
            "presentation_template": "original_markdown_and_cell_order",
            "original_markdown_words": original_words, "adapted_markdown_words": adapted_words}})


def notebook_specs():
    from .dedup_notebook_data import data_specs
    from .dedup_notebook_metadata import metadata_specs
    from .dedup_notebook_images import image_specs
    from .dedup_notebook_evaluation import evaluation_specs
    return sorted(data_specs() + metadata_specs() + image_specs() + evaluation_specs(), key=lambda item: item[0])


def build_notebooks(project_root):
    root = Path(project_root).resolve()
    directory = root / NOTEBOOK_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, title, purpose, sections in notebook_specs():
        notebook = _notebook(title, purpose, sections, filename)
        nbformat.validate(notebook)
        path = directory / filename
        nbformat.write(notebook, path)
        paths.append(path)
    lines = ["# Clean development workflow", "", "Open stages in order. Executing these notebooks starts no training.", "",
        "These notebooks use the original Markdown cells and section order as presentation templates. Colours, figure composition and explanatory detail are retained, while clean inputs, saved numerical findings and methodological qualifications are adapted.", "",
        "All stages explicitly use the frozen dedup_v1 manifest and original patient assignments.", "",
        "Figures are generated from clean records by editable plotting code and exported as PNG/PDF under `results/dedup_v1/clean_workflow/figures/template_v3/<stage>/`. Incomplete image outputs are explicitly pending.", "",
        "See [method and figure coverage](METHOD_COVERAGE.md) for correspondence with the originals and deliberate adaptations.", "",
        "Run `.venv/Scripts/python.exe -B scripts/continue_clean_workflow.py` to complete only missing authorised stages.", "",
        "Affected original metadata notebooks are archived before their report update. Historical predictions, models, figures and receipts remain unchanged.", "",
        "Stages 06 and 07 share their presentation template with the original workflow. Stage 06 is the fixed NN; stage 07 adds the frozen 36-setting, three-seed NN search to the existing conventional grids. New metadata reports and figures use the separate versioned metadata_nn_tuning namespace.", ""]
    lines.extend(f"- [{title}]({filename})" for filename, title, _, _ in notebook_specs())
    (directory / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_coverage(root, directory, paths)
    return paths


def _write_coverage(root, directory, paths, *, receipt_path=None):
    """Document substantive methods and source correspondence, ignoring outputs."""
    coverage = {
        "01": ("Schema, missingness, label/category/age checks, source agreement; exact-copy ledger and image-quality review", "Development IDs only; read frozen cleaning artifacts instead of rewriting source; no test source loaded"),
        "02": ("Image/patient class balance, images per patient, sex/age/site share and malignancy-rate panels; diagnoses", "Regenerate all figures on retained records; patients counted within each category as in the original"),
        "03": ("Fold support table and malignancy-rate figure; outer/inner separation; malignant-patient support", "Verify existing patient folds instead of running a splitter; original test-ID check omitted under no-test-access instruction"),
        "04": ("Dummy/categorical/spline protocols, fold and pooled AP/AUC, ROC/PR, confusion, cutoff diagnostics and coefficients", "Mean-fold ranking primary; OOF-selected cutoff explicitly exploratory"),
        "05": ("Five-model ranking/curves, fold variability, nonlinear age profiles, seven feature sets, coefficients/importances", "Clean fit artifacts only; ablation/model expansion secondary"),
        "06": ("Fixed NN architecture/preprocessing/stopping, histories/epochs, three-seed stability, equal mean, AP/AUC panels with primary mean-fold and secondary pooled views", "Shared original/clean report template; verified existing clean fixed fits; no best-seed selection"),
        "07": ("Four-family nested grids, 36 NN recipes, patient support, candidates/margins, selected settings and parameter frequencies, duration, blue/gold bars, four-row ROC/PR and per-fold gains", "Reconstructed metadata-suite memberships, then exact clean filtering; inner AP selection separate from outer evaluation; additional development experiment"),
        "08": ("Paired patient intervals, reliability/distributions, Brier/log loss, historical training-only decisions, subgroup support; new tuned/fixed NN and spline contrasts", "2,000 within-fold patient draws; historical spline calibration/cutoffs do not apply to tuned NN; no final fit"),
        "09": ("Image preparation/weak previews, recipes/nested protocol, histories/cost, OOF ranking/curves/confusion/reliability and calibration", "Five clean outer folds replace the original one-split pilot as primary evidence; results pending until verified"),
        "10": ("Three-inner-fold duration curves, 12/24 endpoint vs selected-epoch results, matched challenger curves/intervals/cost", "Bounded B0 and ConvNeXt pilots only; DenseNet, stronger augmentation and CLAHE remain deferred, not losing conditions"),
        "11": ("MC/TTA views, provenance/pass coverage, convergence, distributions/scatter/entropy, ranking/reliability/referral and intervals", "Inference from matching five clean held-out checkpoints; frozen base scores for referral; full inference pending until ready"),
        "12": ("Completion, mean-fold/pooled ranking and curves, paired intervals, equal-patient weights, decisions/subgroups and conclusions; separate tuned-NN versus fixed NN and fixed/tuned spline appendix", "Clean models are main comparison; historical image/fixed-metadata contrasts and duplicate-sensitivity appendix retained; new metadata experiment explicitly labelled; no reserved-test claim"),
    }
    rows = []
    text = ["# Method and figure coverage", "", "Every plot uses clean development records or verified clean run artifacts. Historical model figures are not relabelled as clean results.", "",
        "Opening or executing a notebook starts no training. Saved estimators may be used for prediction-only explanations. GPU work remains owned by its separate runners and deadlines.", "",
        "| Clean stage | Original notebook(s) | Retained analytical depth and figures | Declared adaptation |", "|---|---|---|---|"]
    for path in paths:
        stage = path.name[:2]
        notebook = nbformat.read(path, as_version=4)
        originals = []
        for name in ORIGINAL_SOURCES[stage]:
            from .dedup_notebook_templates import original_template_path
            relative = "notebooks/Subset/" + name
            source = original_template_path(root, relative)
            item = {"path": relative, "template_source_path": source.relative_to(root).as_posix()}
            if source.exists():
                original = nbformat.read(source, as_version=4)
                item.update(sha256=sha256_file(source),
                    code_cells=sum(c.cell_type == "code" for c in original.cells),
                    markdown_characters=sum(len(c.source) for c in original.cells if c.cell_type == "markdown"),
                    markdown_words=sum(len(re.findall(r"\b[\w]+\b", c.source)) for c in original.cells if c.cell_type == "markdown"))
            originals.append(item)
        methods, adaptation = coverage[stage]
        rows.append({"notebook":path.relative_to(root).as_posix(), "original_sources":originals,
            "code_cells":sum(c.cell_type == "code" for c in notebook.cells),
            "markdown_characters":sum(len(c.source) for c in notebook.cells if c.cell_type == "markdown"),
            "markdown_words":sum(len(re.findall(r"\b[\w]+\b", c.source)) for c in notebook.cells if c.cell_type == "markdown"),
            "original_markdown_words": notebook.metadata.deduplicated_workflow.original_markdown_words,
            "presentation_template": "original_markdown_and_cell_order",
            "methods_and_figures":methods,"adaptation":adaptation})
        links = "<br>".join(f"[{name}](../Subset/{name})" for name in ORIGINAL_SOURCES[stage])
        text.append(f"| [{path.name}]({path.name}) | {links} | {methods} | {adaptation} |")
    text += ["", "## Markdown coverage", "",
        "Counts exclude code and rendered outputs. Combined stages sum the Markdown of their mapped original templates.", "",
        "| Stage | Original template words | Adapted clean words |", "|---|---:|---:|"]
    text.extend(f"| {row['notebook'].split('/')[-1][:2]} | {row['original_markdown_words']:,} | {row['markdown_words']:,} |" for row in rows)
    text += ["", "The original Markdown is the template: preserve its heading hierarchy, teaching examples, reading instructions, notebook-specific limitations and conclusions. Adapt source paths, clean values and experiment scope explicitly. Source Markdown must retain at least 90% of the mapped originals' word count; the coverage receipt records both counts. This length check complements review of actual explanatory content and figure correspondence.", "",
        "Original plot colours, markers, composition, labels and typography are retained where the analytical question is unchanged. Added clean diagnostics are separately labelled. A changed experiment or estimand is documented rather than silently substituted. Selection methods remain confined to training patients. Pending plots appear only when matching run receipts become complete.", "",
        "All-development fitting, full OOF evaluation of image challengers, stronger augmentation, CLAHE, resolution changes, fusion and reserved-test evaluation require their separate later stage.", ""]
    (directory / "METHOD_COVERAGE.md").write_text("\n".join(text),encoding="utf-8")
    if (root / "pyproject.toml").exists():
        if receipt_path is None:
            receipt_path = ("results/metadata_nn_tuning/v1/report_verification/dedup_depth_coverage.json"
                if (root / "results/metadata_nn_tuning/v1/archive_before_update/preservation_receipt.json").exists()
                else DIRECTORY + "/notebook_validation/depth_coverage.json")
        write_json(root / receipt_path, {
            "revision":3, "notebooks":rows, "source_outputs_used":False, "training_started":False,
            "reserved_test_loaded":False, "interpretation":"Original Markdown and presentation templates adapted to clean evidence; counts complement content and visual review."})


def execute_notebooks(project_root, *, timeout=180, notebook_names=None):
    """Execute reports with the project kernel and save every rendered output."""
    root = Path(project_root).resolve()
    interpreter = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    output = root / DIRECTORY / "notebook_validation"
    output.mkdir(parents=True, exist_ok=True)
    rendered = output / "rendered_template_v3"
    rendered.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"IPYTHONDIR": str(output / ".ipython"), "MPLCONFIGDIR": str(output / ".matplotlib"),
        "JUPYTER_RUNTIME_DIR": str(output / ".jupyter_runtime"), "PYTHONDONTWRITEBYTECODE": "1"})
    paths = sorted((root / NOTEBOOK_DIRECTORY).glob("*.ipynb"))
    rows = []
    if notebook_names is not None:
        requested = set(notebook_names)
        if not requested or not requested.issubset({path.name for path in paths}):
            raise ValueError("Select existing clean notebook filenames for a report refresh.")
        previous = read_json(output / "execution_receipt.json")
        if previous.get("status") != "complete" or previous.get("presentation_revision") != 3:
            raise ValueError("A selective refresh requires a complete current presentation receipt.")
        for row in previous["notebooks"]:
            if Path(row["notebook"]).name not in requested:
                if sha256_file(root / row["notebook"]) != row["sha256"]:
                    raise ValueError("An unselected report changed; include it in the refresh.")
                rows.append(row)
        paths = [path for path in paths if path.name in requested]
    receipt_base = {"presentation_revision": 3, "training_started": False,
        "reserved_test_loaded": False, "kernel_interpreter": str(interpreter)}
    if notebook_names is not None:
        receipt_base.update({key: previous[key] for key in
            ("markdown_only_edits_after_execution", "markdown_edit_code_unchanged") if key in previous})
        if "markdown_only_edits_after_execution" in receipt_base:
            receipt_base["markdown_only_edits_after_execution"] = [name for name in
                receipt_base["markdown_only_edits_after_execution"] if name not in requested]
    write_json(output / "execution_receipt.json", {**receipt_base, "status": "running", "notebooks": rows})
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3", connection_file=str(output / "kernel.json"))
        manager.kernel_spec.argv = [str(interpreter), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(notebook, km=manager, timeout=timeout, resources={"metadata": {"path": str(path.parent)}})
        try:
            client.execute(env=environment)
        except Exception as error:
            write_json(output / "execution_receipt.json", {**receipt_base, "status": "failed",
                "notebooks": rows, "failed_notebook": path.relative_to(root).as_posix(),
                "error": str(error), "updated_utc": datetime.now(timezone.utc).isoformat()})
            raise
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_connection_file()
        nbformat.validate(notebook)
        temporary = path.with_suffix(".executed.tmp")
        nbformat.write(notebook, temporary)
        os.replace(temporary, path)
        images = []
        for index, cell in enumerate(notebook.cells):
            for number, result in enumerate(cell.get("outputs", [])):
                if result.get("output_type") == "error":
                    raise RuntimeError(f"Notebook retains error output: {path.name}")
                payload = result.get("data", {}).get("image/png")
                if payload:
                    figure = rendered / f"{path.stem}_cell{index}_{number}.png"
                    figure.write_bytes(base64.b64decode(payload))
                    images.append(figure.relative_to(root).as_posix())
        rows.append({"notebook": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
            "executed_code_sha256": stable_sha256([cell.source for cell in notebook.cells if cell.cell_type == "code"]),
            "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells), "errors": 0, "figures": images,
            "original_markdown_words": notebook.metadata.deduplicated_workflow.original_markdown_words,
            "adapted_markdown_words": notebook.metadata.deduplicated_workflow.adapted_markdown_words})
        write_json(output / "execution_receipt.json", {**receipt_base, "status": "running", "notebooks": rows,
            "updated_utc": datetime.now(timezone.utc).isoformat()})
        print(f"Executed {path.name}: {rows[-1]['code_cells']} cells, {len(images)} figures", flush=True)
    rows.sort(key=lambda row: row["notebook"])
    receipt = {**receipt_base, "status": "complete", "notebooks": rows, "notebook_count": len(rows),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "executed_in_this_refresh": [path.name for path in paths],
        "visual_inspection": "pending manual figure inspection"}
    write_json(output / "execution_receipt.json", receipt)
    return receipt
