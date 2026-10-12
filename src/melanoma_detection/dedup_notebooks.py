"""Build report notebooks around one frozen development dataset identity."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import os
import re
import subprocess

import nbformat
from .development_dataset import sha256_file
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
ROC_REPORT_DIRECTORY = "results/metadata_nn_tuning/v2/report_verification"
PREPARATION_NOTEBOOKS = ("01_data_cleaning.ipynb", "02_clean_data_eda.ipynb", "03_patient_fold_verification.ipynb")


def _source_project_root(project_root=None):
    """Allow export to another directory while keeping source templates strict."""
    root=Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
    return root if (root/"pyproject.toml").is_file() else Path(__file__).resolve().parents[2]


def _template_source(root,name):
    if int(name[:2])>=4:
        from .metadata_roc_archive import original_snapshot
        return original_snapshot(root,"notebooks/Subset/"+name)
    from .dedup_notebook_templates import original_template_path
    return original_template_path(root,"notebooks/Subset/"+name)


def _image_roc_builder(root):
    """Load the image agent's report-only builder, never its CLI main."""
    path=Path(root)/"scripts/build_image_roc_reports.py"
    spec=importlib.util.spec_from_file_location("_active_image_roc_report_builder",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_notebook


def _sections(notebook):
    return [("__markdown__",cell.source,"") if cell.cell_type=="markdown" else ("__code__","",cell.source)
            for cell in notebook.cells]


def _v2_books(root):
    from .metadata_tuning_v2_reporting import build_notebook,NAMES
    books={}
    for stage in ("04","05","06","07","08"):
        books[NAMES["dedup_v1"][stage]]=build_notebook(root,"dedup_v1",stage)
    image_builder=_image_roc_builder(root)
    for name in ("09_image_modelling.ipynb","10_bounded_image_pilots.ipynb",
                 "11_image_uncertainty.ipynb","12_combined_evaluation.ipynb"):
        books[name]=image_builder(root,"notebooks/Subset_dedup/"+name)
    for name,notebook in books.items():
        presentation=notebook.metadata.setdefault("deduplicated_workflow",{})
        from .metadata_roc_archive import original_snapshot
        preceding_source=original_snapshot(root,"notebooks/Subset_dedup/"+name)
        preceding=nbformat.read(preceding_source,as_version=4)
        preceding_words=sum(len(re.findall(r"\b[\w]+\b",cell.source)) for cell in preceding.cells if cell.cell_type=="markdown")
        mapped_words=sum(len(re.findall(r"\b[\w]+\b",cell.source))
            for original in ORIGINAL_SOURCES[name[:2]]
            for cell in nbformat.read(_template_source(root,original),as_version=4).cells if cell.cell_type=="markdown")
        presentation.update({"dataset_manifest":MANIFEST,"training_on_execute":False,"depth_revision":4,
            "original_method_sources":ORIGINAL_SOURCES[name[:2]],
            "presentation_template":"original_markdown_and_cell_order",
            "experiment_version":2,"primary_metric":"mean_fold_roc_auc"})
        presentation["preceding_template_markdown_words"]=preceding_words
        presentation["presentation_template_source_path"]=preceding_source.relative_to(root).as_posix()
        presentation["presentation_template_sha256"]=sha256_file(preceding_source)
        presentation["mapped_original_markdown_words"]=mapped_words
        presentation["original_markdown_words"]=max(mapped_words,presentation["preceding_template_markdown_words"])
        presentation["adapted_markdown_words"]=sum(len(re.findall(r"\b[\w]+\b",cell.source))
            for cell in notebook.cells if cell.cell_type=="markdown")
        if presentation["adapted_markdown_words"]<.9*presentation["original_markdown_words"]:
            raise ValueError(f"{name}: the ROC report is substantially shorter than its pre-ROC sources")
    return books
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


def _notebook(title, purpose, sections, filename, *, project_root=None):
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
        template_root = _source_project_root(project_root)
        original_words = sum(len(re.findall(r"\b[\w]+\b", cell.source))
            for name in sources
            for cell in nbformat.read(_template_source(template_root,name), as_version=4).cells
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


def notebook_specs(project_root=None):
    from .dedup_notebook_data import data_specs
    root=_source_project_root(project_root)
    specs=list(data_specs())
    for filename,notebook in _v2_books(root).items():
        title=next(cell.source.splitlines()[0].lstrip("# ") for cell in notebook.cells if cell.cell_type=="markdown")
        specs.append((filename,filename[:2]+" · "+title,"",_sections(notebook)))
    return sorted(specs,key=lambda item:item[0])


def build_notebooks(project_root):
    root = Path(project_root).resolve()
    directory = root / NOTEBOOK_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source_root=_source_project_root(root)
    books=_v2_books(source_root)
    if all((directory/name).is_file() for name in PREPARATION_NOTEBOOKS):
        specs=[]
        for name in PREPARATION_NOTEBOOKS:
            prepared=nbformat.read(directory/name,as_version=4)
            title=next(cell.source.splitlines()[0].lstrip("# ") for cell in prepared.cells if cell.cell_type=="markdown")
            specs.append((name,name[:2]+" · "+title,"",_sections(prepared)))
    else:
        from .dedup_notebook_data import data_specs
        specs=list(data_specs())
    for filename,notebook in books.items():
        title=next(cell.source.splitlines()[0].lstrip("# ") for cell in notebook.cells if cell.cell_type=="markdown")
        specs.append((filename,filename[:2]+" · "+title,"",_sections(notebook)))
    specs.sort(key=lambda item:item[0])
    paths = []
    for filename, title, purpose, sections in specs:
        path = directory / filename
        if int(filename[:2])<4 and path.exists():
            # Preserve the already executed preparation reports byte-for-byte.
            paths.append(path)
            continue
        notebook = books.get(filename) or _notebook(title,purpose,sections,filename,project_root=source_root)
        nbformat.validate(notebook)
        temporary=path.with_suffix(".build.tmp")
        nbformat.write(notebook,temporary)
        os.replace(temporary,path)
        paths.append(path)
    lines = ["# Clean development workflow", "", "Open stages in order. Executing these notebooks starts no training.", "",
        "These notebooks use the original Markdown cells and section order as presentation templates. Colours, figure composition and explanatory detail are retained, while clean inputs, saved numerical findings and methodological qualifications are adapted.", "",
        "All stages explicitly use the frozen dedup_v1 manifest and original patient assignments.", "",
        "Stages 04–12 use ROC-AUC-first version-2 reports. Metadata figures are exported under `results/dedup_v1/metadata_nn_tuning/v2/roc_auc_grid36_v1/report/`; image and combined report exports are under `results/metadata_nn_tuning/v2/report_verification/image_reports/`. Preparation 01–03 retains its original saved figures and outputs. Incomplete image experiments are explicitly pending.", "",
        "See [method and figure coverage](METHOD_COVERAGE.md) for correspondence with the originals and deliberate adaptations.", "",
        "Rebuild active reports with `.venv/Scripts/python.exe -B scripts/build_dedup_workflow_notebooks.py`. The command preserves existing preparation 01–03. Execute the current reports separately with `scripts/execute_metadata_roc_reports.py`; its complete version-2 ROC receipt and no-fitting guard are mandatory.", "",
        "Affected original metadata notebooks are archived before their report update. Historical predictions, models, figures and receipts remain unchanged.", "",
        "Both workflows retain 05 models → 06 fixed NN → 07 tuning including NN → 08 robustness. Stage 07 selects the frozen 36-setting, three-seed NN and classical grids by inner ROC-AUC in the separately declared version-2 experiment. Historical AP-selected metadata and image procedures remain preserved; image selection objectives are unchanged. Mean-fold ROC-AUC leads the reports, AP and probability losses are secondary, and pooled results are supporting.", ""]
    lines += ["Read the [ROC-AUC workflow and evidence boundaries](../../docs/metadata_roc_auc_workflow.md) for the explicit version-2 protocol. Historical spline/image calibration maps and training-selected cutoffs remain attached to their fixed procedures; they do not evaluate the newly tuned network. No final all-development model or reserved-test result is produced by these reports.",""]
    lines.extend(f"- [{title}]({filename})" for filename, title, _, _ in specs)
    (directory / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_coverage(root, directory, paths,source_root=source_root)
    return paths


def _write_coverage(root, directory, paths, *, receipt_path=None,source_root=None):
    """Document substantive methods and source correspondence, ignoring outputs."""
    coverage = {
        "01": ("Schema, missingness, label/category/age checks, source agreement; exact-copy ledger and image-quality review", "Development IDs only; read frozen cleaning artifacts instead of rewriting source; no test source loaded"),
        "02": ("Image/patient class balance, images per patient, sex/age/site share and malignancy-rate panels; diagnoses", "Regenerate all figures on retained records; patients counted within each category as in the original"),
        "03": ("Fold support table and malignancy-rate figure; outer/inner separation; malignant-patient support", "Verify existing patient folds instead of running a splitter; original test-ID check omitted under no-test-access instruction"),
        "04": ("Dummy/categorical protocols, primary mean-fold ROC-AUC, secondary AP/error metrics, supporting ROC/PR; historical confusion, cutoff diagnostics and coefficients", "Verified versioned baseline inputs; historical OOF-selected cutoff explicitly exploratory; no fitting in report cells"),
        "05": ("Five-model ROC-AUC-first ranking/curves, fold variability; historical nonlinear age profiles, seven feature sets, coefficients/importances", "Current fixed-family probabilities are versioned; saved historical ablation/age outputs are labelled supporting and never recreated by all-development fitting"),
        "06": ("Fixed NN architecture/preprocessing/stopping, histories/epochs, three-seed stability, equal mean, ROC-AUC/AP panels with primary mean-fold ROC-AUC and supporting pooled views", "Shared pre-ROC original/clean report template; verified existing fixed fits; no best-seed selection"),
        "07": ("Four-family nested grids, 36 NN recipes, patient support, candidates/margins, selected settings and parameter frequencies, duration, blue/gold bars, four-row ROC/PR and per-fold gains", "Version-2 inner ROC-AUC selection separate from outer ROC-AUC evaluation; historical AP-selected version preserved; reuse, resume and rescoring counters distinguished"),
        "08": ("Paired patient intervals, reliability/distributions, Brier/log loss, historical training-only decisions, subgroup support; new tuned/fixed NN and spline contrasts", "2,000 within-fold patient draws; historical spline calibration/cutoffs do not apply to tuned NN; no final fit"),
        "09": ("Image preparation/weak previews, recipes/nested protocol, histories/cost, ROC-first OOF ranking/curves/confusion/reliability and calibration", "Five clean outer folds replace the original one-split pilot as primary evidence; historical AP-based image selection unchanged"),
        "10": ("Three-inner-fold duration curves, 12/24 endpoint vs selected-epoch results, matched challenger curves/intervals/cost", "Bounded B0 and ConvNeXt pilots only; DenseNet, stronger augmentation and CLAHE remain deferred, not losing conditions"),
        "11": ("MC/TTA views, provenance/pass coverage, convergence, distributions/scatter/entropy, ROC-first ranking/reliability/referral and intervals", "Read matching saved five-fold uncertainty artifacts; frozen base scores for referral; no new image inference during report execution"),
        "12": ("Completion, ROC-first mean-fold/pooled ranking and curves, paired intervals, equal-patient weights, decisions/subgroups and conclusions; separate v2 tuned-NN versus fixed NN and fixed/tuned spline appendix", "Clean models are main comparison; historical image/fixed-metadata contrasts and duplicate-sensitivity appendix retained; current ROC-selected metadata and historical AP-selected image procedures distinguished; no reserved-test claim"),
    }
    source_root=_source_project_root(root) if source_root is None else Path(source_root).resolve()
    rows = []
    text = ["# Method and figure coverage", "", "Every plot uses clean development records or verified clean run artifacts. Historical model figures are not relabelled as clean results.", "",
        "Opening or executing a notebook starts no training or new image inference. Preparation 01–03 remains unchanged. Active 04–12 reports lead with mean-fold ROC-AUC, retain AP/error metrics as secondary and pooled results as supporting. Historical AP-era diagnostics are explicitly labelled; image selection objectives remain unchanged.", "",
        "| Clean stage | Original notebook(s) | Retained analytical depth and figures | Declared adaptation |", "|---|---|---|---|"]
    for path in paths:
        stage = path.name[:2]
        notebook = nbformat.read(path, as_version=4)
        originals = []
        for name in ORIGINAL_SOURCES[stage]:
            relative = "notebooks/Subset/" + name
            source = _template_source(source_root,name)
            item = {"path": relative, "template_source_path": source.relative_to(source_root).as_posix()}
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
            "mapped_original_markdown_words":sum(item.get("markdown_words",0) for item in originals),
            "preceding_template_markdown_words":notebook.metadata.deduplicated_workflow.get("preceding_template_markdown_words"),
            "presentation_template_source_path":notebook.metadata.deduplicated_workflow.get("presentation_template_source_path"),
            "presentation_template_sha256":notebook.metadata.deduplicated_workflow.get("presentation_template_sha256"),
            "presentation_template": "original_markdown_and_cell_order",
            "experiment_version":2 if int(stage)>=4 else None,
            "primary_metric":"mean_fold_roc_auc" if int(stage)>=4 else None,
            "historical_source_outputs_used":stage in {"04","05"},
            "methods_and_figures":methods,"adaptation":adaptation})
        links = "<br>".join(f"[{name}](../Subset/{name})" for name in ORIGINAL_SOURCES[stage])
        text.append(f"| [{path.name}]({path.name}) | {links} | {methods} | {adaptation} |")
    text += ["", "## Markdown coverage", "",
        "Counts exclude code and rendered outputs. Active sources are bound to verified pre-ROC snapshots rather than the edited live notebooks. Mapped original source counts are retained in the receipt; metadata reports may preserve the longer preceding clean report as their immediate template.", "",
        "| Stage | Required pre-ROC source words | Adapted clean words |", "|---|---:|---:|"]
    text.extend(f"| {row['notebook'].split('/')[-1][:2]} | {row['original_markdown_words']:,} | {row['markdown_words']:,} |" for row in rows)
    text += ["", "The original Markdown is the template: preserve its heading hierarchy, teaching examples, reading instructions, notebook-specific limitations and conclusions. Adapt source paths, clean values and experiment scope explicitly. Source Markdown must retain at least 90% of the mapped originals' word count; the coverage receipt records both counts. This length check complements review of actual explanatory content and figure correspondence.", "",
        "Original plot colours, markers, composition, labels and typography are retained where the analytical question is unchanged. Added clean diagnostics are separately labelled. A changed experiment or estimand is documented rather than silently substituted. Selection methods remain confined to training patients. Pending plots appear only when matching run receipts become complete.", "",
        "All-development fitting, full OOF evaluation of image challengers, stronger augmentation, CLAHE, resolution changes, fusion and reserved-test evaluation require their separate later stage.", ""]
    (directory / "METHOD_COVERAGE.md").write_text("\n".join(text),encoding="utf-8")
    if (root / "pyproject.toml").exists():
        if receipt_path is None:
            receipt_path = (ROC_REPORT_DIRECTORY+"/dedup_depth_coverage.json"
                if (source_root / "results/metadata_nn_tuning/v2/archive_before_update/snapshot_receipt.json").exists()
                else DIRECTORY + "/notebook_validation/depth_coverage.json")
        write_json(root / receipt_path, {
            "revision":4, "experiment_version":2,"notebooks":rows, "source_outputs_used":True,
            "historical_output_stages":["04","05"],"training_started":False,"image_inference_started":False,
            "reserved_test_loaded":False,"interpretation":"Verified pre-ROC presentation snapshots adapted to ROC-first clean evidence; historical diagnostic outputs remain separately labelled; counts complement content and visual review."})


def execute_notebooks(project_root, *, timeout=900, notebook_names=None):
    """Compatibility dispatch to guarded version-2 execution of clean04–12.

    Saved preparation 01–03 and historical presentation receipts remain intact.
    The current executor verifies complete ROC-selected runs, blocks fitting and
    test access, and confines all generated outputs to the version-2 namespaces.
    """
    root=Path(project_root).resolve()
    if root!=Path(__file__).resolve().parents[2]:
        raise ValueError("Execute reports only in the configured project checkout")
    available={p.name for p in (root/NOTEBOOK_DIRECTORY).glob("[0-9][0-9]_*.ipynb") if int(p.name[:2])>=4}
    selected=sorted(available if notebook_names is None else set(notebook_names))
    if not selected or not set(selected).issubset(available):
        raise ValueError("Select current clean 04–12 filenames; preparation 01–03 remains preserved")
    interpreter=root/".venv"/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    command=[str(interpreter),"-B",str(root/"scripts/execute_metadata_roc_reports.py"),"--timeout",str(timeout)]
    for name in selected:
        command += ["--notebook",NOTEBOOK_DIRECTORY+"/"+name]
    subprocess.run(command,cwd=root,check=True)
    return read_json(root/ROC_REPORT_DIRECTORY/"execution_receipt.json")
