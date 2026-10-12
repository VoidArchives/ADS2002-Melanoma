"""Rebuild eleven saved-result image reports from their verified v1 snapshots.

This creates editable, ROC-first presentation notebooks. It executes no cells
and modifies no image run, checkpoint, prediction, selection or historical file.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from pathlib import Path
import re
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.metadata_roc_archive import original_snapshot, sha256

NOTEBOOKS = [
    "notebooks/Subset/09_image_b0_pilot.ipynb",
    "notebooks/Subset/10_image_architecture_comparison.ipynb",
    "notebooks/Subset/11_image_augmentation.ipynb",
    "notebooks/Subset/12_image_lighting.ipynb",
    "notebooks/Subset/13_image_uncertainty.ipynb",
    "notebooks/Subset/14_image_nested_evaluation.ipynb",
    "notebooks/Subset/15_deduplicated_workflow.ipynb",
    "notebooks/Subset_dedup/09_image_modelling.ipynb",
    "notebooks/Subset_dedup/10_bounded_image_pilots.ipynb",
    "notebooks/Subset_dedup/11_image_uncertainty.ipynb",
    "notebooks/Subset_dedup/12_combined_evaluation.ipynb",
]
FORBIDDEN = {"fit", "fit_transform", "run_smoke", "run_inference", "run_conditions",
             "prepare_conditions", "prepare_architecture_pilot", "run_architecture_pilot",
             "load_selected_checkpoint", "prepare_tensor_cache", "predict_cached",
             "plot_augmentation_preview", "make_transform", "tta_transform",
             "build_augmentation_previews", "build_clahe_previews"}
NOTE = (
    "\n\n**Version 2 reporting boundary.** Primary ranking is unweighted mean outer-fold "
    "**ROC-AUC**; AP is secondary. An inner-validation pilot or uncertainty experiment "
    "has no five-fold estimate, so its ROC-AUC is explicitly exploratory. The saved image "
    "recipes, AP-selected checkpoints/epochs and AP-based admission gates remain unchanged "
    "and are labelled historical selection rules. This is a presentation migration, not "
    "a ROC-selected image refit. Executing every cell reads saved artifacts and writes only "
    "new v2 reports: no image smoke, checkpoint inference, training, calibration fitting, "
    "cutoff selection, final all-development fit or reserved-test access occurs. "
    "The added metadata experiment uses its own explicit v2 ROC selection identity."
)


def _normalise(source):
    for old, new in [("â€”", "—"), ("â€“", "–"), ("âˆ’", "−"), ("Â·", "·"), ("â‰¥", "≥")]:
        source = source.replace(old, new)
    return source


def _markdown(source):
    source = _normalise(source)
    changes = [
        ("The primary measure is **mean outer-fold average precision (AP)**. AP measures", "The primary measure is **unweighted mean outer-fold ROC-AUC**. Secondary AP measures"),
        ("primary ranking measure is **unweighted mean outer-fold AP**", "primary ranking measure is **unweighted mean outer-fold ROC-AUC**"),
        ("primary mean-fold AP", "secondary mean-fold AP"),
        ("unweighted mean-fold AP and ROC-AUC", "unweighted mean-fold ROC-AUC first and AP second"),
        ("unweighted mean outer-fold AP and ROC-AUC", "unweighted mean outer-fold ROC-AUC first and AP second"),
        ("unweighted mean outer-fold AP first. Pooled AP and ROC-AUC remain secondary.", "unweighted mean outer-fold ROC-AUC first. Mean-fold AP and pooled metrics remain secondary."),
        ("Use mean-fold AP and its conditional patient interval", "Use mean-fold ROC-AUC and its conditional patient interval"),
        ("mean-fold AP, while pooled values remain secondary", "mean-fold ROC-AUC, while AP and pooled values remain secondary"),
        ("AP is the primary", "ROC-AUC is the primary"),
        ("AP, the primary", "ROC-AUC, the primary"),
        ("AP is primary", "ROC-AUC is primary"),
        ("Primary mean-fold AP", "Secondary mean-fold AP"),
        ("Average precision (AP) is the primary ranking measure", "Average precision (AP) is the secondary ranking measure"),
        ("mean-fold AP is horizontal", "mean-fold ROC-AUC is horizontal"),
        ("network mean-fold AP minus spline mean-fold AP", "network mean-fold ROC-AUC minus spline mean-fold ROC-AUC"),
        ("The fold table and AP plot", "The fold table and ROC-AUC plot"),
        ("ROC shows a complementary ranking view", "AP supplies a secondary ranking view"),
        ("show selected-checkpoint AP (longer is better)", "show ROC-AUC at the historical AP-selected checkpoint (longer is better)"),
        ("The endpoint AP panels", "The endpoint ROC-AUC panels"),
        ("the small AP difference as final evidence", "the small ranking difference as final evidence"),
        ("unweighted mean-fold AP/AUC remain primary", "unweighted mean-fold ROC-AUC is primary and AP is secondary"),
        ("new reports use `b0_pilot_reporting_v2`", "new reports use the separate v2 presentation namespace"),
    ]
    for old, new in changes:
        source = source.replace(old, new)
    source = source.replace("Primary outcomes are **unweighted mean outer-fold ROC-AUC first and AP second**.",
        "The primary outcome is **unweighted mean outer-fold ROC-AUC**; mean-fold AP is secondary.")
    source = source.replace("Their primary summaries use unweighted mean-fold ROC-AUC first and AP second;",
        "Their primary summaries use unweighted mean-fold ROC-AUC; mean-fold AP,")
    source = source.replace("Primary ranking results are unweighted mean-fold ROC-AUC first and AP second:",
        "Primary ranking results are unweighted mean-fold ROC-AUC, with mean-fold AP secondary:")
    source = source.replace("log loss and Brier describe score errors, and ROC-AUC supplies another ranking\nview.",
        "log loss and Brier describe score errors, and AP supplies the secondary rare-positive ranking\nview.")
    return source


def _original13_markdown(source):
    source = source.replace("We load an existing\n**checkpoint** (saved model weights) and perform inference only.",
        "The saved experiment loaded an existing\n**checkpoint** (saved model weights) and performed inference. This report reads those completed outputs only.")
    source = source.replace("Every current execution writes a new directory\nunder `results/image_uncertainty_runs/`.",
        "The completed inference directory is preserved unchanged. Current execution reads it\nand writes any presentation exports only under the v2 report namespace.")
    source = source.replace("The shared loader rebuilds the trained network and exact deterministic image\npreparation.",
        "The historical loader rebuilt the trained network and exact deterministic image\npreparation. Current cells inspect its saved checkpoint/split provenance without rebuilding a network.")
    source = source.replace("The defaults below select the experimental B0 control. To inspect another\nsupported checkpoint, set `SOURCE_RUN`, `ARCHITECTURE` and `RECIPE_ID` together:",
        "The report is pinned to the completed experimental B0 control's uncertainty run. Historical\nalternative inference required `SOURCE_RUN`, `ARCHITECTURE` and `RECIPE_ID` together:")
    source = source.replace("Any failure stops execution before the 30-pass run.",
        "Any failure stopped the original execution before the 30-pass run. The current cell reads the saved smoke receipt; it authorises no new inference.")
    source = source.replace("checks inference without changing training artifacts.",
        "reads the recorded inference checks without changing training artifacts or rerunning inference.")
    source = source.replace("Raw, summary and figure\noutputs belong only to this new run directory.",
        "The saved raw, summary and figure\noutputs remain in their historical inference directory. Current presentation outputs use a separate v2 directory.")
    return source


def _historical_timing_markdown(source, stage):
    """Keep pilot-time limitations distinct from subsequently completed reports."""
    changes = {
        "09_image_b0_pilot": (
            "A finite image preview cannot certify every lesion; full outer evaluation and the reserved test remain pending.",
            "A finite image preview cannot certify every lesion. At this historical pilot's completion, full outer evaluation was pending; the completed five-fold results now appear in [notebook 14](14_image_nested_evaluation.ipynb) and [clean notebook 09](../Subset_dedup/09_image_modelling.ipynb). The reserved test remains unevaluated.",
        ),
        "10_image_architecture_comparison": (
            "Longer confirmation, calibration, full outer evaluation, fusion and the reserved test remain pending.",
            "At this historical pilot's completion, longer confirmation, calibration and full outer evaluation remained pending. The later five-fold B0 results now appear in [notebook 14](14_image_nested_evaluation.ipynb) and [clean notebook 09](../Subset_dedup/09_image_modelling.ipynb); they do not turn this short architecture pilot into a final family comparison. Fusion and reserved-test evaluation remain deferred.",
        ),
        "15_deduplicated_workflow": (
            "The current complete sensitivity evidence can assess evaluation weighting; a conclusion about clean image training waits for verified five-fold outputs. Preserve the distinction even if the reported rankings agree.",
            "The completed sensitivity evidence assesses evaluation weighting. At the historical sensitivity report's preparation, a conclusion about clean image training awaited verified five-fold outputs; those clean outputs now appear in [clean notebook 09](../Subset_dedup/09_image_modelling.ipynb), alongside the original-data evaluation in [notebook 14](14_image_nested_evaluation.ipynb). Keep evaluation-only filtering separate from clean retraining even when the reported rankings agree.",
        ),
    }
    if stage in changes:
        source = source.replace(*changes[stage])
    return source


def _redirect_setup(source, stage, condition):
    source = source.replace("from melanoma_detection import dedup_notebook_images as image_docs",
                            "from melanoma_detection import image_roc_reporting as image_docs")
    source = source.replace('FIGURE_DIR = ROOT / DIRECTORY / "figures" / "template_v3" / NOTEBOOK_STAGE',
        'from melanoma_detection.image_roc_reporting import report_directory\nNN_REPORT = report_directory(ROOT, "dedup_v1", NOTEBOOK_STAGE)\nFIGURE_DIR = NN_REPORT / "figures"')
    source = re.sub(r'if NOTEBOOK_STAGE\.startswith\(.*?FIGURE_DIR\.mkdir\(parents=True, exist_ok=True\)',
        'from melanoma_detection.image_roc_reporting import report_directory\nNN_REPORT = report_directory(ROOT, "dedup_v1", NOTEBOOK_STAGE)\nFIGURE_DIR = NN_REPORT / "figures"\nFIGURE_DIR.mkdir(parents=True, exist_ok=True)', source, flags=re.S)
    source = re.sub(r'from melanoma_detection\.metadata_nn_tuning import condition_directory, DEFAULT_RUN_ID\nNN_REPORT = .*?\nFIGURE_DIR = NN_REPORT / "figures"',
        'from melanoma_detection.image_roc_reporting import report_directory\nNN_REPORT = report_directory(ROOT, "dedup_v1", NOTEBOOK_STAGE)\nFIGURE_DIR = NN_REPORT / "figures"', source)
    if "from melanoma_detection.dedup_workflow import" in source:
        line = next(line for line in source.splitlines() if line.startswith("from melanoma_detection.dedup_workflow import"))
        source = source.replace(line, line + "\nfrom melanoma_detection.metadata_roc_historical import load_clean_scores, latest_evaluation")
    if "save_metadata_table," in source or "save_metadata_table," in source.splitlines():
        source += "\nfrom melanoma_detection.metadata_roc_historical import save_metadata_table\n"
    source = source.replace('table.sort_values("average_precision")', 'table.sort_values("roc_auc")')
    source = source.replace('table.average_precision, color="#276b91"', 'table.roc_auc, color="#276b91"')
    source = source.replace('"Unweighted mean outer-fold average precision"', '"Unweighted mean outer-fold ROC-AUC"')
    source = source.replace('float(table.average_precision.max())', 'float(table.roc_auc.max())')
    source = source.replace('"model_comparison_ap"', '"model_comparison_roc_auc"')
    return source


def _original_pilot_setup(source, stage):
    source = re.sub(r'from melanoma_detection.image_architecture import .*\n', '', source)
    source = re.sub(r'from melanoma_detection.image_conditions import .*\n', '', source)
    source = source.replace("    plot_augmentation_preview, save_figure)", "    save_figure)")
    source = source.replace("plot_raw_reliability, plot_augmentation_preview, save_figure)", "plot_raw_reliability, save_figure)")
    source = source.replace('TRAIN_ARCHITECTURES = False', 'TRAIN_ARCHITECTURES = False\nimport yaml\nfrom melanoma_detection.image_roc_reporting import report_directory, show_saved_figure, save_figure')
    source = source.replace('TRAIN_CONDITIONS = False', 'TRAIN_CONDITIONS = False\nimport yaml\nfrom melanoma_detection.image_roc_reporting import report_directory, show_saved_figure, save_figure')
    source = re.sub(r'config = load_config\((.*?)\)', r'config = yaml.safe_load((\1).read_text(encoding="utf-8"))', source)
    source = re.sub(r'if TRAIN_ARCHITECTURES:\n.*?else:\n(.*?)bundle = load_architecture_outputs',
        lambda m: "\n".join(line[4:] if line.startswith("    ") else line for line in m.group(1).splitlines()) + "\nbundle = load_architecture_outputs", source, flags=re.S)
    source = re.sub(r'if TRAIN_CONDITIONS:\n.*?summary_path =', 'summary_path =', source, flags=re.S)
    source = source.replace('figures_dir, tables_dir = run_dir / "figures", run_dir / "tables"',
        f'report_dir = report_directory(PROJECT_ROOT, "original", {stage!r})\nfigures_dir, tables_dir = report_dir / "figures", report_dir / "tables"')
    if stage.startswith("09"):
        source = source.replace('run_dir, report_dir = pilot["run_dir"], pilot["report_dir"]',
            f'from melanoma_detection.image_roc_reporting import report_directory, show_saved_figure, save_figure\nrun_dir = pilot["run_dir"]\nreport_dir = report_directory(PROJECT_ROOT, "original", {stage!r})')
    if stage.startswith("10"):
        source += '\nfrom melanoma_detection.image_roc_reporting import plot_resource_comparison\n'
    if stage.startswith(("11", "12")):
        source += f'\nfrom melanoma_detection.image_roc_reporting import report_directory\nreport_dir = report_directory(PROJECT_ROOT, "original", {stage!r})\n'
    return source


def _uncertainty_codes():
    setup = '''from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from melanoma_detection.image_roc_reporting import load_saved_original_uncertainty, show_saved_figure, report_directory, UNCERTAINTY_RUN
stored = load_saved_original_uncertainty(PROJECT_ROOT)
SOURCE_RUN = stored["provenance"]["source_run"]
ARCHITECTURE, RECIPE_ID = stored["provenance"]["architecture"], stored["provenance"]["recipe_id"]
OUTPUT_DIR = report_directory(PROJECT_ROOT, "original", "13_image_uncertainty")
summary, quality = stored["summary"], stored["quality"]
convergence, referral, differences = stored["convergence"], stored["referral"], stored["differences"]
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","axes.facecolor":"white"})
def show_uncertainty_figure(name):
    show_saved_figure(PROJECT_ROOT, f"results/image_uncertainty_runs/{UNCERTAINTY_RUN}/figures/{name}")
print("Pinned completed uncertainty run:", UNCERTAINTY_RUN)
print("No checkpoint loading, smoke, inference, bootstrap recomputation or historical writes")
'''
    return {
        3: setup,
        5: '''checkpoint = stored["checkpoint"]
display(pd.DataFrame([{ "partition":name, "patients":len(checkpoint[field])} for name,field in [("fitting","fitting_patient_ids"),("inner_validation","evaluation_patient_ids"),("outer_excluded","outer_excluded_patient_ids")]]))
display(pd.DataFrame([{"selected_epoch":checkpoint["best_epoch"],"selection":"historical highest inner-validation AP","checkpoint_sha256":checkpoint["checkpoint_sha256"],"evaluation_status":checkpoint["evaluation_status"]}]))
print("Saved evaluation images:",len(summary),"patients:",summary.patient_id.nunique(),"malignant images:",int(summary.target.sum()))''',
        7: 'display(pd.DataFrame(list(json.loads((stored["directory"] / "smoke_receipt.json").read_text()).items()),columns=["Saved smoke check","Result"]))\nprint("Historical smoke receipt only; no new inference starts.")',
        9: 'show_uncertainty_figure("conservative_tta_examples.png")',
        11: '''for name in ("raw_prediction_checks.json","inference_checks.json","inference_provenance.json"):
    display(Markdown("**Saved receipt:** " + name))
    display(pd.DataFrame(list(json.loads((stored["directory"] / name).read_text()).items()),columns=["Recorded check","Value"]))
print("The raw pass files were hash-verified; no pass was generated.")''',
        13: 'display(summary[["image_id","target","base_score","mc_mean_score","mc_score_sd","tta_mean_score","tta_score_sd"]].head(8).round(5))',
        15: 'display(convergence.round(5))\nshow_uncertainty_figure("mc_convergence.png")',
        17: 'display(quality[["prediction","roc_auc","average_precision","log_loss","brier"]].round(5))\ndisplay(stored["reliability"].round(5))\nshow_uncertainty_figure("descriptive_reliability.png")',
        19: 'show_uncertainty_figure("uncertainty_distributions.png")',
        21: 'print("Saved referral at descriptive cutoff 0.5; no threshold fitted")\ndisplay(referral[(referral.referral_fraction == .20) & (referral.ranking != "random")].round(5))',
        22: 'show_uncertainty_figure("referral_comparisons.png")',
        24: 'display(differences[differences.referral_fraction == .20].round(5))',
        25: 'show_uncertainty_figure("paired_patient_bootstrap_differences.png")',
        27: 'display(pd.DataFrame(list(stored["provenance"].items()),columns=["Saved provenance","Value"]))\ndisplay(pd.DataFrame([json.loads((stored["directory"] / "preservation_receipt.json").read_text())]))\nprint("Historical training/inference outputs preserved; v2 reporting only")',
        29: '''observed = quality.set_index("prediction")
display(Markdown(f"**Pinned historical checkpoint:** {ARCHITECTURE}, run `{SOURCE_RUN}`, recipe `{RECIPE_ID}`, AP-selected epoch {stored['checkpoint']['best_epoch']}. **Population:** {len(summary):,} inner-validation images from {summary.patient_id.nunique()} patients, including {int(summary.target.sum())} malignant images. **Exploratory ROC-AUC:** deterministic {observed.loc['Deterministic','roc_auc']:.4f}, MC mean {observed.loc['MC mean','roc_auc']:.4f}, TTA mean {observed.loc['TTA mean','roc_auc']:.4f}. These are selected-inner diagnostics, not mean-fold outer evidence."))
display(Markdown(f"**Secondary AP:** deterministic {observed.loc['Deterministic','average_precision']:.4f}, MC mean {observed.loc['MC mean','average_precision']:.4f}, TTA mean {observed.loc['TTA mean','average_precision']:.4f}. Averaging performance does not establish that uncertainty improves referral."))
for method in ('mc','tta'):
    rows = referral[(referral.referral_fraction == .20) & (referral.method == method)].set_index('ranking')
    pair = differences[(differences.method == method) & (differences.referral_fraction == .20) & (differences.comparator == 'prediction_score')].set_index('metric')
    a, b = pair.loc['malignant_referred_fraction'], pair.loc['retained_error']
    display(Markdown(f"**{method.upper()} at predeclared 20% referral:** SD referred {int(rows.loc['uncertainty_sd','n_malignant_images_referred'])} malignant images versus {int(rows.loc['prediction_score','n_malignant_images_referred'])} by ordinary score. Paired malignant-referral difference {100*a['estimate']:.2f} percentage points (95% interval {100*a['ci_lower']:.2f} to {100*a['ci_upper']:.2f}); retained-error difference {100*b['estimate']:.3f} (interval {100*b['ci_lower']:.3f} to {100*b['ci_upper']:.3f})."))
ordinary = differences[(differences.referral_fraction == .20) & (differences.comparator == 'prediction_score')]
crossing = int((ordinary.ci_lower.le(0) & ordinary.ci_upper.ge(0)).sum())
display(Markdown(f"**Decision:** retain SD as an exploratory diagnostic. {crossing} of {len(ordinary)} intervals against ordinary score include zero. These validation patients helped choose the checkpoint. No final architecture or referral threshold is selected."))''',
    }


def build_notebook(root, relative):
    root = Path(root).resolve()
    if relative not in NOTEBOOKS:
        raise ValueError("This builder owns only the eleven image-facing reports.")
    template_path = original_snapshot(root, relative)
    template = nbformat.read(template_path, as_version=4)
    notebook = deepcopy(template)
    condition = "dedup_v1" if "/Subset_dedup/" in relative else "original"
    stage = Path(relative).stem
    for number, cell in enumerate(notebook.cells, 1):
        if cell.cell_type == "markdown":
            cell.source = _markdown(cell.source)
            if condition == "original":
                cell.source = _historical_timing_markdown(cell.source, stage)
            if stage.startswith("13") and condition == "original":
                cell.source = _original13_markdown(cell.source)
            if condition == "dedup_v1" and stage.startswith("12") and number == 15:
                cell.source = cell.source.replace("horizontal position is AP", "horizontal position is ROC-AUC").replace("Blue circles are observed AP", "Blue circles are observed ROC-AUC").replace("`AP undefined`", "`ROC-AUC undefined`")
                cell.source += "\n\nThis primary subgroup view uses ROC-AUC, with 0.5 as chance ranking; the preserved AP support diagnostic remains secondary. Both require sufficient malignant-patient support."
            if condition == "dedup_v1" and stage.startswith("12") and number == 9:
                cell.source = re.sub(r'\*\*Aggregation adaptation:\*\*.*?multiple comparisons\.',
                    '**Aggregation and metric adaptation:** the original figure displayed pooled AP, and the earlier clean report displayed mean-fold AP. This v2 figure preserves its blue-dot/gold-square layout but displays primary mean-fold ROC-AUC. Both metrics and pooled intervals remain available in supporting tables; their values are not interchangeable. The preserved secondary AP interval for the fixed network mean was [-0.0255, +0.0076], giving no clear AP improvement. The current ROC interval is calculated from the verified fixed-model scores in the table below. These exploratory intervals hold predictions fixed and are not adjusted for multiple comparisons.',cell.source,flags=re.S)
            continue
        source = _normalise(cell.source)
        if condition == "dedup_v1":
            source = _redirect_setup(source, stage, condition)
        if condition == "original" and stage.startswith(("09", "10", "11", "12")):
            if number == (3 if stage.startswith("09") else 2):
                source = _original_pilot_setup(source, stage)
            source = source.replace('"average_precision", "roc_auc"', '"roc_auc", "average_precision"')
            source = source.replace('render_condition_report(bundle, run_dir)', 'render_condition_report(bundle, report_dir)')
            source = source.replace('(run_dir / "reporting_receipt.json").write_text', '(report_dir / "reporting_receipt.json").write_text')
            source = source.replace('path.relative_to(run_dir)', 'path.relative_to(report_dir)')
            source = source.replace('run_dir.mkdir(parents=True, exist_ok=True)', 'report_dir.mkdir(parents=True, exist_ok=True)')
            if stage.startswith("09") and number == 7:
                source = 'show_saved_figure(PROJECT_ROOT,"results/image_d1_runs/b0_pilot_reporting_v2/figures/weak_augmentation_preview.png")'
            if stage.startswith("10") and number == 6:
                source = 'show_saved_figure(PROJECT_ROOT,"results/image_d1_runs/architecture_pilot_3h_v1/figures/weak_augmentation_preview.png")'
            if number in ((19,22) if stage.startswith("09") else (17,) if stage.startswith("10") else (15,)):
                source = 'display(Markdown("**Historical AP selection:** the recorded recommendation/epoch remains unchanged; current ROC summaries do not reselect an image procedure."))\n' + source
        if condition == "original" and stage.startswith("13"):
            source = _uncertainty_codes()[number]
        if condition == "original" and stage.startswith("14"):
            if number == 2:
                source = source.replace('run = load_saved_nested_run(PROJECT_ROOT, RUN_ID)', 'from melanoma_detection.image_roc_reporting import primary_roc_summary\nrun = load_saved_nested_run(PROJECT_ROOT, RUN_ID)')
                source = re.sub(r'    display\(Markdown\(f"\*\*Mean outer-fold AP:.*?patients;.*?\)\)\n', '    primary_roc_summary(run)\n', source, flags=re.S)
                source += '\nfrom melanoma_detection.image_roc_reporting import raw_outer_ranking\nimport matplotlib.pyplot as plt\nplt.rcParams.update({"figure.dpi":160,"font.size":10})\nif run["complete"]:\n    print("Primary mean-fold ROC-AUC:",run["fold_metrics"].roc_auc.mean(),"| secondary mean-fold AP:",run["fold_metrics"].average_precision.mean())\n'
            if number == 10:
                source = 'if run["complete"]:\n    raw_outer_ranking(run, lambda fig,name: (display(fig), plt.close(fig)))\n    display(run["intervals"].query("model == \'image\'").sort_values("metric").round(5))\n    print("Historical image selection used AP; mean-fold ROC-AUC is the v2 reporting primary.")'
            if number == 19:
                source = 'from melanoma_detection.image_roc_reporting import image_takeaways\nimage_takeaways(run, pd.DataFrame())\nif run["complete"]:\n    display(Markdown(run["evaluation"]["development_caveat"]))'
        if condition == "original" and stage.startswith("15"):
            source = source.replace('["model", "weighting", "average_precision", "roc_auc"', '["model", "weighting", "roc_auc", "average_precision"')
            if number in (13,15,17,25):
                source = source.replace('"average_precision"', '"roc_auc"').replace('.average_precision', '.roc_auc').replace(' AP', ' ROC-AUC').replace('average precision', 'ROC-AUC')
                source = source.replace('["roc_auc", "roc_auc"]', '["roc_auc", "average_precision"]')
                source = source.replace('"roc_auc", "roc_auc",', '"roc_auc", "average_precision",')
                source = source.replace('["Mean-fold ROC-AUC", "Mean-fold ROC-AUC"]', '["Primary mean-fold ROC-AUC", "Secondary mean-fold AP"]')
                if number in (13,25):
                    source = source.replace("'average_precision'", "'roc_auc'")
            if number == 19:
                source = source.replace('"average_precision", "roc_auc"', '"roc_auc", "average_precision"')
            if number == 4:
                source = source.replace('check_hashes(metadata_receipt["source_sha256"])',
                    'from melanoma_detection.metadata_roc_archive import verified_source_path\nfor relative,digest in metadata_receipt["source_sha256"].items():\n    verified_source_path(PROJECT_ROOT,relative,digest,parent_receipt="results/dedup_v1/metadata/comparator_receipt.json")')
            if number == 21:
                source = source.replace('mean outer-fold AP', 'mean outer-fold ROC-AUC').replace('average_precision.mean()', 'roc_auc.mean()')
        if condition == "dedup_v1" and stage.startswith("12"):
            if number in (10,16,20,65):
                source = source.replace('"average_precision"', '"roc_auc"').replace('.average_precision', '.roc_auc').replace('average_precision_lower', 'roc_auc_lower').replace('average_precision_upper','roc_auc_upper')
                source = source.replace(' AP', ' ROC-AUC').replace('average precision', 'ROC-AUC').replace('Average precision','ROC-AUC').replace('AP undefined', 'ROC-AUC undefined')
                source = source.replace('["roc_auc", "roc_auc"]', '["roc_auc", "average_precision"]')
                if number == 20:
                    source = source.replace("raw_ap", "raw_auc").replace("nn_ap", "nn_auc")
            source = source.replace('["average_precision","roc_auc"]', '["roc_auc","average_precision"]')
            source = source.replace('[(axes[0],"average_precision","Average precision"),(axes[1],"roc_auc","ROC-AUC")]', '[(axes[0],"roc_auc","ROC-AUC"),(axes[1],"average_precision","Average precision (secondary)")]')
            source = source.replace('title="Primary mean-fold "+label', 'title=("Primary mean-fold " if metric == "roc_auc" else "Secondary mean-fold ")+label')
        cell.source = source
        cell.outputs = []
        cell.execution_count = None
    first_markdown = next(c for c in notebook.cells if c.cell_type == "markdown")
    first_line, _, rest = first_markdown.source.partition("\n")
    first_markdown.source = first_line + NOTE + "\n\n" + rest.lstrip()
    if stage.startswith("14") or (condition == "dedup_v1" and stage.startswith("12")):
        for i,c in enumerate(notebook.cells):
            if c.cell_type == "markdown" and c.source.startswith("## Additional experiment: tuned metadata neural network"):
                c.source = c.source.replace("Stage 07 now searches", "The v2 ROC-selected stage 07 searches")
                from melanoma_detection.metadata_tuning_v2_evaluation import assessment_sections
                shared = assessment_sections(condition,run_id="roc_auc_grid36_v1")
                c.source += "\n\n" + "\n\n".join(markdown for kind,markdown,code in shared if kind == "__markdown__")
                notebook.cells[i+1].source = '''from pathlib import Path
import sys
import matplotlib.pyplot as plt
PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
sys.path.insert(0,str(PROJECT_ROOT / "src"))
plt.rcdefaults()
plt.rcParams.update({"figure.dpi":120,"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
from melanoma_detection.metadata_tuning_v2_evaluation import show_assessment
show_assessment(PROJECT_ROOT, CONDITION, run_id="roc_auc_grid36_v1", experiment_version=2)
'''.replace("CONDITION",repr(condition))
    old_words = sum(len(c.source.split()) for c in template.cells if c.cell_type == "markdown")
    words = sum(len(c.source.split()) for c in notebook.cells if c.cell_type == "markdown")
    if words < .9 * old_words:
        raise ValueError("Image report Markdown is substantially shorter than its template.")
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        code = "\n".join(line for line in cell.source.splitlines() if not line.lstrip().startswith("%"))
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ""
                if name in FORBIDDEN:
                    raise ValueError(f"Image computation remains in {relative}: {name}")
    notebook.metadata["image_roc_report"] = {"condition":condition,"experiment_version":2,"primary_metric":"mean_fold_roc_auc","secondary_metric":"average_precision","image_selection_metric":"historical_average_precision_unchanged","image_compute_on_execute":False,"training_on_execute":False,"reserved_test_loaded":False,"template":str(template_path.relative_to(root)).replace("\\","/"),"template_sha256":sha256(template_path),"template_markdown_words":old_words,"markdown_words":words,"metadata_run_id":"roc_auc_grid36_v1"}
    nbformat.validate(notebook)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--notebook",action="append",choices=NOTEBOOKS)
    args = parser.parse_args()
    for relative in args.notebook or NOTEBOOKS:
        path = args.root / relative
        nbformat.write(build_notebook(args.root,relative),path)
        print(relative)


if __name__ == "__main__":
    main()
