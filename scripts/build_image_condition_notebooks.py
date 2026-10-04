"""Build separate, thin augmentation and optional-lighting experiment notebooks."""

from pathlib import Path
import argparse
import json
import sys

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def build_notebook(*, lighting=False):
    cells = []
    def md(source):
        cells.append(nbf.v4.new_markdown_cell(source.strip() + "\n"))
    def code(source):
        cells.append(nbf.v4.new_code_cell(source.strip() + "\n"))
    run_id = "b0_lighting_12e_v1" if lighting else "b0_augmentation_12e_v1"
    title = "12 — Optional lighting correction with CLAHE" if lighting else "11 — Weak versus stronger B0 augmentation"
    status = ("**Current saved status:** C (lighting correction) was deferred because its\n"
        "required complete, safe A/B comparison was unavailable. No correction previews\n"
        "or C training were performed. The figures show saved control A only." if lighting else
        "**Current saved status:** A (weak augmentation) completed twelve epochs. B\n"
        "(stronger augmentation) failed fitting-image quality review and was not trained.\n"
        "The A/B comparison is incomplete; no performance winner is declared.")
    md(f"""
# {title}

## Purpose

This notebook reads saved results for a controlled B0 experiment. B0 is a
pretrained image network; the fixed control is a reference, not an established
optimum. **Augmentation** varies fitting images to discourage memorising their
orientation or appearance. **Preprocessing** applies a fixed preparation rule.
Changing one factor at a time helps isolate its effect.

An **epoch** is one pass through fitting data; a **checkpoint** stores the model
weights at an epoch. A complete fit runs twelve epochs, while its selected
checkpoint is the epoch with the highest validation average precision (AP).
Training defaults to disabled.

This is an **inner-validation pilot**: held-out fitting-pool patients guide
development. Their labels also select checkpoints, so the results are exploratory.
Full out-of-fold (OOF) evaluation would test the entire selection process on
independent outer patients. Outer fold 0 and the reserved test are excluded here.
No calibration is fitted. See the [concept guide](../concepts.md) for definitions.

{status}
""")
    code(f"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Markdown, Image as DisplayImage
PROJECT_ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
    if (path / "src/melanoma_detection").is_dir() and (path / "configs/image_model.yaml").is_file())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from melanoma_detection.image_conditions import load_config, prepare_conditions, run_conditions
from melanoma_detection.image_condition_reporting import load_condition_outputs, render_condition_report
from melanoma_detection.image_evaluation import sha256

TRAIN_CONDITIONS = False
comparison_label = {"'C - A'" if lighting else "'B - A'"}
config = load_config(PROJECT_ROOT / "configs/image_augmentation.yaml")
run_dir = PROJECT_ROOT / "results/image_d1_runs/{run_id}"
if TRAIN_CONDITIONS:
    prepared = prepare_conditions(PROJECT_ROOT, config, audit_images=True)
    run_conditions(PROJECT_ROOT, config, prepared=prepared, device="cuda",
        condition_ids={"('C',)" if lighting else "('A', 'B')"})
summary_path = run_dir / "summary.json"
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
else:
    summary = {{"run_id": "{run_id}", "status": "deferred" if {lighting!r} else "not_started",
        "conditions": {{}}, "reason": "No executed saved condition results; no results fabricated."}}
report_conditions = {{key: value for key, value in summary.get("conditions", {{}}).items()
    if key in {"('A', 'C')" if lighting else "('A', 'B')"}}}
bundle = load_condition_outputs(PROJECT_ROOT, {{**summary, "conditions": report_conditions}})
plt.rcParams.update({{"font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white"}})
print("Run:", summary["run_id"], "| status:", summary["status"], "| training enabled:", TRAIN_CONDITIONS)
print(summary.get("reason", summary.get("recommendation", "")))
observed_A = summary.get("conditions", {{}}).get("A", {{}})
observed_B = summary.get("conditions", {{}}).get("B", {{}})
if observed_A.get("status") == "complete":
    display(Markdown("**Executed weak control:** best AP "
        + format(observed_A["best_metrics"]["average_precision"], ".4f")
        + " at epoch " + str(observed_A["best_epoch"])
        + "; epoch-12 AP " + format(observed_A["epoch12_metrics"]["average_precision"], ".4f") + "."))
if observed_B.get("status") == "quality_blocked":
    display(Markdown("**B was not trained:** the predefined stronger profile failed fitting-image quality review. "
        "The A/B comparison is incomplete and no performance winner is declared."))
print("Outer evaluation: not performed | reserved test: not loaded")
""")
    md("""
## Controls and patient separation

The saved first inner fold lies within outer-training folds 1–4: 3,660 fitting
images from 225 patients and 1,834 validation images from 113 patients. All images
from a patient stay together, preventing direct training leakage.

Both conditions start from fresh pretrained weights and seed 42. The fixed B0
recipe uses dropout 0.2, fitting-only square-root positive weighting (to give
rare malignant examples more influence), backbone/classifier learning rates
3e-5/1e-3, AdamW weight decay 1e-4 and effective batch 64. BatchNorm is frozen
and automatic mixed precision (AMP) reduces GPU memory use. These settings stay
fixed so any completed comparison isolates the transformation change.

Each admitted fit runs twelve epochs with no early stopping, including one
classifier-only warmup; the fixed learning-rate schedule spans thirty epochs.
Highest AP selects the checkpoint, with exact ties retaining the earlier epoch.
Validation preparation is deterministic (repeatable), without random augmentation.
The shared runner owns saved settings, training, resume and time budgets; excluded
patients' labels do not influence those operations. Read the ledger's completion
status before interpreting a comparison.
""")
    code("""
ledger = pd.DataFrame([{key: record.get(key) for key in ["condition_id", "pipeline_id", "status",
    "completed_epochs", "best_epoch", "initial_model_state_sha256"]} for record in bundle["records"]])
if not ledger.empty:
    display(ledger)
else:
    print("No admitted saved fits. No learning or prediction results fabricated.")
print("Execution budget: 120 minutes total; training cutoff: minute 100; reporting reserve: 20 minutes")
if summary.get("runtime_estimates"):
    display(pd.DataFrame(summary["runtime_estimates"]).T)
""")
    if lighting:
        md("""
## Inspect lighting correction before training

**CLAHE** (contrast-limited adaptive histogram equalisation) adjusts local image
contrast while limiting strong amplification. Condition C would apply it to the
CIELAB lightness channel only (clip limit 2, 8 × 8 tiles), aiming to change
brightness contrast while limiting colour changes. Original files are preserved.
This fixed correction would precede weak fitting augmentation and also apply
before prediction; it differs from B's random brightness/contrast augmentation.

If admitted, original/corrected pairs from the same 24 fitting patients would
be displayed. Read each pair side by side: the full lesion boundary, fine
texture and pigmentation should remain visible, with plausible colour and no
amplified artifacts. Any lost lesion information or implausible colour blocks C.
This finite preview would not certify every image.

Admission requires complete safe A/B fits, passed numerical checks and enough
time, regardless of A/B AP. That gate was not met: there are no saved C preview
pages, and the following missing-preview message is expected. It is not evidence
that CLAHE helps or harms performance.
""")
        pattern = "clahe_pairs_*.png"
    else:
        md("""
## Inspect fitting-image quality before training

The frozen quality sample contains twelve malignant and twelve benign images
from 24 fitting patients, covering framing, hair and lighting variation. It is
a deliberate quality sample, not a prevalence estimate. Each saved page has
originals at left, four A (weak) variants, then four B (stronger) variants.
Each row is one image; `target=1` means melanoma and `target=0` benign. Seeds
make the random variants repeatable and are restored before training.

**Read across each row:** compare the complete lesion edge, texture and pigment
with the original. Look for cropping, distorted boundaries, lost texture,
implausible colour or reflected copies of the lesion at the border. A changed
orientation alone is expected; lost lesion information is not. The recorded
severity scale is 0 (no defect), 1 (minor background-only artifact), 2 (useful
lesion information lost or implausible appearance) and 3 (major damage).
Any severity ≥2 blocks that candidate. An unfinished review is not approval.

A uses flips, ±20° nearest-neighbour rotation and black fill. B adds ±45°
bilinear rotation, ±5% translation, scale 0.9–1.05, reflected padding and
brightness/contrast ±20%, saturation ±10%, with hue unchanged. It has no random
resized crop or masking. The saved review blocked B; this is a quality decision,
not evidence that a stronger model would perform worse. The finite sample cannot
certify every development image.
""")
        pattern = "augmentation_matched_*.png"
    code(f"""
quality_path = run_dir / "quality_review.json"
if quality_path.is_file():
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    print("Reviewer:", quality.get("reviewed_by"), "| time:", quality.get("reviewed_at_utc"))
    print("Gate:", {{key: value.get("verdict") for key, value in quality.get("conditions", {{}}).items()}})
    display(pd.DataFrame(quality.get("selected_examples", []))[["image_id", "patient_id", "target"]])
    print("Sample coverage:", quality.get("sample_coverage"))
    flagged_reviews = []
    for condition_id, condition in quality.get("conditions", {{}}).items():
        for review in condition.get("reviews", []):
            if any(isinstance(review.get(name), int) and review[name] >= 2 for name in
                   ["clipping", "boundary", "texture", "colour", "reflection"]):
                flagged_reviews.append({{"condition_id": condition_id, **review}})
    if flagged_reviews:
        display(Markdown("**Recorded quality failures — these conditions cannot train.**"))
        display(pd.DataFrame(flagged_reviews))
else:
    print("Quality review unavailable; this notebook does not approve it.")
preview_paths = sorted((run_dir / "quality_previews").glob("{pattern}"))
for path in preview_paths:
    display(DisplayImage(filename=str(path)))
if not preview_paths:
    print("No executed preview pages saved for this condition.")
""")
    md("""
## Saved learning and prediction results

Only committed epochs and complete twelve-epoch prediction records are shown.
Each line names an available condition; a missing condition has no performance
result. The current saved figures show the weak control A only, so they cannot
establish an A/B or A/C winner.

**Read the figures in order:**

1. **Learning curves:** completed epoch is horizontal. Upper panels show AP
   (precision across recall levels) and ROC-AUC (overall ranking), where higher
   is better. Guides mark malignant prevalence and chance AUC 0.5. Lower panels
   show log loss (penalty for confident wrong scores) and Brier (mean squared
   score error), where lower is better. Hollow rings mark the AP-selected epoch
   in all panels; the vertical dotted line separates warmup from fine-tuning.
2. **Fitting loss:** epoch is horizontal and binary cross-entropy (BCE, the
   training objective) vertical. Falling fitting loss means a closer fit to
   training labels; compare it with validation trends. All conditions here use
   the same square-root weighting; differently weighted historical losses are
   not directly comparable.
3. **Best-checkpoint ROC/precision–recall:** each line is a condition. ROC plots
   benign false-positive rate horizontally against melanoma recall vertically;
   closer to the upper left beats the chance diagonal. Precision–recall plots
   recall horizontally against precision (malignancy among flags) vertically;
   higher at the same recall is better, with prevalence as a reference. Moving
   along a curve changes the score cutoff. AP is not its trapezoidal area.
4. **Confusion matrices at 0.5:** rows are true class, columns predicted class.
   Diagonal cells are correct; bottom left is missed melanoma and top right
   benign images flagged. Text gives counts and percentages within each row;
   darker blue means a larger row fraction. Large benign counts alone are not
   evidence of good melanoma detection.
5. **Reliability:** mean score within each bin is horizontal, observed melanoma
   fraction vertical. The diagonal means agreement; below it scores exceed
   observed frequency, above it the reverse. Check bin support before interpreting
   sparse positives. These raw scores are not calibrated probabilities.
6. **Epoch-12 ROC/precision–recall:** read the axes as in figure 3. These use the
   final scheduled checkpoint rather than the AP-selected checkpoint, helping
   show whether later fitting improved the same patients' ranking.

The two metric tables keep best and epoch-12 scores separate. AP is primary
because malignancy is rare; loss and Brier can look small even with poor recall.
Reliability is descriptive and does not fit a calibrator. The resource table
sums measured epoch time and reports peak allocated GPU memory, excluding setup.
""")
    code("""
report = render_condition_report(bundle, run_dir)
if not report["resources"].empty:
    display(Markdown("### Measured training resources"))
    display(report["resources"])
for figure in report["figures"]:
    display(figure)
    plt.close(figure)
for name, table in [("Highest-AP checkpoints", report["metrics"]),
                    ("Epoch-12 checkpoints", report["epoch12_metrics"])]:
    display(Markdown("### " + name))
    if table.empty:
        print("No complete twelve-epoch records available.")
    else:
        display(table[["pipeline_id", "selected_epoch", "n_images", "n_patients", "n_melanoma",
            "average_precision", "roc_auc", "log_loss", "brier"]].style.format(
            {name: "{:.4f}" for name in ["average_precision", "roc_auc", "log_loss", "brier"]}))
""")
    md("""
## Classification at the declared cutoff

The **raw score ≥0.5** cutoff is a fixed reporting scenario. Recall/sensitivity
measures malignant images found; specificity measures benign images correctly
unflagged. Precision measures malignancy among flags; F1 balances precision
and recall. TP/TN are correct malignant/benign decisions; FN/FP are missed
malignancies/false alarms. Undefined precision remains missing.

Scores are uncalibrated, and no cutoff is selected using these validation labels.
The later 80%-recall scenario and any fitted score-to-probability mapping require
training-only OOF predictions. Good ranking alone does not make 0.5 useful.
""")
    code("""
if not report["metrics"].empty:
    display(report["metrics"][["pipeline_id", "selected_epoch", "threshold", "score_type",
        "TP", "FP", "TN", "FN", "precision", "recall", "sensitivity", "specificity", "f1"]]
        .style.format({name: "{:.4f}" for name in
            ["precision", "recall", "sensitivity", "specificity", "f1"]}, na_rep="Undefined"))
else:
    print("No complete prediction records available for classification reporting.")
""")
    md("""
## Compare changes only when both fits are complete

The table reports candidate AP minus control AP: positive favours the candidate.
Compare B (stronger augmentation) or C (lighting correction) with fresh A only
when both completed the same twelve-epoch budget and passed quality checks.
A missing AP difference is an unavailable comparison, not zero improvement.

The predeclared AP gain of 0.005 is a screening margin for a promising B0
hypothesis, not statistical proof, a clinical rule or an established optimum.
Smaller gains retain weak/unchanged as the reference. Quality-blocked, paused
or deferred conditions cannot be called worse-performing. B0 findings cannot
select transformations for DenseNet or ConvNeXt. A later combined augmentation
and CLAHE experiment would need individual quality gates and a matched comparison
against the better individual change.
""")
    code("""
display(report["comparisons"].loc[report["comparisons"].comparison.eq(comparison_label)])
completed = [row for row in bundle["records"] if row.get("status") == "complete"]
if completed:
    late = [row["condition_id"] for row in completed if row.get("best_epoch", 0) >= 10]
    if late:
        print("Highest-AP epochs near the twelve-epoch cap:", ", ".join(late),
            "— further training is a later validation question.")
    else:
        print("Highest-AP checkpoints precede the cap. Increasing the epoch budget alone is not supported by this split; "
            "confirm duration across three patient-grouped inner folds with the later stopping rules.")
receipt = {"run_id": summary["run_id"], "training_enabled_in_notebook": TRAIN_CONDITIONS,
    "status": summary["status"], "threshold": .5, "threshold_tuned": False,
    "calibration_fitted": False, "reserved_test_loaded": False, "outer_evaluation": False,
    "reporting_source_sha256": {relative: sha256(PROJECT_ROOT / relative) for relative in [
        "src/melanoma_detection/image_condition_reporting.py", "scripts/build_image_condition_notebooks.py"]},
    "summary_sha256": sha256(summary_path) if summary_path.is_file() else None,
    "quality_review_sha256": sha256(quality_path) if quality_path.is_file() else None,
    "figures": [str(path.relative_to(run_dir)) for path in report["figure_paths"]],
    "figure_sha256": {str(path.relative_to(run_dir)): sha256(path) for path in report["figure_paths"]},
    "prediction_csv_sha256": {record[key]: sha256(Path(record[key]) if Path(record[key]).is_absolute()
        else PROJECT_ROOT / record[key]) for record in bundle["records"] if record.get("status") == "complete"
        for key in ["predictions_file", "epoch12_predictions_file"] if record.get(key)}}
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "reporting_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("Saved outputs:", run_dir)
print("No optimal transformation claimed | no final architecture selected")
""")
    md("""
## Later work

First inspect a safer augmentation profile and complete a fresh matching
comparison. CLAHE still requires its admission gate. Hair removal, combined
changes, longer training and three-inner-fold confirmation remain separate.
Every architecture needs its own recipe and preparation choices.

The entire selection process must later repeat within each outer-training
partition before full OOF evaluation, calibration and any threshold fitting.
Current [uncertainty diagnostics](13_image_uncertainty.ipynb) do not replace that
evaluation. Fusion and the single reserved-test evaluation remain pending.
""")
    md("""
## Limitations

One inner split contains only 33 malignant validation images, so results can
vary with the patient sample. Control and epoch choices reuse development labels;
these scores are not unbiased outer-fold evidence. A finite quality preview
cannot certify every lesion, and an untrained condition has no performance
result. Class-weighted scores are uncalibrated; 0.5 is a reporting cutoff rather
than a clinical rule. B0 findings do not establish another architecture's choices.

## Summary

Read completion and image-quality status before the curves. The saved control
provides a reference, but the unavailable candidate comparison supports no
performance winner. The observations below guide the next safe, matched
experiment; full outer evaluation remains necessary.
""")
    code("""
if observed_A.get("status") == "complete":
    print("Fresh weak B0 control: twelve epochs complete; highest AP",
        format(observed_A["best_metrics"]["average_precision"], ".4f"),
        "at epoch", observed_A["best_epoch"], "; epoch-12 AP",
        format(observed_A["epoch12_metrics"]["average_precision"], ".4f"))
comparison = report["comparisons"].loc[report["comparisons"].comparison.eq(comparison_label)]
display(comparison)
if summary.get("status") in ("incomplete", "deferred"):
    print("No comparative advancement decision. Preserve weak/unchanged as the development reference.")
print("Next: declare and inspect a safer augmentation profile, then complete a fresh matched comparison. "
    "CLAHE still requires its admission gate; longer confirmation and architecture-specific selection remain later stages.")
""")
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python (ADS2002 Melanoma)",
        "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python"}
    nbf.validate(notebook)
    for index, cell in enumerate(cells):
        if cell.cell_type == "code":
            compile(cell.source, f"condition_report_cell_{index}", "exec")
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="Execute report-only notebooks with this Python; training remains disabled.")
    parser.add_argument("--only", choices=["augmentation", "lighting", "both"], default="both")
    args = parser.parse_args()
    for lighting, filename in [(False, "11_image_augmentation.ipynb"),
                                (True, "12_image_lighting.ipynb")]:
        if args.only != "both" and lighting != (args.only == "lighting"):
            continue
        path = ROOT / "notebooks/Subset" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        notebook = build_notebook(lighting=lighting)
        if args.execute:
            from jupyter_client import KernelManager
            from nbclient import NotebookClient
            manager = KernelManager(kernel_name="python3")
            manager.kernel_spec.argv[0] = sys.executable
            NotebookClient(notebook, km=manager, timeout=600,
                resources={"metadata": {"path": str(path.parent)}}).execute()
            nbf.validate(notebook)
        nbf.write(notebook, path)
        if args.execute:
            sys.path.insert(0, str(ROOT / "src"))
            from melanoma_detection.image_evaluation import sha256
            outputs = [output for cell in notebook.cells if cell.cell_type == "code"
                       for output in cell.get("outputs", [])]
            errors = [output for output in outputs if output.output_type == "error"]
            run_id = "b0_lighting_12e_v1" if lighting else "b0_augmentation_12e_v1"
            receipt_path = ROOT / "results/image_d1_runs" / run_id / "notebook_execution_receipt.json"
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps({"notebook": str(path.relative_to(ROOT)),
                "notebook_sha256": sha256(path), "training_enabled": False,
                "executed_code_cells": sum(cell.cell_type == "code" and cell.execution_count is not None
                                           for cell in notebook.cells),
                "error_count": len(errors), "embedded_png_count": sum("image/png" in output.get("data", {})
                    for output in outputs)}, indent=2), encoding="utf-8")
        print(path)
