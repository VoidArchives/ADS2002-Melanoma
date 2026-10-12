"""Build the read-only notebook for the complete five-fold image evaluation."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip() + "\n"))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip() + "\n"))


md("""
# 14 — Image model: five outer folds

## tl;dr

This report evaluates the bounded B0 recipe and duration selection process on five patient
groups. Each outer fold supplies predictions from a model that did not use its
patients for fitting, recipe selection or epoch selection in this run.
The primary measure is **mean outer-fold average precision (AP)**. AP measures
how effectively melanoma images appear near the top of the model's ranking.
Higher is better; fold prevalence supplies a simple ranking baseline.

The notebook reads saved results only. It does not start training, fit
calibration, choose a cutoff or open the reserved test. A partial run displays
its progress and withholds final five-fold conclusions.
""")
code("""
import json
import sys
from pathlib import Path
import pandas as pd
from IPython.display import display, Markdown, Image

PROJECT_ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
    if (path / "src/melanoma_detection").is_dir()
    and (path / "data/subset/processed/train_folds.csv").is_file())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from melanoma_detection.image_nested_reporting import load_saved_nested_run
from melanoma_detection.image_nested_calibration_reporting import load_saved_calibration_run

RUN_ID = "b0_nested_5fold_5h_v1"
CALIBRATION_RUN_ID = "b0_nested_calibration_5h_v1"
run = load_saved_nested_run(PROJECT_ROOT, RUN_ID)
if run["complete"]:
    result = run["evaluation"]
    lower, upper = result["primary_mean_fold_ap_ci95"]
    display(Markdown(f"**Mean outer-fold AP: {result['primary_mean_fold_average_precision']:.3f}** "
        f"(conditional 95% patient-bootstrap interval {lower:.3f}–{upper:.3f}). "
        f"{result['n_images']:,} images from {result['n_patients']:,} patients; "
        f"{result['n_positive']:,} melanoma images across all five folds."))
    print("Verified:", run["verification"])
else:
    display(Markdown(f"**Five-fold evaluation is incomplete: `{run['status']}`.** "
        "Rerun this notebook after training finishes. Final performance is withheld until "
        "all five folds have verified predictions and a final evaluation receipt."))
    progress = {key: value for key, value in run["summary"].items()
        if key in {"status", "completed_outer_folds", "stopping_reason", "updated_utc", "error"}}
    if progress:
        display(pd.DataFrame([progress]).astype(str).T.rename(columns={0: "Saved progress"}))
""")
md("""
## Context & Methods

The outer folds are the five existing patient groups in the canonical d1
development CSV. The complete process repeats independently inside every outer
training pool:

1. Build three inner patient folds. On the first inner fold, screen all **four
   B0 recipes for six epochs**: learning rate 3e-5 or 1e-4, crossed with unweighted
   or square-root-weighted loss. Only fitting patients determine the loss weight.
2. Choose the recipe with the highest screening AP. Scores within 0.0001 of the
   maximum tie; prefer unweighted loss and then the lower learning rate.
3. Confirm that selected recipe for **twelve total epochs on all three inner
   folds**. The compatible six-epoch first-fold checkpoint continues to twelve;
   the other two confirmation fits begin afresh.
4. Select the highest-AP epoch separately in each confirmation fit; exact ties
   keep the earlier epoch. The **median of those three best epochs** fixes the
   duration of a fresh refit on all outer training patients.
5. Predict the outer fold only after the fixed-duration refit completes. Outer
   labels do not guide fitting, recipe selection, epoch choice or stopping.

Confirmation AP chooses epochs inside training data; it is not outer performance.
The recipe and refit duration can differ between outer folds. Each canonical
image must receive exactly one outer prediction; missing, duplicate, extra or
relabelled rows fail verification.

### Key Assumptions

Images from one patient remain together in every partition. AP and other
metrics concern individual images; the uncertainty calculation resamples
**patients**, keeping all their images together. Patients are sampled with
replacement separately within each fixed outer fold. Draws lacking either
outcome in any fold are discarded consistently across all reported measures.
The report records requested, retained and rejected draws.

These intervals condition on the saved models, recipe choices and folds. They
do not represent variability from retraining or repeating model selection.
Mean fold AP is primary because scores from separate fitted models can have
different scales; pooled AP is a secondary description.

Earlier architecture, augmentation and recipe pilots used this development
dataset, and archived exploratory predictions used outer fold 0. This is
stronger **development evaluation**, but prior adaptation limits how fully it
represents an untouched evaluation. Freeze remaining decisions before the
single reserved-test evaluation.
""")
code("""
manifest = run["manifest"]
protocol = manifest.get("protocol")
if protocol:
    display(Markdown(f"Recorded protocol: `{protocol}`"))
config = manifest.get("config", {})
if config:
    display(pd.DataFrame([config.get("training", {})]).astype(str).T.rename(columns={0: "Recorded training setting"}))
    selection = config.get("selection", {})
    settings = {name: value for name, value in selection.items() if name != "recipes"}
    display(pd.DataFrame([settings]).astype(str).T.rename(columns={0: "Recorded selection setting"}))
    display(pd.DataFrame(selection.get("recipes", [])))
print("Run directory:", run["run_dir"])
print("Canonical development input:", manifest.get("development_csv", "Run not started"))
print("Reserved test loaded by this report: False")
""")
md("""
### Recorded recipe and refit duration by outer fold

Each completed outer fold records the recipe chosen on its first inner split
and the fixed refit duration obtained from three confirmation fits. This table
documents what was selected, rather than ranking recipes using outer labels.
An unfinished fold keeps missing settings until its selection is saved.
""")
code("""
fold_records = run["summary"].get("folds", [])
if fold_records:
    rows = [{"outer_fold": record["fold"], "status": record["status"],
        "selected_recipe": record.get("selected_recipe"),
        "refit_epochs": record.get("refit_epochs"),
        "incomplete_stage": record.get("stage")} for record in fold_records]
    display(pd.DataFrame(rows).sort_values("outer_fold").style.format(na_rep="Pending"))
else:
    print("No completed outer selection has been recorded yet.")

if run["complete"]:
    confirmation_rows = []
    for outer_fold in range(5):
        selection_path = run["run_dir"] / f"folds/{outer_fold}/selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        for confirmation in selection["confirmation"]:
            confirmation_rows.append({"outer_fold": outer_fold,
                "recipe": selection["recipe"]["id"],
                "inner_fold": confirmation["inner_fold"],
                "best_epoch": confirmation["best_epoch"],
                "inner_selection_ap": confirmation["best_metrics"]["average_precision"],
                "outer_refit_epochs": selection["refit_epochs"]})
    display(pd.DataFrame(confirmation_rows).style.format({"inner_selection_ap": "{:.4f}"}))
    print("Inner confirmation AP selects epochs. Held-out outer AP appears in Results below.")
""")
md("""
## Data

The verified population is the complete canonical development cohort. Counts
by fold make rare melanoma support and unequal image counts visible. All
models are scored on the same images when a metadata comparison is available.
No inner join can silently reduce the evaluation population.
""")
code("""
if run["complete"]:
    columns = ["fold", "n_images", "n_patients", "n_positive", "prevalence"]
    display(run["fold_metrics"][columns].style.format({"prevalence": "{:.2%}"}))
else:
    print("Complete population counts will appear after five-fold verification.")
""")
md("""
## Results

### Ranking and fold variation

The fold table and AP plot show how ranking performance varies between patient
groups. The spread between five fold scores is descriptive variability, not a
confidence interval. Precision–recall is particularly informative here because
melanoma images are uncommon; ROC shows a complementary ranking view.
""")
code("""
if run["complete"]:
    display(run["fold_metrics"].style.format({name: "{:.4f}" for name in
        ["prevalence", "average_precision", "roc_auc", "log_loss", "brier"]}))
    display(Image(filename=str(run["run_dir"] / "figures/outer_fold_average_precision.png")))
    display(Image(filename=str(run["run_dir"] / "figures/outer_fold_pr_roc.png")))
    intervals = run["intervals"]
    display(intervals[(intervals.model == "image")].drop(columns=["interval"]).style.format(
        {"estimate": "{:.4f}", "ci_lower": "{:.4f}", "ci_upper": "{:.4f}"}))
    print(run["evaluation"]["bootstrap_method"])
""")
md("""
### Raw probabilities and the declared 0.5 cutoff

Log loss and Brier score examine probability quality; lower is better. The
reliability plot compares mean raw score with the observed melanoma fraction
in fixed bins, and shows how many images support each point. It is descriptive:
sparse bins can be unstable. Class weighting can alter raw probability scale.

The confusion counts use the declared rule **raw score ≥ 0.5**. This cutoff is
a transparent default illustration, not a tuned operating point or a clinical
recommendation. No new threshold or calibration model is fitted here.
""")
code("""
if run["complete"]:
    display(run["threshold_metrics"].T.rename(columns={0: "Raw cutoff 0.5"}))
    display(Image(filename=str(run["run_dir"] / "figures/raw_reliability.png")))
    display(run["reliability"].style.format({"lower": "{:.1f}", "upper": "{:.1f}",
        "mean_probability": "{:.4f}", "observed_prevalence": "{:.4f}"}, na_rep="Empty bin"))
""")
md("""
### Optional training-only calibration and the 80%-recall scenario

This separate stage has its own run directory. It may start after the raw
five-fold evaluation finishes, if three fresh fixed-duration inner refits per
outer fold fit the same session deadline. Each uses that outer pool's already
selected recipe and median epoch duration. Its training-only OOF predictions
fit a logistic calibration map on logits (`C=1`) and choose the highest cutoff
reaching 80% malignant recall on those training predictions. A failed or
non-positive-slope map is recorded and leaves the separate calibration stage
incomplete; the raw report remains available. Complete frozen maps and cutoffs
are then applied to the unchanged raw outer predictions.

All five calibration folds must finish before the comparisons below appear.
Calibration is assessed through held-out log loss, Brier and reliability.
Positive-slope maps preserve ordering within a fold, so fold AP/ROC-AUC should
stay the same; numerical clipping or ties require inspection if they change.
Pooled AP may change when the five folds' score scales change.

The 80% recall value is a **training selection target**, not a promise about
actual outer recall or a clinically validated target. Report the actual held-out
recall, precision, F1, specificity and confusion counts. Calibration maps and
cutoffs remain fixed in the reported conditional bootstrap intervals. The raw
five-fold status and matched metadata comparison remain unchanged.
""")
code("""
calibration = load_saved_calibration_run(PROJECT_ROOT, CALIBRATION_RUN_ID)
if calibration["complete"]:
    display(Markdown("**The separate five-fold calibration report is complete and verified.**"))
    print("Verified:", calibration["verification"])
    display(pd.DataFrame(calibration["evaluation"]["mean_fold_metrics"]).T.style.format("{:.4f}"))
    probability_intervals = calibration["intervals"]
    display(probability_intervals[probability_intervals.metric.isin(["log_loss", "brier"])]
        .drop(columns="interval").style.format({"estimate": "{:.4f}", "ci_lower": "{:.4f}", "ci_upper": "{:.4f}"}))
    differences = calibration["paired_differences"]
    display(differences[differences.metric.isin(["log_loss", "brier"])].style.format(
        {"difference": "{:+.4f}", "ci_lower": "{:+.4f}", "ci_upper": "{:+.4f}"}))
    display(Image(filename=str(calibration["run_dir"] / "figures/raw_vs_calibrated_reliability.png")))
    print("Negative calibrated-minus-raw log-loss/Brier differences favour calibration.")
    print(calibration["evaluation"]["bootstrap_method"])
else:
    display(Markdown(f"**Separate calibration stage: `{calibration['status']}`.** "
        "No aggregate calibration or training-selected-threshold result is claimed until "
        "all five calibration folds and reporting checks complete."))
    progress = {key: value for key, value in calibration["summary"].items()
        if key in {"completed_outer_folds", "stopping_reason", "updated_utc", "error"}}
    if progress:
        display(pd.DataFrame([progress]).astype(str).T.rename(columns={0: "Calibration progress"}))
""")
code("""
if calibration["complete"]:
    display(calibration["ranking_invariance"].style.format({"raw_average_precision": "{:.4f}",
        "calibrated_average_precision": "{:.4f}", "ap_difference": "{:+.6f}",
        "raw_roc_auc": "{:.4f}", "calibrated_roc_auc": "{:.4f}"}))
    if calibration["evaluation"]["ranking_warning"]:
        display(Markdown(calibration["evaluation"]["ranking_warning"]))
    if calibration["evaluation"]["decision_warning"]:
        display(Markdown(calibration["evaluation"]["decision_warning"]))
    scenario_columns = ["fold", "model", "threshold", "n_images", "n_positive", "TP", "FN", "FP", "TN",
        "recall", "precision", "f1", "specificity", "flagged_fraction"]
    formats = {name: "{:.4f}" for name in ["threshold", "recall", "precision", "f1", "specificity", "flagged_fraction"]}
    display(calibration["scenario_by_fold"][scenario_columns].style.format(formats, na_rep="Undefined"))
    display(calibration["scenario_pooled"][scenario_columns].drop(columns="threshold")
        .style.format(formats, na_rep="Undefined"))
    display(Image(filename=str(calibration["run_dir"] / "figures/training_selected_threshold_counts.png")))
    display(Markdown(calibration["evaluation"]["threshold_interpretation"]))
""")
md("""
### Matched metadata comparison

When present, metadata scores come from the existing hash-recorded OOF
prediction bundle with verified canonical identities. The image-minus-metadata
interval uses the **same patient draw** for both models. Positive AP or ROC-AUC
differences favour the image model; negative log-loss or Brier differences
favour it. These are paired development comparisons after prior diagnostics.
""")
code("""
if run["complete"] and not run["metadata_comparison"].empty:
    display(run["metadata_comparison"].style.format({"difference": "{:+.4f}",
        "ci_lower": "{:+.4f}", "ci_upper": "{:+.4f}"}))
elif run["complete"]:
    print("No verified matched metadata comparison was saved for this run.")
""")
md("""
## Takeaways

Use mean-fold AP and its conditional patient interval together with fold
variation when assessing this image model. Pooled AP, raw-score reliability and
the declared-cutoff confusion counts answer complementary questions. Preserve
the development caveat when presenting these results; the reserved test is the
remaining independent assessment after all decisions are frozen.
""")
code("""
if run["complete"]:
    result = run["evaluation"]
    display(Markdown(f"The five fold AP scores range from "
        f"**{run['fold_metrics'].average_precision.min():.3f} to "
        f"{run['fold_metrics'].average_precision.max():.3f}**. "
        f"Pooled AP is **{result['pooled_metrics_secondary']['average_precision']:.3f}** "
        f"on a cohort with **{result['prevalence']:.2%}** melanoma images."))
    display(Markdown(result["development_caveat"]))
else:
    print("Performance takeaways remain pending until every outer fold completes.")
""")

if __name__ == "__main__":
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3 (project .venv)", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.11"}
    path = ROOT / "notebooks/Subset/14_image_nested_evaluation.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.validate(notebook)
    nbf.write(notebook, path)
    print(path)
