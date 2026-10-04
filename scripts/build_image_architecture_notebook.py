"""Build a thin architecture experiment using shared training/reporting utilities."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip() + "\n"))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))

md("""
# 10 — Compare image-model architectures

Does changing the image network improve early ranking at an affordable cost?
This notebook compares pretrained convolutional neural networks on matching
patients: **EfficientNet-B0** is a compact reference; **DenseNet121** passes
earlier features to later layers; **ConvNeXt-Tiny** uses a modernised convolutional
design. Different designs may learn useful patterns at different speeds.

An architecture is the network design; a **recipe** specifies how it is trained.
A **checkpoint** stores weights after an epoch (one pass through fitting data).
For each recipe, the highest validation AP checkpoint is reported.

These are **inner-validation diagnostics**: held-out fitting-pool patients guide
recipe and epoch choices. Full out-of-fold (OOF) evaluation would test the entire
selection process on independent outer patients. Outer fold 0 and the reserved
test are excluded here. Eight new fits were budgeted; the saved completion
ledger establishes what actually ran. Short pilots cannot eliminate a slower
family or select the final model.

Training defaults to disabled. The notebook reads executed results using shared
training/reporting utilities. See the [concept guide](../concepts.md) for terms.
""")
code("""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Markdown
PROJECT_ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
    if (path / "src/melanoma_detection").is_dir() and (path / "configs/image_model.yaml").is_file())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from melanoma_detection.image_architecture import load_config, prepare_architecture_pilot, run_architecture_pilot
from melanoma_detection.image_evaluation import load_historical_pilot, load_architecture_outputs, metric_table, curve_tables, sha256
from melanoma_detection.image_reporting import (plot_learning_curves, plot_training_losses,
    plot_roc_precision_recall, plot_confusion_matrices, plot_resource_comparison,
    plot_raw_reliability, plot_augmentation_preview, save_figure)

TRAIN_ARCHITECTURES = False
config = load_config(PROJECT_ROOT / "configs/image_architecture_pilot.yaml")
run_dir = PROJECT_ROOT / config["run"]["output_dir"] / config["run"]["run_id"]
historical = load_historical_pilot(PROJECT_ROOT)
if TRAIN_ARCHITECTURES:
    prepared = prepare_architecture_pilot(PROJECT_ROOT, config, audit_images=True)
    summary = run_architecture_pilot(PROJECT_ROOT, config, prepared=prepared, resume=True, device="cuda")
else:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Prepare/run the shared architecture CLI first; this notebook does not fabricate results.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
bundle = load_architecture_outputs(PROJECT_ROOT, summary)
predictions, matched, history = bundle["predictions"], bundle["matched_predictions"], bundle["histories"]
all_records = [*summary.get("b0_reference", []), *summary.get("recipes", [])]
epochs = {record["pipeline_id"]: record.get("best_epoch") for record in all_records}
figures_dir, tables_dir = run_dir / "figures", run_dir / "tables"
tables_dir.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white"})
print("Run:", summary["run_id"], "| status:", summary["status"], "| training enabled:", TRAIN_ARCHITECTURES)
print("Matched completed slots:", summary["matched_recipe_slots"])
print("Unmatched completed trials:", summary["unmatched_completed_trials"])
display(Markdown(summary["recommendation"]))
""")
md("""
## Shared preparation and the completion ledger

Every family uses the same patient memberships and whole RGB images resized
to 224 × 224 pixels, with the same weak fitting flips/rotation and ImageNet
colour normalisation. Ordinary validation has no random augmentation. These
controls make the network comparison easier to interpret.

Each family tries its own low/high backbone learning rates (the size of weight
updates), with either no class weighting or fitting-only square-root positive
weighting (more influence for rare malignant examples). Thus `low_sqrt` is a
matched recipe **slot**, not an identical numerical learning rate across networks.
Effective batch size is 64 images. Each admitted fit runs six epochs, including
one classifier-only warmup, under a fixed 30-epoch learning-rate schedule. The
selected epoch is the highest AP so far; exact ties retain the earlier epoch.

B0/DenseNet keep BatchNorm statistics and parameters fixed, limiting changes
to their pretrained normalisation with small image batches. ConvNeXt retains
its native LayerNorm, trained during full fine-tuning. All new classifiers use
dropout 0.2 and one logit, converted to a score between 0 and 1. The
[concept guide](../concepts.md) explains these terms.

Read `status` and `completed_epochs` before comparing results. Only completed
matching slots rank families; a paused trial is unfinished, not inferior. The
original session allowed at most 50 training minutes within its wider deadline.
""")
code("""
ledger_rows = []
for record in all_records:
    ledger_rows.append({"pipeline_id": record["pipeline_id"], "family": record["architecture"],
        "recipe_slot": record["recipe_slot"], "status": record["status"],
        "completed_epochs": record.get("completed_epochs"), "selected_epoch": record.get("best_epoch"),
        "backbone_lr": record.get("backbone_lr"), "loss_weight": record.get("loss_weight"),
        "matched_comparison": record["recipe_slot"] in summary["matched_recipe_slots"]})
ledger = pd.DataFrame(ledger_rows)
ledger.to_csv(tables_dir / "completion_ledger.csv", index=False)
display(ledger)
display(pd.DataFrame(summary.get("runtime_estimates", {})).T)
print("Ordinary validation: deterministic | calibrated scores: unavailable | tuned thresholds: deferred")
""")
md("""
**Read the preparation preview:** each row contains one fitting image; the
original is left and three seeded weak variants follow. `target=1` means melanoma,
`target=0` benign. Compare the full lesion boundary, texture and colour. Rotation
can create black corners or clip peripheral pixels. This small deliberate label
mix checks image preparation; it does not certify every lesion or represent
dataset prevalence. Later augmentation and lighting stages are separate.
""")
code("""
preview = plot_augmentation_preview(historical["fitting"], config["model"])
save_figure(preview, figures_dir / "weak_augmentation_preview.png", dpi=130)
plt.show()
""")
md("""
## Learning over epochs

For each family, the first figure uses completed epoch horizontally and four
validation metrics vertically. Each line is a recipe; paused runs are labelled.
**AP** summarises precision across recall levels and is useful for rare melanoma;
**ROC-AUC** measures overall ranking. Higher is better for both; their guides
are malignant-image prevalence and chance AUC 0.5. **Log loss** penalises confident
wrong scores and **Brier** averages squared score errors; lower is better. A low
loss can still accompany poor melanoma detection, so read the ranking panels too.

Hollow rings mark the AP-selected epoch across all panels. The dotted vertical
line separates warmup from whole-network fine-tuning. The next figure plots
fitting BCE (binary cross-entropy, the training objective) against epoch, grouped
by weighting. Falling fitting loss should be checked against validation trends.
Weighted and unweighted fitting-loss magnitudes are not comparable. Six epochs
show early progress, not each family's full learning potential.
""")
code("""
if not history.empty:
    for family, family_history in history.groupby("architecture", sort=False):
        fig = plot_learning_curves(family_history, historical["validation"].target.mean(),
            f"{family}: exploratory six-epoch inner validation")
        save_figure(fig, figures_dir / f"{family}_learning_curves.png")
        plt.show()
        loss = plot_training_losses(family_history, title=f"{family}: fitting objectives separated by loss weighting")
        save_figure(loss, figures_dir / f"{family}_training_loss.png")
        plt.show()
else:
    print("No completed trial histories available; no curves fabricated.")
""")
md("""
## Compare completed matching recipes

Only slots completed for **all three families** enter these tables and plots.
All scores concern the same validation images and patients. AP is primary;
log loss and Brier describe score errors, and ROC-AUC supplies another ranking
view. Raw scores remain uncalibrated.

**Read the horizontal bars:** each row is a network/recipe. The three panels
show selected-checkpoint AP (longer is better), total measured training minutes
and peak allocated GPU memory in GiB (shorter is cheaper). Time sums the recorded
epochs despite the figure's "epoch time" title and excludes downloads/setup;
memory is allocated by training, not total device usage. Assess the trade-off
without treating the small AP difference as final evidence.

**Read each matched-slot curve pair:** each line is one family. The left ROC
panel plots benign false-positive rate horizontally against melanoma recall
vertically; nearer the upper left is better than the chance diagonal. The right
precision–recall panel plots recall horizontally against precision (malignancy
among flags) vertically; higher at the same recall is better. The horizontal
guide is prevalence. Moving along a curve changes the cutoff; the legend names
the selected epoch and metric. AP is not the trapezoidal area under the plotted
steps. No curve chooses a referral or classification cutoff here.
""")
code("""
if matched:
    metrics = metric_table(matched, threshold=.5, epochs=epochs)
    metrics.to_csv(tables_dir / "matched_metrics_raw_threshold_0_5.csv", index=False)
    display(metrics[["pipeline_id", "selected_epoch", "average_precision", "roc_auc", "log_loss", "brier",
        "n_images", "n_patients"]].style.format({key: "{:.4f}" for key in
        ["average_precision", "roc_auc", "log_loss", "brier"]}))
    matched_history = history.loc[history.pipeline_id.isin(matched)]
    resource = plot_resource_comparison(matched_history, metrics)
    save_figure(resource, figures_dir / "matched_performance_time_memory.png")
    plt.show()
    for slot in summary["matched_recipe_slots"]:
        paired = {key: frame for key, frame in matched.items() if frame.recipe_slot.iloc[0] == slot}
        curves = plot_roc_precision_recall(paired, epochs=epochs,
            title=f"Matched {slot}: exploratory family ROC / precision–recall")
        save_figure(curves, figures_dir / f"{slot}_roc_precision_recall.png")
        plt.show()
        for identity, frame in paired.items():
            for name, table in curve_tables(frame.target, frame.base_probability).items():
                if isinstance(table, pd.DataFrame):
                    table.to_csv(tables_dir / f"{identity.replace('/', '__')}_{name}.csv", index=False)
else:
    print("No matched DenseNet/ConvNeXt/B0 slot completed; no family ranking reported.")
""")
md("""
## Fixed-cutoff errors and descriptive reliability

Flag a raw score `>= 0.5`; the stored `base_probability` column is an uncalibrated
score despite its name. The cutoff is declared rather than fitted to these labels.
Recall/sensitivity measures malignant images found; specificity measures benign
images correctly unflagged. Precision measures malignancy among flags; F1
balances precision and recall. Undefined precision remains missing.

**Read the confusion matrices:** one panel per family, rows are true class and
columns predicted class. Diagonal cells are correct; bottom left is missed
melanoma, top right is benign images flagged. Text gives counts and percentages
within each true-class row; darker blue means a larger row fraction. Compare
missed malignancies and false alarms alongside AP.

**Read the reliability panels:** horizontal position is mean score in a bin;
vertical position is observed melanoma fraction. Points on the diagonal agree;
below it scores exceed the observed frequency, above it scores fall below it.
Panels share the observed-range axes and the table gives bin support. Few
malignant images and shared patients make bins noisy. This describes reliability
without fitting a calibrator or validating clinical risk. An 80%-recall cutoff
requires later training-only OOF predictions.
""")
code("""
if matched:
    display(metrics[["pipeline_id", "selected_epoch", "threshold", "score_type", "TP", "FP", "TN", "FN",
        "precision", "recall", "sensitivity", "specificity", "f1"]].style.format(
        {name: "{:.4f}" for name in ["precision", "recall", "sensitivity", "specificity", "f1"]}, na_rep="Undefined"))
    for slot in summary["matched_recipe_slots"]:
        paired = {key: frame for key, frame in matched.items() if frame.recipe_slot.iloc[0] == slot}
        confusion = plot_confusion_matrices(paired, epochs=epochs,
            title=f"Matched {slot}: exploratory classification at raw threshold 0.5")
        save_figure(confusion, figures_dir / f"{slot}_confusion_threshold_0_5.png")
        plt.show()
        reliability, bins = plot_raw_reliability(paired, title=f"Matched {slot}: exploratory raw-score checks")
        save_figure(reliability, figures_dir / f"{slot}_raw_reliability.png")
        bins.to_csv(tables_dir / f"{slot}_reliability_support.csv", index=False)
        plt.show()
        display(bins)
""")
md("""
## Incomplete records and next steps

The table below lists any completed unmatched trials separately. They cannot
rank families. Paused fits preserve model and training states for compatible
resumption at epoch boundaries; reaching a time budget is not evidence against
their performance. Complete matching comparisons before expanding conclusions.

Next, validate every family with longer training and three inner patient folds.
Choose augmentation and preparation separately for each architecture, then
repeat the complete selection procedure within each outer-training partition.
The current [uncertainty notebook](13_image_uncertainty.ipynb) supplies inference
diagnostics only. Fusion, final outer evaluation, any eligible Swin experiment
and reserved-test evaluation remain separate work.
""")
code("""
unmatched = {key: frame for key, frame in predictions.items()
    if key in set(summary["unmatched_completed_trials"])}
if unmatched:
    unmatched_metrics = metric_table(unmatched, threshold=.5, epochs=epochs)
    unmatched_metrics.to_csv(tables_dir / "unmatched_exploratory_records.csv", index=False)
    display(unmatched_metrics)
else:
    print("Unmatched completed new trials: none.")
(run_dir / "reporting_receipt.json").write_text(json.dumps({
    "run_id": summary["run_id"], "training_enabled_in_notebook": TRAIN_ARCHITECTURES,
    "matched_recipe_slots": summary["matched_recipe_slots"], "threshold": .5,
    "threshold_tuned": False, "calibration_fitted": False, "test_loaded": False,
    "status": summary["status"],
    "trial_statuses": {record["pipeline_id"]: record["status"] for record in all_records},
    "reporting_source_sha256": {relative: sha256(PROJECT_ROOT / relative) for relative in [
        "src/melanoma_detection/image_evaluation.py", "src/melanoma_detection/image_reporting.py",
        "scripts/build_image_architecture_notebook.py"]},
    "summary_sha256": sha256(run_dir / "summary.json"),
    "prediction_csv_sha256": {str(Path(record["predictions_file"]).relative_to(PROJECT_ROOT)):
        sha256(record["predictions_file"]) for record in all_records
        if record.get("status") == "complete" and record.get("predictions_file")},
    "figures": sorted(path.name for path in figures_dir.glob("*.png"))}, indent=2), encoding="utf-8")
print("Artifacts:", run_dir)
print("Outer evaluation: not performed | reserved test: not loaded")
""")
md("""
## Limitations

- Six epochs on one inner split measure early learning rather than final potential.
- Choosing recipes and epochs on these validation labels makes the results optimistic.
- The shared weak preparation controls this comparison; it does not establish each family's best preparation.
- Weighted fitting objectives and raw score scales differ; 0.5 is an untuned reporting cutoff.
- Paused or unmatched trials cannot lose a comparison; short results cannot rule out slower families.
- Longer confirmation, calibration, full outer evaluation, fusion and the reserved test remain pending.
""")
md("""
## Summary

The saved comparison shows early ranking, fixed-cutoff errors, reliability and
cost on matching patients. The family observations below are exploratory.
No final architecture is selected; confirming the selection process on independent
outer patients remains necessary.
""")
code("""
if matched:
    observed = metric_table(matched, threshold=.5, epochs=epochs)
    observed["architecture"] = observed.pipeline_id.str.split("/").str[0]
    family_best = observed.sort_values(["architecture", "average_precision", "selected_epoch", "pipeline_id"],
        ascending=[True, False, True, True]).groupby("architecture", sort=False).head(1)
    family_best = family_best.rename(columns={"pipeline_id": "observed_recipe", "selected_epoch": "best_epoch"})
    print("All eight new trials completed." if summary["comparison_complete"] else
        "Partial comparison: family observations use completed matching slots only.")
    display(family_best[["architecture", "observed_recipe", "best_epoch", "average_precision", "roc_auc"]])
    late = family_best.loc[family_best.best_epoch.ge(5), "architecture"].tolist()
    if late:
        display(Markdown("Best observed epochs are near the six-epoch cap for **" +
            ", ".join(late) + "**. Their late peaks support additional training; they do not establish optimal durations."))
else:
    print("No matching slot completed for all three families; there is no comparative family summary yet.")
if not summary["comparison_complete"]:
    outstanding = ledger.loc[ledger.status.ne("complete"),
        ["pipeline_id", "status", "completed_epochs", "selected_epoch"]]
    display(outstanding)
    display(Markdown("Finish missing or paused matching trials before expanding conclusions. "
        "Budget exhaustion is not evidence against unfinished candidates."))
display(Markdown("Next, independently validate **all three families** with their own recipes, "
    "a **30-epoch cap** and **three inner patient folds**. The cap is a maximum budget, "
    "distinct from the validation-selected best epoch. No longer training starts in this reporting notebook."))
print("No final architecture selected | outer evaluation: not performed | reserved test: not loaded")
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name": "Python (ADS2002 Melanoma)", "language": "python", "name": "python3"}
notebook.metadata.language_info = {"name": "python"}
nbf.validate(notebook)
for index, cell in enumerate(cells):
    if cell.cell_type == "code": compile(cell.source, f"architecture_report_cell_{index}", "exec")
path = ROOT / "notebooks/Subset/10_image_architecture_comparison.ipynb"
path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, path)
print(f"Created {path} ({len(cells)} cells)")
