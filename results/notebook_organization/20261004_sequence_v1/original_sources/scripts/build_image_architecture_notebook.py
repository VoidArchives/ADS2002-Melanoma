"""Build a thin architecture experiment using shared training/reporting utilities."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip() + "\n"))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))

md("""
# 10 — d1 architecture exploration within a three-hour session

EfficientNet-B0 is the reference, not a predetermined winner. Compare its saved
six-epoch records with independently configured pretrained DenseNet121 and
ConvNeXt-Tiny on the same inner validation patients. All eight new fits are a
maximum target, not a completion promise. A short trial cannot eliminate a
slower family. Outer fold 0 and the reserved test remain excluded.

The notebook defaults to reading executed results. Training, when explicitly
enabled, calls the shared runner and existing training loop. No independent
training implementation appears here.
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
## Controlled preparation, recipes and completion ledger

The same patient memberships, RGB whole-image 224-pixel bilinear preparation,
weak flips/rotation and unchanged preprocessing apply to every family. ImageNet
normalisation remains fixed. B0/DenseNet freeze BatchNorm statistics and affine
parameters; ConvNeXt preserves its native LayerNorm and trains it during full
fine-tuning. Each new output uses explicit dropout 0.2 and one logit.

Each family has its own lower/higher backbone learning rates crossed with
unweighted or fitting-only square-root positive weighting. Effective batch is
64. Use one classifier-only warmup and a fixed 30-epoch schedule, executing six
epochs with no early stopping. The selected checkpoint is actual highest AP;
exact ties retain the earlier epoch.

Finish DenseNet/ConvNeXt in matched pairs before starting the next slot. Paused
trials cannot lose a comparison. At most 50 minutes are available for training,
subject to the overall session deadline and reporting reserve.
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
Fitting-only weak-preparation preview. The existing rotation can create black
corners and clip peripheral pixels; the limited preview does not certify every
lesion. Stronger augmentation, hair removal and lighting correction are deferred.
""")
code("""
preview = plot_augmentation_preview(historical["fitting"], config["model"])
save_figure(preview, figures_dir / "weak_augmentation_preview.png", dpi=130)
plt.show()
""")
md("""
## Executed learning trajectories

The panels below use committed epochs only, including paused trials explicitly
marked in their legends. Hollow rings mark highest-AP epochs so far.
Weighted and unweighted fitting losses represent different objectives and are
shown separately; validation log loss and Brier are unweighted for all trials.
Checkpoints from a six-epoch cap remain exploratory duration choices.
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
## Matched family comparison

Only recipe slots completed for **all three families** enter the comparative
tables and plots. All use matching validation images and patients. AP is primary;
ROC-AUC and actual precision–recall/ROC curves describe ranking. Measured epoch
time excludes weight-download/setup overhead. Peak memory is measured allocated
GPU memory, not total device usage. These are early learning results, not a
final architecture selection.
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
## Fixed-threshold exploratory classification and raw calibration

Flag scores `>= 0.5` using raw base probabilities. The cutoff is declared, not
fitted to validation labels. Confusion plots show counts and percentages within
each true class; report F1, precision, recall/sensitivity and specificity with
the cutoff, checkpoint and support. Undefined precision remains missing.

Reliability plots are descriptive raw-score checks, not fitted calibration or
clinical risk. The academic 80%-recall cutoff awaits training-only OOF scores.
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
## Unmatched executed records and next steps

Completed unmatched trials are saved below separately. They are not used to
rank families. Paused fits retain compatible model/optimiser/scheduler/scaler/RNG
states and resume at epoch boundaries; budget exhaustion supplies no evidence
against them. Finish paused comparisons before independently validating every
family with longer training and three inner folds.

Later stages test augmentation/preprocessing separately for each architecture
and repeat all selection inside five outer-training partitions. Short-pilot
results cannot globally prune candidates. Swin, final evaluation, uncertainty,
fusion and reserved-test evaluation remain outside this session.
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

- Six epochs on one inner patient split measure early behaviour, not each family's full learning potential.
- Best recipes and epochs reuse validation labels. These are exploratory observations, not final outer performance.
- The fixed weak/unchanged profile controls this comparison; each architecture still requires independent augmentation/preprocessing selection later.
- Fitting objectives differ under weighting. Raw scores are uncalibrated and threshold 0.5 is not tuned.
- Paused/unmatched trials cannot lose a comparison, and short results cannot exclude slower families.
- Three-fold confirmation, calibration/threshold fitting, nested evaluation, uncertainty, fusion and the reserved test remain separate.
""")
md("""
## Summary

The observations below come from executed saved predictions on matching patients.
They describe the six-epoch experiment; no final architecture winner is selected.
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
path = ROOT / "notebooks/d1/10_image_architecture_comparison.ipynb"
path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, path)
print(f"Created {path} ({len(cells)} cells)")
