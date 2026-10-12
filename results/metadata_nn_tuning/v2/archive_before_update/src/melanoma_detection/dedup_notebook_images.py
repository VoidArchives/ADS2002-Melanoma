"""Detailed read-only image report cells for the clean numbered workflow.

Only saved clean predictions/histories and development image previews are read.
No checkpoint is loaded into a model, no inference runs, and no training starts.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from IPython.display import display, Markdown
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .development_dataset import load_preserved_inner_memberships
from .image_model import DEFAULT_SPEC, make_transform
from .image_uncertainty import TTA_PASSES, tta_transform
from .image_uncertainty_evaluation import predictive_entropy
from .metadata_evaluation import calibration_bins, probability_metrics

RAW_RUN = "b0_nested_dedup_v1"
CALIBRATION_RUN = "b0_nested_calibration_dedup_v1"
FOLD_COLORS = {0: "#235789", 1: "#C47A14", 2: "#66743A", 3: "#A45078", 4: "#333333"}
REFERRAL_COLORS = {"uncertainty_sd": "#245D8C", "prediction_score": "#B46A20",
                   "predictive_entropy": "#747133", "random": "#666666"}
REFERRAL_LINES = {"uncertainty_sd": "-", "prediction_score": "--",
                  "predictive_entropy": "-.", "random": ":"}


def original_style(source):
    """Use the original notebook's typography; figure-specific styles stay local."""
    size = 10 if source == "uncertainty" else 11
    plt.rcParams.update({"font.size": size, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.facecolor": "white"})


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pending(message):
    display(Markdown("**Pending:** " + message))


def _finish(fig, name, finish):
    finish(fig, name)


def training_preview_frame(root, data, identity):
    members = load_preserved_inner_memberships(root, identity, 0)
    fitting_ids = set(members.loc[members.inner_fold.ne(0), "image_name"])
    fitting = data.loc[data.image_name.isin(fitting_ids)].sort_values("image_name")
    return pd.concat([fitting.loc[fitting.target.eq(label)].head(2) for label in (0, 1)]).reset_index(drop=True)


def show_preparation(root, data, identity, finish):
    """Preview only fitting images; weak transforms never alter stored images."""
    import torch
    spec = {**DEFAULT_SPEC, "architecture": "efficientnet_b0", "weights": "IMAGENET1K_V1",
            "image_size": 224, "augmentation": "weak", "preprocessing": "none"}
    selected = training_preview_frame(root, data, identity)
    ordinary, weak = make_transform(spec, training=False), make_transform(spec, training=True)
    mean = np.array([.485, .456, .406]).reshape(1, 1, 3)
    std = np.array([.229, .224, .225]).reshape(1, 1, 3)
    def visible(tensor):
        return np.clip(tensor.permute(1, 2, 0).numpy() * std + mean, 0, 1)
    fig, axes = plt.subplots(len(selected), 4, figsize=(11, 2.6 * len(selected)), squeeze=False, constrained_layout=True)
    state = torch.get_rng_state()
    try:
        for index, row in enumerate(selected.itertuples()):
            with PILImage.open(Path(root) / "data/subset/train" / f"{row.image_name}.jpg") as opened:
                image = opened.convert("RGB")
                axes[index, 0].imshow(image)
                for column in range(1, 4):
                    # Transform randomness uses CPU torch; do not initialise CUDA.
                    torch.random.default_generator.manual_seed(42 + index * 10 + column)
                    axes[index, column].imshow(visible(weak(image)))
            axes[index, 0].set_title(f"{row.image_name} | target {row.target}", fontsize=9)
            for column in range(1, 4):
                axes[index, column].set_title(f"Seeded weak variant {column}", fontsize=9)
            for ax in axes[index]:
                ax.axis("off")
    finally:
        torch.set_rng_state(state)
    fig.suptitle("Weak augmentation QA: retained fitting patients only; deliberate label mix", fontsize=13)
    _finish(fig, "image_preparation_and_weak_augmentation", finish)
    display(Markdown("Colour is displayed after reversing ImageNet normalisation. Weak rotation can leave black corners or clip framing; these previews document the preserved transform and do not approve a new augmentation profile."))


def show_pipeline(finish):
    fig, ax = plt.subplots(figsize=(12, 2.4), layout="constrained")
    labels = ["Retained RGB\nimage", "Whole-image\n224 × 224 resize", "Fitting only:\nweak flips/rotation", "ImageNet\nnormalisation", "Pretrained B0\n+ fresh output head", "One sigmoid\nraw score"]
    for index, label in enumerate(labels):
        x = index * 1.95
        ax.text(x, 0, label, ha="center", va="center", fontsize=9, bbox={"boxstyle": "round,pad=.55", "facecolor": "#edf3f7", "edgecolor": "#276b91"})
        if index:
            ax.annotate("", xy=(x - .79, 0), xytext=(x - 1.15, 0), arrowprops={"arrowstyle": "->", "color": "#276b91"})
    ax.set(xlim=(-1, 11), ylim=(-.7, .7), title="Fitting pipeline; ordinary validation omits random augmentation")
    ax.axis("off")
    _finish(fig, "image_pipeline", finish)


def image_settings(root):
    import yaml
    config = yaml.safe_load((Path(root) / "configs/image_nested_dedup.yaml").read_text(encoding="utf-8"))
    display(pd.DataFrame([{"Setting": key, "Value": str(value)} for section in ("model", "training") for key, value in config[section].items()]))
    display(pd.DataFrame(config["selection"]["recipes"]))
    display(pd.DataFrame([{"Outer folds": 5, "Preserved inner folds": 3, "Screen epochs": 6,
                           "Confirm epochs": 12, "Duration": "median of three saved AP-best epochs", "Raw GPU fits": 35,
                           "Calibration GPU fits": 15, "Fresh start": "IMAGENET1K_V1 for each distinct fit"}]))


def image_partition_table(root, data, identity):
    members = load_preserved_inner_memberships(root, identity, 0)
    fitting = set(members.loc[members.inner_fold.ne(0), "image_name"])
    validation = set(members.loc[members.inner_fold.eq(0), "image_name"])
    partitions = {"fitting": data.loc[data.image_name.isin(fitting)],
                  "inner_validation": data.loc[data.image_name.isin(validation)],
                  "outer_excluded": data.loc[data.fold.eq(0)]}
    display(pd.DataFrame([{"partition": name, "images": len(frame),
                          "patients": frame.patient_id.nunique(),
                          "malignant_images": int(frame.target.sum())}
                         for name, frame in partitions.items()]))


def image_selection_table(root, data):
    directory = Path(root) / "results/image_d1_runs" / RAW_RUN
    summary = _read(directory / "summary.json") if (directory / "summary.json").exists() else {}
    completed = {row["fold"] for row in summary.get("folds", []) if row.get("status") == "complete"}
    records = []
    for fold in range(5):
        path = directory / f"folds/{fold}/selection.json"
        selection = _read(path) if path.exists() else {}
        records.append({"outer_fold": fold, "outer_images": int(data.fold.eq(fold).sum()),
                        "outer_malignant_images": int(data.loc[data.fold.eq(fold), "target"].sum()),
                        "selected_recipe": selection.get("recipe", {}).get("id", "pending"),
                        "inner_best_epochs": [row["best_epoch"] for row in selection.get("confirmation", [])],
                        "fixed_refit_epochs": selection.get("refit_epochs"),
                        "status": "complete" if fold in completed else "pending"})
    display(pd.DataFrame(records))


def deferred_condition_status(condition):
    display(pd.DataFrame([{"clean_experiment": condition, "status": "deferred",
                           "clean_quality_review": "not performed", "clean_matching_fits": 0,
                           "clean_performance_comparison": "unavailable"}]))


def image_takeaways(raw, comparison):
    if not raw.get("complete"):
        _pending("Verified complete retained five-fold results are required for a numerical image conclusion.")
        return
    folds = raw["fold_metrics"]
    display(Markdown(f"**Executed clean B0 result:** mean outer-fold AP **{folds.average_precision.mean():.4f}** "
        f"and mean ROC-AUC **{folds.roc_auc.mean():.4f}**. AP ranges from "
        f"**{folds.average_precision.min():.3f} to {folds.average_precision.max():.3f}** across the five retained "
        "patient groups. The observed fold spread is descriptive variability, not a confidence interval. "
        "These values replace the historical pilot findings; they remain development evidence after earlier exploration."))
    if not comparison.empty:
        display(comparison.loc[comparison.model.isin(["b0_raw", "spline_raw", "metadata_nn_mean"])])


def uncertainty_takeaways(bundle):
    if not bundle["complete"]:
        _pending("Completed clean uncertainty artifacts are needed before interpreting score variation.")
        return
    display(pd.read_csv(bundle["directory"] / "mean_fold_metrics.csv"))
    table = pd.read_csv(bundle["directory"] / "paired_patient_bootstrap_differences.csv")
    selected = table.loc[table.referral_fraction.eq(.2) & table.comparator.eq("prediction_score")]
    crossing = selected.ci_lower.le(0) & selected.ci_upper.ge(0)
    display(Markdown(f"**Executed clean scope:** {len(bundle['predictions']):,} retained outer-held-out images "
        f"from {bundle['predictions'].patient_id.nunique()} patients. At the prescribed 20% referral, "
        f"**{int(crossing.sum())} of {len(selected)}** SD-minus-score intervals include zero. "
        "Read both retained errors and malignant referral support before claiming an ordering benefit. "
        "A favourable mean prediction or conditional interval does not establish a clinical referral policy; "
        "the reserved test remains outside this report."))


def image_progress(root, data, identity, finish):
    """Expose saved inner diagnostics independently of full OOF completion."""
    directory = Path(root) / "results/image_d1_runs" / RAW_RUN
    if not (directory / "run_manifest.json").exists():
        _pending("Clean B0 has no saved run yet.")
        return
    manifest = _read(directory / "run_manifest.json")
    if manifest.get("dataset_identity") != identity:
        raise ValueError("Image histories belong to another dataset.")
    selections, screens, cost = [], [], []
    histories = []
    for fold in range(5):
        folder = directory / f"folds/{fold}"
        selection_path = folder / "selection.json"
        selection = _read(selection_path) if selection_path.exists() else {}
        summary_path = directory / "summary.json"
        summary = _read(summary_path) if summary_path.exists() else {}
        completed = {row["fold"] for row in summary.get("folds", []) if row.get("status") == "complete"}
        selections.append({"outer_fold": fold, "outer_images": int(data.fold.eq(fold).sum()),
                           "outer_malignant_images": int(data.loc[data.fold.eq(fold), "target"].sum()),
                           "selected_recipe": selection.get("recipe", {}).get("id", "pending"),
                           "inner_best_epochs": [row["best_epoch"] for row in selection.get("confirmation", [])],
                           "fresh_outer_refit_epochs": selection.get("refit_epochs"),
                           "outer_status": "complete" if fold in completed else "pending/in progress"})
        for path in sorted((folder / "inner").glob("*/*/fit_result.json")):
            result = _read(path)
            recipe, inner = path.parent.parent.name, int(path.parent.name)
            rows = result.get("history", [])
            if rows:
                histories.extend({"outer_fold": fold, "recipe": recipe, "inner_fold": inner, "fit_status": result.get("status"), **row} for row in rows)
                cost.append({"outer_fold": fold, "recipe": recipe, "inner_fold": inner,
                             "epochs": len(rows), "status": result.get("status"),
                             "training_minutes": sum(row.get("epoch_seconds", 0) for row in rows) / 60,
                             "peak_GPU_GiB": max(row.get("peak_memory_bytes", 0) for row in rows) / 2**30})
                first_six = rows[:6]
                best = max(first_six, key=lambda row: row["average_precision"])
                if inner == 0:
                    screens.append({"outer_fold": fold, "recipe": recipe, "screen_epochs_available": len(first_six),
                                    "best_screen_epoch": best["epoch"], "screen_AP": best["average_precision"],
                                    "screen_AUC": best["roc_auc"], "scope": "inner selection diagnostic"})
    display(pd.DataFrame(selections))
    if not histories:
        _pending("No committed inner histories are saved yet.")
        return
    display(pd.DataFrame(screens))
    display(pd.DataFrame(cost))
    history = pd.DataFrame(histories)
    # The historical six-epoch screen remains identifiable even when the
    # selected trajectory later continues to its 12-epoch confirmation.
    first_outer = int(history.outer_fold.min())
    shown = history.loc[history.outer_fold.eq(first_outer) & history.inner_fold.eq(0) & history.epoch.le(6)].copy()
    from .image_reporting import plot_learning_curves, plot_training_losses
    shown["pipeline_id"] = "efficientnet_b0/" + shown.recipe
    shown["loss_weight"] = np.where(shown.recipe.str.endswith("sqrt"), "sqrt", "none")
    shown["selected_epoch"] = shown.groupby("recipe")["average_precision"].transform(lambda values: shown.loc[values.idxmax(), "epoch"])
    members = load_preserved_inner_memberships(root, identity, first_outer)
    prevalence = members.loc[members.inner_fold.eq(0), "target"].mean()
    fig = plot_learning_curves(shown, prevalence, f"Clean B0 six-epoch screen: outer pool {first_outer}, inner 0; exploratory selection")
    _finish(fig, "b0_inner_recipe_learning_curves", finish)
    fig = plot_training_losses(shown)
    _finish(fig, "b0_training_loss_and_schedule", finish)


def plot_probability_views(frame, columns, finish, prefix, title, *, reliability=True):
    """ROC/PR and reliability read directly from matching saved scores."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    styles = {"Deterministic": ("#666666", "-"), "MC mean": ("#245D8C", "--"),
              "TTA mean": ("#B46A20", "-."), "Raw B0": ("#235789", "-"),
              "Training-only calibrated B0": ("#C47A14", "--"),
              "b0_raw": ("#235789", "-"), "spline_raw": ("#2563eb", "--"),
              "metadata_nn_mean": ("#333333", "-.")}
    for index, (label, column) in enumerate(columns.items()):
        color, line = styles.get(label, (plt.get_cmap("tab20")(index), ["-", "--", "-.", ":"][index % 4]))
        fpr, tpr, _ = roc_curve(frame.target, frame[column])
        precision, recall, _ = precision_recall_curve(frame.target, frame[column])
        metrics = probability_metrics(frame.target, frame[column])
        axes[0].plot(fpr, tpr, color=color, linestyle=line, linewidth=1.7, label=f"{label} | AUC {metrics['roc_auc']:.3f}")
        axes[1].step(recall, precision, where="post", color=color, linestyle=line, linewidth=1.7, label=f"{label} | AP {metrics['average_precision']:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle=":", color="#666666", label="Chance ranking")
    axes[1].axhline(frame.target.mean(), linestyle=":", color="#666666", label=f"Prevalence {frame.target.mean():.3%}")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate / sensitivity", title="ROC")
    axes[1].set(xlabel="Recall / sensitivity", ylabel="Precision", title="Precision–recall (AP is not trapezoidal area)")
    for ax in axes:
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.18))
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
    fig.suptitle(title + f"\n{len(frame):,} retained images | {frame.patient_id.nunique()} patients | pooled curves are secondary", fontsize=13)
    _finish(fig, prefix + "_roc_pr", finish)
    if not reliability:
        return
    bins_table = pd.concat([calibration_bins(frame.target, frame[column], n_bins=5, groups=frame.patient_id).assign(model=label)
                             for label, column in columns.items()], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    observed_max = max(.05, float(bins_table.mean_probability.max()), float(bins_table.observed_rate.max()))
    for ax, limit, subtitle in zip(axes, (1., min(1., observed_max * 1.12)), ("Full score range", "Enlarged observed range")):
        for index, (label, group) in enumerate(bins_table.groupby("model", sort=False)):
            color, line = styles.get(label, (plt.get_cmap("tab20")(index), ["-", "--", "-.", ":"][index % 4]))
            ax.plot(group.mean_probability, group.observed_rate, marker="o", color=color, linestyle=line, label=label)
        ax.plot([0, limit], [0, limit], linestyle=":", color="#333333", label="Score–frequency agreement")
        ax.set(xlabel="Mean prediction score (uncalibrated)", ylabel="Observed malignant-image fraction", title=subtitle,
               xlim=(0, limit), ylim=(0, limit))
    axes[0].legend(fontsize=9)
    fig.suptitle(title + " · same five equal-count bins; descriptive")
    _finish(fig, prefix + "_reliability", finish)
    display(bins_table)


def plot_confusions(frame, scenarios, finish, prefix):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.3 * len(scenarios), 3.9), squeeze=False, constrained_layout=True)
    rows = []
    for ax, (label, decisions) in zip(axes.flat, scenarios.items()):
        matrix = confusion_matrix(frame.target, np.asarray(decisions, dtype=int), labels=[0, 1])
        percentages = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.full((2, 2), np.nan), where=matrix.sum(axis=1, keepdims=True) != 0)
        ax.imshow(percentages, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                fraction = f"{percentages[i, j]:.1%}" if np.isfinite(percentages[i, j]) else "undefined"
                ax.text(j, i, f"{matrix[i, j]:,}\n{fraction}", ha="center", va="center", color="white" if percentages[i, j] > .55 else "black", fontsize=9)
        ax.set(xticks=[0, 1], xticklabels=["Benign", "Flagged"], yticks=[0, 1], yticklabels=["Benign", "Malignant"],
               xlabel="Predicted decision", ylabel="Recorded label", title=label)
        tn, fp, fn, tp = matrix.ravel()
        rows.append({"scenario": label, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                     "recall": tp / (tp + fn) if tp + fn else np.nan,
                     "precision": tp / (tp + fp) if tp + fp else np.nan,
                     "specificity": tn / (tn + fp) if tn + fp else np.nan, "flagged_images": int(np.asarray(decisions).sum())})
    fig.suptitle("Counts and percentages within each true class", fontsize=13)
    _finish(fig, prefix + "_confusion_counts", finish)
    display(pd.DataFrame(rows))


def raw_oof_figures(raw, finish):
    if not raw.get("complete"):
        _pending("All five clean B0 folds and their reporting receipts must complete before OOF plots are shown.")
        return
    raw_outer_ranking(raw, finish)
    raw_outer_reliability(raw, finish)


def raw_outer_ranking(raw, finish):
    """Original stage-14 composition, colours and fold line identities."""
    if not raw.get("complete"):
        _pending("All five retained outer folds are required for ranking conclusions.")
        return
    from .image_evaluation import curve_tables
    frame, metrics = raw["predictions"], raw["fold_metrics"]
    display(metrics.style.format({name: "{:.4f}" for name in ("prevalence", "average_precision", "roc_auc", "log_loss", "brier")}))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for fold, rows in frame.groupby("fold", sort=True):
        curves = curve_tables(rows.target, rows.base_probability)
        label = f"Fold {fold} ({int(rows.target.sum())} positive images)"
        axes[0].plot(curves["precision_recall"].recall, curves["precision_recall"].precision, color=FOLD_COLORS[fold], label=label, linewidth=1.6, linestyle="--" if fold % 2 else "-")
        axes[1].plot(curves["roc"].false_positive_rate, curves["roc"].true_positive_rate, color=FOLD_COLORS[fold], linewidth=1.6, linestyle="--" if fold % 2 else "-")
    pooled = curve_tables(frame.target, frame.base_probability)
    axes[0].plot(pooled["precision_recall"].recall, pooled["precision_recall"].precision, color="#111111", linewidth=2.5, linestyle=":", label="Pooled OOF (secondary)")
    axes[0].axhline(frame.target.mean(), color="#999999", linestyle=":", label=f"Pooled prevalence {frame.target.mean():.2%}")
    axes[1].plot([0, 1], [0, 1], color="#999999", linestyle=":")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Raw OOF precision–recall", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="Raw OOF ROC by outer fold", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=8, loc="upper right")
    _finish(fig, "outer_fold_pr_roc", finish)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for row in metrics.itertuples():
        ax.scatter(row.fold, row.average_precision, s=90, color=FOLD_COLORS[row.fold])
        ax.annotate(f"{row.average_precision:.3f}", (row.fold, row.average_precision), xytext=(0, 8), textcoords="offset points", ha="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.scatter(metrics.fold, metrics.prevalence, marker="x", color="#666666", label="Fold prevalence")
    ax.axhline(metrics.average_precision.mean(), color="#235789", linestyle="--", label="Mean fold AP")
    ax.set(xlabel="Outer fold", ylabel="Average precision / prevalence", xticks=range(5), title="Outer-fold variation in image ranking", ylim=(0, min(1, max(.1, metrics.average_precision.max() * 1.3))))
    ax.legend(loc="upper right")
    _finish(fig, "outer_fold_average_precision", finish)


def raw_outer_reliability(raw, finish):
    """Keep original fixed-width reliability with support and score histogram."""
    if not raw.get("complete"):
        _pending("Verified five-fold raw predictions are required for reliability.")
        return
    from .image_nested_reporting import reliability_table
    frame = raw["predictions"]
    reliability = reliability_table(frame.target, frame.base_probability, n_bins=10)
    nonempty = reliability.loc[reliability.n_images.gt(0)]
    display(reliability)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], color="#666666", linestyle="--", label="Ideal agreement")
    axes[0].plot(nonempty.mean_probability, nonempty.observed_prevalence, "o-", color="#235789", label="Raw image scores")
    for row in nonempty.itertuples():
        axes[0].annotate(str(row.n_images), (row.mean_probability, row.observed_prevalence), xytext=(3, 5), textcoords="offset points", fontsize=8)
    axes[0].set(xlabel="Mean raw probability in bin", ylabel="Observed melanoma fraction", xlim=(0, 1), ylim=(0, 1), title="Pooled raw reliability (bin counts shown)")
    axes[0].legend(fontsize=9)
    axes[1].hist(frame.base_probability, bins=np.linspace(0, 1, 11), color="#235789", edgecolor="white")
    axes[1].set(xlabel="Raw image probability", ylabel="Number of images", title="OOF score distribution", xlim=(0, 1))
    _finish(fig, "raw_reliability", finish)
    plot_confusions(frame, {"Raw score ≥ 0.5 (descriptive)": frame.base_probability.ge(.5)}, finish, "raw_b0")


def calibration_figures(calibration, finish):
    if not calibration.get("complete"):
        _pending("Training-only image calibration and frozen cutoffs are unfinished; no calibrated curve or cutoff result is inferred.")
        return
    frame = calibration["predictions"]
    display(calibration["probability_metrics"])
    display(calibration["ranking_invariance"])
    display(calibration["scenario_by_fold"])
    from .image_nested_reporting import reliability_table
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    reliability_parts = []
    for ax, name, column, color in zip(axes, ("Raw", "Calibrated"), ("base_probability", "calibrated_probability"), ("#235789", "#C47A14")):
        bins = reliability_table(frame.target, frame[column], n_bins=10)
        reliability_parts.append(bins.assign(model=name))
        rows = bins.loc[bins.n_images.gt(0)]
        ax.plot([0, 1], [0, 1], "--", color="#666666", label="Ideal agreement")
        ax.plot(rows.mean_probability, rows.observed_prevalence, "o-", color=color, label=name)
        for row in rows.itertuples():
            ax.annotate(str(row.n_images), (row.mean_probability, row.observed_prevalence), xytext=(3, -12) if row.observed_prevalence >= .95 else (3, 5), textcoords="offset points", fontsize=8)
        ax.set(xlabel="Mean score in bin", ylabel="Observed melanoma fraction", xlim=(0, 1), ylim=(0, 1.04), title=f"{name} pooled OOF reliability")
        ax.legend(fontsize=9)
    fig.suptitle("Fixed bins; annotations show image counts")
    _finish(fig, "raw_vs_calibrated_reliability", finish)
    display(pd.concat(reliability_parts, ignore_index=True))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    matrices = [confusion_matrix(frame.target, frame[column].astype(int), labels=[0, 1])
                for column in ("decision_raw", "decision_calibrated")]
    maximum = max(1, max(matrix.max() for matrix in matrices))
    for ax, name, matrix in zip(axes, ("Raw", "Calibrated"), matrices):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for y in range(2):
            for x in range(2):
                ax.text(x, y, f"{matrix[y, x]:,}", ha="center", va="center", color="white" if matrix[y, x] > matrix.max() / 2 else "#111111", fontsize=15)
        recall = matrix[1, 1] / matrix[1].sum() if matrix[1].sum() else np.nan
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not flagged", "Flagged"],
               yticklabels=["Benign", "Melanoma"], xlabel="Frozen training-selected decision", ylabel="Actual image outcome",
               title=f"{name}: actual recall {recall:.1%}")
        ax.grid(False)
    fig.suptitle("Training-selected 80%-recall scenario on held-out outer images")
    _finish(fig, "training_selected_threshold_counts", finish)
    display(Markdown("A monotonic map preserves within-fold ranking. Different fold-specific maps can change pooled ordering. Calibration is assessed with log loss, Brier and reliability; the academic training-only 80%-recall rule does not guarantee 80% recall on held-out patients."))


def matched_comparator_figures(bundle, finish):
    names = [name for name in ("b0_raw", "spline_raw", "metadata_nn_mean") if name in bundle["scores"]]
    if "b0_raw" not in names:
        _pending("Complete clean B0 predictions are required for an image-versus-metadata curve.")
        return
    frame = bundle["expected"].copy()
    for name in names:
        frame[name] = bundle["scores"][name]
    plot_probability_views(frame, {name: name for name in names}, finish, "matched_b0_metadata", "Same retained records: B0, fixed spline and NN mean")


def pilot_figures(root, pilot_loader, finish):
    complete = {}
    progress, costs = [], []
    for run_id, name in (("b0_duration_dedup_v1", "B0"), ("convnext_duration_dedup_v1", "ConvNeXt-Tiny")):
        directory, summary = pilot_loader(run_id)
        progress.append({"family": name, "run_id": run_id, "status": summary.get("status", "pending"),
                         "completed_inner_folds": summary.get("completed_inner_folds", []), "stop_reason": summary.get("stopping_reason")})
        if summary.get("status") != "complete":
            continue
        complete[name] = (directory, summary)
        costs.append({"family": name, "distinct_fits": summary["fits"], "training_epochs": summary["training_epochs"],
                      "training_minutes": summary["recorded_training_seconds"] / 60,
                      "peak_allocated_GiB": summary["peak_memory_bytes"] / 2**30,
                      "peak_reserved_GiB": summary["peak_reserved_memory_bytes"] / 2**30})
    display(pd.DataFrame(progress))
    if not complete:
        _pending("No complete three-inner-fold pilot is available. Stronger augmentation, CLAHE and full alternative-architecture OOF remain deferred.")
        return
    display(pd.DataFrame(costs))
    metric_parts, curve_parts, paired_parts = [], [], []
    for family, (directory, summary) in complete.items():
        metric_parts.append(pd.read_csv(directory / "evaluation/inner_metrics.csv").assign(family=family))
        curve_parts.append(pd.read_csv(directory / "evaluation/learning_curves.csv").assign(family=family))
        paired_parts.append(pd.read_csv(directory / "evaluation/paired_patient_comparisons.csv").assign(family=family))
        if summary.get("screening"):
            display(pd.DataFrame([{"recipe": row["recipe"]["id"], "epochs": row["completed_epochs"], "AP-best epoch": row["best_epoch"], **row["best_metrics"]} for row in summary["screening"]]))
    metrics, curves, paired = pd.concat(metric_parts), pd.concat(curve_parts), pd.concat(paired_parts)
    display(metrics)
    display(paired)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, metric in zip(axes.flat, ("average_precision", "roc_auc", "log_loss", "brier")):
        for (family, inner), group in curves.groupby(["family", "inner_fold"]):
            ax.plot(group.epoch, group[metric], label=f"{family}, inner {inner}", linestyle="-" if family == "B0" else "--")
        ax.axvline(12, color="#888888", linestyle=":")
        ax.axvline(24, color="#888888", linestyle=":")
        ax.set(xlabel="Epoch in the same fresh trajectory", ylabel=metric.replace("_", " "), title=metric.replace("_", " "))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Bounded pilots: original 30-epoch scheduler; stop at epoch 24")
    _finish(fig, "pilot_four_metric_learning_curves", finish)
    fixed = metrics.loc[metrics.selection_kind.eq("endpoint")].groupby(["family", "cap"])[["average_precision", "roc_auc"]].mean().reset_index()
    selected = metrics.loc[metrics.selection_kind.eq("best_in_cap")].groupby(["family", "cap"])[["average_precision", "roc_auc"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    positions = np.arange(len(fixed))
    axes[0].barh(positions, fixed.average_precision, color="#276b91")
    axes[0].set(yticks=positions, yticklabels=[f"{r.family}: epoch {r.cap}" for r in fixed.itertuples()], xlabel="Mean inner-fold AP", title="Fixed endpoints")
    costs = pd.DataFrame(costs)
    axes[1].bar(costs.family, costs.training_minutes, color="#276b91")
    axes[1].set(ylabel="Recorded fitting minutes", title="Computing time")
    axes[2].bar(costs.family, costs.peak_allocated_GiB, color="#276b91")
    axes[2].set(ylabel="Peak allocated GiB", title="GPU memory")
    _finish(fig, "pilot_endpoint_quality_time_memory", finish)
    display(Markdown("**Fixed endpoints** report epochs 12 and 24 without picking a favourable epoch. **Best in cap** reuses these inner labels for selection and is more optimistic."))
    display(selected)
    ap_pairs = paired.loc[paired.metric.eq("average_precision")].reset_index(drop=True) if "metric" in paired else pd.DataFrame()
    if not ap_pairs.empty:
        estimate_column = "difference" if "difference" in ap_pairs else "estimate"
        fig, ax = plt.subplots(figsize=(10, max(3.4, .4 * len(ap_pairs))), layout="constrained")
        for index, row in ap_pairs.iterrows():
            ax.hlines(index, row.ci_lower, row.ci_upper, color="#276b91")
            ax.scatter(row[estimate_column], index, color="#276b91")
        ax.axvline(0, color="#777777", linestyle=":")
        ax.set(yticks=np.arange(len(ap_pairs)), yticklabels=ap_pairs.comparison,
               xlabel="Candidate minus control mean inner AP", title="Paired patient intervals; selected inner evidence")
        _finish(fig, "pilot_paired_patient_ap_intervals", finish)
    # Curves/error/reliability for matched fixed epoch 24, never historical scores.
    prediction_frames = {family: pd.read_csv(directory / "evaluation/predictions_endpoint_24.csv") for family, (directory, _) in complete.items()}
    first_name = next(iter(prediction_frames))
    reference = prediction_frames[first_name][["image_id", "patient_id", "target", "inner_fold"]].sort_values("image_id").reset_index(drop=True)
    scenarios = {}
    for family, prediction in prediction_frames.items():
        ordered = prediction.sort_values("image_id").reset_index(drop=True)
        if not ordered[["image_id", "patient_id", "target", "inner_fold"]].equals(reference[["image_id", "patient_id", "target", "inner_fold"]]):
            raise ValueError("Pilot families have mismatched retained inner records.")
        reference[family] = ordered.base_probability
        scenarios[f"{family}: fixed score 0.5"] = ordered.base_probability.ge(.5)
    plot_probability_views(reference, {name: name for name in prediction_frames}, finish, "pilot_epoch24", "Fixed epoch 24 · selected inner recipe evidence")
    plot_confusions(reference, scenarios, finish, "pilot_epoch24")


def load_pilot_bundle(pilot_loader):
    """Separate verified pilot loading from each focused presentation section."""
    complete, progress, costs = {}, [], []
    for run_id, family in (("b0_duration_dedup_v1", "B0"), ("convnext_duration_dedup_v1", "ConvNeXt-Tiny")):
        directory, summary = pilot_loader(run_id)
        progress.append({"family": family, "run_id": run_id, "status": summary.get("status", "pending"),
                         "completed_inner_folds": summary.get("completed_inner_folds", []), "stop_reason": summary.get("stopping_reason")})
        if summary.get("status") == "complete":
            complete[family] = (directory, summary)
            costs.append({"family": family, "distinct_fits": summary["fits"], "training_epochs": summary["training_epochs"],
                          "training_minutes": summary["recorded_training_seconds"] / 60,
                          "peak_allocated_GiB": summary["peak_memory_bytes"] / 2**30,
                          "peak_reserved_GiB": summary["peak_reserved_memory_bytes"] / 2**30})
    display(pd.DataFrame(progress))
    if not complete:
        _pending("No complete three-inner-fold pilot is available; remaining sections wait for matching committed exports.")
        return {"complete": False}
    metrics, curves, pairs = [], [], []
    for family, (directory, summary) in complete.items():
        metrics.append(pd.read_csv(directory / "evaluation/inner_metrics.csv").assign(family=family))
        curves.append(pd.read_csv(directory / "evaluation/learning_curves.csv").assign(family=family))
        pairs.append(pd.read_csv(directory / "evaluation/paired_patient_comparisons.csv").assign(family=family))
    frames = {family: pd.read_csv(directory / "evaluation/predictions_endpoint_24.csv").sort_values("image_id").reset_index(drop=True)
              for family, (directory, _) in complete.items()}
    columns = ["image_id", "patient_id", "target", "inner_fold"]
    reference = next(iter(frames.values()))[columns].copy()
    for family, frame in frames.items():
        if not frame[columns].equals(reference[columns]):
            raise ValueError("Completed pilot families do not cover matching clean inner records.")
        reference[family] = frame.base_probability
    return {"complete": True, "runs": complete, "metrics": pd.concat(metrics, ignore_index=True),
            "curves": pd.concat(curves, ignore_index=True), "paired": pd.concat(pairs, ignore_index=True),
            "costs": pd.DataFrame(costs), "predictions": reference, "families": list(frames)}


def pilot_screening(bundle):
    if not bundle["complete"]:
        return
    for family, (_, summary) in bundle["runs"].items():
        screens = summary.get("screening", [])
        if screens:
            display(Markdown(f"**{family}: six-epoch recipe screen on inner fold 0.**"))
            display(pd.DataFrame([{"recipe": row["recipe"]["id"], "completed_epochs": row["completed_epochs"],
                                   "AP-best epoch": row["best_epoch"], **row["best_metrics"]} for row in screens]))
        display(pd.DataFrame([{"family": family, "inner_fold": row["inner_fold"], "selected_recipe": row["recipe"]["id"]}
                              for row in summary.get("records", [])]))


def pilot_learning(bundle, finish):
    if not bundle["complete"]:
        return
    curves = bundle["curves"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, metric in zip(axes.flat, ("average_precision", "roc_auc", "log_loss", "brier")):
        for (family, inner), group in curves.groupby(["family", "inner_fold"]):
            ax.plot(group.epoch, group[metric], label=f"{family}, inner {inner}", linestyle="-" if family == "B0" else "--")
        ax.axvline(12, color="#888888", linestyle=":")
        ax.axvline(24, color="#888888", linestyle=":")
        ax.set(xlabel="Epoch in the same fresh trajectory", ylabel=metric.replace("_", " "), title=metric.replace("_", " "))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Bounded pilots: fixed 30-epoch scheduler; stop at epoch 24")
    _finish(fig, "pilot_four_metric_learning_curves", finish)


def pilot_endpoint_metrics(bundle, finish):
    if not bundle["complete"]:
        return
    metrics = bundle["metrics"]
    display(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for ax, kind, title in zip(axes, ("endpoint", "best_in_cap"), ("Predeclared fixed endpoint", "AP-selected best epoch within cap")):
        table = metrics.loc[metrics.selection_kind.eq(kind)].groupby(["family", "cap"])[["average_precision", "roc_auc", "log_loss", "brier"]].mean().reset_index()
        display(table.assign(selection_kind=kind))
        ax.barh(np.arange(len(table)), table.average_precision, color="#276b91")
        ax.set(yticks=np.arange(len(table)), yticklabels=[f"{row.family}: cap {row.cap}" for row in table.itertuples()],
               xlabel="Unweighted mean inner-fold AP", title=title)
    _finish(fig, "pilot_fixed_endpoints_vs_selected_epochs", finish)


def pilot_cost(bundle, finish):
    if not bundle["complete"]:
        return
    costs = bundle["costs"]
    display(costs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].bar(costs.family, costs.training_minutes, color="#276b91")
    axes[0].set(ylabel="Recorded fitting minutes", title="Total training work")
    axes[1].bar(costs.family, costs.peak_allocated_GiB, color="#276b91")
    axes[1].set(ylabel="Peak allocated GiB", title="GPU memory")
    _finish(fig, "pilot_training_time_and_memory", finish)


def pilot_matching_curves(bundle, finish):
    if bundle["complete"]:
        plot_probability_views(bundle["predictions"], {family: family for family in bundle["families"]}, finish,
                               "pilot_epoch24", "Matching fixed epoch 24 · selected inner recipes")


def pilot_decisions(bundle, finish):
    if bundle["complete"]:
        frame = bundle["predictions"]
        plot_confusions(frame, {f"{family}: raw score ≥0.5": frame[family].ge(.5) for family in bundle["families"]}, finish, "pilot_epoch24")


def pilot_interval_support(bundle, finish):
    if not bundle["complete"]:
        return
    paired = bundle["paired"]
    display(paired)
    support = bundle["predictions"].groupby("inner_fold").agg(images=("image_id", "size"), patients=("patient_id", "nunique"), malignant_images=("target", "sum"))
    display(support)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, .32 * len(paired))), layout="constrained")
    for ax, metric in zip(axes, ("average_precision", "roc_auc")):
        rows = paired.loc[paired.metric.eq(metric)].reset_index(drop=True)
        for index, row in enumerate(rows.itertuples()):
            ax.hlines(index, row.ci_lower, row.ci_upper, color="#276b91")
            ax.scatter(row.difference, index, color="#276b91")
        ax.axvline(0, color="#777777", linestyle=":")
        ax.set(yticks=np.arange(len(rows)), yticklabels=rows.comparison, xlabel="Candidate minus control", title=metric.replace("_", " "))
        ax.invert_yaxis()
    _finish(fig, "pilot_paired_patient_ranking_intervals", finish)


def load_uncertainty(root, identity):
    from .dedup_workflow import load_uncertainty_report, UNCERTAINTY
    receipt = load_uncertainty_report(root)
    directory = Path(root) / UNCERTAINTY
    if receipt.get("status") != "complete":
        display(pd.DataFrame([{"stage": "Clean five-fold MC/TTA inference", "status": receipt.get("status", "pending"), "training_fits": 0}]))
        _pending("All five clean outer refits and the separate uncertainty pass exports must complete. Opening this notebook does not start inference.")
        return {"complete": False, "receipt": receipt, "directory": directory}
    if receipt["dataset_identity"] != identity:
        raise ValueError("Uncertainty predictions belong to another development manifest.")
    summary = pd.read_csv(directory / "prediction_summary.csv")
    display(pd.DataFrame([{key: receipt[key] for key in ("status", "images", "patients", "malignant_images", "malignant_patients", "mc_passes", "tta_passes", "training_fits")}]))
    return {"complete": True, "receipt": receipt, "directory": directory, "predictions": summary}


def uncertainty_views(root, data, identity, finish):
    selected = training_preview_frame(root, data, identity).groupby("target", sort=True).head(1)
    spec = {**DEFAULT_SPEC, "architecture": "efficientnet_b0", "weights": "IMAGENET1K_V1", "image_size": 224, "augmentation": "weak", "preprocessing": "none"}
    ordinary = make_transform(spec, training=False)
    mean, std = np.array([.485, .456, .406]), np.array([.229, .224, .225])
    fig, axes = plt.subplots(len(selected), 4, figsize=(11, 2.8 * len(selected)), squeeze=False, layout="constrained")
    for index, row in enumerate(selected.itertuples()):
        with PILImage.open(Path(root) / "data/subset/train" / f"{row.image_name}.jpg") as opened:
            tensor = ordinary(opened.convert("RGB")).unsqueeze(0)
        for column, view in enumerate(TTA_PASSES):
            picture = tta_transform(tensor, view)[0].permute(1, 2, 0).numpy()
            axes[index, column].imshow(np.clip(picture * std + mean, 0, 1))
            axes[index, column].set_title(view)
            axes[index, column].set_xticks([])
            axes[index, column].set_yticks([])
        axes[index, 0].set_ylabel(f"{row.image_name}\n{'Malignant' if row.target else 'Benign'}", fontsize=9)
    fig.suptitle("Conservative whole-image views · illustrative retained fitting examples")
    _finish(fig, "uncertainty_four_view_preview", finish)


def uncertainty_checks(bundle):
    if not bundle["complete"]:
        return
    directory, summary = bundle["directory"], bundle["predictions"]
    display(summary[["image_id", "patient_id", "fold", "target", "base_score", "mc_mean_score", "mc_score_sd", "tta_mean_score", "tta_score_sd"]].head(8))
    display(pd.DataFrame(_read(directory / "inference_checks.json")["folds"]))
    counts = summary.groupby("fold").agg(images=("image_id", "size"), patients=("patient_id", "nunique"), malignant_images=("target", "sum"))
    counts["expected_MC_rows"] = counts.images * 30
    counts["expected_TTA_rows"] = counts.images * 4
    display(counts)
    display(pd.DataFrame([{"Total MC rows": len(summary) * 30, "Total TTA rows": len(summary) * 4,
                           "Cutoff fitting": False, "Calibration fitting": False, "BN updates": False,
                           "Referral score": "Frozen deterministic saved score"}]))


def uncertainty_convergence(bundle, finish):
    if not bundle["complete"]:
        return
    table = pd.read_csv(bundle["directory"] / "mc_convergence.csv")
    display(table)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    # The original two-panel composition is retained for every clean fold;
    # fold colours identify independent checkpoints and dashed SD separates it
    # from the mean-score line without introducing a third metric panel.
    for fold, rows in table.groupby("fold"):
        axes[0].plot(rows.n_passes, rows.mean_score_mae, marker="o", color=FOLD_COLORS[fold], label=f"Fold {fold}: mean score")
        axes[0].plot(rows.n_passes, rows.score_sd_mae, marker="s", linestyle="--", color=FOLD_COLORS[fold], label=f"Fold {fold}: score SD")
        axes[1].plot(rows.n_passes, rows.sd_spearman, marker="o", color=FOLD_COLORS[fold], label=f"Fold {fold}")
    axes[0].set(title="Numerical stability", xlabel="First MC passes used", ylabel="Mean absolute difference versus 30 passes", xticks=[5, 10, 20])
    axes[1].set(title="SD ranking stability", xlabel="First MC passes used", ylabel="Spearman correlation versus 30 passes", xticks=[5, 10, 20], ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    _finish(fig, "uncertainty_mc_convergence_by_fold", finish)


def uncertainty_quality(bundle, data, finish):
    if not bundle["complete"]:
        return
    summary = bundle["predictions"]
    metrics = pd.read_csv(bundle["directory"] / "fold_metrics.csv")
    display(metrics)
    means = pd.read_csv(bundle["directory"] / "mean_fold_metrics.csv")
    baseline = []
    for fold, rows in summary.groupby("fold"):
        prevalence = data.loc[data.fold.ne(fold), "target"].mean()
        baseline.append({"fold": fold, "mode": "constant fitting prevalence", **probability_metrics(rows.target, np.repeat(prevalence, len(rows)))})
    display(pd.concat([means, pd.DataFrame(baseline).groupby("mode")[["average_precision", "roc_auc", "log_loss", "brier"]].mean().reset_index()], ignore_index=True))
    plot_probability_views(summary, {"Deterministic": "base_score", "MC mean": "mc_mean_score", "TTA mean": "tta_mean_score"},
                           finish, "uncertainty_mean_scores", "Clean held-out score averaging · descriptive raw outputs")


def uncertainty_distributions(bundle, finish):
    if not bundle["complete"]:
        return
    summary = bundle["predictions"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axes[0].hist(summary.mc_score_sd, bins=35, color="#245D8C")
    axes[1].hist(summary.tta_score_sd, bins=35, color="#B46A20")
    axes[2].scatter(summary.mc_score_sd, summary.tta_score_sd, s=9, alpha=.4, color="#666666")
    axes[0].set(xlabel="MC score SD (30 passes)", ylabel="Images", title="Dropout-mask sensitivity")
    axes[1].set(xlabel="TTA score SD (4 views)", ylabel="Images", title="Orientation sensitivity")
    axes[2].set(xlabel="MC score SD", ylabel="TTA score SD", title="Agreement between diagnostics")
    _finish(fig, "uncertainty_sd_histograms_scatter", finish)
    entropy = predictive_entropy(summary.base_score)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].scatter(summary.base_score, entropy, s=9, alpha=.35, color="#666666")
    axes[0].set(xlabel="Frozen deterministic score", ylabel="Predictive entropy (natural-log units)", title="Entropy ranks scores near 0.5")
    axes[1].scatter(summary.base_score, summary.mc_score_sd, s=9, alpha=.35, label="MC SD")
    axes[1].scatter(summary.base_score, summary.tta_score_sd, s=9, alpha=.35, label="TTA SD")
    axes[1].set(xlabel="Frozen deterministic score", ylabel="Score SD", title="Malignant ranking and variation answer different questions")
    axes[1].legend()
    _finish(fig, "uncertainty_entropy_and_score_variation", finish)


def uncertainty_referral(bundle, finish):
    if not bundle["complete"]:
        return
    referral = pd.read_csv(bundle["directory"] / "referral_rankings.csv")
    selected = referral.loc[referral.referral_fraction.eq(.2) & referral.ranking.ne("random")]
    columns = ["method", "ranking", "n_images", "n_patients", "n_malignant_images", "n_malignant_patients", "n_referred", "n_malignant_images_referred", "n_malignant_patients_referred", "retained_error", "malignant_referred_fraction"]
    display(selected[columns])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for row, method in enumerate(("mc", "tta")):
        for ranking in ("uncertainty_sd", "prediction_score", "predictive_entropy", "random"):
            group = referral.loc[referral.method.eq(method) & referral.ranking.eq(ranking)]
            curve = group.groupby("referral_fraction")[["retained_error", "malignant_referred_fraction"]].mean().reset_index()
            for column, metric in enumerate(("retained_error", "malignant_referred_fraction")):
                axes[row, column].plot(curve.referral_fraction * 100, curve[metric] * 100, marker="o", markersize=4, color=REFERRAL_COLORS[ranking], linestyle=REFERRAL_LINES[ranking], label=ranking.replace("_", " "))
        axes[row, 0].set(title=f"{method.upper()} SD: common deterministic-score error", ylabel="Classification error at score 0.5 (%)")
        axes[row, 1].set(title=f"{method.upper()} SD: malignant images referred", ylabel="Fraction of malignant images referred (%)", ylim=(0, 100))
        for ax in axes[row]:
            ax.set_xlabel("Images referred (%)")
            ax.legend(fontsize=8)
    _finish(fig, "uncertainty_referral_order_comparison", finish)
    fold_referral = pd.read_csv(bundle["directory"] / "referral_per_fold.csv")
    display(fold_referral.loc[fold_referral.referral_fraction.eq(.2) & fold_referral.ranking.eq("uncertainty_sd")][["fold", *columns]])


def uncertainty_intervals(bundle, finish):
    if not bundle["complete"]:
        return
    table = pd.read_csv(bundle["directory"] / "paired_patient_bootstrap_differences.csv")
    selected = table.loc[table.referral_fraction.eq(.2)]
    display(selected)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, metric, title in zip(axes, ("retained_error", "malignant_referred_fraction"), ("Retained error: negative favours SD", "Malignant referral: positive favours SD")):
        rows = selected.loc[selected.metric.eq(metric)].reset_index(drop=True)
        for index, row in enumerate(rows.itertuples()):
            color = "#245D8C" if row.method == "mc" else "#B46A20"
            ax.hlines(index, row.ci_lower * 100, row.ci_upper * 100, color=color, linewidth=2)
            ax.plot(row.estimate * 100, index, marker="o" if row.method == "mc" else "s", color=color)
        ax.axvline(0, color="#333333", linestyle=":")
        ax.set(yticks=np.arange(len(rows)), yticklabels=[f"{r.method.upper()} − {r.comparator.replace('_', ' ')}" for r in rows.itertuples()], xlabel="SD minus comparator (percentage points)", title=title)
        ax.invert_yaxis()
    fig.suptitle("Paired patient-bootstrap 95% intervals at predeclared 20% referral")
    _finish(fig, "uncertainty_paired_patient_intervals_at20", finish)
    ordinary = selected.loc[selected.comparator.eq("prediction_score")]
    including_zero = ordinary.ci_lower.le(0) & ordinary.ci_upper.ge(0)
    display(Markdown(f"At the predeclared 20% referral, **{int(including_zero.sum())} of {len(ordinary)}** displayed intervals against score ranking include zero. Read the direction and support counts; these conditional intervals do not establish a clinical policy or a final architecture."))


def uncertainty_examples(bundle, root, finish):
    if not bundle["complete"]:
        return
    summary = bundle["predictions"]
    chosen = pd.concat([summary.sort_values([column, "image_id"], ascending=[False, True]).head(1).assign(reason=reason)
                        for column, reason in (("mc_score_sd", "Largest MC SD"), ("tta_score_sd", "Largest TTA SD"), ("base_score", "Largest ordinary score"))])
    display(chosen[["reason", "image_id", "patient_id", "fold", "target", "base_score", "mc_score_sd", "tta_score_sd"]])
    fig, axes = plt.subplots(1, len(chosen), figsize=(11, 4), squeeze=False, layout="constrained")
    for ax, row in zip(axes.flat, chosen.itertuples()):
        with PILImage.open(Path(root) / "data/subset/train" / f"{row.image_id}.jpg") as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{row.reason}\n{row.image_id}, fold {row.fold}\nscore={row.base_score:.3f}; MC SD={row.mc_score_sd:.3f}; TTA SD={row.tta_score_sd:.3f}", fontsize=9)
        ax.axis("off")
    _finish(fig, "uncertainty_saved_ranking_examples", finish)


def legacy_image_specs():
    return [
        ("09_image_modelling.ipynb", "09 · Clean B0: preparation, nested selection and OOF evaluation",
         "EfficientNet-B0 starts from ImageNet features and learns a new melanoma output. The full bounded selection procedure repeats inside five preserved patient groups; the notebook explains preparation, learning, ranking, probability quality and decisions using saved clean records.", [
            ("Purpose, settings and the image pipeline", "An architecture is the network design; a recipe fixes learning rate and weighting. An epoch is a pass through fitting images. Every distinct fitting job begins with original IMAGENET1K_V1 weights, a new classifier and fresh optimiser/scheduler. Dropout remains 0.2; batch-normalisation statistics stay frozen. The table records the exact 224-pixel weak-preparation settings.",
             'from melanoma_detection import dedup_notebook_images as image_docs\nimage_docs.image_settings(ROOT)\nimage_docs.show_pipeline(finish_figure)'),
            ("Inspect preserved fitting preparation", "Read each row from original RGB through the deterministic validation view and three seeded weak fitting variants. Whole-image bilinear resize and ImageNet colour normalisation are shared. Flips/rotation occur only during fitting. These examples come from retained fitting patients in outer pool 0, inner fitting set; no model is trained and no source image changes.",
             'image_docs.show_preparation(ROOT, data, dataset_identity, finish_figure)'),
            ("Nested selection and computing cost", "Within each outer-training pool, screen four learning-rate/weighting recipes for six epochs on inner fold 0. Confirm the selected recipe to 12 epochs across three preserved inner patient folds. Take the median of three AP-best epochs, then freshly refit every outer-training patient for that fixed duration. Only then predict the held-out outer group. Square-root weights use deduplicated fitting class counts. Thirty-five raw fits measure this bounded procedure; 15 additional fits support training-only calibration.",
             'image_docs.image_progress(ROOT, data, dataset_identity, finish_figure)'),
            ("Read the inner learning curves", "AP and ROC-AUC are higher-is-better ranking measures; log loss and Brier are lower-is-better score errors. Hollow rings mark AP-selected epochs across every panel. The vertical guide marks the six-epoch screen. Histories can be displayed before full OOF completion, so they are explicitly provisional selection diagnostics. The AP reference is the malignant-image fraction in this preserved inner-validation fold. Weighted fitting-loss scales cannot rank recipes against each other.",
             'display(Markdown("The saved selection table, committed histories and measured minutes above explain each fold’s procedure; unfinished fold outcomes remain pending."))'),
            ("Verify raw and calibration completion", "A completed fold is not a completed five-fold experiment. Strict loaders check dataset identity, prediction coverage and hashes before final figures. Reporting-pending and training-pending are shown separately; no historical predictions fill an unfinished clean fold.",
             'from melanoma_detection.image_nested_calibration_reporting import load_saved_calibration_run\nraw = load_image_report(ROOT)\ncalibration = load_saved_calibration_run(ROOT, "b0_nested_calibration_dedup_v1")\ndisplay(pd.DataFrame([{ "stage": "raw B0", "status": raw["status"], "complete": raw["complete"]}, {"stage": "image calibration", "status": calibration["status"], "complete": calibration["complete"]}]))'),
            ("Outer-fold ranking, ROC and precision–recall", "Unweighted mean-fold AP is primary because malignant images are rare. Each bar uses independent held-out patients within this run; five-fold spread is descriptive variability, not a confidence interval. ROC plots recall against false alarms; PR plots precision against recall. The pooled curves are secondary because fold-specific models can have different score scales. Reliability bins compare mean score with observed melanoma frequency and show their support.",
             'image_docs.raw_oof_figures(raw, finish_figure)'),
            ("Training-only calibration and frozen decisions", "Three fresh fixed-duration fits per outer-training pool produce training-only OOF scores. Those scores fit the logistic-on-logits map and academic 80%-recall cutoffs. Outer patients never fit maps or cutoffs. Read confusion matrices by true-label rows and decision columns: FN are missed malignant images; FP are benign flags. The raw 0.5 cutoff is descriptive, while the training-selected scenarios use fold-specific frozen rules.",
             'image_docs.calibration_figures(calibration, finish_figure)'),
            ("Matched clean metadata comparison", "B0, fixed spline and the three-seed NN mean must cover exactly the same retained image IDs and patients. AP is the predeclared ranking objective; probability and cutoff measures address different questions. Prior development exploration remains a limitation, and a patient bootstrap conditions on saved fits rather than repeating training.",
             'table = comparison_table()\nselected = table[table.model.isin(["b0_raw", "metadata_nn_mean", "spline_raw"])] if not table.empty else table\ndisplay(selected)\nmodel_chart(selected, "Clean B0 and metadata: mean outer-fold AP")\nimage_docs.matched_comparator_figures(score_bundle(), finish_figure)')]),
        ("10_bounded_image_pilots.ipynb", "10 · Bounded duration and ConvNeXt-Tiny pilots",
         "Ask two focused optimisation questions after the clean baseline: whether 24 epochs improve the fixed 12-epoch endpoint, and whether ConvNeXt-Tiny improves the matching B0 pilot. Keep optimisation evidence separate from independent outer-fold evaluation.", [
            ("Controls, fitting scope and deferred work", "Both pilots use outer fold 0’s training pool and its preserved three inner patient folds. Common settings are retained image IDs, 224-pixel RGB preparation, weak augmentation, effective batch 64 and the same ranking objective. Every distinct fit starts from original pretrained weights. The original 30-epoch scheduler remains fixed while the new cap is 24. DenseNet sweeps, full alternative-architecture OOF, stronger augmentation, CLAHE and resolution changes remain deferred.",
             'from melanoma_detection import dedup_notebook_images as image_docs\ninner = load_preserved_inner_memberships(ROOT, dataset_identity, 0)\ndisplay(inner.groupby("inner_fold").agg(images=("image_name", "size"), patients=("patient_id", "nunique"), malignant_images=("target", "sum")))\ndisplay(pd.DataFrame([{"Pilot": "B0 duration", "Fits": 3, "Epochs": 72, "Endpoints": "12 and 24", "Recipe": "clean baseline’s selected outer-pool recipe"}, {"Pilot": "ConvNeXt-Tiny", "Fits": 6, "Epochs": 90, "Endpoints": "12 and 24", "Recipe": "four six-epoch screens on inner 0, then selected confirmations"}]))'),
            ("Completion ledger and selected recipes", "ConvNeXt-Tiny changes network design while preserving preparation. Its four six-epoch recipes screen on inner 0; the selected clean trajectory continues to 24 and the same recipe fits inner 1 and 2. A completed family is admitted only after endpoint predictions, memberships and receipt hashes verify. Missing or budget-stopped families do not lose a comparison.",
             'pilots = image_docs.load_pilot_bundle(pilot_report)\nimage_docs.pilot_screening(pilots)'),
            ("Learning over epochs", "Read completed epoch horizontally and AP/AUC, log loss and Brier vertically. Each line is a family and preserved inner split; solid B0 and dashed ConvNeXt lines share the same 12/24 guides. AP and AUC are higher-is-better; log loss and Brier are lower-is-better. These curves describe selected inner recipes, not independent outer performance.",
             'image_docs.pilot_learning(pilots, finish_figure)'),
            ("Matching fixed endpoints and best-in-cap metrics", "The same fresh trajectories supply frozen epoch-12 and epoch-24 snapshots. The left panel compares predeclared fixed endpoints; the right reports AP-selected best epochs within each cap. Best-in-cap metrics reuse validation labels for selection and are more optimistic. Mean inner-fold AP is primary, with AUC and score-quality metrics alongside it. The cap remains 24 even if the best epoch reaches its boundary.",
             'image_docs.pilot_endpoint_metrics(pilots, finish_figure)'),
            ("ROC, precision–recall and descriptive reliability", "Only completed fixed-24 exports on exactly matching retained inner image/patient records enter these figures. ROC plots recall against false alarms; PR plots precision against recall, with the malignant-image fraction as a reference. Equal-count reliability compares mean raw score with observed malignancy and shows bin support. Recipe selection already used these patients, so favourable curves are exploratory.",
             'image_docs.pilot_matching_curves(pilots, finish_figure)'),
            ("Fixed-cutoff errors", "Flag raw scores at the declared 0.5 cutoff; no inner-validation cutoff is selected here. Confusion rows are recorded benign/malignant labels and columns are predicted decisions. FN are missed malignancies; FP are benign flags. Recall, specificity, precision and flagged counts complement ranking; undefined ratios remain missing. Different class-weighting and network score scales make this a descriptive scenario.",
             'image_docs.pilot_decisions(pilots, finish_figure)'),
            ("Recorded training time and GPU memory", "The bars compare recorded fitting minutes and peak allocated GPU GiB. The table also records reserved memory, distinct fits and epochs. ConvNeXt includes unsuccessful recipe screens in its workload. Peak memory and fitting minutes describe this machine and protocol; they cannot establish a network’s general speed or justify ignoring matching prediction quality.",
             'image_docs.pilot_cost(pilots, finish_figure)'),
            ("Paired patient intervals and support", "Whole-patient draws within the three inner folds serve both sides of each 24-minus-12 or ConvNeXt-minus-B0 comparison. Positive AP/AUC differences favour the candidate. The interval table records requested, valid and single-class-rejected draws, and the support table exposes malignant-image counts. These conditional intervals omit retraining and selection variability; best-in-cap comparisons remain selected evidence.",
             'image_docs.pilot_interval_support(pilots, finish_figure)'),
            ("Qualified interpretation and next step", "Duration and architecture effects are kept separate from the clean baseline. A positive fixed-endpoint AP change can motivate another procedure but cannot automatically promote it. An incomplete family supplies no comparison result; neither a favourable epoch nor a favourable raw cutoff proves generalisation.",
             'display(Markdown("**Interpretation:** a promising pilot is a candidate procedure. Promotion requires selection repeated within every outer-training pool and a separate five-fold evaluation. Full DenseNet/ConvNeXt OOF, stronger augmentation, CLAHE, resolution changes and fusion remain deferred."))')]),
        ("11_image_uncertainty.ipynb", "11 · Clean outer-fold uncertainty and referral diagnostics",
         "Score variation under dropout masks and conservative image views can prioritise review differently from a high melanoma score. Repeat the original diagnostic depth on clean five-fold held-out records, with no historical checkpoints or inferred missing results.", [
            ("Explicit clean checkpoint scope and completion", "Each retained image uses its own outer-held-out B0 checkpoint, fitted without that patient. Strict receipts bind the exact clean manifest, final selected duration, checkpoint bytes, source hashes and saved deterministic predictions. Full uncertainty waits for all five raw refits. A smoke check verifies repeatability, seeded dropout, identity TTA and unchanged model/BatchNorm state before costly inference.",
             'from melanoma_detection import dedup_notebook_images as image_docs\nuncertainty = image_docs.load_uncertainty(ROOT, dataset_identity)'),
            ("Inspect conservative whole-image views", "Columns are identity, horizontal flip, vertical flip and both flips after deterministic preparation. Flips rearrange existing pixels; they add no crop, reflected border or interpolation. The examples are retained fitting photographs and illustrate framing, not a performance sample. This cell previews images only and starts no inference.",
             'image_docs.uncertainty_views(ROOT, data, dataset_identity, finish_figure)'),
            ("Keep MC dropout and TTA separate", "MC dropout makes 30 passes at the trained explicit-dropout rate. BatchNorm and stochastic depth remain in evaluation mode. Seed 20261004 and recorded pass seeds make masks repeatable. TTA uses four deterministic views with dropout disabled. Both use batch 64 and full precision. Means and sample SDs are computed separately; real held-out targets are joined only after prediction.",
             'display(pd.DataFrame([{"Mode": "MC dropout", "Passes": 30, "Stochastic component": "explicit trained dropout only", "Seed": 20261004}, {"Mode": "TTA", "Passes": 4, "Stochastic component": "none; fixed tensor flips", "Seed": 20261004}]))'),
            ("Validate pass coverage before aggregation", "Every image/pass combination must occur exactly once with finite scores in [0,1] and matching patient/fold identities. Retained records total 6,706 images and 422 patients, including 127 malignant images from 88 patients. Saved raw passes allow independent means/SD recalculation. Deterministic scores must reproduce the checkpoint export within 1e-6, and the frozen saved values drive referral.",
             'image_docs.uncertainty_checks(uncertainty)'),
            ("Numerical and ranking stability with more passes", "Compare prefixes of 5, 10 and 20 passes against the same 30-pass reference. Smaller score/SD mean absolute differences suggest numerical stability; Spearman correlation closer to one suggests a stable SD ordering. Draws and pass count are prescribed rather than selected from labels. Lines are preserved outer folds; stability is not predictive correctness.",
             'image_docs.uncertainty_convergence(uncertainty, finish_figure)'),
            ("Ranking and descriptive reliability of averaged scores", "Report deterministic, MC-mean and TTA-mean AP/AUC, log loss and Brier on each outer fold, with unweighted mean-fold summaries primary. Constant fitting prevalence is computed separately from each fold’s training patients. Pooled ROC/PR and equal-count reliability are descriptive. A favourable mean score does not prove SD improves referral and does not replace the training-only calibration/cutoff procedure.",
             'image_docs.uncertainty_quality(uncertainty, data, finish_figure)'),
            ("What SD and predictive entropy capture", "Histograms show dropout-mask and orientation sensitivity on their own scales. The scatter compares MC and TTA SD on the same images. Predictive entropy −s log(s) − (1−s) log(1−s) prioritises scores near 0.5; it is not epistemic uncertainty. High malignant score and high score variation answer different questions.",
             'image_docs.uncertainty_distributions(uncertainty, finish_figure)'),
            ("Referral order with fixed underlying decisions", "Set aside the prescribed fractions 0%, 5%, 10%, 20%, 30%, 40% and 50% for review. Compare descending SD, ordinary score, predictive entropy and the mean of 100 seeded random rankings (seed 17042). All methods retain the same deterministic classification score and descriptive cutoff 0.5, so differences isolate ordering. Patient and malignant-patient counts show support; referred images are not assumed correctly diagnosed.",
             'image_docs.uncertainty_referral(uncertainty, finish_figure)'),
            ("Paired whole-patient uncertainty intervals", "Use 2,000 draws, seed 20242, resampling the original patient count independently within each outer fold. Every ranking and MC/TTA method receives identical patient multiplicities and paired random orders. Recompute rankings and referral counts on every draw. At the predeclared 20% referral, negative retained-error differences favour SD; positive malignant-referral differences favour SD. Undefined malignant fractions remain missing, with valid-draw counts shown.",
             'image_docs.uncertainty_intervals(uncertainty, finish_figure)'),
            ("Examples from completed clean rankings", "Choose examples deterministically by largest MC SD, largest TTA SD and largest ordinary score, breaking ties by image ID. These are case illustrations from saved clean predictions, not automatic exclusions, proof of correctness or a representative quality sample.",
             'image_docs.uncertainty_examples(uncertainty, ROOT, finish_figure)'),
            ("Provenance and limits", "Dropout SD covers explicit masks, not backbone stochastic depth, training-seed variation or every source of uncertainty. Four flips cover orientation changes, not new patients or image conditions. Patient-bootstrap intervals condition on fitted checkpoints, selected procedures and the observed inference draws; they omit retraining and selection variability. Keep calibrated decisions and raw diagnostic means separate. No clinical referral policy, final architecture, fusion result or reserved-test performance is established.",
             'if uncertainty["complete"]:\n    receipt = uncertainty["receipt"]\n    display(pd.DataFrame([{key: receipt[key] for key in ("dataset_manifest_sha256", "bootstrap_sampling", "referral_score", "test_set_loaded", "calibration_fitted", "cutoff_fitted")}]))\nelse:\n    display(Markdown("**Conclusion pending:** completed clean uncertainty exports are required before score variation or referral benefit is interpreted."))')])
    ]


def image_specs():
    """Generate editable reports from the original narrative/cell templates."""
    from .dedup_notebook_image_templates import image_template_specs
    return image_template_specs()
