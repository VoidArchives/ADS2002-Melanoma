"""Inspectable pilot figures; no training, selection or test-data access."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .image_model import make_transform, seed_everything
from .image_evaluation import assert_matching_predictions, classification_metrics, curve_tables, metric_table
from .metadata_evaluation import calibration_bins


STYLES = {
    "lr3e5_none": ("#2563a6", "o", "-"),
    "lr1e4_none": ("#c87622", "s", "--"),
    "lr3e5_sqrt": ("#3b8270", "^", "-."),
    "lr1e4_sqrt": ("#8550a2", "D", ":"),
}


def load_histories(run_dir):
    frames = []
    for folder in sorted((Path(run_dir) / "recipes").glob("*")):
        path = folder / "training_history.csv"
        if not path.is_file():
            path = folder / "history.csv"
        if path.is_file() and path.stat().st_size:
            frame = pd.read_csv(path)
            if not frame.empty:
                frames.append(frame.assign(recipe_id=folder.name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _style(identity, index=0):
    recipe = identity.split("/")[-1]
    if recipe in STYLES and identity.startswith("efficientnet_b0/"):
        return STYLES[recipe]
    palette = plt.get_cmap("tab20")
    return palette(index % 20), ["o", "s", "^", "D"][index % 4], ["-", "--", "-.", ":"][index % 4]


def _identity_column(history):
    return "pipeline_id" if "pipeline_id" in history else "recipe_id"


def plot_learning_curves(history, validation_prevalence, title="Exploratory inner validation"):
    if history.empty:
        raise ValueError("No completed epochs available for plotting.")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        ("average_precision", "Melanoma ranking: average precision", "Average precision"),
        ("roc_auc", "Ranking: ROC-AUC", "ROC-AUC"),
        ("log_loss", "Raw score quality: log loss", "Unweighted validation log loss"),
        ("brier", "Raw score quality: Brier score", "Brier score"),
    ]
    for ax, (metric, panel_title, ylabel) in zip(axes.flat, panels):
        for index, (recipe_id, rows) in enumerate(history.groupby(_identity_column(history), sort=False)):
            colour, marker, line = _style(recipe_id, index)
            rows = rows.sort_values("epoch")
            status = str(rows.run_status.iloc[0]) if "run_status" in rows else "complete"
            label = recipe_id + (f" [{status}]" if status != "complete" else "")
            ax.plot(rows.epoch, rows[metric], color=colour, marker=marker,
                    linestyle=line, linewidth=1.8, markersize=5, label=label)
            if "selected_epoch" in rows:
                selected = rows.loc[rows.epoch.eq(rows.selected_epoch.iloc[0])]
                if len(selected):
                    ax.scatter(selected.epoch, selected[metric], s=130, facecolors="none",
                               edgecolors=[colour], linewidths=2, zorder=5)
        if metric == "average_precision":
            ax.axhline(validation_prevalence, color="#555555", linestyle="--",
                       linewidth=1, label="Validation prevalence")
            ax.set_ylim(bottom=0)
        elif metric == "roc_auc":
            ax.axhline(.5, color="#555555", linestyle="--", linewidth=1)
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=0)
        ax.axvline(1.5, color="#999999", linestyle=":", linewidth=1)
        ax.set(title=panel_title, xlabel="Completed epoch", ylabel=ylabel)
        ax.set_xticks(sorted(history.epoch.unique()))
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="outside lower center", ncol=2)
    fig.suptitle(title + "\nHollow rings mark highest-AP epochs", fontsize=13)
    return fig


def plot_training_losses(history, title="Training objectives: magnitudes are not comparable across weights"):
    if history.empty:
        raise ValueError("No executed training histories.")
    groups = list(history.groupby("loss_weight", sort=True)) if "loss_weight" in history else [("recorded objective", history)]
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.5),
                             squeeze=False, constrained_layout=True)
    for ax, (weight, group) in zip(axes.flat, groups):
        for index, (identity, rows) in enumerate(group.groupby(_identity_column(group), sort=False)):
            colour, marker, line = _style(identity, index)
            rows = rows.sort_values("epoch")
            status = str(rows.run_status.iloc[0]) if "run_status" in rows else "complete"
            label = identity + (f" [{status}]" if status != "complete" else "")
            ax.plot(rows.epoch, rows.train_loss, color=colour, marker=marker,
                    linestyle=line, label=label, linewidth=1.6)
            if "selected_epoch" in rows:
                best = rows.loc[rows.epoch.eq(rows.selected_epoch.iloc[0])]
                ax.scatter(best.epoch, best.train_loss, s=130, facecolors="none", edgecolors=[colour])
        ax.set(title=f"Loss weighting: {weight}", xlabel="Completed epoch", ylabel="Fitting BCE objective")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=12)
    return fig


def plot_roc_precision_recall(frames, epochs=None, title="Exploratory inner-validation ROC / precision–recall"):
    if not frames:
        raise ValueError("No completed prediction records.")
    assert_matching_predictions(frames)
    table = metric_table(frames, epochs=epochs).set_index("pipeline_id")
    first = next(iter(frames.values()))
    prevalence = first.target.mean()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for index, (identity, frame) in enumerate(frames.items()):
        colour, _, line = _style(identity, index)
        curves = curve_tables(frame.target, frame.base_probability)
        suffix = f" | epoch {(epochs or {}).get(identity, '?')}"
        axes[0].plot(curves["roc"].false_positive_rate, curves["roc"].true_positive_rate,
                     color=colour, linestyle=line, linewidth=1.7,
                     label=f"{identity}{suffix} | AUC {table.loc[identity, 'roc_auc']:.3f}")
        axes[1].step(curves["precision_recall"].recall, curves["precision_recall"].precision,
                     where="post", color=colour, linestyle=line, linewidth=1.7,
                     label=f"{identity}{suffix} | AP {table.loc[identity, 'average_precision']:.3f}")
    axes[0].plot([0, 1], [0, 1], color="#666666", linestyle=":", label="Chance ranking")
    axes[1].axhline(prevalence, color="#666666", linestyle=":", label=f"Prevalence {prevalence:.3%}")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate / sensitivity", title="ROC")
    axes[1].set(xlabel="Recall / sensitivity", ylabel="Precision", title="Precision–recall (AP is not trapezoidal area)")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.18))
    fig.suptitle(f"{title}\n{len(first):,} images | {first.patient_id.nunique()} patients | raw scores", fontsize=13)
    return fig


def plot_confusion_matrices(frames, epochs=None, threshold=.5,
                            title="Exploratory classification: fixed raw threshold 0.5"):
    if not frames:
        raise ValueError("No completed prediction records.")
    assert_matching_predictions(frames)
    columns = min(3, len(frames))
    rows = (len(frames) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 3.9 * rows),
                             squeeze=False, constrained_layout=True)
    for ax, (identity, frame) in zip(axes.flat, frames.items()):
        m = classification_metrics(frame.target, frame.base_probability, threshold)
        counts = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        totals = counts.sum(axis=1, keepdims=True)
        percentages = np.divide(counts, totals, out=np.full((2, 2), np.nan), where=totals != 0)
        ax.imshow(percentages, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                fraction = f"{percentages[i, j]:.1%}" if np.isfinite(percentages[i, j]) else "undefined"
                ax.text(j, i, f"{counts[i, j]}\n{fraction}", ha="center", va="center",
                        color="white" if percentages[i, j] > .55 else "black", fontsize=9)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benign", "Melanoma"],
               yticklabels=["Benign", "Melanoma"], xlabel="Predicted class", ylabel="True class",
               title=f"{identity}\nepoch {(epochs or {}).get(identity, '?')} | raw ≥ {threshold:g}")
        ax.tick_params(axis="both", length=0)
    for ax in list(axes.flat)[len(frames):]:
        ax.set_visible(False)
    fig.suptitle(title + "\nCounts and percentages within each true class", fontsize=13)
    return fig


def plot_resource_comparison(history, metric_rows):
    if history.empty or metric_rows.empty:
        raise ValueError("No completed comparable trials.")
    resources = history.groupby(_identity_column(history)).agg(
        measured_training_minutes=("epoch_seconds", lambda value: value.sum() / 60),
        peak_allocated_GiB=("peak_memory_bytes", lambda value: value.max() / 2**30)).reset_index()
    resources = resources.rename(columns={_identity_column(history): "pipeline_id"})
    table = metric_rows.merge(resources, on="pipeline_id", validate="one_to_one")
    fig, axes = plt.subplots(1, 3, figsize=(14, max(4.5, .35 * len(table))), constrained_layout=True)
    labels = table.pipeline_id.tolist()
    for ax, column, title in zip(axes, ["average_precision", "measured_training_minutes", "peak_allocated_GiB"],
                                  ["Selected-checkpoint AP", "Measured epoch time (minutes)", "Peak allocated GPU memory (GiB)"]):
        ax.barh(labels, table[column], color="#387ca6")
        ax.invert_yaxis()
        ax.set(title=title, xlabel=column.replace("_", " "))
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Matched six-epoch trials: recorded time excludes weight downloads", fontsize=13)
    return fig


def save_figure(fig, path, dpi=150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=.2)
    return path


def plot_raw_reliability(frames, title="Exploratory raw-score reliability; no calibration fitted"):
    if not frames:
        raise ValueError("No completed prediction records.")
    assert_matching_predictions(frames)
    tables = {identity: calibration_bins(frame.target, frame.base_probability, n_bins=5,
              groups=frame.patient_id) for identity, frame in frames.items()}
    columns = min(3, len(tables))
    n_rows = (len(tables) + columns - 1) // columns
    upper = min(1, max(.1, 1.15 * max(max(table.mean_probability.max(), table.observed_rate.max())
                                    for table in tables.values())))
    fig, axes = plt.subplots(n_rows, columns, figsize=(4.2 * columns, 3.7 * n_rows),
                             squeeze=False, constrained_layout=True)
    for index, (ax, (identity, table)) in enumerate(zip(axes.flat, tables.items())):
        colour, marker, _ = _style(identity, index)
        ax.plot([0, upper], [0, upper], color="#666666", linestyle=":")
        ax.plot(table.mean_probability, table.observed_rate, marker=marker, color=colour)
        ax.set(xlim=(0, upper), ylim=(0, upper), title=identity,
               xlabel="Mean raw score", ylabel="Observed melanoma fraction")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(tables):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13)
    return fig, pd.concat([table.assign(pipeline_id=identity) for identity, table in tables.items()], ignore_index=True)


def plot_reliability_checks(run_dir):
    available = [(folder.name, folder / "reliability.csv") for folder in
                 sorted((Path(run_dir) / "recipes").glob("*"))
                 if (folder / "reliability.csv").is_file()]
    if not available:
        raise ValueError("No raw-score reliability tables available.")
    tables = [(recipe_id, pd.read_csv(path)) for recipe_id, path in available]
    maximum = max(max(bins.mean_probability.max(), bins.observed_rate.max())
                  for _, bins in tables)
    upper = min(1, max(.1, maximum * 1.15))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (recipe_id, bins) in zip(axes.flat, tables):
        colour, marker, _ = _style(f"efficientnet_b0/{recipe_id}")
        ax.plot([0, upper], [0, upper], color="#555555", linestyle="--", linewidth=1)
        ax.plot(bins.mean_probability, bins.observed_rate, color=colour,
                marker=marker, linewidth=1.8)
        ax.text(.03, .95, f"{bins.n_images.sum():,} images | "
                f"{bins.n_malignant_images.sum()} melanoma images", transform=ax.transAxes,
                va="top", fontsize=9)
        ax.set(xlim=(0, upper), ylim=(0, upper), title=recipe_id,
               xlabel="Mean raw score in bin", ylabel="Observed melanoma fraction")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(available):]:
        ax.set_visible(False)
    fig.suptitle("Raw-score calibration checks: shared axes cover observed bin means", fontsize=14)
    return fig


def plot_augmentation_preview(fitting, spec):
    # Purposeful QA sample, not a prevalence-representative image sample.
    chosen = pd.concat([
        fitting.loc[fitting.target.eq(1)].sample(n=min(2, int(fitting.target.sum())), random_state=42),
        fitting.loc[fitting.target.eq(0)].sample(n=min(4, int(fitting.target.eq(0).sum())), random_state=42),
    ]).reset_index(drop=True)
    transform = make_transform(spec, training=True)
    mean = np.array([.485, .456, .406])
    std = np.array([.229, .224, .225])
    fig, axes = plt.subplots(len(chosen), 4, figsize=(11, 2.6 * len(chosen)),
                             squeeze=False, constrained_layout=True)
    for row_index, row in chosen.iterrows():
        with Image.open(row.image_path) as source:
            original = source.convert("RGB")
            axes[row_index, 0].imshow(original)
            axes[row_index, 0].set_title(f"{row.image_name} | target {row.target}", fontsize=9)
            for variant in range(3):
                seed_everything(4200 + row_index * 3 + variant)
                array = transform(original).permute(1, 2, 0).numpy() * std + mean
                axes[row_index, variant + 1].imshow(np.clip(array, 0, 1))
                axes[row_index, variant + 1].set_title(f"Seeded weak variant {variant + 1}", fontsize=9)
        for ax in axes[row_index]:
            ax.axis("off")
    fig.suptitle("Weak augmentation QA: fitting patients only; deliberate label mix", fontsize=13)
    return fig
