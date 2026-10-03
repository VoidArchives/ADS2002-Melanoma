"""Inspectable pilot figures; no training, selection or test-data access."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .image_model import make_transform, seed_everything


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


def plot_learning_curves(history, validation_prevalence):
    if history.empty:
        raise ValueError("No completed epochs available for plotting.")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        ("average_precision", "Melanoma ranking: average precision", "Average precision"),
        ("roc_auc", "Ranking: ROC-AUC", "ROC-AUC"),
        ("log_loss", "Raw score quality: log loss", "Unweighted validation log loss"),
        ("brier", "Raw score quality: Brier score", "Brier score"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        for recipe_id, rows in history.groupby("recipe_id", sort=False):
            colour, marker, line = STYLES[recipe_id]
            rows = rows.sort_values("epoch")
            ax.plot(rows.epoch, rows[metric], color=colour, marker=marker,
                    linestyle=line, linewidth=1.8, markersize=5, label=recipe_id)
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
        ax.set(title=title, xlabel="Completed epoch", ylabel=ylabel)
        ax.set_xticks(sorted(history.epoch.unique()))
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=9, loc="best")
    fig.suptitle("B0 pilot: one inner patient split; outer fold 0 excluded", fontsize=14)
    return fig


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
        colour, marker, _ = STYLES[recipe_id]
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
