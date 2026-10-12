"""Version-two presentation of saved image experiments, without image compute.

The historical image recipes, AP-selected epochs and admission gates are read
unchanged. Only saved scores are summarised with mean-fold ROC-AUC first. No
checkpoint is instantiated and no training, smoke, inference or gate is run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display
from .metadata_roc_historical import load_clean_scores, latest_evaluation, save_metadata_table

REPORT_ROOT = "results/metadata_nn_tuning/v2/report_verification"
UNCERTAINTY_RUN = "efficientnet_b0_lr3e5_sqrt_mc30_tta4_20261003T235530258956Z"
MANIFEST = "data/subset/processed/dedup_v1/dataset_manifest.json"
DIRECTORY = "results/dedup_v1/clean_workflow"
METADATA_SUITE = "results/dedup_v1/metadata_suite/metadata_full_dedup_v1"
UNCERTAINTY = "results/dedup_v1/uncertainty/b0_uncertainty_dedup_v1"
CORE_RECEIPT = "results/dedup_v1/metadata/comparator_receipt.json"


def _path(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Saved image report inputs must stay inside the project.")
    return path


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def report_directory(root, condition, stage):
    if condition not in ("original", "dedup_v1"):
        raise ValueError("An explicit development condition is required.")
    if not stage or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in stage):
        raise ValueError("A simple report stage is required.")
    folder = _path(root, f"{REPORT_ROOT}/image_reports/{condition}/{stage}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_figure(figure, path, *, dpi=150):
    """Reject accidental historical figure writes before delegating formatting."""
    path = Path(path).resolve()
    root = next(p for p in path.parents if (p / "pyproject.toml").is_file())
    if not path.is_relative_to((root / REPORT_ROOT).resolve()):
        raise ValueError("Image presentation exports must stay in the v2 report namespace.")
    from .image_reporting import save_figure as original_save
    return original_save(figure, path, dpi=dpi)


def show_saved_figure(root, relative):
    """Show a previously rendered preview instead of decoding/transformation."""
    path = _path(root, relative)
    if not path.is_file():
        display(Markdown(f"**Pending saved preview:** `{relative}`. This report does not recreate image transformations."))
        return
    preservation = _read(Path(root) / "results/metadata_nn_tuning/v2/archive_before_update/preservation_receipt.json")
    expected = preservation["protected_files"].get(path.relative_to(Path(root).resolve()).as_posix())
    if not expected or _sha(path) != expected:
        raise ValueError("Saved image preview changed or lacks its pre-update binding.")
    display(Image(filename=str(path)))


def show_preparation(root, data, identity, finish):
    show_saved_figure(root, "results/dedup_v1/clean_workflow/figures/template_v3/09_image_modelling/image_preparation_and_weak_augmentation.png")
    display(Markdown("**Saved fitting-image QA:** the original weak-transformation preview is read unchanged. Its image selection, random draws and admission decision are not rerun."))


def uncertainty_views(root, data, identity, finish):
    show_saved_figure(root, "results/dedup_v1/clean_workflow/figures/template_v3/11_image_uncertainty/uncertainty_four_view_preview.png")


def raw_outer_ranking(raw, finish):
    """Original fold scatter and two-curve composition, ROC-AUC presented first."""
    if not raw.get("complete"):
        display(Markdown("**Pending:** all five image outer folds are required."))
        return
    from .image_evaluation import curve_tables
    from .dedup_notebook_images import FOLD_COLORS
    frame, metrics = raw["predictions"], raw["fold_metrics"]
    columns = [c for c in ("fold", "n_images", "n_patients", "n_positive", "roc_auc", "average_precision", "prevalence", "log_loss", "brier") if c in metrics]
    display(metrics[columns].style.format({c: "{:.4f}" for c in ("roc_auc", "average_precision", "prevalence", "log_loss", "brier")}))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for row in metrics.itertuples():
        ax.scatter(row.fold, row.roc_auc, s=90, color=FOLD_COLORS[row.fold])
        ax.annotate(f"{row.roc_auc:.3f}", (row.fold, row.roc_auc), xytext=(0, 8), textcoords="offset points", ha="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.scatter(metrics.fold, np.full(len(metrics), .5), marker="x", color="#666666", label="Chance ROC-AUC")
    ax.axhline(metrics.roc_auc.mean(), color="#235789", linestyle="--", label="Mean fold ROC-AUC")
    ax.set(xlabel="Outer fold", ylabel="ROC-AUC", xticks=range(5), title="Outer-fold variation in image ranking", ylim=(min(.45, metrics.roc_auc.min() - .05), min(1, metrics.roc_auc.max() + .09)))
    ax.legend(loc="lower right")
    finish(fig, "outer_fold_roc_auc")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for fold, rows in frame.groupby("fold", sort=True):
        curves = curve_tables(rows.target, rows.base_probability)
        label = f"Fold {fold} ({int(rows.target.sum())} positive images)"
        line = "--" if fold % 2 else "-"
        axes[0].plot(curves["roc"].false_positive_rate, curves["roc"].true_positive_rate, color=FOLD_COLORS[fold], label=label, linewidth=1.6, linestyle=line)
        axes[1].plot(curves["precision_recall"].recall, curves["precision_recall"].precision, color=FOLD_COLORS[fold], linewidth=1.6, linestyle=line)
    pooled = curve_tables(frame.target, frame.base_probability)
    axes[0].plot(pooled["roc"].false_positive_rate, pooled["roc"].true_positive_rate, color="#111111", linewidth=2.5, linestyle=":", label="Pooled OOF (secondary)")
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle=":", label="Chance ranking")
    axes[1].plot(pooled["precision_recall"].recall, pooled["precision_recall"].precision, color="#111111", linewidth=2.5, linestyle=":")
    axes[1].axhline(frame.target.mean(), color="#999999", linestyle=":")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Raw OOF ROC by outer fold", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Raw OOF precision–recall (secondary)", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=8, loc="lower right")
    finish(fig, "outer_fold_roc_pr")


def primary_roc_summary(raw):
    if not raw.get("complete"):
        return
    folds, frame = raw["fold_metrics"], raw["predictions"]
    ci = raw["intervals"].loc[lambda t: t.model.eq("image") & t.aggregation.eq("mean_fold") & t.metric.eq("roc_auc")]
    interval = f" (conditional 95% patient interval {ci.iloc[0].ci_lower:.4f}–{ci.iloc[0].ci_upper:.4f})" if len(ci) else ""
    display(Markdown(f"**Primary mean outer-fold ROC-AUC: {folds.roc_auc.mean():.4f}**{interval}. Secondary mean-fold AP is **{folds.average_precision.mean():.4f}**. {len(frame):,} images from {frame.patient_id.nunique():,} patients, including {int(frame.target.sum())} malignant images. The saved image settings and epochs were selected using historical AP rules; no image reselection occurred."))


def image_takeaways(raw, comparison):
    if not raw.get("complete"):
        display(Markdown("**Pending:** verified five-fold image predictions are required."))
        return
    folds = raw["fold_metrics"]
    display(Markdown(f"**Saved image procedure:** primary mean outer-fold ROC-AUC **{folds.roc_auc.mean():.4f}**; secondary mean-fold AP **{folds.average_precision.mean():.4f}**. ROC-AUC ranges from **{folds.roc_auc.min():.3f} to {folds.roc_auc.max():.3f}** across five patient-held-out folds. The historical recipes and duration were selected using AP, and are unchanged. This presentation migration is not a new ROC-selected image training experiment or independent final test."))
    if not comparison.empty:
        selected = comparison.loc[comparison.model.isin(["b0_raw", "spline_raw", "metadata_nn_mean"])]
        columns = ["model"] + [c for c in ("roc_auc", "average_precision", "log_loss", "brier") if c in selected]
        display(selected[columns])


def pilot_endpoint_metrics(bundle, finish):
    if not bundle["complete"]:
        return
    metrics = bundle["metrics"]
    display(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for ax, kind, title in zip(axes, ("endpoint", "best_in_cap"), ("Predeclared fixed endpoint", "Historical AP-selected best epoch within cap")):
        table = metrics.loc[metrics.selection_kind.eq(kind)].groupby(["family", "cap"])[["roc_auc", "average_precision", "log_loss", "brier"]].mean().reset_index()
        display(table.assign(selection_kind=kind))
        ax.barh(np.arange(len(table)), table.roc_auc, color="#276b91")
        ax.set(yticks=np.arange(len(table)), yticklabels=[f"{r.family}: cap {r.cap}" for r in table.itertuples()], xlabel="Unweighted mean inner-fold ROC-AUC", title=title)
    finish(fig, "pilot_fixed_endpoints_vs_selected_epochs_roc_auc")


def pilot_interval_support(bundle, finish):
    if not bundle["complete"]:
        return
    paired = bundle["paired"]
    display(paired)
    display(bundle["predictions"].groupby("inner_fold").agg(images=("image_id","size"),patients=("patient_id","nunique"),malignant_images=("target","sum")))
    fig,axes=plt.subplots(1,2,figsize=(12,max(3.5,.32*len(paired))),layout="constrained")
    for ax,metric in zip(axes,("roc_auc","average_precision")):
        rows=paired.loc[paired.metric.eq(metric)].reset_index(drop=True)
        for index,row in enumerate(rows.itertuples()):
            ax.hlines(index,row.ci_lower,row.ci_upper,color="#276b91")
            ax.scatter(row.difference,index,color="#276b91")
        ax.axvline(0,color="#777777",linestyle=":")
        title="Exploratory ROC-AUC" if metric=="roc_auc" else "Secondary exploratory AP"
        ax.set(yticks=np.arange(len(rows)),yticklabels=rows.comparison,xlabel="Candidate minus control",title=title)
        ax.invert_yaxis()
    finish(fig,"pilot_paired_patient_ranking_intervals")


def uncertainty_quality(bundle, data, finish):
    if not bundle["complete"]:
        return
    from .metadata_evaluation import probability_metrics
    from .dedup_notebook_images import plot_probability_views
    summary=bundle["predictions"]
    folds=pd.read_csv(bundle["directory"] / "fold_metrics.csv")
    means=pd.read_csv(bundle["directory"] / "mean_fold_metrics.csv")
    order=lambda t:[c for c in t if c not in ("roc_auc","average_precision","log_loss","brier")]+[c for c in ("roc_auc","average_precision","log_loss","brier") if c in t]
    display(folds[order(folds)])
    baseline=[]
    for fold,rows in summary.groupby("fold"):
        prevalence=data.loc[data.fold.ne(fold),"target"].mean()
        baseline.append({"fold":fold,"mode":"constant fitting prevalence",**probability_metrics(rows.target,np.repeat(prevalence,len(rows)))})
    table=pd.concat([means,pd.DataFrame(baseline).groupby("mode")[["roc_auc","average_precision","log_loss","brier"]].mean().reset_index()],ignore_index=True)
    display(table[order(table)])
    display(Markdown("Mean-fold ROC-AUC is primary; AP and descriptive raw-score reliability are secondary. These saved passes do not change the historical AP-selected checkpoints."))
    plot_probability_views(summary,{"Deterministic":"base_score","MC mean":"mc_mean_score","TTA mean":"tta_mean_score"},finish,"uncertainty_mean_scores","Clean held-out score averaging · descriptive raw outputs")


def uncertainty_takeaways(bundle):
    if not bundle["complete"]:
        return
    table=pd.read_csv(bundle["directory"] / "mean_fold_metrics.csv")
    leading=[c for c in table if c not in ("roc_auc","average_precision","log_loss","brier")]
    display(table[leading+[c for c in ("roc_auc","average_precision","log_loss","brier") if c in table]])
    differences=pd.read_csv(bundle["directory"] / "paired_patient_bootstrap_differences.csv")
    selected=differences.loc[differences.referral_fraction.eq(.2)&differences.comparator.eq("prediction_score")]
    crossing=selected.ci_lower.le(0)&selected.ci_upper.ge(0)
    display(Markdown(f"**Saved clean scope:** {len(bundle['predictions']):,} outer-held-out images from {bundle['predictions'].patient_id.nunique()} patients. At predeclared 20% referral, {int(crossing.sum())} of {len(selected)} SD-minus-score intervals include zero. Mean-fold ROC-AUC is the ranking primary; referral intervals concern the separate review-order question. No clinical referral rule, new image selection or reserved-test evaluation is established."))


def plot_resource_comparison(history, metric_rows):
    """Keep the original three-panel resource composition and blue bars."""
    identity = "pipeline_id" if "pipeline_id" in history else "recipe_id"
    resources = history.groupby(identity).agg(
        measured_training_minutes=("epoch_seconds", lambda v: v.sum() / 60),
        peak_allocated_GiB=("peak_memory_bytes", lambda v: v.max() / 2**30)).reset_index().rename(columns={identity:"pipeline_id"})
    table = metric_rows.merge(resources,on="pipeline_id",validate="one_to_one")
    fig, axes = plt.subplots(1,3,figsize=(14,max(4.5,.35*len(table))),constrained_layout=True)
    for ax,column,title in zip(axes,["roc_auc","measured_training_minutes","peak_allocated_GiB"],
        ["ROC-AUC at historical AP-selected checkpoint","Measured epoch time (minutes)","Peak allocated GPU memory (GiB)"]):
        ax.barh(table.pipeline_id.tolist(),table[column],color="#387ca6")
        ax.invert_yaxis();ax.set(title=title,xlabel=column.replace("_"," "))
        ax.grid(axis="x",alpha=.2);ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Matched six-epoch trials: recorded time excludes weight downloads",fontsize=13)
    return fig


def load_saved_original_uncertainty(root):
    """Verify the explicitly approved completed run, without loading a network."""
    root = Path(root).resolve()
    directory = _path(root, f"results/image_uncertainty_runs/{UNCERTAINTY_RUN}")
    provenance = _read(directory / "analysis_provenance.json")
    if provenance.get("reserved_test_loaded") is not False or provenance.get("outer_fold_predictions_generated") is not False:
        raise ValueError("Original uncertainty report requires exploratory development-only evidence.")
    for name, digest in provenance["output_sha256"].items():
        path = _path(directory, name)
        if _sha(path) != digest:
            raise ValueError(f"Saved uncertainty output changed: {name}")
    checkpoint = _read(directory / "checkpoint_and_split_provenance.json")
    for field, digest_field in (("checkpoint", "checkpoint_sha256"), ("split_memberships", "split_memberships_sha256"), ("saved_predictions", "saved_predictions_sha256")):
        if _sha(_path(root, checkpoint[field])) != checkpoint[digest_field]:
            raise ValueError(f"Saved uncertainty checkpoint/split binding changed: {field}")
    for relative, digest in provenance["source_sha256"].items():
        from .image_evaluation import preserved_artifact_path
        # The existing rename registry explicitly binds the older image-source
        # snapshots. It preserves the recorded receipt and accepts exact bytes.
        canonical = relative.replace("\\", "/")
        try:
            preserved_artifact_path(root, canonical, digest)
        except ValueError:
            from .metadata_roc_archive import verified_source_path
            verified_source_path(root, canonical, digest, parent_receipt=directory / "analysis_provenance.json")
    summary = pd.read_csv(directory / "prediction_summary.csv", float_precision="round_trip")
    expected = set(checkpoint["evaluation_image_ids"])
    if not summary.image_id.is_unique or set(summary.image_id) != expected or set(summary.patient_id) != set(checkpoint["evaluation_patient_ids"]):
        raise ValueError("Saved uncertainty predictions do not match their declared inner patients.")
    if set(summary.patient_id) & set(checkpoint["fitting_patient_ids"]) or set(summary.patient_id) & set(checkpoint["outer_excluded_patient_ids"]):
        raise ValueError("Uncertainty report patient roles overlap.")
    names = {"quality": "prediction_quality.csv", "convergence": "mc_convergence.csv", "reliability": "descriptive_reliability.csv", "referral": "referral_rankings.csv", "differences": "paired_patient_bootstrap_differences.csv"}
    return {"directory": directory, "provenance": provenance, "checkpoint": checkpoint, "summary": summary,
            **{name: pd.read_csv(directory / file, float_precision="round_trip") for name, file in names.items()}}


def __getattr__(name):
    """Reuse existing report-only numerical/plot helpers with the same styles."""
    from . import dedup_notebook_images
    return getattr(dedup_notebook_images, name)
