"""Verified, inference-only reports for the original and clean NN searches.

Training lives in metadata_nn_tuning.  This module deliberately does not import
that runner: opening a report can never start an experiment.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, precision_recall_curve, roc_auc_score,
                             roc_curve)


BASE_SEEDS = (42, 142, 242)
BLUE, GOLD, ORANGE, CHARCOAL = "#2563EB", "#C58A00", "#D97706", "#303030"
NN_COMPARISON_COLORS = [BLUE, ORANGE, ORANGE, ORANGE, CHARCOAL]
FAMILIES = (
    ("Spline logistic", "p_spline", "p_spline_tuned"),
    ("Random forest", "p_random_forest", "p_random_forest_tuned"),
    ("XGBoost", "p_xgboost", "p_xgboost_tuned"),
    ("Metadata NN", "p_nn_mean", "p_nn_tuned"),
)
ARCHIVE = Path("results/metadata_nn_tuning/v1/archive_before_update/notebooks/Subset")
NAMES = {"06": "06_metadata_neural_network.ipynb", "07": "07_metadata_tuning.ipynb"}


def _root(root=None):
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]


def _condition(value):
    if value == "deduplicated":
        return "dedup_v1"
    if value not in ("original", "dedup_v1"):
        raise ValueError("condition must be original or dedup_v1")
    return value


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _contained(root, path):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Report input escapes the project: {path}")
    return path


def _run_directory(root, base, run_id):
    if not isinstance(run_id,str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}",run_id) is None:
        raise ValueError("A simple versioned run identity is required")
    base = _contained(root,base)
    return _contained(base,base/run_id)


def load_tuning(root, condition, run_id=None):
    """Load only complete, hash-verified development outputs; otherwise pending.

    Receipt output hashes use repository-relative paths.  No partial CSV is
    interpreted as a completed result, and reserved-test inputs are never read.
    """
    root = _root(root)
    condition = _condition(condition)
    base = root / ("results/metadata_nn_tuning/v1/original" if condition == "original"
                   else "results/dedup_v1/metadata_nn_tuning/v1")
    if run_id is not None:
        directory = _run_directory(root, base, run_id)
        receipts = [directory / "receipt.json"]
    else:
        pointer = base / "latest.json"
        selected_run = (json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
                        if pointer.exists() else "nn_grid36_v1")
        directory = _run_directory(root, base, selected_run)
        receipts = [directory / "receipt.json"]
    pending = {"ready": False, "status": "pending", "condition": condition,
               "output_dir": directory, "root": root,
               "reason": "No completed, verified tuning receipt is available."}
    if not receipts or not receipts[0].exists():
        return pending
    path = receipts[0]
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        pending["reason"] = f"Run status is {receipt.get('status', 'unrecorded')}; results remain pending."
        return pending
    if receipt.get("condition") != condition:
        raise ValueError("Tuning receipt condition does not match this report")
    if receipt.get("test_set_loaded") is not False or receipt.get("final_model_fitted") is not False:
        raise ValueError("This report requires development-only, no-final-fit evidence")
    hashes = receipt.get("output_sha256", {})
    if not hashes:
        raise ValueError("Completed tuning receipt has no output fingerprints")
    for relative, expected in hashes.items():
        source = _contained(root, root / relative)
        if not source.is_relative_to(directory):
            raise ValueError("Tuning receipt output escapes its versioned run")
        if not source.is_file() or _hash(source) != expected:
            raise ValueError(f"Tuning output fingerprint mismatch: {relative}")
    directory = path.parent
    if receipt.get("run_id") != directory.name or _contained(root, root / receipt["output_root"]) != directory:
        raise ValueError("Receipt does not identify the requested run directory")
    artifacts = {}
    for name in ("oof", "candidates", "selected_parameters", "epochs", "histories",
                 "memberships", "fit_log", "classical_selected_parameters", "classical_candidates", "classical_inner_audit"):
        source = directory / f"{name}.csv"
        if not source.exists():
            if name.startswith("classical_"):
                artifacts[name] = pd.DataFrame()
                continue
            raise ValueError(f"Completed run is missing {source.name}")
        relative = source.relative_to(root).as_posix()
        if relative not in hashes and str(source.relative_to(root)) not in hashes:
            raise ValueError(f"Report artifact is not covered by the receipt: {source.name}")
        try:
            artifacts[name] = pd.read_csv(source,float_precision="round_trip")
        except pd.errors.EmptyDataError:
            artifacts[name] = pd.DataFrame()
    oof = artifacts["oof"]
    required = ["image_name", "patient_id", "fold", "target"]
    required += [c for _, a, b in FAMILIES for c in (a, b)]
    required += [f"p_nn_{prefix}seed_{seed}" for prefix in ("", "tuned_") for seed in BASE_SEEDS]
    missing = set(required) - set(oof)
    if missing:
        raise ValueError(f"OOF columns missing: {sorted(missing)}")
    if oof[required].isna().any().any() or oof.image_name.duplicated().any():
        raise ValueError("OOF identities must be complete and unique")
    if set(oof.target.unique()) != {0, 1} or set(oof.fold.unique()) != set(range(5)):
        raise ValueError("OOF report requires binary labels and five preserved outer folds")
    if oof.groupby("patient_id").fold.nunique().max() != 1:
        raise ValueError("A patient appears in multiple assessment folds")
    values = oof[[c for c in required if c.startswith("p_")]].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("OOF probabilities must be finite and between zero and one")
    for _, fold in oof.groupby("fold"):
        if set(fold.target.unique()) != {0, 1}:
            raise ValueError("Every assessment fold needs both classes")
    if len(oof) != receipt.get("rows") or oof.patient_id.nunique() != receipt.get("patients"):
        raise ValueError("OOF population differs from its completion receipt")
    if int(oof.target.sum()) != receipt.get("positive_images"):
        raise ValueError("OOF malignant-image count differs from its completion receipt")
    from .development_dataset import load_development_dataset
    manifest = receipt["dataset_identity"].get("dataset_manifest") if condition == "dedup_v1" else None
    development, identity = load_development_dataset(root, manifest)
    if receipt["dataset_identity"] != identity:
        raise ValueError("Completed tuning input identity differs from current frozen development data")
    expected = development[["image_name","patient_id","fold","target"]].sort_values("image_name").reset_index(drop=True)
    actual = oof[["image_name","patient_id","fold","target"]].sort_values("image_name").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    return {"ready": True, "status": "complete", "condition": condition,
            "output_dir": directory, "root": root, "receipt": receipt,
            "receipt_path": path, **artifacts}


def _scores(frame, column):
    y, p = frame.target.to_numpy(int), frame[column].to_numpy(float)
    return {"average_precision": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p),
            "brier_score": brier_score_loss(y, p),
            "log_loss": log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])}


def metric_table(oof, columns=None):
    """Compute equal-fold primary metrics and pooled secondary metrics."""
    columns = list(columns) if columns is not None else [c for _, a, b in FAMILIES for c in (a, b)]
    rows = []
    for column in columns:
        folds = pd.DataFrame([_scores(frame, column) for _, frame in oof.groupby("fold", sort=True)])
        for scope, values in (("mean_fold", folds.mean().to_dict()), ("pooled", _scores(oof, column))):
            rows.append({"model": column, "scope": scope, **values,
                         "fold_ap_sd": folds.average_precision.std(ddof=1),
                         "fold_auc_sd": folds.roc_auc.std(ddof=1)})
    return pd.DataFrame(rows)


def fold_changes(oof):
    rows = []
    for fold, frame in oof.groupby("fold", sort=True):
        for family, fixed, tuned in FAMILIES:
            a, b = _scores(frame, fixed), _scores(frame, tuned)
            rows.append({"model": family, "outer_fold": fold,
                         "fixed_average_precision": a["average_precision"],
                         "tuned_average_precision": b["average_precision"],
                         "average_precision_change": b["average_precision"] - a["average_precision"],
                         "roc_auc_change": b["roc_auc"] - a["roc_auc"]})
    return pd.DataFrame(rows)


def tuning_change_table(oof, scope="mean_fold"):
    scores = metric_table(oof).query("scope == @scope").set_index("model")
    changes = fold_changes(oof)
    rows = []
    for family,fixed,tuned in FAMILIES:
        values = changes[changes.model.eq(family)].average_precision_change
        ties = np.isclose(values,0)
        rows.append({"model":family,"scope":scope,
                     "fixed_average_precision":scores.loc[fixed,"average_precision"],
                     "tuned_average_precision":scores.loc[tuned,"average_precision"],
                     "average_precision_change":scores.loc[tuned,"average_precision"]-scores.loc[fixed,"average_precision"],
                     "fixed_roc_auc":scores.loc[fixed,"roc_auc"],
                     "tuned_roc_auc":scores.loc[tuned,"roc_auc"],
                     "roc_auc_change":scores.loc[tuned,"roc_auc"]-scores.loc[fixed,"roc_auc"],
                     "folds_improved":int(((values>0)&~ties).sum()),
                     "folds_unchanged":int(ties.sum()),"folds_worse":int(((values<0)&~ties).sum())})
    return pd.DataFrame(rows)


def plot_nn_comparison(oof, scope="mean_fold"):
    columns = ["p_spline", *[f"p_nn_seed_{seed}" for seed in BASE_SEEDS], "p_nn_mean"]
    scores = metric_table(oof, columns).query("scope == @scope").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    labels = ["Spline", "NN 42", "NN 142", "NN 242", "NN mean"]
    for ax, metric, title in zip(axes, ("average_precision", "roc_auc"), ("Average precision", "ROC-AUC")):
        values = scores.loc[columns, metric].to_numpy()
        bars = ax.bar(labels, values, color=NN_COMPARISON_COLORS, edgecolor=CHARCOAL, linewidth=.6)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width()/2, value),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
        ax.set(title=title, ylabel=title, ylim=(0, max(values.max() * 1.22, .01)))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    prevalence = (np.mean([rows.target.mean() for _,rows in oof.groupby("fold")])
                  if scope == "mean_fold" else oof.target.mean())
    axes[0].axhline(prevalence, color="#606060", ls="--", lw=1,
                   label="Mean-fold malignant share" if scope == "mean_fold" else "Malignant-image share")
    axes[0].legend(frameon=False)
    fig.suptitle("Equal-fold development scores (primary)" if scope == "mean_fold" else "Pooled development scores (secondary)")
    fig.tight_layout()
    return fig


def plot_tuning_bars(oof, metric="average_precision", scope="mean_fold"):
    table = metric_table(oof).query("scope == @scope").set_index("model")
    labels = [name for name, _, _ in FAMILIES]
    positions = np.arange(len(FAMILIES))
    width = .34
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for state, offset, color, edge, index in (("Untuned", -width/2, BLUE, "#1E3A8A", 1),
                                            ("Nested-tuned", width/2, GOLD, "#7C5E10", 2)):
        values = [table.loc[entry[index], metric] for entry in FAMILIES]
        bars = ax.bar(positions + offset, values, width, label=state, color=color,
                      edgecolor=edge, linewidth=.8)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.4f}", (bar.get_x()+bar.get_width()/2, value),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10)
    title = "Average precision" if metric == "average_precision" else "ROC-AUC"
    maximum = table[metric].max()
    ax.set(xticks=positions, xticklabels=labels, ylabel=title,
           title=f"{title}: {'equal-fold primary' if scope == 'mean_fold' else 'pooled secondary'} comparison",
           ylim=(0, maximum*1.22))
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#CBD5E1", alpha=.65)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_tuning_curves(oof):
    fig, axes = plt.subplots(4, 2, figsize=(14, 52/3), sharex="col", sharey="col")
    y = oof.target.to_numpy(int)
    for row, (family, fixed, tuned) in enumerate(FAMILIES):
        for name, column, color, style, marker in (("Untuned", fixed, BLUE, "--", "o"),
                                                  ("Nested-tuned", tuned, GOLD, "-", "s")):
            p = oof[column].to_numpy(float)
            fpr, tpr, _ = roc_curve(y, p)
            precision, recall, _ = precision_recall_curve(y, p)
            axes[row, 0].plot(fpr, tpr, color=color, ls=style, lw=2.4,
                              marker=marker, markevery=max(1, len(fpr)//10), markersize=4,
                              label=f"{name} (AUC {roc_auc_score(y,p):.4f})")
            axes[row, 1].plot(recall, precision, color=color, ls=style, lw=2.4,
                              marker=marker, markevery=max(1, len(recall)//10), markersize=4,
                              label=f"{name} (AP {average_precision_score(y,p):.4f})")
        axes[row, 0].plot([0, 1], [0, 1], color="#64748B", ls=":", lw=1.2)
        axes[row, 1].axhline(y.mean(), color="#64748B", ls=":", lw=1.2,
                             label=f"Malignant share {y.mean():.2%}")
        axes[row, 0].set(title=f"{family}: ROC", xlabel="False positive rate", ylabel="True positive rate",
                         xlim=(0,1), ylim=(0,1))
        axes[row, 1].set(title=f"{family}: precision-recall", xlabel="Recall", ylabel="Precision",
                         xlim=(0,1), ylim=(0,.30))
        for ax in axes[row]:
            ax.legend(frameon=False, loc="best")
            ax.grid(alpha=.2)
    fig.suptitle("Pooled outer-held-out curves: secondary ranking summaries", y=1.002)
    fig.tight_layout()
    return fig


def plot_candidate_grid(candidates):
    """All 36 recipes within each selection run; common quantitative scale."""
    folds = sorted(candidates.outer_fold.unique())
    dropouts = sorted(candidates.dropout.unique())
    widths = sorted(candidates.hidden_units.unique())
    rates = sorted(candidates.learning_rate.unique())
    fig, axes = plt.subplots(len(folds), len(dropouts), figsize=(14, 4*len(folds)), squeeze=False)
    low, high = candidates.mean_inner_average_precision.min(), candidates.mean_inner_average_precision.max()
    for row, fold in enumerate(folds):
        for col, dropout in enumerate(dropouts):
            ax = axes[row, col]
            source = candidates[(candidates.outer_fold == fold) & (candidates.dropout == dropout)]
            matrix = source.pivot(index="hidden_units", columns="learning_rate", values="mean_inner_average_precision").reindex(index=widths, columns=rates)
            mark = ax.imshow(matrix, cmap="Blues", vmin=low, vmax=high, aspect="auto")
            ax.set(xticks=range(len(rates)), xticklabels=[f"{lr:g}" for lr in rates],
                   yticks=range(len(widths)), yticklabels=widths,
                   xlabel="Learning rate", ylabel="Hidden units",
                   title=f"Outer {fold}: dropout {dropout:g}")
            for i, width in enumerate(widths):
                for j, lr in enumerate(rates):
                    value = matrix.loc[width, lr]
                    color = "white" if value > (low+high)/2 else CHARCOAL
                    ax.text(j, i, f"{value:.4f}", ha="center", va="center", color=color, fontsize=10)
            chosen = source[source.selected.astype(str).str.lower().isin(["true", "1"])]
            for _, entry in chosen.iterrows():
                ax.add_patch(Rectangle((rates.index(entry.learning_rate)-.5, widths.index(entry.hidden_units)-.5),
                                       1, 1, fill=False, edgecolor=GOLD, linewidth=2.5))
    fig.subplots_adjust(left=.07, right=.89, top=.945, bottom=.035, hspace=.48, wspace=.3)
    cax = fig.add_axes([.92, .25, .017, .5])
    fig.colorbar(mark, cax=cax, label="Mean inner-fold ensemble AP")
    fig.suptitle("36 NN recipes in each outer-training selection: gold outline marks the chosen recipe",y=.988)
    return fig


def plot_selection_frequency(selected):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, column, title in zip(axes, ("hidden_units", "learning_rate", "dropout"),
                                 ("Hidden units", "Learning rate", "Dropout")):
        counts = selected[column].value_counts().sort_index()
        bars = ax.bar([f"{v:g}" for v in counts.index], counts, color=GOLD, edgecolor="#7C5E10")
        ax.bar_label(bars, padding=4, fmt="%d")
        ax.set(title=title, ylabel="Outer-training selections", ylim=(0, len(selected)+.8))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.suptitle("Selected NN settings across five development selection runs")
    fig.tight_layout()
    return fig


def plot_selection_margins(selected):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars = ax.bar(selected.outer_fold.astype(str), selected.selection_margin,
                  color=GOLD, edgecolor="#7C5E10")
    ax.bar_label(bars, fmt="%.6f", padding=5)
    ax.set(xlabel="Outer fold", ylabel="Selected minus runner-up mean inner AP",
           title="NN selection margins within training data")
    ax.set_ylim(0, max(float(selected.selection_margin.max())*1.25, .001))
    fig.tight_layout()
    return fig


def plot_fold_gains(oof):
    changes = fold_changes(oof)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    limit = max(changes.average_precision_change.abs().max()*1.2, .001)
    for ax, (family, _, _) in zip(axes.ravel(), FAMILIES):
        source = changes[changes.model.eq(family)]
        ax.axhline(0, color="#64748B", ls="--", lw=1)
        ax.plot(source.outer_fold, source.average_precision_change, color=GOLD,
                marker="s", markersize=6, linewidth=1.4)
        ax.set(title=family, xlabel="Outer fold", ylabel="Tuned minus fixed AP", ylim=(-limit,limit))
        ax.set_xticks(sorted(oof.fold.unique()))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Changes within the same held-out patient folds")
    fig.tight_layout()
    return fig


def plot_duration(epochs, stage="nn_outer_fixed"):
    source = epochs[epochs.stage.eq(stage)] if "stage" in epochs else pd.DataFrame()
    if source.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for offset, seed in enumerate(BASE_SEEDS):
        rows = source[source.base_seed.eq(seed)].sort_values("outer_fold")
        ax.scatter(rows.outer_fold + (offset-1)*.12, rows.selected_epochs,
                   color=ORANGE if stage == "nn_outer_fixed" else GOLD,
                   marker=("o", "s", "^")[offset], s=48, label=f"Base seed {seed}")
    median = source.groupby("outer_fold").selected_epochs.median()
    ax.plot(median.index, median, color=CHARCOAL, marker="D", label="Median across seeds")
    ax.set(xlabel="Outer fold", ylabel="Selected training epochs",
           title="Training-only stopping choices: fixed NN" if stage == "nn_outer_fixed" else "Training-only stopping choices: tuned NN",
           xticks=sorted(source.outer_fold.unique()), ylim=(0, max(source.selected_epochs.max()*1.15, 1)))
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    return fig


def plot_nn_histories(histories):
    if "stage" not in histories:
        return None
    source = histories[histories.stage.eq("nn_outer_fixed")]
    if source.empty:
        return None
    folds = sorted(source.outer_fold.unique())
    fig, axes = plt.subplots(len(folds), 2, figsize=(12, 3.6*len(folds)), squeeze=False)
    for row, fold in enumerate(folds):
        for col, phase in enumerate(("stopping", "refit")):
            ax = axes[row,col]
            rows = source[(source.outer_fold == fold) & source.phase.eq(phase)]
            for seed, style in zip(BASE_SEEDS, ("-", "--", ":")):
                run = rows[rows.base_seed.eq(seed)].sort_values("epoch")
                if run.empty:
                    continue
                ax.plot(run.epoch, run.fit_bce, color=ORANGE, ls=style, label=f"Fit BCE: {seed}")
                if phase == "stopping" and "stop_bce" in run:
                    ax.plot(run.epoch, run.stop_bce, color=CHARCOAL, ls=style, label=f"Stopping BCE: {seed}")
            ax.set(title=f"Outer {fold}: {phase}", xlabel="Epoch", ylabel="Binary cross-entropy")
            ax.grid(alpha=.2)
            if row == 0:
                ax.legend(frameon=False, fontsize=9, ncols=2)
    fig.suptitle("Fixed NN training histories: orange fit loss, charcoal stopping loss")
    fig.tight_layout()
    return fig


def plot_seed_stability(oof):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    folds = sorted(oof.fold.unique())
    for seed, marker in zip(BASE_SEEDS,("o","s","^")):
        values = [average_precision_score(rows.target,rows[f"p_nn_seed_{seed}"])
                  for _,rows in oof.groupby("fold",sort=True)]
        ax.scatter(folds,values,color=ORANGE,marker=marker,s=48,label=f"NN {seed}")
    for column,color,label in (("p_spline",BLUE,"Fixed spline"),("p_nn_mean",CHARCOAL,"Equal NN mean")):
        values = [average_precision_score(rows.target,rows[column]) for _,rows in oof.groupby("fold",sort=True)]
        ax.plot(folds,values,color=color,lw=1.8,marker="D",label=label)
    ax.set(xticks=folds,xlabel="Outer fold",ylabel="Average precision",
           title="Fixed NN seed and stopping-group variation on the same held-out patients")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False,ncols=3)
    ax.grid(axis="y",alpha=.2)
    fig.tight_layout()
    return fig


def plot_paired_intervals(paired):
    rows = paired[(paired.aggregation == "mean_fold") & paired.metric.eq("average_precision")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5,4.8))
    for i,row in rows.iterrows():
        ax.plot([row.ci_lower,row.ci_upper],[i,i],color=GOLD,lw=2)
        ax.scatter(row.difference,i,color="#7C5E10",marker="s",s=45,zorder=3)
    ax.axvline(0,color="#64748B",ls=":",lw=1)
    ax.set(yticks=range(len(rows)),yticklabels=rows.comparison.str.replace("_"," ",regex=False),
           xlabel="Tuned NN minus comparator: mean outer-fold AP",
           title="95% paired patient-bootstrap intervals, conditional on saved predictions")
    ax.grid(axis="x",alpha=.2)
    fig.tight_layout()
    return fig


def _assessment(report):
    directory = report["output_dir"] / "report" / "evaluation"
    source = directory / "receipt.json"
    if not source.exists():
        return None
    receipt = json.loads(source.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete" or receipt.get("identity",{}).get("oof_sha256") != _hash(report["output_dir"] / "oof.csv"):
        raise ValueError("Assessment receipt does not match this completed OOF run")
    for name,digest in receipt["output_sha256"].items():
        path = _contained(directory, directory/name)
        if not path.is_file() or _hash(path) != digest:
            raise ValueError("Assessment table fingerprint mismatch")
    return pd.read_csv(directory / "paired_comparisons.csv")


def _display_table(frame, limit=36, precision=4):
    from IPython.display import Markdown, display
    display(frame.head(limit).round(precision))
    if len(frame) > limit:
        display(Markdown(f"Showing {limit} of {len(frame)} rows; the complete table is saved with the run."))


def _show_figure(report, figure, name):
    from IPython.display import display
    if figure is None:
        return
    directory = report["output_dir"] / "report" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(directory / f"{name}.{suffix}", dpi=report.get("export_dpi",150), bbox_inches="tight")
    display(figure)
    plt.close(figure)


def _summary(report, section):
    table = metric_table(report["oof"]).set_index(["scope", "model"])
    scope = table.loc["mean_fold"]
    pooled = table.loc["pooled"]
    if section == "nn_summary":
        spline, nn = scope.loc["p_spline"], scope.loc["p_nn_mean"]
        return (f"The fixed three-seed NN average has primary mean-fold AP **{nn.average_precision:.4f}**, "
                f"versus **{spline.average_precision:.4f}** for the fixed spline (NN minus spline "
                f"**{nn.average_precision-spline.average_precision:+.4f}**). Mean-fold ROC-AUC is "
                f"{nn.roc_auc:.4f} and {spline.roc_auc:.4f}, respectively. Pooled secondary AP is "
                f"{pooled.loc['p_nn_mean','average_precision']:.4f} versus {pooled.loc['p_spline','average_precision']:.4f}. "
                "These are point estimates on reused development folds. They do not establish categorical superiority "
                "or inferiority; retain paired uncertainty and simplicity in the decision. Stage 07 evaluates the "
                "separately prescribed NN tuning procedure rather than selecting the luckiest seed.")
    sentences = []
    changes = fold_changes(report["oof"])
    for family, fixed, tuned in FAMILIES:
        difference = scope.loc[tuned,"average_precision"]-scope.loc[fixed,"average_precision"]
        counts = changes[changes.model.eq(family)].average_precision_change
        improved = ((counts > 0) & ~np.isclose(counts,0)).sum()
        sentences.append(f"**{family}:** mean-fold AP {scope.loc[fixed,'average_precision']:.4f} → "
                         f"{scope.loc[tuned,'average_precision']:.4f} ({difference:+.4f}); "
                         f"{improved}/5 folds improve. Pooled AP {pooled.loc[fixed,'average_precision']:.4f} → "
                         f"{pooled.loc[tuned,'average_precision']:.4f}.")
    return "\n\n".join(sentences) + ("\n\nThe comparison estimates each training-and-selection procedure. "
        "Selected settings can differ between folds; this does not yet specify one all-development recipe. "
        "Small gains and inner selection margins are not sampling confidence intervals. Keep the final model decision "
        "separate from reporting these development results and freeze preprocessing, recipe selection, calibration, "
        "cutoff rules and evaluation before opening the reserved test.")


def render_section(report, section):
    """Display a bounded report section from verified artifacts; never train."""
    from IPython.display import Markdown, display
    if not report["ready"]:
        display(Markdown(f"**Result pending.** {report['reason']} Expected source: `{report['output_dir']}`. "
                         "This cell reads saved results and starts no training."))
        return
    oof = report["oof"]
    if section == "inputs":
        _display_table(oof.groupby("fold").agg(images=("target","size"), patients=("patient_id","nunique"),
                                               malignant_images=("target","sum")))
        receipt = report["receipt"]
        display(Markdown(f"Verified **{len(oof):,} development images**, **{oof.patient_id.nunique()} patients**, "
                         f"**{int(oof.target.sum())} malignant images** ({oof.target.mean():.2%}); "
                         f"condition `{report['condition']}`, run `{receipt['run_id']}`. "
                         "All declared output hashes match. Reserved-test loading and final fitting are recorded false."))
    elif section == "nn_procedure":
        display(Markdown("The fixed comparison uses 32 hidden units, dropout 0.2, learning rate 0.001, weight decay 0.001 "
                         "and equal averaging of seeds 42/142/242. A fresh refit follows training-only epoch selection. "
                         "The NN tuning extension is documented separately in stage 07."))
        epochs = report["epochs"]
        if "encoded_input_width" in epochs:
            shape = epochs.loc[epochs.stage.eq("nn_outer_fixed"),
                               ["stage","encoded_input_width"]].drop_duplicates().copy()
            shape["fixed_32_unit_parameter_count"] = (shape.encoded_input_width+1)*32+33
            _display_table(shape)
    elif section == "nn_metrics":
        columns = ["p_spline", *[f"p_nn_seed_{seed}" for seed in BASE_SEEDS], "p_nn_mean"]
        table = metric_table(oof,columns)
        _display_table(table.set_index(["scope", "model"]), precision=5)
        _show_figure(report, plot_nn_comparison(oof), "nn_fixed_mean_fold_comparison")
        _show_figure(report, plot_nn_comparison(oof,"pooled"), "nn_fixed_pooled_comparison")
        individual = table[(table.scope == "mean_fold") & table.model.str.startswith("p_nn_seed_")]
        _display_table(individual[["average_precision","roc_auc"]].agg(["mean","std","min","max"]), precision=5)
        _show_figure(report, plot_seed_stability(oof), "nn_fixed_seed_fold_stability")
        display(Markdown("The three-run spread reflects seeds and stopping groups on the same patients; it is not a sampling confidence interval."))
    elif section == "nn_epochs":
        epochs = report["epochs"]
        rows = epochs[epochs.stage.eq("nn_outer_fixed")] if "stage" in epochs else epochs
        _display_table(rows)
        _show_figure(report, plot_duration(epochs), "nn_fixed_selected_epochs")
        history = plot_nn_histories(report["histories"])
        if history is None:
            display(Markdown("Fixed NN loss histories were not recorded in the reused historical run. "
                             "No training curve is reconstructed or fabricated; epoch records and prediction fingerprints remain available."))
        else:
            _show_figure(report, history, "nn_fixed_histories")
    elif section == "nn_runs":
        _display_table(pd.DataFrame([{"base_seed":seed,"held_out_predictions":len(oof),
                                     "minimum_probability":oof[f"p_nn_seed_{seed}"].min(),
                                     "maximum_probability":oof[f"p_nn_seed_{seed}"].max()}
                                    for seed in BASE_SEEDS]))
        display(Markdown("Every image has one outer-held-out prediction per prescribed seed. The spline and NN predictions use the same assessment patients."))
    elif section == "run_settings":
        receipt = report["receipt"]
        _display_table(pd.DataFrame([{"field":key,"value":str(receipt.get(key))} for key in
                                    ("condition","run_id","rows","patients","positive_images","executed_fits","reused_fits","test_set_loaded","final_model_fitted","config_sha256","compatibility_sha256")]))
        display(Markdown(f"Receipt: `{report['receipt_path'].relative_to(report['root']).as_posix()}`. "
                         "The report writes only under this versioned run's `report/` directory. Historical experiment files are preserved."))
    elif section == "tuning_grid":
        classical = report.get("classical_candidates",pd.DataFrame())
        if not classical.empty:
            settings = []
            for model,group in classical.groupby("model"):
                parameters = [json.loads(text) for text in group.parameters.unique()]
                for key in sorted({key for entry in parameters for key in entry}):
                    values = sorted({str(entry[key]) for entry in parameters},key=str)
                    settings.append({"model":model,"parameter":key.split("__")[-1],
                                     "tried_values":", ".join(values),"recipes":group.candidate_index.nunique()})
            _display_table(pd.DataFrame(settings))
        candidates = report["candidates"]
        recipe = candidates[["recipe_id","recipe_order","hidden_units","learning_rate","dropout"]].drop_duplicates().sort_values("recipe_order")
        _display_table(recipe)
        display(Markdown(f"The NN search contains **{len(recipe)} recipes** in each outer-training selection. "
                         "Each recipe is assessed through the equal three-seed ensemble on three inner patient folds. "
                         "Inner assessment labels choose settings; disjoint stopping patients choose duration. "
                         "Neither role uses that run's outer assessment patients."))
    elif section == "tuning_memberships":
        audit = report.get("classical_inner_audit",pd.DataFrame())
        if not audit.empty:
            for column in ("patient_overlap","outer_patient_overlap"):
                if column in audit and not audit[column].eq(0).all():
                    raise ValueError("Saved inner patient audit reports leakage")
            _display_table(audit)
        epochs = report["epochs"]
        if "stage" in epochs:
            source = epochs[epochs.stage.eq("nn_inner")]
            columns = [column for column in ("outer_fold","inner_fold","base_seed","epoch_fit_images","epoch_stop_images",
                "refit_images","score_images","epoch_fit_patients","epoch_stop_patients","refit_patients","score_patients") if column in source]
            if columns:
                _display_table(source[columns].drop_duplicates(),limit=45)
        display(Markdown("Inner patient membership was reconstructed from the original row order using the original seed rule, "
                         "then retained for the original run and filtered for the clean run. The historical notebook did not save a membership receipt; "
                         "the reconstruction and current membership hashes make that qualification explicit. "
                         "Fitting, stopping and assessment patients have separate roles; the outer assessment group stays outside all inner roles."))
    elif section == "tuning_candidates":
        _show_figure(report, plot_candidate_grid(report["candidates"]), "nn_all_36_candidate_grids")
        _display_table(report["candidates"].sort_values(["outer_fold","recipe_order"]), limit=36, precision=6)
    elif section == "tuning_choices":
        classical = report["classical_selected_parameters"]
        if not classical.empty:
            _display_table(classical)
            _display_table(classical.groupby(["model","parameters"]).size().rename("outer_training_selections").reset_index())
        else:
            display(Markdown("Classical selected-parameter records are not included in this NN run; historical selections remain in the archived stage 07 outputs."))
        _display_table(report["selected_parameters"], precision=6)
        _display_table(report["selected_parameters"].groupby(["recipe_id","hidden_units","learning_rate","dropout"]).size()
                       .rename("outer_training_selections").reset_index())
        _show_figure(report, plot_selection_frequency(report["selected_parameters"]), "nn_selected_setting_frequency")
        _show_figure(report, plot_selection_margins(report["selected_parameters"]), "nn_selection_margins")
    elif section == "tuning_metrics":
        _display_table(metric_table(oof))
        _display_table(fold_changes(oof))
        _show_figure(report, plot_fold_gains(oof), "four_family_fold_ap_gains")
    elif section == "tuning_bars":
        for scope in ("mean_fold","pooled"):
            for metric in ("roc_auc","average_precision"):
                _show_figure(report, plot_tuning_bars(oof,metric,scope), f"four_family_{metric}_{scope}")
    elif section == "tuning_curves":
        _show_figure(report, plot_tuning_curves(oof), "four_family_pooled_roc_pr")
    elif section == "tuning_changes":
        for scope in ("mean_fold","pooled"):
            _display_table(tuning_change_table(oof,scope))
    elif section == "tuning_duration":
        epochs = report["epochs"]
        rows = epochs[epochs.stage.eq("nn_outer_tuned")] if "stage" in epochs else epochs
        _display_table(rows)
        _show_figure(report, plot_duration(epochs,"nn_outer_tuned"), "nn_tuned_selected_epochs")
        log = report["fit_log"].copy()
        if not log.empty:
            if "stage" in log:
                missing_stage = log.stage.isna()
                log.loc[missing_stage, "stage"] = "classical"
                if "phase" in log:
                    reused_fixed = missing_stage & log.phase.eq("reused_fixed_nn")
                    log.loc[reused_fixed, "stage"] = "nn_outer_fixed_reused"
            grouping = [key for key in ("stage","phase") if key in log]
            counts = [key for key in ("fits","reused_fits") if key in log]
            if grouping and counts:
                _display_table(log.groupby(grouping)[counts].sum().reset_index())
    elif section in ("nn_summary","tuning_summary"):
        display(Markdown(_summary(report,section)))
    elif section == "tuning_intervals":
        paired = _assessment(report)
        if paired is None:
            display(Markdown("**Paired assessment pending:** completed OOF predictions are available, but the separately cached patient-bootstrap assessment has not been verified yet."))
        else:
            primary = paired[paired.aggregation.eq("mean_fold")]
            _display_table(primary, precision=6)
            _show_figure(report,plot_paired_intervals(paired),"nn_tuned_paired_mean_fold_ap")
            display(Markdown("The interval uses the same whole-patient resampling draw for both models within each preserved outer fold. "
                             "An interval spanning zero leaves both directions compatible with this conditional assessment; it does not prove equivalence."))
    else:
        raise ValueError(f"Unknown report section: {section}")


def _adapt_markdown(source, stage, cell, condition):
    clean = condition == "dedup_v1"
    source = source.replace("Subset data", "deduplicated Subset data" if clean else "original Subset data")
    if clean:
        source = source.replace("`data/subset/processed/train_folds.csv`", "the frozen deduplicated development manifest")
        source = source.replace("128 malignant images", "127 malignant images")
        source = source.replace("1.86% malignant-image rate", "observed malignant-image rate displayed in the figure")
        source = source.replace("shared Subset fold file", "frozen retained-development manifest and preserved Subset folds")
    if stage == "06":
        if cell == 1:
            source = re.sub(r"The saved three-network average.*?The network reads no image pixels\.",
                "The verified outputs below compare the fixed three-network average with the spline and report every prescribed seed. "
                "Primary mean-fold AP and secondary pooled scores are distinguished explicitly. The computed findings, rather than historical "
                "numbers in prose, determine the observed comparison. The network reads no image pixels.",source,flags=re.S)
        elif cell == 4:
            source = source.replace("before training", "in the saved experiment")
        elif cell == 9:
            source = source.replace("Train the prescribed runs", "Review the prescribed runs")
            source += "\n\nThis cell verifies and reads saved experiment outputs. Training is performed separately by the versioned runner; executing this report starts no fits."
        elif cell == 11:
            source = source.replace("Each score pools held-out predictions from all five outer folds.",
                "The primary score averages the five outer-fold scores equally. A separate secondary chart pools all held-out predictions; "
                "these aggregations can differ because cross-fold score scales and malignant-image counts differ.")
            source = source.replace("The network's higher ROC-AUC does not overturn its lower AP, the chosen primary measure.",
                "A higher ROC-AUC does not by itself overturn the comparison on AP, the chosen primary measure. Read the computed findings below.")
        elif cell == 13:
            source = source.replace("Save predictions and run settings", "Read predictions and run settings")
            source = re.sub(r"These are development outputs\..*", "These are development outputs. Reports read hash-verified versioned "
                "predictions and settings and write presentation copies only inside the new run's `report/` directory. "
                "Historical `d1` results and the previous notebook are preserved in the archive.", source,flags=re.S)
        elif cell == 15:
            source = source.replace("We test one small architecture, so this experiment cannot identify the best hidden-layer size, dropout rate or network design.",
                "This stage compares one prescribed small architecture. Stage 07 separately searches hidden-layer size, learning rate and dropout "
                "through nested patient checks; even that bounded search cannot identify the best possible network design.")
        elif cell == 16:
            source = re.sub(r"The three-network average has AP.*?Next,", "The computed summary below reports both aggregations and the "
                "observed NN-minus-spline difference. Keep all prescribed seeds and their equal average in view: choosing the best individual "
                "run after seeing its score could favour a lucky result. Paired uncertainty and the simpler model's practical advantages "
                "remain relevant alongside the ranking scores.\n\nNext,", source,flags=re.S)
    else:
        if cell == 1:
            source = source.replace("We return to the spline and tree models in [05_metadata_models.ipynb](05_metadata_models.ipynb),",
                "We return to the spline and tree models in [05_metadata_models.ipynb](05_metadata_models.ipynb) "
                "and add the metadata NN from [06_metadata_neural_network.ipynb](06_metadata_neural_network.ipynb),")
            source = re.sub(r"The saved search raises.*?The reserved test is not used\.",
                "The computed results below compare fixed and selected versions of all four model families. "
                "The NN extension adds a prescribed 36-recipe search of hidden-layer size, learning rate and dropout. "
                "Results remain development evidence; the reserved test is not used.",source,flags=re.S)
        elif cell == 2:
            source = source.replace("Split the other patients into three new inner folds", "Use three preserved inner patient folds within the other patients")
            source += "\n\n**Experiment-scope addition:** the NN now receives its own 36-recipe search, after the fixed NN comparison in stage 06. "
            source += "The selected method is the three-seed average, not the best individual seed. **Aggregation clarification:** mean outer-fold AP is "
            source += "the primary selection-procedure comparison; pooled scores and curves are secondary descriptive views."
        elif cell == 3:
            source = source.replace("Re-executing the modelling cells repeats the searches and replaces their named outputs.",
                "Re-executing this report reads completed receipts and writes only presentation copies inside the new versioned run. "
                "The separate runner performs the searches; legacy named outputs are never overwritten.")
        elif cell == 13:
            source = source.replace("The first table pools predictions across all outer folds.",
                "The first table labels equal-fold primary and pooled secondary scores separately.")
            source = source.replace("Read the pooled score and fold changes together", "Read the primary score, pooled secondary score and fold changes together")
        elif cell == 15:
            source = source.replace("Spline AP rises while ROC-AUC falls; both tree models fall on both.",
                "The direction and size of each change are calculated from the versioned predictions below; do not carry a historical direction into a new dataset.")
        elif cell == 17:
            source = source.replace("1.86% malignant-image rate", "observed malignant-image rate")
            source += "\n\nThe added fourth row is the metadata NN. Circle markers identify fixed settings and squares identify tuned settings, "
            source += "reinforcing the original blue/gold and dashed/solid encodings. Curves pool predictions and therefore complement the primary mean-fold scores."
        elif cell == 19:
            source = re.sub(r"The spline's AP improves.*?This is useful", "Fold counts and score changes are computed for every family below. This is useful", source,flags=re.S)
        elif cell == 21:
            source = source.replace("best possible spline or tree model", "best possible spline, tree or neural-network model")
            source = source.replace("We do not report a paired uncertainty interval for the tuned-minus-fixed differences in this notebook, so the small spline AP gain remains uncertain.",
                "The added NN paired intervals condition on saved predictions and omit retraining and broader adaptive-selection variability. "
                "Any small gain must be interpreted with these limits; an interval spanning zero leaves its direction unresolved.")
            source = source.replace("the small spline AP gain remains uncertain", "any small observed gain remains uncertain")
            source += "\n\nThe NN grid keeps weight decay, batch size, activation and depth fixed. It evaluates one hidden-layer family; "
            source += "wider networks, alternative optimizers, class weighting and additional engineered features remain outside scope. "
            source += "Settings-selection frequency and runner-up margins describe training sensitivity, not new-patient confidence intervals."
        elif cell == 22:
            source = re.sub(r"Spline tuning raises AP.*?Next,", "The computed summary below states the fixed-to-tuned changes and fold consistency "
                "for spline logistic, Random Forest, XGBoost and the metadata NN. Their searches include the fixed settings, but selecting the "
                "best inner score can still reduce outer performance through sampling variability. A favourable point estimate alone does not "
                "establish a dependable gain or calibrated probabilities.\n\nDo not choose an individual seed or a global recipe from the outer "
                "results after inspecting them. Any final candidate change requires the same paired uncertainty and practical trade-off review.\n\nNext,",source,flags=re.S)
            source += "\n\nFreezing the procedure means recording the complete data cleaning, feature transformation, model or search recipe, "
            source += "seed averaging, stopping, calibration and threshold rules before final testing. The final model is then trained using "
            source += "development data according to those rules, and the reserved test is evaluated once. Changing a rule after seeing its test result "
            source += "would make that test part of development rather than an independent final check. That final-fit and test stage is not performed here."
    return source


def metadata_nn_assessment_sections(condition):
    """Give every added NN assessment the same original robustness typography."""
    from .metadata_nn_tuning_evaluation import assessment_sections
    from .dedup_notebook_templates import original_plot_style
    style = original_plot_style("08_metadata_robustness.ipynb")
    sections = []
    for kind, markdown, code in assessment_sections(condition):
        if kind == "__code__":
            code = code.replace("show_assessment(PROJECT_ROOT,",
                "import matplotlib.pyplot as plt\nplt.rcdefaults()\n" + style +
                "\nshow_assessment(PROJECT_ROOT,")
        sections.append((kind, markdown, code))
    return sections


def metadata_nn_notebook_sections(condition, stage, root=None, run_id=None):
    """Full original cell structure plus clearly labelled NN-search additions."""
    root = _root(root)
    stage = str(stage).zfill(2)
    condition = _condition(condition)
    if stage not in NAMES:
        raise ValueError("Expected original/deduplicated and stage 06/07")
    source_path = root / ARCHIVE / NAMES[stage]
    original = json.loads(source_path.read_text(encoding="utf-8"))
    from .dedup_notebook_templates import original_plot_style, original_export_dpi
    style = original_plot_style(NAMES[stage])
    export_dpi = original_export_dpi(NAMES[stage])
    setup = f'''from pathlib import Path
import sys
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

PROJECT_ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
                    if (path / "src" / "melanoma_detection").exists())
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
from melanoma_detection.metadata_nn_tuning_reporting import (
    load_tuning, render_section, plot_nn_comparison, plot_tuning_bars, plot_tuning_curves)
plt.rcdefaults()
{style}
ORIGINAL_EXPORT_DPI = {export_dpi}
NN_COMPARISON_COLORS = ['#2563EB', '#D97706', '#D97706', '#D97706', '#303030']
report = load_tuning(PROJECT_ROOT, {condition!r}, run_id={run_id!r})
report["export_dpi"] = ORIGINAL_EXPORT_DPI
display(Markdown("**Saved-results report:** " + report["status"] +
                 ". Executing cells verifies and presents artifacts; training runs separately."))
'''
    sections = []
    code_sections = ({3:None,5:"inputs",7:"nn_procedure",8:None,10:"nn_runs",12:"nn_metrics",14:"run_settings"}
                     if stage == "06" else
                     {4:None,6:"inputs",8:"tuning_grid",10:"tuning_candidates",12:"tuning_choices",
                      14:"tuning_metrics",16:"tuning_bars",18:"tuning_curves",20:"tuning_changes"})
    for number, cell in enumerate(original["cells"],1):
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            source = _adapt_markdown(source,stage,number,condition)
            if number == 1:
                source += ("\n\n**Scope and preservation:** this notebook uses the complete archived original as its section "
                           "and visual template. Numerical findings are computed from the explicitly identified versioned run. "
                           "The original experiments remain archived; no reserved-test labels or final all-development fit are used here.")
            sections.append(("__markdown__",source,""))
        else:
            if number not in code_sections:
                raise ValueError(f"Unmapped original code cell {stage}:{number}")
            section = code_sections[number]
            code = setup if number == min(code_sections) else (
                f"render_section(report, {section!r})" if section else
                'display(Markdown("Training functions are kept in the separate runner. This report reads completed results only."))')
            sections.append(("__code__","",code))
        if stage == "06" and number == 12:
            sections += [("__markdown__", "### Read the recorded training duration and histories\n\nThe epoch choices are made on training-only stopping groups. "
                          "Orange marks show the individual fixed-NN runs; charcoal shows their median. Loss curves, when recorded, "
                          "show fit and stopping probability error. They describe optimization, not performance on the outer patients.", ""),
                         ("__code__","", "render_section(report, 'nn_epochs')")]
        if stage == "07" and number == 8:
            sections += [("__markdown__", "### Explain the neural-network settings being searched\n\n"
                "The hidden layer combines the encoded age, sex and site inputs with learned weights. We compare 8, 16, 32 and 64 units. "
                "A larger layer can express more interactions, but has more parameters relative to the few malignant patients. "
                "A smaller layer may miss useful patterns; increasing size is therefore a question for held-out training groups, not an automatic improvement.\n\n"
                "AdamW's learning rate controls the size of each weight update. Rates 0.0003, 0.001 and 0.003 compare slower, baseline and faster steps. "
                "Too small a rate may not reach a useful solution within the epoch cap; too large a rate may overshoot or fluctuate. "
                "Early stopping cannot fully replace this setting because it controls duration, rather than the update size.\n\n"
                "Dropout randomly masks hidden activations during training. Rates 0, 0.2 and 0.4 compare no masking, the fixed baseline and stronger masking. "
                "It can reduce reliance on individual units but may also weaken a small network. Prediction always disables dropout. "
                "The Cartesian product gives 4 × 3 × 3 = 36 recipes, including the existing 32/0.001/0.2 procedure. "
                "Weight decay remains 0.001; depth, batch size 128, ReLU, unweighted binary cross-entropy and the epoch cap remain fixed.\n\n"
                "For each candidate and inner assessment fold, separate inner-training patients choose duration through a stopping split. "
                "Preprocessing is learned only on fitting patients for that choice. A fresh preprocessor and network then refit on all inner-training "
                "patients for the selected duration and predict the disjoint inner assessment patients. The three prescribed seed predictions are averaged "
                "before AP is computed. The recipe with highest mean AP across the three inner assessment folds wins; deterministic recipe order resolves ties. "
                "A fresh selected procedure predicts outer patients once. The outer labels neither choose duration nor select settings.\n\n"
                "The stopping split is prescribed rather than searched. For inner assessment group v, the remaining groups are "
                "[(v+1)%3, (v+2)%3]. Seed indices 0 and 2 stop on the first and fit on the second; index 1 reverses these roles. "
                "For example, assessing group 0 leaves groups 1 and 2: seeds 42 and 242 stop on group 1, while seed 142 stops on group 2. "
                "The fresh inner refit uses both groups before predicting group 0. Exact recipe ties prefer the current 32/0.001/0.2 setting, "
                "then ascending size, learning rate and dropout.\n\n"
                "For the selected outer procedure, stopping fold = (outer fold + 1 + seed index) % 5. "
                "The training seed is base seed + outer fold for both duration selection and fresh refitting. "
                "These rules are fixed for all 36 candidates and both datasets; no outer score chooses a stopping group or a seed.", "")]
        if stage == "07" and number == 6:
            sections += [("__markdown__", "### Check the preserved inner patient groups\n\n"
                "Compare fitting and inner-assessment patient counts, malignant support and overlap checks. "
                "The NN adds a third role: training-only stopping patients choose epoch count inside the inner fitting pool. "
                "This prevents the labels used to assess a candidate from also choosing its training duration. "
                "The clean condition filters original memberships; it does not rebuild folds after removing copies.", ""),
                         ("__code__","", "render_section(report, 'tuning_memberships')")]
        if stage == "07" and number == 12:
            sections += [("__markdown__", "### Read selection margins and training duration\n\nThe margin is selected mean inner AP minus the runner-up's mean inner AP. "
                          "A small margin shows that settings were close in this training sample; it is not a confidence interval. "
                          "Selected epochs differ by training partition and seed. Executed and reused fit counts make the extra search effort visible.", ""),
                         ("__code__","", "render_section(report, 'tuning_duration')")]
        if stage == "07" and number == 20:
            sections += [("__markdown__", "### Interpret paired differences on the primary measure\n\n"
                "The additional assessment resamples whole patients within each preserved outer fold, using identical draws for every compared model. "
                "Two thousand draws estimate 95% intervals for tuned NN minus fixed NN, fixed spline and tuned spline. "
                "These comparisons use mean-fold AP as primary; secondary pooled results remain available in the assessment table. "
                "An interval spanning zero allows either direction under this assessment. It does not establish equivalence, nor account for "
                "all variability from retraining or expanding the search after earlier development results were examined. "
                "Read the paired comparison before treating a small score change as a reason to replace the final candidate.", ""),
                         ("__code__","", "render_section(report, 'tuning_intervals')")]
        if number == (16 if stage == "06" else 22):
            sections.append(("__code__","",f"render_section(report, {'nn_summary' if stage == '06' else 'tuning_summary'!r})"))
    return sections


def build_notebook(root, condition, stage, run_id=None):
    import nbformat
    root = _root(root)
    stage = str(stage).zfill(2)
    condition = _condition(condition)
    sections = metadata_nn_notebook_sections(condition,stage,root,run_id)
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_markdown_cell(text) if kind == "__markdown__"
                      else nbformat.v4.new_code_cell(code) for kind,text,code in sections]
    notebook.metadata.update({"kernelspec":{"display_name":"Python 3", "language":"python", "name":"python3"},
                              "language_info":{"name":"python"},
                              "metadata_nn_tuning_report":{"condition":condition,"stage":stage,
                                  "training_on_execution":False,"template_archive":(ARCHIVE/NAMES[stage]).as_posix(),
                                  "template_sha256":_hash(root/ARCHIVE/NAMES[stage]),
                                  "primary_metric":"mean_fold_average_precision","run_id":run_id}})
    if condition == "dedup_v1":
        original = nbformat.read(root/ARCHIVE/NAMES[stage],as_version=4)
        count = lambda cells: sum(len(re.findall(r"\b[\w]+\b",cell.source))
                                  for cell in cells if cell.cell_type == "markdown")
        notebook.metadata["deduplicated_workflow"] = {
            "dataset_manifest":"data/subset/processed/dedup_v1/dataset_manifest.json",
            "training_on_execute":False,"depth_revision":3,
            "original_method_sources":[NAMES[stage]],
            "presentation_template":"original_markdown_and_cell_order",
            "original_markdown_words":count(original.cells),
            "adapted_markdown_words":count(notebook.cells)}
    nbformat.validate(notebook)
    return notebook
