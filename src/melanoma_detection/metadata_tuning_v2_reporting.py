"""ROC-AUC-first metadata reports; never fit models or read reserved-test data.

Version 1 remains unchanged. Its plotting primitives are reused where colour,
typography and composition have the same meaning; version-2 selection and inputs
are verified explicitly before results can be displayed.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from . import metadata_nn_tuning_reporting as _v1

BASE_SEEDS = _v1.BASE_SEEDS
BLUE, GOLD, ORANGE, CHARCOAL = _v1.BLUE, _v1.GOLD, _v1.ORANGE, _v1.CHARCOAL
FAMILIES = _v1.FAMILIES
NN_COMPARISON_COLORS = _v1.NN_COMPARISON_COLORS
DEFAULT_RUN_ID = "roc_auc_grid36_v1"
EXPERIMENT_VERSION = 2
ARCHIVE = Path("results/metadata_nn_tuning/v2/archive_before_update")
METRIC_ORDER = ("roc_auc", "average_precision", "log_loss", "brier_score")
NAMES = {
    "original": {"04": "04_dummy_logistic_baseline.ipynb", "05": "05_metadata_models.ipynb",
                 "06": "06_metadata_neural_network.ipynb", "07": "07_metadata_tuning.ipynb",
                 "08": "08_metadata_robustness.ipynb"},
    "dedup_v1": {"04": "04_metadata_baselines.ipynb", "05": "05_metadata_models.ipynb",
                 "06": "06_metadata_neural_network.ipynb", "07": "07_metadata_tuning.ipynb",
                 "08": "08_metadata_robustness.ipynb"},
}


def _root(root=None):
    return _v1._root(root)


def _condition(condition):
    return _v1._condition(condition)


def original_snapshot(root, relative):
    """Resolve only a hash-verified snapshot made before the ROC migration."""
    root = _root(root)
    receipt_path = root / ARCHIVE / "snapshot_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("The pre-migration archive must be ready before building reports")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        raise RuntimeError("The pre-migration archive is incomplete")
    relative = Path(relative).as_posix()
    digest = receipt.get("notebook_snapshots", {}).get(relative)
    snapshot = (root / ARCHIVE / relative).resolve()
    if digest is None or not snapshot.is_relative_to(root / ARCHIVE):
        raise ValueError("Notebook is not bound to the pre-migration archive")
    if not snapshot.is_file() or _v1._hash(snapshot) != digest:
        raise ValueError("The pre-migration notebook snapshot changed")
    return snapshot


def _snapshot_notebook(root, condition, stage):
    import nbformat
    folder = "Subset" if _condition(condition) == "original" else "Subset_dedup"
    relative = f"notebooks/{folder}/{NAMES[_condition(condition)][stage]}"
    return nbformat.read(original_snapshot(root, relative), as_version=4), relative


def load_tuning(root, condition, run_id=DEFAULT_RUN_ID):
    """Read only the explicitly selected, complete ROC-AUC experiment."""
    from .metadata_tuning_v2 import load_tuning_outputs
    root, condition = _root(root), _condition(condition)
    bundle = load_tuning_outputs(root, condition, experiment_version=2,
                                 run_id=run_id, expected_selection_metric="roc_auc")
    report = {"ready": bundle.get("status") == "complete", "status": bundle.get("status", "pending"),
              "condition": condition, "root": root, "output_dir": Path(bundle["directory"]),
              "reason": "A complete, verified version-2 ROC-AUC receipt is required.",
              "run_id": run_id, "experiment_version": 2}
    if not report["ready"]:
        return report
    receipt = bundle["receipt"]
    if receipt.get("experiment_version") != 2 or receipt.get("selection_metric") != "roc_auc":
        raise ValueError("An AP-selected or unversioned experiment cannot supply this report")
    if receipt.get("test_set_loaded") is not False or receipt.get("final_model_fitted") is not False:
        raise ValueError("Only development-only, no-final-fit evidence is accepted")
    directory = report["output_dir"]
    hashes = receipt["output_sha256"]
    frames = {"oof": bundle["oof"]}
    for name in ("candidates", "selected_parameters", "epochs", "histories", "memberships", "fit_log",
                 "classical_selected_parameters", "classical_candidates", "classical_inner_audit"):
        source = directory / f"{name}.csv"
        relative = source.relative_to(root).as_posix()
        if relative not in hashes or _v1._hash(source) != hashes[relative]:
            raise ValueError(f"Report table is not covered by the completed receipt: {name}")
        try:
            frames[name] = pd.read_csv(source, float_precision="round_trip")
        except pd.errors.EmptyDataError:
            frames[name] = pd.DataFrame()
    return {**report, **frames, "receipt": receipt, "receipt_path": directory / "receipt.json"}


def metric_table(oof, columns=None):
    table = _v1.metric_table(oof, columns)
    prefix = [name for name in table if name not in METRIC_ORDER and not name.startswith("fold_")]
    return table[[*prefix, *METRIC_ORDER, "fold_auc_sd", "fold_ap_sd"]]


def plot_nn_comparison(oof, scope="mean_fold"):
    columns = ["p_spline", *[f"p_nn_seed_{seed}" for seed in BASE_SEEDS], "p_nn_mean"]
    table = metric_table(oof, columns).query("scope == @scope").set_index("model").loc[columns]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, metric, title in zip(axes, ("roc_auc", "average_precision"), ("ROC-AUC", "Average precision")):
        bars = ax.bar(["Spline", "NN 42", "NN 142", "NN 242", "NN mean"], table[metric],
                      color=NN_COMPARISON_COLORS, edgecolor=CHARCOAL, linewidth=.6)
        for bar, value in zip(bars, table[metric]):
            ax.annotate(f"{value:.4f}", (bar.get_x()+bar.get_width()/2, value),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
        if metric == "average_precision":
            prevalence = oof.groupby("fold").target.mean().mean() if scope == "mean_fold" else oof.target.mean()
            ax.axhline(prevalence, color="#606060", linestyle="--", linewidth=1,
                       label="Mean-fold malignant share" if scope == "mean_fold" else "Malignant-image share")
            ax.legend(frameon=False)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_ylim(0, max(.01, float(table[metric].max())*1.22))
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.suptitle("Equal-fold scores: ROC-AUC primary, AP secondary" if scope == "mean_fold"
                 else "Pooled development scores (supporting)")
    fig.tight_layout()
    return fig


def plot_tuning_bars(oof, metric="roc_auc", scope="mean_fold"):
    fig = _v1.plot_tuning_bars(oof, metric, scope)
    role = ("primary metric" if metric == "roc_auc" else "secondary metric") if scope == "mean_fold" else "supporting summary"
    aggregation = "Mean outer-fold" if scope == "mean_fold" else "Pooled"
    fig.axes[0].set_title(f"{aggregation} {'ROC-AUC' if metric == 'roc_auc' else 'AP'} — {role}")
    return fig


def plot_tuning_curves(oof):
    fig = _v1.plot_tuning_curves(oof)
    fig.suptitle("Pooled outer-held-out curves: supporting ROC / precision–recall", y=1.002)
    return fig


def plot_candidate_grid(candidates):
    required = "mean_inner_roc_auc"
    if required not in candidates:
        raise ValueError("ROC-AUC candidate scores are required")
    shown = candidates.drop(columns=["mean_inner_average_precision"], errors="ignore").rename(
        columns={required: "mean_inner_average_precision"})
    fig = _v1.plot_candidate_grid(shown)
    fig.axes[-1].set_ylabel("Mean inner-fold ensemble ROC-AUC")
    fig.suptitle("36 NN recipes in each outer-training selection: gold outline marks the ROC-AUC winner", y=.988)
    return fig


def plot_selection_margins(selected):
    fig = _v1.plot_selection_margins(selected)
    fig.axes[0].set_ylabel("Selected minus runner-up mean inner ROC-AUC")
    return fig


def plot_fold_gains(oof, metric="roc_auc"):
    changes = _v1.fold_changes(oof)
    column = "roc_auc_change" if metric == "roc_auc" else "average_precision_change"
    label = "ROC-AUC" if metric == "roc_auc" else "AP"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    limit = max(.01, float(changes[column].abs().max())*1.2)
    for ax, (family, _, _) in zip(axes.flat, FAMILIES):
        rows = changes[changes.model.eq(family)].sort_values("outer_fold")
        ax.plot(rows.outer_fold, rows[column], marker="s", color=GOLD)
        ax.axhline(0, color="#64748B", linestyle="--", linewidth=1)
        ax.set(title=family, xlabel="Outer fold", ylabel=f"Tuned minus fixed {label}", ylim=(-limit, limit))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"{label} changes within the same held-out patient folds")
    fig.tight_layout()
    return fig


def plot_seed_stability(oof):
    fig = _v1.plot_seed_stability(oof)
    ax = fig.axes[0]
    ax.clear()
    for seed, marker in zip(BASE_SEEDS, ("o", "s", "^")):
        rows = oof.groupby("fold").apply(lambda g: roc_auc_score(g.target, g[f"p_nn_seed_{seed}"]), include_groups=False)
        ax.scatter(rows.index, rows, marker=marker, color=ORANGE, label=f"NN {seed}")
    for column, label, colour in [("p_spline", "Fixed spline", BLUE), ("p_nn_mean", "Equal NN mean", CHARCOAL)]:
        rows = oof.groupby("fold").apply(lambda g: roc_auc_score(g.target, g[column]), include_groups=False)
        ax.plot(rows.index, rows, marker="D", color=colour, label=label)
    ax.set(xlabel="Outer fold", ylabel="ROC-AUC", title="Fixed NN seed and stopping-group variation on the same held-out patients", ylim=(0, 1))
    ax.legend(ncol=3, frameon=False); ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    return fig


def _display_table(frame, limit=36, precision=4):
    roc=[c for c in frame if "roc_auc" in str(c)]
    ap=[c for c in frame if "average_precision" in str(c)]
    error=[c for c in frame if ("log_loss" in str(c) or "brier" in str(c))]
    if roc or ap:
        spread=[c for c in ("fold_auc_sd","fold_ap_sd") if c in frame]
        others=[c for c in frame if c not in roc+ap+error+spread]
        frame=frame[others+roc+ap+error+spread]
    _v1._display_table(frame, limit=limit, precision=precision)


def _show_figure(report, figure, name):
    _v1._show_figure(report, figure, name)


def _summary(report, stage):
    table = metric_table(report["oof"])
    mean = table.query("scope == 'mean_fold'").set_index("model")
    pooled = table.query("scope == 'pooled'").set_index("model")
    if stage == "nn_summary":
        a, b = mean.loc["p_spline"], mean.loc["p_nn_mean"]
        return (f"**Primary mean-fold ROC-AUC:** fixed spline {a.roc_auc:.4f}; fixed NN mean {b.roc_auc:.4f}; "
                f"NN minus spline {b.roc_auc-a.roc_auc:+.4f}. **Secondary mean-fold AP:** spline {a.average_precision:.4f}; "
                f"NN mean {b.average_precision:.4f}. Pooled ROC-AUC is {pooled.loc['p_spline','roc_auc']:.4f} versus "
                f"{pooled.loc['p_nn_mean','roc_auc']:.4f} and remains supporting evidence. Report every prescribed seed; "
                "three runs do not supply a sampling confidence interval. Stage 08 reports the paired comparison. "
                "The changed reporting priority does not itself replace a final candidate or supply a calibrated decision rule.")
    pieces = []
    changes = _v1.fold_changes(report["oof"])
    for family, fixed, tuned in FAMILIES:
        rows = changes[changes.model.eq(family)].roc_auc_change
        count = int(((rows>0) & ~np.isclose(rows, 0)).sum())
        a, b = mean.loc[fixed], mean.loc[tuned]
        pieces.append(f"**{family}:** primary mean-fold ROC-AUC {a.roc_auc:.4f} → {b.roc_auc:.4f} "
                      f"({b.roc_auc-a.roc_auc:+.4f}); {count}/5 folds improve. Secondary mean-fold AP "
                      f"{a.average_precision:.4f} → {b.average_precision:.4f}.")
    return "\n\n".join(pieces) + ("\n\nThis is the separately declared ROC-AUC-selected version-2 experiment. "
        "Version-1 AP selections and outputs remain historical. Paired intervals condition on saved predictions and "
        "cannot establish dependable superiority from small point gains. No final procedure is replaced automatically; "
        "all-development fitting, calibration and cutoff selection for a new final model, and reserved-test evaluation remain separate.")


def render_section(report, section):
    from IPython.display import Markdown, display
    if not report["ready"]:
        display(Markdown(f"**Result pending.** {report['reason']} Expected source: `{report['output_dir']}`. No training starts here."))
        return
    oof = report["oof"]
    if section in {"inputs", "nn_procedure", "nn_runs", "nn_epochs", "run_settings", "tuning_memberships"}:
        _v1.render_section(report, section)
        if section=="run_settings":
            _display_table(pd.DataFrame([{"field":name,"value":str(report["receipt"].get(name))}
                for name in ("experiment_version","selection_metric","protocol","parent_receipt_sha256")]))
    elif section == "nn_metrics":
        columns = ["p_spline", *[f"p_nn_seed_{seed}" for seed in BASE_SEEDS], "p_nn_mean"]
        table = metric_table(oof, columns)
        _display_table(table.set_index(["scope", "model"]), precision=5)
        for scope in ("mean_fold", "pooled"):
            _show_figure(report, plot_nn_comparison(oof, scope), f"nn_fixed_{scope}_roc_ap")
        seeds = table[(table.scope == "mean_fold") & table.model.str.startswith("p_nn_seed_")]
        _display_table(seeds[["roc_auc", "average_precision"]].agg(["mean", "std", "min", "max"]), precision=5)
        _show_figure(report, plot_seed_stability(oof), "nn_fixed_seed_fold_roc_stability")
        display(Markdown("The same patients and prescribed seeds are compared; run spread is not a patient-sampling interval."))
    elif section == "tuning_grid":
        _v1.render_section(report, section)
        display(Markdown("**Version-2 selection:** mean inner-fold ROC-AUC selects every family’s settings. AP is a secondary diagnostic; the historical AP-selected version is preserved."))
    elif section == "tuning_candidates":
        _show_figure(report, plot_candidate_grid(report["candidates"]), "nn_all_36_roc_candidate_grids")
        _display_table(report["candidates"].sort_values(["outer_fold", "recipe_order"]), precision=6)
    elif section == "tuning_choices":
        classical = report["classical_selected_parameters"]
        _display_table(classical, precision=6)
        if not classical.empty:
            _display_table(classical.groupby(["model", "parameters"]).size().rename("outer_training_selections").reset_index())
        selected = report["selected_parameters"]
        _display_table(selected, precision=6)
        _display_table(selected.groupby(["recipe_id", "hidden_units", "learning_rate", "dropout"]).size()
                       .rename("outer_training_selections").reset_index())
        _show_figure(report, _v1.plot_selection_frequency(selected), "nn_selected_setting_frequency")
        _show_figure(report, plot_selection_margins(selected), "nn_roc_selection_margins")
    elif section == "tuning_metrics":
        _display_table(metric_table(oof))
        change = _v1.fold_changes(oof)
        first = ["model", "outer_fold", "roc_auc_change"]
        _display_table(change[first+[c for c in change if c not in first]])
        for metric in ("roc_auc", "average_precision"):
            _show_figure(report, plot_fold_gains(oof, metric), f"four_family_fold_{metric}_gains")
    elif section == "tuning_bars":
        for scope in ("mean_fold", "pooled"):
            for metric in ("roc_auc", "average_precision"):
                _show_figure(report, plot_tuning_bars(oof, metric, scope), f"four_family_{metric}_{scope}")
    elif section == "tuning_curves":
        _show_figure(report, plot_tuning_curves(oof), "four_family_pooled_roc_pr")
    elif section == "tuning_changes":
        for scope in ("mean_fold", "pooled"):
            scores = metric_table(oof).query("scope == @scope").set_index("model")
            rows = []
            gains = _v1.fold_changes(oof)
            for family, fixed, tuned in FAMILIES:
                values = gains[gains.model.eq(family)].roc_auc_change
                tied = np.isclose(values, 0)
                rows.append({"model": family, "scope": scope,
                             "fixed_roc_auc": scores.loc[fixed,"roc_auc"], "tuned_roc_auc": scores.loc[tuned,"roc_auc"],
                             "roc_auc_change": scores.loc[tuned,"roc_auc"]-scores.loc[fixed,"roc_auc"],
                             "fixed_average_precision": scores.loc[fixed,"average_precision"],
                             "tuned_average_precision": scores.loc[tuned,"average_precision"],
                             "average_precision_change": scores.loc[tuned,"average_precision"]-scores.loc[fixed,"average_precision"],
                             "roc_folds_improved": int(((values>0)&~tied).sum()), "roc_folds_tied": int(tied.sum()),
                             "roc_folds_worse": int(((values<0)&~tied).sum())})
            _display_table(pd.DataFrame(rows))
    elif section == "tuning_intervals":
        from .metadata_tuning_v2_evaluation import show_assessment
        show_assessment(report["root"], report["condition"], run_id=report["run_id"], experiment_version=2)
    elif section == "tuning_duration":
        epochs=report["epochs"]
        _display_table(epochs[epochs.stage.eq("nn_outer_tuned")])
        _show_figure(report,_v1.plot_duration(epochs,"nn_outer_tuned"),"nn_tuned_selected_epochs")
        log=report["fit_log"].copy()
        if not log.empty:
            if "stage" in log:
                log["stage"]=log.stage.fillna("classical")
            grouping=[c for c in ("stage","phase","model","provenance") if c in log]
            counters=[c for c in ("fits","executed_fits","external_reused_fits","reused_fits",
                       "resumed_fits","executed_this_invocation","rescoring_operations") if c in log]
            _display_table(log.groupby(grouping,dropna=False)[counters].sum().reset_index())
        display(Markdown("Reusing a verified fit and rescoring its predictions are separate from a newly executed fit. "
                         "The receipt and phase log distinguish external reuse, compatible resume and work in this invocation."))
    elif section in {"nn_summary", "tuning_summary"}:
        display(Markdown(_summary(report, section)))
    else:
        raise ValueError(f"Unknown metadata report section: {section}")


def load_baseline_report(root, condition, stage, run_id=DEFAULT_RUN_ID):
    from .metadata_roc_baselines import load_baselines
    from .development_dataset import load_development_dataset
    root, condition = _root(root), _condition(condition)
    stage = str(stage).zfill(2)
    baseline = load_baselines(root, condition)
    directory = root / ("results/metadata_nn_tuning/v2/original" if condition == "original"
                        else "results/dedup_v1/metadata_nn_tuning/v2") / run_id
    report = {"root": root, "condition": condition, "stage": stage, "run_id": run_id,
              "output_dir": directory, "ready": baseline.get("status") == "complete",
              "reason": "Verified baseline diagnostics are pending.", "status": baseline.get("status", "pending")}
    if not report["ready"]:
        return report
    manifest = "data/subset/processed/dedup_v1/dataset_manifest.json" if condition == "dedup_v1" else None
    frame, identity = load_development_dataset(root, manifest)
    keys = ["image_name", "patient_id", "fold", "target"]
    oof = baseline["oof"].copy()
    pd.testing.assert_frame_equal(oof[keys].sort_values("image_name").reset_index(drop=True),
                                  frame[keys].sort_values("image_name").reset_index(drop=True), check_dtype=False)
    if stage == "05":
        tuned = load_tuning(root, condition, run_id)
        if not tuned["ready"]:
            return {**report, "ready": False, "status": "pending", "reason": tuned["reason"]}
        fixed = tuned["oof"][keys+["p_spline", "p_random_forest", "p_xgboost"]]
        oof = oof.merge(fixed, on=keys, validate="one_to_one")
    return {**report, "oof": oof, "frame": frame, "baseline_receipt": baseline["receipt"],
            "baseline_directory": baseline["directory"], "dataset_identity": identity, "status": "complete"}


def _baseline_columns(stage):
    rows = [("dummy", "Prevalence dummy", "p_dummy", "#94A3B8"),
            ("logistic", "Logistic regression" if stage=="04" else "Standard logistic regression", "p_categorical_logistic", BLUE)]
    if stage == "05":
        rows += [("spline_logistic", "Spline logistic regression", "p_spline", "#E07A5F"),
                 ("random_forest", "Random forest", "p_random_forest", "#3D9970"),
                 ("xgboost", "XGBoost", "p_xgboost", "#9467BD")]
    return rows


def plot_baseline_curves(report):
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    oof, stage = report["oof"], report["stage"]
    models = _baseline_columns(stage)
    if stage == "04":
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        for _, label, column, colour in models:
            RocCurveDisplay.from_predictions(oof.target, oof[column], name=label,
                                             curve_kwargs={"color": colour}, ax=axes[0])
            PrecisionRecallDisplay.from_predictions(oof.target, oof[column], name=label,
                                                     curve_kwargs={"color": colour}, ax=axes[1])
        axes[0].plot([0,1],[0,1], linestyle="--", color="#475569", label="Chance (AUC = 0.50)")
        axes[0].set_title("Out-of-fold ROC curves (supporting pooled)")
        axes[1].axhline(oof.target.mean(), linestyle="--", color="#475569", label=f"Positive prevalence ({oof.target.mean():.2%})")
        axes[1].set_title("Out-of-fold precision-recall curves (supporting pooled)")
        for ax in axes:
            ax.legend(frameon=False); ax.grid(alpha=.25)
        fig.tight_layout()
    else:
        # Use the archived plotting code's literal model colours and complete
        # four-row ROC/PR composition; only inputs and export path change.
        nb, _ = _snapshot_notebook(report["root"], report["condition"], "05")
        plotting_cell = nb.cells[20]
        source = plotting_cell.source
        if plotting_cell.cell_type != "code" or "PrecisionRecallDisplay" not in source or "plt.subplots" not in source:
            raise ValueError("The verified stage-05 curve template has changed composition")
        source = re.sub(r"^.*fig\.savefig\(.*$", "", source, flags=re.M)
        source = "\n".join(line for line in source.splitlines() if "CURVES_PATH" not in line)
        source = source.replace("plt.show()", "")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform"}:
                raise ValueError("Historical plotting source contains a fitting call")
        aliases = {name: column for name, _, column, _ in models}
        prediction = oof[["image_name", "patient_id", "fold", "target"]].copy()
        for name, column in aliases.items():
            prediction[name+"_probability"] = oof[column]
        table = metric_table(oof, list(aliases.values())).query("scope == 'pooled'").set_index("model")
        table = table.rename(index={column: label for _,label,column,_ in models})
        table["brier_score"] = table.brier_score
        namespace = {"plt": plt, "np": np, "pd": pd, "RocCurveDisplay": RocCurveDisplay,
                     "PrecisionRecallDisplay": PrecisionRecallDisplay, "train": report["frame"],
                     "oof_predictions": prediction, "estimators": dict(aliases), "overall_metrics": table,
                     "MODEL_LABELS": {name: label for name,label,_,_ in models},
                     "finish_figure": lambda *args: None}
        exec(compile(source, "archived_metadata_curves_plot_only", "exec"), namespace)
        fig = namespace["fig"]
    return fig


def display_historical_cell(root, condition, stage, number):
    """Display saved output, never execute the archived cell's source."""
    from IPython.display import display, Image, HTML, Markdown
    notebook, relative = _snapshot_notebook(root, condition, stage)
    cell = notebook.cells[number-1]
    display(Markdown(f"**Historical supporting diagnostic:** saved output of pre-migration `{relative}`, cell {number}. "
                     "Its AP-era interpretation and fitted age illustration are preserved; no model is fitted to recreate it."))
    shown = False
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if "image/png" in data:
            encoded = data["image/png"]
            encoded = "".join(encoded) if isinstance(encoded,list) else encoded
            display(Image(data=base64.b64decode(encoded))); shown = True
        elif "text/html" in data:
            display(HTML("".join(data["text/html"]))); shown = True
        elif output.get("output_type") == "stream":
            print("".join(output.get("text", []))); shown = True
        elif "text/plain" in data:
            print("".join(data["text/plain"])); shown = True
    if not shown:
        display(Markdown("No saved output was recorded for this historical cell; it is not reconstructed by fitting."))


def display_historical_source(root,condition,stage,number):
    from IPython.display import display,Markdown,Code
    notebook,_=_snapshot_notebook(root,condition,stage)
    display(Markdown("**Historical method reference:** this source explains the original fitting procedure. "
                     "It is displayed as text and is not executed by this report."))
    display(Code(notebook.cells[number-1].source,language="python"))


def render_baseline_section(report, section):
    from IPython.display import display, Markdown
    if not report["ready"]:
        display(Markdown(f"**Result pending.** {report['reason']} No models are fitted by this report.")); return
    oof = report["oof"]
    models = _baseline_columns(report["stage"])
    columns = [column for _,_,column,_ in models]
    labels = {column:label for _,label,column,_ in models}
    if section == "inputs":
        _display_table(oof.groupby("fold").agg(images=("target","size"), patients=("patient_id","nunique"), malignant_images=("target","sum")))
        display(Markdown("Image, patient, fold and target identities match the frozen development records. Baseline probabilities have verified diagnostic provenance."))
    elif section == "metrics":
        table = metric_table(oof,columns)
        table["model"] = table.model.map(labels)
        _display_table(table.set_index(["scope","model"]), precision=5)
    elif section == "fold_metrics":
        rows=[]
        for fold,g in oof.groupby("fold"):
            for column in columns:
                rows.append({"fold":fold,"model":labels[column],**_v1._scores(g,column)})
        table=pd.DataFrame(rows)
        _display_table(table[["fold","model",*METRIC_ORDER]].set_index(["fold","model"]), precision=5)
    elif section == "dummy_check":
        rows=[]
        for fold,g in oof.groupby("fold"):
            if g.p_dummy.nunique()!=1:
                raise ValueError("The prevalence dummy must be constant within every outer fold")
            rows.append({"fold":fold,"dummy_roc_auc":roc_auc_score(g.target,g.p_dummy),
                         "training_prevalence_score":g.p_dummy.iloc[0],"held_out_malignant_share":g.target.mean(),
                         "secondary_dummy_ap":average_precision_score(g.target,g.p_dummy)})
        _display_table(pd.DataFrame(rows),precision=5)
    elif section == "curves":
        _show_figure(report,plot_baseline_curves(report),"baseline_pooled_roc_pr")
    elif section == "summary":
        table=metric_table(oof,columns).query("scope == 'mean_fold'").set_index("model")
        values="; ".join(f"{labels[c]} {table.loc[c,'roc_auc']:.4f}" for c in columns)
        display(Markdown(f"**Primary mean-fold ROC-AUC:** {values}. AP, log loss and Brier remain secondary; pooled curves support interpretation. "
                         "These are patient-held-out development comparisons. Historical ablations, cutoffs and age illustrations are separately labelled supporting diagnostics. "
                         "A point ordering does not automatically choose a final model; no all-development fit or reserved-test evaluation is performed."))
    else:
        raise ValueError(f"Unknown baseline section: {section}")


def _style_and_dpi(notebook):
    declarations=[]; dpi=[]
    for cell in notebook.cells:
        if cell.cell_type!="code": continue
        source="\n".join(line for line in cell.source.splitlines() if not line.lstrip().startswith(("%","!")))
        try: tree=ast.parse(source)
        except SyntaxError: continue
        for node in tree.body:
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=="plt.rcParams.update":
                try: value=ast.literal_eval(node.value.args[0])
                except (ValueError,TypeError,IndexError): continue
                if isinstance(value,dict): declarations.append(f"plt.rcParams.update({value!r})")
            if isinstance(node,ast.Assign) and len(node.targets)==1 and ast.unparse(node.targets[0])=="ORIGINAL_EXPORT_DPI":
                try: dpi.append(int(ast.literal_eval(node.value)))
                except (TypeError,ValueError): pass
            if isinstance(node,ast.Assign) and len(node.targets)==1 and ast.unparse(node.targets[0]).startswith("plt.rcParams["):
                try: ast.literal_eval(node.value)
                except (TypeError,ValueError): continue
                declarations.append(ast.unparse(node))
        dpi += [int(value) for value in re.findall(r"\bdpi\s*=\s*(\d+)",source)]
    return "\n".join(dict.fromkeys(declarations)), (dpi[0] if dpi else 150)


def _setup(root,condition,stage,run_id,notebook):
    style,dpi=_style_and_dpi(notebook)
    loader=(f"load_tuning(PROJECT_ROOT, {condition!r}, run_id={run_id!r})" if stage in {"06","07"}
            else f"load_baseline_report(PROJECT_ROOT, {condition!r}, {stage!r}, run_id={run_id!r})")
    return f'''from pathlib import Path
import sys
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "src/melanoma_detection").is_dir() and (p / "pyproject.toml").is_file())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from melanoma_detection.metadata_tuning_v2_reporting import (
    load_tuning, load_baseline_report, render_section, render_baseline_section,
    display_historical_cell, display_historical_source, original_snapshot,
)
plt.rcdefaults()
{style}
ORIGINAL_EXPORT_DPI = {dpi}
NN_COMPARISON_COLORS = ['#2563EB', '#D97706', '#D97706', '#D97706', '#303030']
report = {loader}
report["export_dpi"] = ORIGINAL_EXPORT_DPI
display(Markdown("**Version-2 saved-results report:** " + report["status"] +
    ". ROC-AUC is primary; AP and probability errors are secondary. No training, final fitting or reserved-test loading starts here."))
'''


def _roc_markdown(source,condition,stage):
    """Adapt report claims selectively; retain AP teaching and BCE stopping."""
    source=source.replace("original Subset data","deduplicated Subset data" if condition=="dedup_v1" else "original Subset data")
    if condition=="dedup_v1":
        source=source.replace("`data/subset/processed/train_folds.csv`","the frozen `dedup_v1` development manifest")
        source=source.replace("128 malignant images","127 malignant images")
    if stage in {"04","05"}:
        # These numerical examples belong to saved AP-era diagnostics. Changing
        # their metric names would turn valid AP values into false ROC claims.
        source=source.replace("The saved run favours", "The historical AP-era saved run favours")
        source=source.replace("The clean workflow's primary ranking measure is **unweighted mean outer-fold AP**.",
            "The historical fixed-model comparison used **unweighted mean outer-fold AP**. Current tables lead with **unweighted mean outer-fold ROC-AUC**.")
        source=source.replace("primary mean-fold AP", "historical mean-fold AP")
        source=source.replace("Primary mean-fold AP", "Historical mean-fold AP")
        source=source.replace("Average precision (AP), the main measure", "Average precision (AP), the historical selection measure")
        source=source.replace("Use AP first", "Use primary mean-fold ROC-AUC first, then secondary AP")
        source=source.replace("Use the added mean-fold AP table first", "Use the ROC-AUC-first mean-fold table first")
        source=source.replace("AP and ROC-AUC", "ROC-AUC and AP")
        if stage=="04":
            source=source.replace("### 2. Generate held-out probabilities", "### 2. Load and verify held-out probabilities")
        else:
            source=source.replace("### 1. Generate patient-safe out-of-fold predictions", "### 1. Load patient-safe out-of-fold predictions")
        if source.startswith("## Summary"):
            heading, _, body=source.partition("\n")
            source=heading+"\n\n**Historical fixed-model interpretation:** the AP-era numerical examples and retained-model recommendation below describe the preserved comparison. The current ROC-AUC-first recap follows separately.\n"+body
        return source
    replacements={
        "Primary mean-fold AP":"Primary mean-fold ROC-AUC", "primary mean-fold AP":"primary mean-fold ROC-AUC",
        "mean-fold AP as primary":"mean-fold ROC-AUC as primary", "mean-fold AP is primary":"mean-fold ROC-AUC is primary",
        "Mean-fold AP is primary":"Mean-fold ROC-AUC is primary", "mean-fold AP primary":"mean-fold ROC-AUC primary",
        "Primary mean fold AP":"Primary mean fold ROC-AUC", "primary mean fold AP":"primary mean fold ROC-AUC",
        "mean outer-fold AP as primary":"mean outer-fold ROC-AUC as primary",
        "mean inner AP":"mean inner ROC-AUC", "mean inner-fold AP":"mean inner-fold ROC-AUC",
        "AP ties":"ROC-AUC ties", "AP changes within each fold":"ROC-AUC changes within each fold",
        "AP improvements":"ROC-AUC improvements", "mean-fold AP primary":"mean-fold ROC-AUC primary",
        "AP/AUC":"ROC-AUC/AP", "AP and ROC-AUC":"ROC-AUC and AP",
        "A separate secondary chart pools":"A separate supporting chart pools",
        "equal-fold primary and pooled secondary scores":"equal-fold scores with primary ROC-AUC and secondary AP, plus supporting pooled scores",
        "Its AP examples":"Its historical AP examples",
        "Average precision (AP), the primary measure":"Average precision (AP), a secondary measure",
        "mean outer-fold AP is the primary selection-procedure comparison; pooled scores and curves are secondary descriptive views":
            "mean outer-fold ROC-AUC is the primary selection-procedure comparison; AP is secondary and pooled scores and curves are supporting descriptive views",
        "The inner best AP":"The inner best ROC-AUC",
        "Read the primary score, pooled secondary score and fold changes together":
            "Read primary ROC-AUC, secondary AP, supporting pooled scores and fold changes together",
        "Primary mean-fold ROC-AUC and secondary pooled scores":"Primary mean-fold ROC-AUC, secondary AP and supporting pooled scores",
        "secondary pooled results remain available":"secondary AP and supporting pooled results remain available",
    }
    for old,new in replacements.items(): source=source.replace(old,new)
    source=re.sub(r"Average precision \(AP\) is primary because.*?ROC-AUC measures how often a malignant image scores above a benign image\.",
                  "ROC-AUC is the primary ranking measure in this version: it measures how often a malignant image scores above a benign image. "
                  "Average precision (AP) remains a secondary measure of how strongly rare malignant images concentrate near the top of the ranking.",
                  source,flags=re.S)
    source=re.sub(r"The vertical axis is AP on the left\s*and ROC-AUC on the right", "The vertical axis is ROC-AUC on the left and AP on the right",source)
    source=source.replace("Average precision (AP) is primary because it rewards ranking rare malignant images near the top. ROC-AUC measures how often a malignant image scores above a benign one; 0.5 is chance.",
        "ROC-AUC is primary in this version: it measures how often a malignant image scores above a benign one; 0.5 is chance. Average precision (AP) is secondary and rewards ranking rare malignant images near the top.")
    source=source.replace("A higher ROC-AUC does not by itself overturn the comparison on AP, the chosen primary measure.",
        "A higher primary ROC-AUC can coexist with lower secondary AP. Read both measures and the paired uncertainty before interpreting the trade-off.")
    source=source.replace("mean-fold AP as primary; secondary pooled results", "mean-fold ROC-AUC as primary; secondary AP and supporting pooled results")
    return source


def metadata_nn_notebook_sections(condition,stage,root=None,run_id=DEFAULT_RUN_ID):
    """Keep every prior current 06/07 cell and its teaching depth as a template."""
    root,condition=_root(root),_condition(condition)
    stage=str(stage).zfill(2)
    if stage not in {"06","07"}: raise ValueError("NN template stage must be 06 or 07")
    # Both conditions share the complete original prior-current report template.
    notebook,_=_snapshot_notebook(root,"original",stage)
    setup=_setup(root,condition,stage,run_id,notebook)
    sections=[]; used_setup=False
    for cell in notebook.cells:
        if cell.cell_type=="markdown":
            text=_roc_markdown(cell.source,condition,stage)
            if not sections:
                text+=("\n\n**Experiment-scope change: version 2.** The completed AP-selected version 1 is preserved. "
                       "This separately declared experiment selects settings by mean inner-fold ROC-AUC and reports mean outer-fold ROC-AUC as primary. "
                       "AP, log loss and Brier are secondary; pooled summaries are supporting. The prior report’s heading structure, "
                       "model colours, fonts and explanatory detail are retained. No all-development model or reserved-test result is produced.")
            sections.append(("__markdown__",text,""))
        else:
            if not used_setup:
                code=setup; used_setup=True
            else:
                code=cell.source
                if "render_section" not in code:
                    code='display(Markdown("The separate runner owns model fitting. This report reads completed version-2 outputs only."))'
            sections.append(("__code__","",code))
    return sections


def baseline_notebook_sections(condition,stage,root=None,run_id=DEFAULT_RUN_ID):
    root,condition=_root(root),_condition(condition)
    stage=str(stage).zfill(2)
    if stage not in {"04","05"}: raise ValueError("Baseline stage must be 04 or 05")
    notebook,relative=_snapshot_notebook(root,condition,stage)
    sections=[]; first_code=True
    dispatch=({6:"inputs",8:"metrics",9:"dummy_check",11:"curves"} if stage=="04" else
              {11:"inputs",13:"inputs",15:"metrics",18:"fold_metrics",21:"curves"})
    for number,cell in enumerate(notebook.cells,1):
        if cell.cell_type=="markdown":
            text=_roc_markdown(cell.source,condition,stage)
            if not sections:
                text+=("\n\n**Current reporting priority:** primary unweighted mean outer-fold ROC-AUC, followed by secondary AP, log loss and Brier. "
                       "Pooled curves support the comparison. Verified versioned baseline diagnostics supply the dummy/logistic predictions; "
                       "stage 05 uses matching fixed-family version-2 predictions. Original AP-era numerical examples, ablations, age illustrations "
                       "and exploratory cutoffs below remain clearly historical supporting evidence. No all-development fit is recreated.")
            if number not in {1,2} and any(word in text for word in ("ablation","marginal","cutoff","confusion","coefficient","importance")):
                text+="\n\n**Historical supporting section:** its saved pre-migration diagnostic is retained below; it does not select the current final procedure."
            sections.append(("__markdown__",text,""))
        else:
            if first_code:
                code=_setup(root,condition,stage,run_id,notebook); first_code=False
            elif number in dispatch:
                code=f"render_baseline_section(report, {dispatch[number]!r})"
            elif stage=="05" and condition=="original" and number in {7,9}:
                code=f"display_historical_source(PROJECT_ROOT, {condition!r}, {stage!r}, {number})"
            else:
                code=f"display_historical_cell(PROJECT_ROOT, {condition!r}, {stage!r}, {number})"
            sections.append(("__code__","",code))
    sections += [("__markdown__","### Current ROC-AUC-first recap\n\nRead the mean-fold ROC-AUC comparison first. "
                  "Keep rare-positive AP, probability error, patient support and the historical diagnostics alongside it. "
                  "The comparison supplies development evidence rather than a new final model or test result.",""),
                 ("__code__","","render_baseline_section(report, 'summary')")]
    return sections


def metadata_nn_assessment_sections(condition,run_id=DEFAULT_RUN_ID):
    from .metadata_tuning_v2_evaluation import assessment_sections
    return assessment_sections(_condition(condition),run_id=run_id)


def _bind_v2_table_writer(source):
    """Keep historical read helpers while preventing a late legacy writer bind."""
    lines=source.splitlines(keepends=True)
    parsed="\n".join("" if line.lstrip().startswith(("%","!")) else line.rstrip("\r\n") for line in lines)
    offsets=[0]
    for line in lines: offsets.append(offsets[-1]+len(line))
    edits=[]
    for node in ast.walk(ast.parse(parsed)):
        if not isinstance(node,ast.ImportFrom) or node.module!="melanoma_detection.dedup_notebook_metadata":continue
        writer=[alias for alias in node.names if alias.name=="save_metadata_table"]
        if not writer:continue
        other=[alias for alias in node.names if alias.name!="save_metadata_table"]
        replacements=[]
        if other:
            replacements.append(ast.unparse(ast.ImportFrom(module=node.module,names=other,level=node.level)))
        replacements.append(ast.unparse(ast.ImportFrom(module="melanoma_detection.metadata_roc_historical",names=writer,level=0)))
        indentation=" "*node.col_offset
        replacement=("\n"+indentation).join(replacements)
        edits.append((offsets[node.lineno-1]+node.col_offset,offsets[node.end_lineno-1]+node.end_col_offset,replacement))
    for start,end,replacement in sorted(edits,reverse=True):
        source=source[:start]+replacement+source[end:]
    return source


def robustness_notebook_sections(condition,root=None,run_id=DEFAULT_RUN_ID):
    root,condition=_root(root),_condition(condition)
    notebook,_=_snapshot_notebook(root,condition,"08")
    sections=[]
    for number,cell in enumerate(notebook.cells,1):
        if cell.cell_type=="markdown":
            if "## Additional experiment: tuned metadata neural network" in cell.source:
                break
            text=cell.source
            if not sections:
                text+=("\n\n**Version-2 reporting and evidence boundary:** the current added assessment uses the separately declared "
                       "ROC-AUC-selected experiment, with mean outer-fold ROC-AUC primary and AP, log loss and Brier secondary. "
                       "The fixed-model calibration, cutoff, subgroup and AP-led comparisons retained below are historical supporting "
                       "analyses from the pre-migration report. Their maps and thresholds do not evaluate the new tuned network. "
                       "The new paired ROC-AUC and secondary AP appendix follows those preserved sections; neither branch fits a final model or opens the reserved test.")
            text=text.replace("Average precision (AP) is the primary ranking measure:",
                              "Average precision (AP) was the primary ranking measure for these historical fixed-model diagnostics:")
            text=text.replace("AP is the primary ranking measure:","AP was the primary ranking measure for this historical fixed-model analysis:")
            text=text.replace("The clean workflow's primary ranking measure is **unweighted mean outer-fold AP**.",
                              "The preserved clean fixed-model analysis used **unweighted mean outer-fold AP** as its historical primary measure.")
            text=text.replace("primary mean-fold AP", "historical primary mean-fold AP")
            text=text.replace("displays the primary mean-fold AP", "displays the historical primary mean-fold AP")
            if text.startswith("## Summary"):
                text+="\n\n**Historical fixed-model summary:** read the separately declared version-2 assessment below for the current ROC-AUC-first model comparison."
            text=text.replace("### 7. Record the final-fit boundary", "### 8. Record the final-fit boundary")
            text=text.replace("### 3. Select calibration and cutoffs using training patients", "### 3. Verify calibration and cutoffs selected using training patients")
            text=text.replace("### 8. Prepare the retained model for later final fitting and testing", "### 8. Record the final-fit boundary")
            sections.append(("__markdown__",text,""))
        else:
            source=cell.source
            # All historical reads remain bound to their verified old sources;
            # presentation writes move to the new version-2 report directory.
            source=source.replace("from melanoma_detection.metadata_nn_tuning import condition_directory, DEFAULT_RUN_ID",
                                  "from melanoma_detection.metadata_tuning_v2 import condition_directory, DEFAULT_RUN_ID")
            source=source.replace("results/dedup_v1/clean_workflow", "results/dedup_v1/metadata_nn_tuning/v2/roc_auc_grid36_v1/report/historical")
            if number==4 and condition=="dedup_v1":
                source+=("\nfrom melanoma_detection.metadata_roc_historical import (load_clean_scores, latest_evaluation, "
                         "load_matched_metadata, save_metadata_table)\n")
            if "show_assessment(" in source:
                continue
            source=_bind_v2_table_writer(source)
            # No archived fitting definitions can silently re-enter execution.
            cleaned="\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%","!")))
            for node in ast.walk(ast.parse(cleaned)):
                if isinstance(node,ast.Call):
                    name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ""
                    if name in {"fit","fit_transform","fit_final_bundle","evaluate_outer_fold","run_tuning","run_inference"}:
                        raise ValueError(f"Historical robustness cell {number} contains a prohibited fitting/inference call")
            sections.append(("__code__","",source))
    assessment=metadata_nn_assessment_sections(condition,run_id)
    kind,text,code=assessment[0]
    # Robustness has finished its Summary: the additional experiment is a new
    # peer section, matching the archived level-2 explanatory hierarchy.
    assessment[0]=(kind,text.replace("### Additional ROC-AUC-selected metadata experiment",
                                   "## Additional ROC-AUC-selected metadata experiment",1),code)
    sections+=assessment
    return sections


def metadata_notebook_sections(condition,stage,root=None,run_id=DEFAULT_RUN_ID):
    stage=str(stage).zfill(2)
    if stage in {"04","05"}: return baseline_notebook_sections(condition,stage,root,run_id)
    if stage in {"06","07"}: return metadata_nn_notebook_sections(condition,stage,root,run_id)
    if stage=="08": return robustness_notebook_sections(condition,root,run_id)
    raise ValueError("Metadata stage must be 04 through 08")


def build_notebook(root,condition,stage,run_id=DEFAULT_RUN_ID):
    import nbformat
    root,condition=_root(root),_condition(condition)
    stage=str(stage).zfill(2)
    sections=metadata_notebook_sections(condition,stage,root,run_id)
    notebook=nbformat.v4.new_notebook()
    notebook.cells=[nbformat.v4.new_markdown_cell(text) if kind=="__markdown__" else nbformat.v4.new_code_cell(code)
                    for kind,text,code in sections]
    template,relative=_snapshot_notebook(root,condition,stage)
    count=lambda cells: sum(len(re.findall(r"\b[\w]+\b",cell.source)) for cell in cells if cell.cell_type=="markdown")
    original_words,adapted_words=count(template.cells),count(notebook.cells)
    notebook.metadata.update({"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"},"metadata_tuning_v2_report":{
            "condition":condition,"stage":stage,"experiment_version":2,"selection_metric":"roc_auc",
            "primary_metric":"mean_fold_roc_auc","secondary_metrics":["average_precision","log_loss","brier"],
            "pooled_role":"supporting","training_on_execution":False,"reserved_test_loaded":False,
            "run_id":run_id,"template_archive":(ARCHIVE/relative).as_posix(),
            "template_sha256":_v1._hash(original_snapshot(root,relative)),
            "original_markdown_words":original_words,"adapted_markdown_words":adapted_words}})
    if condition=="dedup_v1":
        notebook.metadata["deduplicated_workflow"]={"dataset_manifest":"data/subset/processed/dedup_v1/dataset_manifest.json",
            "training_on_execute":False,"depth_revision":4,"original_method_sources":[NAMES["original"][stage]],
            "original_markdown_words":original_words,"adapted_markdown_words":adapted_words,
            "experiment_version":2,"primary_metric":"mean_fold_roc_auc"}
    for cell in notebook.cells:
        if cell.cell_type=="code":
            source="\n".join(line for line in cell.source.splitlines() if not line.lstrip().startswith(("%","!")))
            compile(source,"metadata_roc_report_cell","exec")
    nbformat.validate(notebook)
    return notebook
