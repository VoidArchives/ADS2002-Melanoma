"""Read-only model assessment of completed, versioned NN tuning runs."""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import pandas as pd

from .dedup_metadata_suite import metrics_table
from .image_nested_reporting import patient_fold_bootstrap, reliability_table


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_assessment(root, condition, run_id=None, *, repetitions=2000):
    from .metadata_tuning_v2 import load_tuning_outputs
    bundle = load_tuning_outputs(root, condition, experiment_version=2, run_id=run_id or "roc_auc_grid36_v1", expected_selection_metric="roc_auc")
    if bundle.get("status") != "complete":
        raise FileNotFoundError("NN tuning has not completed all five outer folds.")
    directory, predictions = bundle["directory"], bundle["oof"]
    output = directory / "report" / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    identity = {"oof_sha256": _sha(directory / "oof.csv"), "bootstrap_repetitions": repetitions,
                "seed": 42, "experiment_version": 2, "selection_metric": "roc_auc", "primary_metric": "roc_auc", "evaluation_source_sha256": _sha(__file__)}
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity"] != identity:
            raise ValueError("Assessment inputs changed; select a new run identity.")
        for name, digest in receipt["output_sha256"].items():
            if _sha(output / name) != digest:
                raise ValueError("Assessment output changed.")
        return {**bundle, "assessment_directory": output, "assessment": receipt}
    selected_columns = ["image_name", "patient_id", "fold", "target", "p_spline", "p_spline_tuned",
                        "p_random_forest", "p_random_forest_tuned", "p_xgboost", "p_xgboost_tuned",
                        "p_nn_mean", "p_nn_tuned"]
    table = predictions[selected_columns].copy()
    metrics = metrics_table(table, table)
    metrics = metrics[["model", "scope", "fold", "roc_auc", "average_precision", "log_loss", "brier"]]
    metrics.to_csv(output / "metrics.csv", index=False)
    canonical = table.rename(columns={"image_name": "image_id"}).sort_values("image_id").reset_index(drop=True)
    absolute, paired = patient_fold_bootstrap(canonical, {
        "image": canonical.p_nn_tuned.to_numpy(), "fixed_nn": canonical.p_nn_mean.to_numpy(),
        "fixed_spline": canonical.p_spline.to_numpy(), "tuned_spline": canonical.p_spline_tuned.to_numpy(),
    }, n_bootstrap=repetitions, seed=42)
    absolute["model"] = absolute.model.replace({"image": "nn_tuned"})
    paired["comparison"] = paired.comparison.str.replace("image minus ", "nn_tuned minus ", regex=False)
    absolute.to_csv(output / "absolute_intervals.csv", index=False)
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    reliability = []
    for name, column in [("Fixed spline", "p_spline"), ("Fixed NN", "p_nn_mean"), ("Tuned NN", "p_nn_tuned")]:
        rows = reliability_table(table.target, table[column])
        rows["model"] = name
        reliability.append(rows)
    pd.concat(reliability, ignore_index=True).to_csv(output / "reliability.csv", index=False)
    names = ["metrics.csv", "absolute_intervals.csv", "paired_comparisons.csv", "reliability.csv"]
    receipt = {"status": "complete", "identity": identity,
               "output_sha256": {name: _sha(output / name) for name in names},
               "interpretation": "Additional ROC-AUC-selected development experiment after earlier results were examined; conditional patient bootstrap of saved OOF predictions, without retraining. No final-model replacement or reserved-test evaluation."}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return {**bundle, "assessment_directory": output, "assessment": receipt}


def assessment_sections(condition, run_id="roc_auc_grid36_v1"):
    """Shared report appendix, pinned to the ROC experiment rather than latest."""
    markdown = """### Additional ROC-AUC-selected metadata experiment

The new experiment selects spline, Random Forest, XGBoost and NN settings by unweighted mean inner-fold ROC-AUC, with preprocessing and duration selection confined to outer-training patients. The completed earlier experiment remains AP-selected history; it is not retrospectively described as ROC-AUC-tuned. The new three-seed network still trains with binary cross-entropy and selects its duration using stopping loss. Its probabilities are averaged before scoring.

Mean outer-fold ROC-AUC is the primary model comparison. Mean outer-fold AP, log loss and Brier score are secondary; pooled metrics are supporting summaries. These are image-level metrics within patient-held-out folds, not patient-level diagnoses. Original results retain duplicate-bearing development records; the deduplicated condition supplies the main clean-data evidence.

The paired comparisons use 2,000 whole-patient resamples within each preserved outer fold. Every model receives the same draw. The intervals condition on saved predictions and do not include repeating the complete fitting and selection process. An interval containing zero leaves the difference unresolved and does not establish equivalence. An observed leading score does not prove superiority.

This is an additional development experiment after earlier outer-fold findings were examined. The revised objective and search were frozen before execution, without expanding the grid in response to new outer scores. Historical calibration maps and recall cutoffs remain attached to their original fixed procedures; they cannot be transferred to the newly ROC-selected NN or spline. This appendix does not replace the final model, fit an all-development model or open the reserved test.
"""
    code = ("from melanoma_detection.metadata_tuning_v2_evaluation import show_assessment\n"
            "import matplotlib.pyplot as plt\n"
            "plt.rcdefaults()\n"
            "plt.rcParams.update({'figure.dpi': 120, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})\n"
            f"show_assessment({'ROOT' if condition == 'dedup_v1' else 'PROJECT_ROOT'}, {condition!r}, run_id={run_id!r}, experiment_version=2)")
    return [("__markdown__", markdown, None), ("assessment", None, code)]


def show_assessment(root, condition, *, run_id="roc_auc_grid36_v1", experiment_version=2):
    from IPython.display import display, Markdown
    if experiment_version != 2:
        raise ValueError("The ROC assessment requires experiment version 2.")
    try:
        bundle = build_assessment(root, condition, run_id=run_id)
    except FileNotFoundError:
        display(Markdown("**Pending:** all five ROC-selected folds must pass verification before findings are reported."))
        return
    output = bundle["assessment_directory"]
    metrics = pd.read_csv(output / "metrics.csv")
    selected = metrics.loc[metrics.scope.eq("mean_fold") & metrics.model.isin(["spline", "spline_tuned", "nn_mean", "nn_tuned"])].sort_values("roc_auc", ascending=False)
    display(selected[["model", "roc_auc", "average_precision", "log_loss", "brier"]].round(6))
    intervals = pd.read_csv(output / "paired_comparisons.csv")
    primary = intervals.loc[intervals.aggregation.eq("mean_fold")].copy()
    order = {"roc_auc": 0, "average_precision": 1, "log_loss": 2, "brier": 3}
    primary["metric_order"] = primary.metric.map(order)
    primary = primary.sort_values(["metric_order", "comparison"])
    display(primary[["comparison", "metric", "difference", "ci_lower", "ci_upper"]].round(6))
    labels = {"nn_tuned minus fixed_nn": "ROC-selected NN minus fixed NN",
              "nn_tuned minus fixed_spline": "ROC-selected NN minus fixed spline",
              "nn_tuned minus tuned_spline": "ROC-selected NN minus ROC-selected spline"}
    for metric, title, filename in [("roc_auc", "ROC-AUC (primary)", "paired_roc_auc"),
                                     ("average_precision", "AP (secondary)", "paired_ap")]:
        rows = primary.loc[primary.metric.eq(metric)].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10.5, 4.3))
        for index, row in rows.iterrows():
            ax.plot([row.ci_lower, row.ci_upper], [index, index], color="#C58A00", linewidth=2)
            ax.scatter(row.difference, index, color="#C58A00", marker="s", zorder=3)
        ax.axvline(0, color="#64748B", linestyle=":")
        ax.set(yticks=range(len(rows)), yticklabels=[labels.get(value, value) for value in rows.comparison],
               xlabel=f"Tuned NN minus comparator: mean outer-fold {title}",
               title="Paired patient-bootstrap differences; 95% conditional intervals")
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        fig.savefig(output / (filename + ".png"), dpi=120, bbox_inches="tight")
        fig.savefig(output / (filename + ".pdf"), bbox_inches="tight")
        plt.show()
        plt.close(fig)
    contrast = primary.loc[primary.metric.eq("roc_auc") & primary.comparison.eq("nn_tuned minus fixed_spline")].iloc[0]
    fixed = selected.loc[selected.model.eq("spline"), "roc_auc"].iloc[0]
    tuned = selected.loc[selected.model.eq("nn_tuned"), "roc_auc"].iloc[0]
    display(Markdown(f"**Observed primary comparison:** fixed spline ROC-AUC {fixed:.4f}; ROC-selected NN ROC-AUC {tuned:.4f}. "
        f"The mean-fold difference is {contrast.difference:+.4f}, with a 95% conditional patient interval "
        f"[{contrast.ci_lower:+.4f}, {contrast.ci_upper:+.4f}]. "
        "Read secondary AP and probability losses alongside the primary result. Model choice remains an explicit decision; "
        "no calibration map, cutoff or final fitted model is produced for the newly tuned procedures."))


