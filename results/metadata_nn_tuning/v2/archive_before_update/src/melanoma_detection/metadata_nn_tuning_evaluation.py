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
    from .metadata_nn_tuning import load_tuning_outputs
    bundle = load_tuning_outputs(root, condition, run_id=run_id)
    if bundle.get("status") != "complete":
        raise FileNotFoundError("NN tuning has not completed all five outer folds.")
    directory, predictions = bundle["directory"], bundle["oof"]
    output = directory / "report" / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    identity = {"oof_sha256": _sha(directory / "oof.csv"), "bootstrap_repetitions": repetitions,
                "seed": 42, "evaluation_source_sha256": _sha(__file__)}
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
               "interpretation": "Additional development experiment; conditional patient bootstrap of saved OOF predictions, without retraining. No final-model replacement or reserved-test evaluation."}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return {**bundle, "assessment_directory": output, "assessment": receipt}


def assessment_sections(condition):
    """Identical, inference-only appendix cells for both robustness workflows."""
    return [
        ("__markdown__", "## Additional experiment: tuned metadata neural network\n\n"
         "Stage 07 now searches 36 predefined combinations of network width, learning rate and dropout. "
         "The three-seed mean is selected using inner patient groups, then evaluated on the excluded outer patients. "
         "This extension was specified after earlier development results had been examined, so it remains development evidence. "
         "It does not automatically replace the spline.\n\n"
         "Read unweighted mean outer-fold AP first. Pooled AP and ROC-AUC remain secondary. "
         "Paired intervals resample whole patients within each outer fold while holding the fitted models fixed. "
         "They describe uncertainty conditional on these saved predictions and do not include the full variability of retraining "
         "or adaptively choosing another search. Log loss, Brier score and reliability assess raw probability quality; "
         "completed fixed-spline calibration and training-selected cutoffs do not transfer to the tuned NN.", ""),
        ("__code__", "", f'''from pathlib import Path
import sys
PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from melanoma_detection.metadata_nn_tuning_evaluation import show_assessment
show_assessment(PROJECT_ROOT, {condition!r})
'''),
    ]


def show_assessment(root, condition):
    from IPython.display import display, Markdown
    try:
        bundle = build_assessment(root, condition)
    except FileNotFoundError:
        display(Markdown("**Pending:** all five NN tuning folds must finish before this appendix reports findings."))
        return
    output = bundle["assessment_directory"]
    metrics = pd.read_csv(output / "metrics.csv")
    selected = metrics.loc[metrics.scope.eq("mean_fold") & metrics.model.isin(["spline", "spline_tuned", "nn_mean", "nn_tuned"])]
    display(selected[["model", "average_precision", "roc_auc", "log_loss", "brier"]].round(5))
    intervals = pd.read_csv(output / "paired_comparisons.csv")
    intervals = intervals.loc[intervals.aggregation.eq("mean_fold")]
    display(intervals[["comparison", "metric", "difference", "ci_lower", "ci_upper"]].round(5))
    ap = intervals.loc[intervals.metric.eq("average_precision")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    for index, row in ap.iterrows():
        ax.plot([row.ci_lower, row.ci_upper], [index, index], color="#C58A00", linewidth=2)
        ax.scatter(row.difference, index, color="#C58A00", marker="s", zorder=3)
    ax.axvline(0, color="#64748B", linestyle=":")
    ax.set(yticks=range(len(ap)), yticklabels=ap.comparison,
           xlabel="Tuned NN minus comparator: mean outer-fold AP",
           title="Paired patient-bootstrap differences; 95% conditional intervals")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "paired_ap.png", dpi=150, bbox_inches="tight")
    fig.savefig(output / "paired_ap.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    fixed = selected.loc[selected.model.eq("spline"), "average_precision"].iloc[0]
    tuned = selected.loc[selected.model.eq("nn_tuned"), "average_precision"].iloc[0]
    contrast = ap.loc[ap.comparison.eq("nn_tuned minus fixed_spline")].iloc[0]
    display(Markdown(f"**Observed primary comparison:** fixed spline AP {fixed:.4f}; tuned NN AP {tuned:.4f}. "
        f"The difference is {contrast.difference:+.4f}, with a 95% conditional patient interval "
        f"[{contrast.ci_lower:+.4f}, {contrast.ci_upper:+.4f}]. "
        "These are development results. The selected final procedure, its calibration and its decision threshold "
        "remain separate decisions before reserved-test evaluation."))
