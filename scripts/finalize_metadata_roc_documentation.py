"""Write verified ROC development findings and combined fit counts, without fitting."""
import json
import re
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.metadata_tuning_v2 import load_tuning_outputs
from melanoma_detection.metadata_tuning_v2_evaluation import build_assessment
from melanoma_detection.metadata_roc_baselines import load_baselines
from melanoma_detection.metadata_model import file_sha256


def main():
    summaries, counts, fingerprints = [], {}, {}
    pairs = {}
    for condition in ["original", "dedup_v1"]:
        bundle = load_tuning_outputs(ROOT, condition, experiment_version=2, run_id="roc_auc_grid36_v1", expected_selection_metric="roc_auc")
        if bundle["status"] != "complete":
            raise RuntimeError("Only both completed five-fold experiments may be published")
        assessment = build_assessment(ROOT, condition, run_id="roc_auc_grid36_v1", repetitions=2000)
        metrics = pd.read_csv(assessment["assessment_directory"] / "metrics.csv")
        for _, row in metrics.loc[metrics.scope.eq("mean_fold")].iterrows():
            summaries.append({"condition": condition, **row.to_dict()})
        pairs[condition] = pd.read_csv(assessment["assessment_directory"] / "paired_comparisons.csv")
        counts[condition] = {name: bundle["receipt"][name] for name in ["executed_fits", "external_reused_fits", "resumed_fits", "executed_this_invocation", "rescoring_operations"]}
        fingerprints[condition] = file_sha256(bundle["directory"] / "receipt.json")
    baseline = load_baselines(ROOT, "original")
    baseline_count = int(baseline["receipt"]["executed_fits"])
    total = sum(c["executed_fits"] for c in counts.values()) + baseline_count
    if total > 725 or baseline_count != 5:
        raise ValueError("The approved fitting bound or baseline reconstruction differs")
    ledger = dict(status="complete", experiment_version=2, selection_metric="roc_auc", conditions=counts,
                  original_baseline_executed_fits=baseline_count, dummy_fits=0, total_executed_fits=total,
                  externally_reused_core_fits=sum(c["external_reused_fits"] for c in counts.values()),
                  rescoring_operations=sum(c["rescoring_operations"] for c in counts.values()),
                  core_completion_sha256=fingerprints, baseline_receipt_sha256=file_sha256(baseline["directory"] / "receipt.json"),
                  source_sha256=file_sha256(Path(__file__)), final_model_fitted=False, test_set_loaded=False)
    report = ROOT / "results/metadata_nn_tuning/v2/report_verification"
    report.mkdir(parents=True, exist_ok=True)
    (report / "combined_fit_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    table = pd.DataFrame(summaries)
    table.to_csv(report / "mean_fold_comparison.csv", index=False)
    text = ["## Completed ROC-selected experiment: `roc_auc_grid36_v1`", "",
            "Both conditions passed all five outer folds. These are additional development findings after earlier results were examined.", "",
            "| Dataset | Procedure | Mean-fold ROC-AUC (primary) | Mean-fold AP | Log loss | Brier |",
            "|---|---|---:|---:|---:|---:|"]
    names = {"spline": "Fixed spline", "spline_tuned": "ROC-selected spline", "nn_mean": "Fixed NN", "nn_tuned": "ROC-selected NN",
             "random_forest": "Fixed Random Forest", "random_forest_tuned": "ROC-selected Random Forest",
             "xgboost": "Fixed XGBoost", "xgboost_tuned": "ROC-selected XGBoost"}
    for condition in ["original", "dedup_v1"]:
        for _, row in table.loc[table.condition.eq(condition)].sort_values("roc_auc", ascending=False).iterrows():
            text.append(f"| {'Original' if condition == 'original' else 'Deduplicated'} | {names[row.model]} | {row.roc_auc:.4f} | {row.average_precision:.4f} | {row.log_loss:.5f} | {row.brier:.5f} |")
    text.extend(["", f"Actual new fitting calls: **{total}**, comprising {counts['original']['executed_fits']} original core fits, {counts['dedup_v1']['executed_fits']} clean core fits, and five original baseline logistic fits. The core ledger explicitly reuses **{ledger['externally_reused_core_fits']}** historical fit phases and performs **{ledger['rescoring_operations']}** candidate rescoring operations. Resumed native phases are recorded separately and are not added to executed or external-reuse totals. Dummy scores require zero fits.", "",
                 "| Clean-data comparison | Mean-fold ROC-AUC difference | 95% conditional paired patient interval | AP difference | 95% conditional paired patient interval |",
                 "|---|---:|---|---:|---|"])
    for comparison in ["nn_tuned minus fixed_nn", "nn_tuned minus fixed_spline", "nn_tuned minus tuned_spline"]:
        rows = pairs["dedup_v1"].loc[pairs["dedup_v1"].aggregation.eq("mean_fold") & pairs["dedup_v1"].comparison.eq(comparison)].set_index("metric")
        roc, ap = rows.loc["roc_auc"], rows.loc["average_precision"]
        text.append(f"| {comparison.replace('_', ' ')} | {roc.difference:+.4f} | [{roc.ci_lower:+.4f}, {roc.ci_upper:+.4f}] | {ap.difference:+.4f} | [{ap.ci_lower:+.4f}, {ap.ci_upper:+.4f}] |")
    clean = table.loc[table.condition.eq("dedup_v1")].sort_values("roc_auc", ascending=False)
    winner = clean.iloc[0]
    spline = clean.loc[clean.model.eq("spline")].iloc[0]
    text.extend(["", f"The clean-data ROC-AUC point leader is **{names[winner.model]}** ({winner.roc_auc:.4f}). Its AP is {winner.average_precision:.4f}, compared with fixed spline AP {spline.average_precision:.4f}. The paired intervals above indicate whether the NN advantage is resolved against each specified comparator; zero-crossing intervals leave it uncertain. These intervals condition on saved predictions and omit repeating fitting and selection. The chosen primary metric supports consideration of the ROC leader, with the AP and probability-error trade-offs retained; no final model is automatically replaced.", "",
                 "All bootstrap analyses use 2,000 whole-patient replicates within outer folds. Historical image selection remains AP-based. No calibration maps or recall cutoffs were transferred to new models; no image model was retrained, no all-development final fit was performed, and reserved-test data was not accessed.", ""])
    document = ROOT / "docs/metadata_roc_auc_workflow.md"
    existing = document.read_text(encoding="utf-8").split("## Completed ROC-selected experiment:")[0].rstrip()
    pieces = re.split(r"(```.*?```|`[^`]*`)", existing, flags=re.DOTALL)
    for index in range(0, len(pieces), 2):
        pieces[index] = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", pieces[index])
        pieces[index] = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", pieces[index])
        pieces[index] = re.sub(r"(?<=[A-Za-z]),(?=\d)", ", ", pieces[index])
    existing = "".join(pieces)
    document.write_text(existing + "\n\n" + "\n".join(text), encoding="utf-8")
    print(json.dumps(ledger, indent=2), flush=True)


if __name__ == "__main__":
    main()
