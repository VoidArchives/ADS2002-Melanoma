"""Read-only evaluation of a separate training-only calibration/threshold stage.

Raw outer fits and their reports remain unchanged. No model or calibrator is
fitted here, and no reserved-test data are read. The whole five-fold calibration
stage must complete before pooled comparisons or final metrics are exposed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

from .image_evaluation import sha256
from .image_nested_reporting import (
    BOOTSTRAP_CAVEAT, CAVEAT, METRICS, _canonical, _json_safe, fold_metric_table,
    patient_fold_bootstrap, reliability_table, validate_outer_predictions,
)
from .metadata_evaluation import decision_metrics, select_recall_threshold, validate_probabilities


def validate_calibration_predictions(calibrated_predictions, raw_predictions, expected_frame):
    """Require unchanged raw identities/scores and explicitly frozen decisions."""
    raw = validate_outer_predictions(raw_predictions, expected_frame)
    calibrated = validate_outer_predictions(calibrated_predictions, expected_frame)
    required = {"calibrated_probability", "raw_threshold", "calibrated_threshold", "decision_raw", "decision_calibrated"}
    if not required.issubset(calibrated):
        raise ValueError(f"Missing calibration scenario columns: {sorted(required - set(calibrated))}")
    if not np.allclose(calibrated.base_probability, raw.base_probability, rtol=0, atol=1e-14):
        raise ValueError("Calibration results changed the frozen raw outer predictions.")
    _, calibrated["calibrated_probability"] = validate_probabilities(calibrated.target, calibrated.calibrated_probability)
    for name in ("raw_threshold", "calibrated_threshold"):
        threshold = pd.to_numeric(calibrated[name], errors="raise")
        if not np.isfinite(threshold).all() or ((threshold < 0) | (threshold > 1)).any():
            raise ValueError(f"{name} must contain finite cutoffs in [0, 1].")
        calibrated[name] = threshold
        if calibrated.groupby("fold")[name].nunique().ne(1).any():
            raise ValueError(f"Each outer fold must use one frozen training-only {name}.")
    for decision, probability, threshold in (
        ("decision_raw", "base_probability", "raw_threshold"),
        ("decision_calibrated", "calibrated_probability", "calibrated_threshold"),
    ):
        decision_metrics(calibrated.target, calibrated[decision])  # Reject original non-binary values.
        expected_decisions = (calibrated[probability] >= calibrated[threshold]).astype(int)
        if not np.array_equal(calibrated[decision].to_numpy(dtype=int), expected_decisions):
            raise ValueError(f"{decision} disagrees with the declared frozen score >= threshold rule.")
    return calibrated


def _decision_table(frame, *, by_fold):
    groups = frame.groupby("fold", sort=True) if by_fold else [("pooled", frame)]
    records = []
    for fold, rows in groups:
        for model, decision, threshold in (("raw", "decision_raw", "raw_threshold"),
                                           ("calibrated", "decision_calibrated", "calibrated_threshold")):
            counts = decision_metrics(rows.target, rows[decision])
            denominator = 2 * counts["TP"] + counts["FP"] + counts["FN"]
            records.append({"fold": fold, "model": model, "n_images": len(rows),
                "n_patients": rows.patient_id.nunique(), "n_positive": int(rows.target.sum()),
                "training_target_recall": .8, "threshold": float(rows[threshold].iloc[0]) if by_fold else np.nan,
                "threshold_scope": "fixed per outer fold from training-only OOF scores",
                **counts, "recall": counts["sensitivity"],
                "f1": 2 * counts["TP"] / denominator if denominator else np.nan})
    return pd.DataFrame(records)


def _calibration_figures(reliability, decisions, output):
    output.mkdir(parents=True, exist_ok=True)
    colors = {"raw": "#235789", "calibrated": "#C47A14"}
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
        for ax, model in zip(axes, ("raw", "calibrated")):
            rows = reliability[(reliability.model == model) & (reliability.n_images > 0)]
            ax.plot([0, 1], [0, 1], "--", color="#666666", label="Ideal agreement")
            ax.plot(rows.mean_probability, rows.observed_prevalence, "o-", color=colors[model], label=model.capitalize())
            for row in rows.itertuples():
                label_offset = (3, -12) if row.observed_prevalence >= .95 else (3, 5)
                ax.annotate(str(row.n_images), (row.mean_probability, row.observed_prevalence),
                            xytext=label_offset, textcoords="offset points", fontsize=8)
            ax.set(xlabel="Mean score in bin", ylabel="Observed melanoma fraction", xlim=(0, 1), ylim=(0, 1.04),
                   title=f"{model.capitalize()} pooled OOF reliability")
            ax.legend(fontsize=9)
        fig.suptitle("Fixed bins; annotations show image counts")
        fig.savefig(output / "raw_vs_calibrated_reliability.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
        for ax, model in zip(axes, ("raw", "calibrated")):
            row = decisions[decisions.model == model].iloc[0]
            matrix = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=int)
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, decisions[["TN", "FP", "FN", "TP"]].to_numpy().max()))
            for y in range(2):
                for x in range(2):
                    ax.text(x, y, f"{matrix[y, x]:,}", ha="center", va="center", color="white" if matrix[y, x] > matrix.max() / 2 else "#111111", fontsize=15)
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not flagged", "Flagged"],
                   yticklabels=["Benign", "Melanoma"], xlabel="Frozen training-selected decision", ylabel="Actual image outcome",
                   title=f"{model.capitalize()}: actual recall {row.recall:.1%}")
            ax.grid(False)
        fig.suptitle("Training-selected 80%-recall scenario on held-out outer images")
        fig.savefig(output / "training_selected_threshold_counts.png", dpi=160)
        plt.close(fig)


def build_calibration_evaluation(calibrated_predictions, raw_predictions, expected_frame, output_dir,
                                 bootstrap_replicates=2000, seed=42):
    """Compare a complete calibration stage with its unchanged raw reference."""
    frame = validate_calibration_predictions(calibrated_predictions, raw_predictions, expected_frame)
    raw_folds = fold_metric_table(frame).assign(model="raw")
    calibrated_folds = fold_metric_table(frame, probability_column="calibrated_probability").assign(model="calibrated")
    probability_table = pd.concat([raw_folds, calibrated_folds], ignore_index=True)
    rank_audit = pd.DataFrame({"fold": raw_folds.fold, "raw_average_precision": raw_folds.average_precision,
        "calibrated_average_precision": calibrated_folds.average_precision,
        "ap_difference": calibrated_folds.average_precision - raw_folds.average_precision,
        "raw_roc_auc": raw_folds.roc_auc, "calibrated_roc_auc": calibrated_folds.roc_auc})
    rank_audit["ap_invariant"] = np.isclose(rank_audit.ap_difference, 0, rtol=0, atol=1e-12)
    rank_audit["roc_auc_invariant"] = np.isclose(rank_audit.raw_roc_auc, rank_audit.calibrated_roc_auc, rtol=0, atol=1e-12)
    rank_audit["changed_training_selected_decisions"] = [int((rows.decision_raw != rows.decision_calibrated).sum())
        for _, rows in frame.groupby("fold", sort=True)]
    intervals, differences = patient_fold_bootstrap(frame,
        {"image": frame.calibrated_probability.to_numpy(), "raw": frame.base_probability.to_numpy()},
        n_bootstrap=bootstrap_replicates, seed=seed)
    intervals["model"] = intervals.model.replace({"image": "calibrated"})
    differences["comparison"] = "calibrated minus raw"
    by_fold = _decision_table(frame, by_fold=True)
    pooled = _decision_table(frame, by_fold=False)
    reliability = pd.concat([reliability_table(frame.target, frame[column]).assign(model=model)
        for model, column in (("raw", "base_probability"), ("calibrated", "calibrated_probability"))], ignore_index=True)
    output_dir = Path(output_dir)
    tables, figures = output_dir / "evaluation", output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    emitted = {"probability_metrics_by_fold.csv": probability_table, "ranking_invariance_by_fold.csv": rank_audit,
               "bootstrap_intervals.csv": intervals, "paired_calibration_differences.csv": differences,
               "threshold_scenario_by_fold.csv": by_fold, "threshold_scenario_pooled.csv": pooled,
               "reliability_bins.csv": reliability}
    for name, table in emitted.items():
        table.to_csv(tables / name, index=False)
    _calibration_figures(reliability, pooled, figures)
    summary = {"schema_version": 1, "status": "complete_five_fold_calibration_evaluation",
        "n_images": len(frame), "n_patients": frame.patient_id.nunique(), "outer_folds": list(range(5)),
        "mean_fold_metrics": probability_table.groupby("model")[list(METRICS)].mean().to_dict("index"),
        "mean_fold_ap_invariant": bool(rank_audit.ap_invariant.all()),
        "fold_roc_auc_invariant": bool(rank_audit.roc_auc_invariant.all()),
        "ranking_interpretation": "Positive-slope logistic calibration is monotonic within each fold. AP/ROC-AUC should remain unchanged except for numerical ties or clipped endpoints. Pooled AP can change because fold scales change.",
        "ranking_warning": None if rank_audit.ap_invariant.all() and rank_audit.roc_auc_invariant.all() else "One or more fold ranking metrics changed; inspect clipping/numerical ties before claiming invariance.",
        "training_selected_decisions_identical": bool(rank_audit.changed_training_selected_decisions.sum() == 0),
        "decision_warning": None if rank_audit.changed_training_selected_decisions.sum() == 0 else "Raw/calibrated training-selected decisions differ; inspect clipped endpoints, score ties and frozen threshold mappings.",
        "training_target_recall": .8,
        "threshold_interpretation": "The 80% target is used only on training-only OOF scores. Actual held-out recall can be lower or higher. This is an academic scenario, without clinical validation.",
        "threshold_metrics_pooled": _json_safe(pooled.to_dict("records")),
        "bootstrap_replicates": int(bootstrap_replicates), "valid_bootstrap_replicates": int(intervals.valid_replicates.iloc[0]),
        "rejected_single_class_bootstrap_replicates": int(intervals.rejected_single_class_replicates.iloc[0]),
        "bootstrap_method": BOOTSTRAP_CAVEAT + " Calibration maps and thresholds are also held fixed.",
        "development_caveat": CAVEAT, "reserved_test_loaded": False,
        "metadata_comparison": "Use the separate raw-reference matched metadata report; no comparator is selected using these calibrated outer results."}
    summary = _json_safe(summary)
    summary_path = tables / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    paths = [summary_path, *[tables / name for name in emitted],
             figures / "raw_vs_calibrated_reliability.png", figures / "training_selected_threshold_counts.png"]
    summary["evaluation_artifact_sha256"] = {path.relative_to(output_dir).as_posix(): sha256(path) for path in paths}
    return summary


def load_saved_calibration_run(project_root, run_id="b0_nested_calibration_5h_v1"):
    """Expose complete separate calibration reports only after hash/coverage checks."""
    root = Path(project_root).resolve()
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one directory name.")
    run_dir = root / "results/image_d1_runs" / run_id
    manifest_path, summary_path = run_dir / "run_manifest.json", run_dir / "summary.json"
    if not manifest_path.is_file():
        return {"status": "not_started", "complete": False, "run_dir": run_dir, "manifest": {}, "summary": {}}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    result = {"status": summary.get("status", "running"), "complete": False, "run_dir": run_dir,
              "manifest": manifest, "summary": summary}
    if summary.get("status") != "complete":
        return result
    if summary.get("completed_outer_folds") != list(range(5)):
        raise ValueError("Calibration stage marked complete without all five folds.")
    reference_id = summary.get("reference_run_id") or manifest.get("reference_run_id")
    if not isinstance(reference_id, str) or not reference_id or Path(reference_id).name != reference_id or "/" in reference_id or "\\" in reference_id:
        raise ValueError("Calibration reference must name one saved run directory.")
    raw_dir = root / "results/image_d1_runs" / reference_id
    raw_manifest = json.loads((raw_dir / "run_manifest.json").read_text(encoding="utf-8"))
    raw_summary = json.loads((raw_dir / "summary.json").read_text(encoding="utf-8"))
    if raw_summary.get("status") != "complete" or raw_summary.get("completed_outer_folds") != list(range(5)):
        raise ValueError("Calibration reference raw run is incomplete.")
    canonical = root / "data/subset/processed/train_folds.csv"
    for declaration in (raw_manifest, manifest):
        if (root / declaration.get("development_csv", "")).resolve() != canonical.resolve() or sha256(canonical) != declaration.get("development_csv_sha256"):
            raise ValueError("Calibration or raw-reference canonical input receipt changed.")
    paths = [(run_dir / "outer_oof_predictions.csv", summary.get("outer_oof_predictions_sha256")),
             (raw_dir / "outer_oof_predictions.csv", raw_summary.get("outer_oof_predictions_sha256"))]
    for path, digest in paths:
        if sha256(path) != digest:
            raise ValueError("Calibration or raw-reference OOF prediction hash changed.")
    expected = pd.read_csv(canonical)
    raw_predictions = pd.read_csv(paths[1][0])
    predictions = validate_calibration_predictions(pd.read_csv(paths[0][0]), raw_predictions, expected)
    fold_records = summary.get("folds", [])
    if len(fold_records) != 5 or {row.get("fold") for row in fold_records} != set(range(5)) or any(row.get("status") != "complete" for row in fold_records):
        raise ValueError("Calibration completion requires exactly five complete fold receipts.")
    for record in fold_records:
        folder = run_dir / "folds" / str(record["fold"])
        for name, digest_key in (("calibrator.json", "calibrator_sha256"),
                                 ("training_oof_predictions.csv", "training_oof_sha256"),
                                 ("outer_predictions.csv", "predictions_sha256")):
            if sha256(folder / name) != record.get(digest_key):
                raise ValueError(f"Calibration fold {record['fold']} changed: {name}")
        training_oof = _canonical(pd.read_csv(folder / "training_oof_predictions.csv"))
        pool = _canonical(expected[expected.fold != record["fold"]])
        identity_columns = ["image_id", "patient_id", "fold", "target"]
        if not training_oof[identity_columns].equals(pool[identity_columns]):
            raise ValueError("Calibration training OOF does not exactly match the outer training pool.")
        if "inner_fold" not in training_oof or not pd.to_numeric(training_oof.inner_fold).isin(range(3)).all() or set(training_oof.inner_fold) != set(range(3)) or training_oof.groupby("patient_id").inner_fold.nunique().ne(1).any():
            raise ValueError("Calibration training OOF does not keep patients in three inner folds.")
        validate_probabilities(training_oof.target, training_oof.base_probability)
        calibrator = json.loads((folder / "calibrator.json").read_text(encoding="utf-8"))
        slope, intercept = calibrator.get("slope"), calibrator.get("intercept")
        if calibrator.get("status") != "complete" or slope is None or intercept is None or not np.isfinite([slope, intercept]).all() or slope <= 0:
            raise ValueError("A complete finite positive-slope calibration map is required.")
        if calibrator.get("logit_clip") != 1e-7 or calibrator.get("C") != 1 or calibrator.get("target_recall") != .8 or calibrator.get("outer_labels_used") is not False or calibrator.get("training_oof_sha256") != record["training_oof_sha256"]:
            raise ValueError("Calibration map does not match the declared training-only protocol.")
        selection_path = raw_dir / "folds" / str(record["fold"]) / "selection.json"
        if sha256(selection_path) != record.get("training_selection_sha256") or calibrator.get("training_selection_sha256") != record.get("training_selection_sha256"):
            raise ValueError("The selected raw recipe/duration changed after calibration.")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if calibrator.get("fixed_epochs") != selection.get("refit_epochs") or record.get("fixed_epochs") != selection.get("refit_epochs") or calibrator.get("recipe") != selection.get("recipe"):
            raise ValueError("Calibration inner refits changed the selected recipe or fixed duration.")
        outer = predictions[predictions.fold == record["fold"]]
        for name in ("slope", "intercept", "raw_threshold", "calibrated_threshold"):
            if not np.isclose(calibrator[name], record[name], rtol=0, atol=1e-14):
                raise ValueError("Frozen calibration coefficients/cutoffs disagree with the fold receipt.")
        for name in ("raw_threshold", "calibrated_threshold"):
            if not np.allclose(outer[name], calibrator[name], rtol=0, atol=1e-14):
                raise ValueError("Outer thresholds disagree with the frozen training-selected cutoffs.")
        def transform(probabilities):
            clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
            return expit(slope * np.log(clipped / (1 - clipped)) + intercept)
        if not np.allclose(outer.calibrated_probability, transform(outer.base_probability), rtol=0, atol=1e-14):
            raise ValueError("Outer calibrated probabilities do not reproduce the frozen map.")
        training_calibrated = transform(training_oof.base_probability)
        for name, scores in (("raw_threshold", training_oof.base_probability), ("calibrated_threshold", training_calibrated)):
            if not np.isclose(calibrator[name], select_recall_threshold(training_oof.target, scores, .8), rtol=0, atol=1e-14):
                raise ValueError("Saved cutoff does not reproduce the training-only 80%-recall rule.")
    evaluation = summary.get("evaluation", {})
    hashes = evaluation.get("evaluation_artifact_sha256", {})
    if "evaluation/metrics_summary.json" not in hashes:
        result["status"] = "training_complete_reporting_pending"
        result.update({"ready_for_reporting": True, "predictions": predictions, "raw_predictions": raw_predictions, "expected": expected})
        return result
    for relative, digest in hashes.items():
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir.resolve()) or path.parent not in ((run_dir / "evaluation").resolve(), (run_dir / "figures").resolve()) or sha256(path) != digest:
            raise ValueError(f"Calibration evaluation artifact changed: {relative}")
    recorded = pd.read_csv(run_dir / "evaluation/probability_metrics_by_fold.csv")
    recomputed = pd.concat([fold_metric_table(predictions).assign(model="raw"),
        fold_metric_table(predictions, probability_column="calibrated_probability").assign(model="calibrated")], ignore_index=True)
    if not recorded[["fold", "model"]].equals(recomputed[["fold", "model"]]) or not np.allclose(recorded[list(METRICS)], recomputed[list(METRICS)], rtol=1e-10, atol=1e-12):
        raise ValueError("Calibration probability metrics disagree with outer predictions.")
    for name, by_fold in (("threshold_scenario_by_fold.csv", True), ("threshold_scenario_pooled.csv", False)):
        actual = pd.read_csv(run_dir / "evaluation" / name)
        computed = _decision_table(predictions, by_fold=by_fold)
        if not np.array_equal(actual[["TN", "FP", "FN", "TP"]], computed[["TN", "FP", "FN", "TP"]]):
            raise ValueError("Calibration scenario decision counts disagree with frozen predictions.")
    result.update({"complete": True, "ready_for_reporting": True, "predictions": predictions, "raw_predictions": raw_predictions, "expected": expected,
        "evaluation": json.loads((run_dir / "evaluation/metrics_summary.json").read_text(encoding="utf-8")),
        "probability_metrics": recorded, "intervals": pd.read_csv(run_dir / "evaluation/bootstrap_intervals.csv"),
        "paired_differences": pd.read_csv(run_dir / "evaluation/paired_calibration_differences.csv"),
        "scenario_by_fold": pd.read_csv(run_dir / "evaluation/threshold_scenario_by_fold.csv"),
        "scenario_pooled": pd.read_csv(run_dir / "evaluation/threshold_scenario_pooled.csv"),
        "ranking_invariance": pd.read_csv(run_dir / "evaluation/ranking_invariance_by_fold.csv"),
        "verification": "both canonical/prediction receipts, complete fold receipts, final artifact hashes and recomputed outer metrics/counts"})
    return result
