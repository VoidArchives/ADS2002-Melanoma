"""Read-only data/report interfaces for the separate clean notebook workflow.

Training is deliberately absent. Complete receipts are verified before scores
are exposed, and unfinished image runs cannot become five-fold comparisons.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .development_dataset import (load_development_dataset, load_preserved_inner_memberships,
                                  sha256_file, stable_sha256)
from .dedup_metadata import verify_saved_receipt
from .dedup_sensitivity import align_predictions, metric_tables, subgroup_table
from .image_nested_reporting import (load_matched_metadata, load_saved_nested_run,
                                     patient_fold_bootstrap)
from .image_nested_calibration_reporting import load_saved_calibration_run
from .metadata_evaluation import decision_metrics, probability_metrics

MANIFEST = "data/subset/processed/dedup_v1/dataset_manifest.json"
DIRECTORY = "results/dedup_v1/clean_workflow"
METADATA_SUITE = "results/dedup_v1/metadata_suite/metadata_full_dedup_v1"
UNCERTAINTY = "results/dedup_v1/uncertainty/b0_uncertainty_dedup_v1"
RAW_RUN = "b0_nested_dedup_v1"
CALIBRATION_RUN = "b0_nested_calibration_dedup_v1"
CORE_RECEIPT = "results/dedup_v1/metadata/comparator_receipt.json"
LABELS = {"b0_raw": "EfficientNet-B0", "spline_raw": "Fixed spline logistic",
          "metadata_nn_mean": "Metadata NN mean"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def project_path(root, relative, prefix="results/dedup_v1"):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to((Path(root) / prefix).resolve()):
        raise ValueError(f"Artifact must stay under {prefix}: {relative}")
    return path


def load_context(project_root):
    root = Path(project_root).resolve()
    frame, identity = load_development_dataset(root, dataset_manifest=MANIFEST)
    manifest = read_json(root / MANIFEST)
    ledger = pd.read_csv(root / "data/subset/processed/dedup_v1/retention_ledger.csv")
    before = ledger.groupby("fold").agg(images=("image_name", "size"),
        malignant_images=("target", "sum"), patients=("patient_id", "nunique")).reset_index()
    after = frame.groupby("fold").agg(images=("image_name", "size"),
        malignant_images=("target", "sum"), patients=("patient_id", "nunique")).reset_index()
    folds = before.merge(after, on="fold", suffixes=("_before", "_after"), validate="one_to_one")
    folds["excluded_images"] = folds.images_before - folds.images_after
    folds["excluded_malignant"] = folds.malignant_images_before - folds.malignant_images_after
    return {"root": root, "frame": frame, "identity": identity, "manifest": manifest,
            "ledger": ledger, "fold_counts": folds}


def verify_cleaning(project_root, *, verify_pixels=False, progress=print):
    """Recheck frozen sources and, optionally, every required JPEG without edits."""
    context = load_context(project_root)
    root, manifest, ledger = context["root"], context["manifest"], context["ledger"]
    checked_sources = 0
    for relative, digest in manifest["audited_input_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or "test" in path.name.lower():
            raise ValueError("Audit source may not resolve to reserved-test data.")
        if sha256_file(path) != digest:
            raise ValueError(f"Audited development source changed: {relative}")
        checked_sources += 1
    pixels_checked = 0
    if verify_pixels:
        from .dedup_audit import _decode, image_fingerprint
        for index, row in enumerate(ledger.itertuples(index=False), 1):
            path = root / "data/subset/train" / f"{row.image_name}.jpg"
            pixels = _decode(path)
            if (sha256_file(path) != row.source_sha256 or image_fingerprint(pixels) != row.pixel_sha256
                    or pixels.size != (row.width, row.height)):
                raise ValueError(f"Required JPEG changed: {row.image_name}")
            pixels_checked += 1
            if progress and index % 1000 == 0:
                progress(f"Cleaning verification: {index}/{len(ledger)} development images", flush=True)
        for _, group in ledger.groupby("pixel_sha256", sort=False):
            if len(group) <= 1:
                continue
            arrays = [np.asarray(_decode(root / "data/subset/train" / f"{name}.jpg")) for name in group.image_name]
            if any(not np.array_equal(arrays[0], other) for other in arrays[1:]):
                raise ValueError("Fingerprint group failed confirmed pixel equality.")
    frame = context["frame"]
    for fold in range(5):
        load_preserved_inner_memberships(root, context["identity"], fold)
    excluded = ledger[ledger.decision.eq("excluded")]
    record = {"status": "complete", "dataset_identity": context["identity"], "counts": manifest["counts"],
              "excluded_images": len(excluded), "excluded_benign": int(excluded.target.eq(0).sum()),
              "excluded_malignant": int(excluded.target.eq(1).sum()), "sources_verified": checked_sources,
              "pixels_verified": pixels_checked, "pixel_verification_requested": verify_pixels,
              "all_outer_and_preserved_inner_assignments_verified": True,
              "reserved_test_loaded": False, "images_deleted": False,
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    output = root / DIRECTORY / "cleaning"
    output.mkdir(parents=True, exist_ok=True)
    context["fold_counts"].to_csv(output / "fold_counts.csv", index=False)
    write_json(output / "receipt.json", record)
    return record


def _verify_hashes(root, receipt):
    hashes = receipt.get("output_sha256") or receipt.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Complete output receipt lacks artifact hashes.")
    for relative, digest in hashes.items():
        path = project_path(root, relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Complete artifact is missing or changed: {relative}")
    for relative, digest in receipt.get("source_sha256", {}).items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError(f"Recorded source changed: {relative}")
    inputs = receipt.get("input_sha256", {})
    if isinstance(inputs, dict):
        for relative, digest in inputs.items():
            path = (root / relative).resolve()
            if not path.is_relative_to((root / "results").resolve()) or sha256_file(path) != digest:
                raise ValueError(f"Report input changed: {relative}")


def load_clean_scores(project_root):
    context = load_context(project_root)
    root, frame, identity = context["root"], context["frame"], context["identity"]
    expected = frame.rename(columns={"image_name": "image_id"}).sort_values("image_id").reset_index(drop=True)
    receipt = read_json(root / CORE_RECEIPT)
    verify_saved_receipt(root, receipt)
    comparators = load_matched_metadata(root, frame, receipt_path=CORE_RECEIPT, dataset_identity=identity)
    scores = {"metadata_nn_mean": comparators["metadata_mlp_mean"].base_probability.to_numpy(),
              "spline_raw": comparators["metadata_spline_raw"].base_probability.to_numpy()}
    inputs = {CORE_RECEIPT: sha256_file(root / CORE_RECEIPT)}
    stages = {"metadata_core": "complete", "metadata_suite": "not_started"}
    suite_path = root / METADATA_SUITE / "summary.json"
    if suite_path.exists():
        suite = read_json(suite_path)
        stages["metadata_suite"] = suite.get("status", "incomplete")
        if suite.get("status") == "complete":
            suite_receipt = read_json(root / METADATA_SUITE / "receipt.json")
            if suite_receipt.get("dataset_manifest_sha256") != identity["dataset_manifest_sha256"]:
                raise ValueError("Metadata suite uses another development dataset.")
            _verify_hashes(root, suite_receipt)
            for model, relative in suite["prediction_paths"].items():
                source = align_predictions(pd.read_csv(project_path(root, relative)), expected, ["score"])
                if model in scores and not np.allclose(scores[model], source.score, rtol=0, atol=1e-14):
                    raise ValueError(f"Reused comparator changed: {model}")
                scores[model] = source.score.to_numpy()
            inputs[str(suite_path.relative_to(root).as_posix())] = sha256_file(suite_path)
    raw = load_image_report(root)
    stages["image_raw"] = raw["status"]
    if raw["complete"]:
        if raw["manifest"].get("dataset_identity") != identity:
            raise ValueError("Image run uses another development identity.")
        predictions = align_predictions(raw["predictions"], expected, ["base_probability"])
        scores["b0_raw"] = predictions.base_probability.to_numpy()
        relative = f"results/image_d1_runs/{RAW_RUN}/outer_oof_predictions.csv"
        inputs[relative] = sha256_file(root / relative)
    return {**context, "expected": expected, "scores": scores, "inputs": inputs,
            "stages": stages, "raw_bundle": raw}


def load_image_report(project_root):
    """Treat the short training-finished/reporting-pending interval explicitly."""
    root = Path(project_root).resolve()
    path = root / f"results/image_d1_runs/{RAW_RUN}/summary.json"
    if path.exists():
        summary = read_json(path)
        if summary.get("status") == "complete" and not summary.get("evaluation", {}).get("evaluation_artifact_sha256"):
            return {"status": "training_complete_reporting_pending", "complete": False,
                    "summary": summary, "manifest": {}, "run_dir": path.parent}
    return load_saved_nested_run(root, RAW_RUN)


def primary_bootstrap(frame, scores, *, replicates=2000, seed=42):
    """Predeclared contrasts with identical fold-wise whole-patient draws."""
    available = {name: values for name, values in scores.items() if name in LABELS}
    reference = "b0_raw" if "b0_raw" in available else "metadata_nn_mean"
    renamed = {"image" if name == reference else name: values for name, values in available.items()}
    intervals, differences = patient_fold_bootstrap(frame, renamed, n_bootstrap=replicates, seed=seed)
    intervals["model"] = intervals.model.replace({"image": reference})
    differences["comparison"] = differences.comparison.str.replace("image minus", reference + " minus", regex=False)
    if reference == "b0_raw":
        _, nn_difference = patient_fold_bootstrap(frame, {"image": scores["metadata_nn_mean"],
            "spline_raw": scores["spline_raw"]}, n_bootstrap=replicates, seed=seed)
        nn_difference["comparison"] = nn_difference.comparison.str.replace("image minus", "metadata_nn_mean minus", regex=False)
        differences = pd.concat([differences, nn_difference], ignore_index=True)
    differences["role"] = "predeclared development comparison"
    return intervals, differences


def _decision_report(frame, table, model, score_columns):
    records = []
    source = align_predictions(table, frame, list(score_columns))
    for label, (score, threshold, decision) in score_columns.items():
        if not {score, threshold, decision}.issubset(source):
            raise ValueError("Decision report lacks a frozen cutoff or decision.")
        if not np.array_equal(source[score].ge(source[threshold]).astype(int), source[decision].astype(int)):
            raise ValueError("Saved decisions disagree with their frozen cutoff.")
        for fold, rows in [*source.groupby("fold"), ("pooled", source)]:
            records.append({"model": model, "score_scale": label, "fold": fold, "n_images": len(rows),
                "n_patients": rows.patient_id.nunique(), "malignant_images": int(rows.target.sum()),
                "malignant_patients": rows.loc[rows.target.eq(1), "patient_id"].nunique(),
                "n_flagged": int(rows[decision].sum()), "cutoff_minimum": float(rows[threshold].min()),
                "cutoff_maximum": float(rows[threshold].max()),
                **probability_metrics(rows.target, rows[score]), **decision_metrics(rows.target, rows[decision]),
                "cutoff_source": "training-only OOF; academic target recall 0.8"})
    return pd.DataFrame(records)


def build_evaluation(project_root, *, replicates=2000, seed=42):
    bundle = load_clean_scores(project_root)
    root, frame, scores = bundle["root"], bundle["expected"], bundle["scores"]
    inputs = dict(bundle["inputs"])
    calibration = load_saved_calibration_run(root, CALIBRATION_RUN)
    bundle["stages"]["image_calibration"] = calibration["status"]
    core = read_json(root / CORE_RECEIPT)
    calibration_path = root / core["calibration_prediction_path"]
    inputs[core["calibration_prediction_path"]] = sha256_file(calibration_path)
    calibrated_metadata = pd.read_csv(calibration_path)
    decisions = [_decision_report(frame, calibrated_metadata, "fixed_spline", {
        "p_spline_raw": ("p_spline_raw", "raw_threshold", "decision_raw"),
        "p_spline_calibrated": ("p_spline_calibrated", "calibrated_threshold", "decision_calibrated")})]
    if calibration["complete"]:
        if calibration["manifest"].get("dataset_identity") != bundle["identity"]:
            raise ValueError("Calibration uses another development variant.")
        predictions = calibration["predictions"]
        decisions.append(_decision_report(frame, predictions, "b0", {
            "base_probability": ("base_probability", "raw_threshold", "decision_raw"),
            "calibrated_probability": ("calibrated_probability", "calibrated_threshold", "decision_calibrated")}))
        relative = f"results/image_d1_runs/{CALIBRATION_RUN}/outer_calibrated_predictions.csv"
        # Actual artifact name is recorded by the complete summary, not guessed.
        for name in ("outer_calibrated_predictions.csv", "outer_oof_predictions.csv"):
            candidate = root / f"results/image_d1_runs/{CALIBRATION_RUN}" / name
            if candidate.is_file():
                inputs[candidate.relative_to(root).as_posix()] = sha256_file(candidate)
                break
        inputs[f"results/image_d1_runs/{CALIBRATION_RUN}/summary.json"] = sha256_file(calibration["run_dir"] / "summary.json")
    complete = all(bundle["stages"].get(name) == "complete" for name in
                   ("metadata_core", "metadata_suite", "image_raw", "image_calibration"))
    fingerprint = stable_sha256({"identity": bundle["identity"], "inputs": inputs,
        "replicates": replicates, "seed": seed, "source": sha256_file(Path(__file__))})
    output = root / DIRECTORY / "evaluation" / fingerprint[:16]
    summary_path = output / "summary.json"
    if summary_path.exists():
        existing = read_json(summary_path)
        _verify_hashes(root, existing)
        write_json(root / DIRECTORY / "evaluation" / "latest.json", {"summary_path": summary_path.relative_to(root).as_posix()})
        return existing
    output.mkdir(parents=True, exist_ok=True)
    metrics = metric_tables(frame, scores, "clean")
    subgroups = subgroup_table(frame, scores, "clean")
    intervals, paired = primary_bootstrap(frame, scores, replicates=replicates, seed=seed)
    for name, table in (("metrics.csv", metrics), ("subgroups.csv", subgroups),
        ("bootstrap_intervals.csv", intervals), ("paired_comparisons.csv", paired),
        ("training_only_cutoff_metrics.csv", pd.concat(decisions, ignore_index=True))):
        table.to_csv(output / name, index=False)
    fixed = []
    for model, values in scores.items():
        fixed.append({"model": model, "cutoff": .5, "role": "fixed descriptive diagnostic",
                      "n_flagged": int(np.sum(np.asarray(values) >= .5)),
                      **decision_metrics(frame.target, np.asarray(values) >= .5)})
    pd.DataFrame(fixed).to_csv(output / "fixed_cutoff_05.csv", index=False)
    main = metrics.loc[metrics.aggregation.eq("mean_fold") & metrics.weighting.eq("image")].copy()
    main["role"] = np.where(main.model.isin(LABELS), "predeclared", "secondary")
    main.to_csv(output / "model_comparison.csv", index=False)
    conclusion = "Pending complete clean image and calibration predictions."
    if complete:
        core_ap = main.loc[main.model.isin(LABELS)].set_index("model").average_precision
        b0 = float(core_ap["b0_raw"])
        best_metadata = float(core_ap[["spline_raw", "metadata_nn_mean"]].max())
        contrasts = paired.loc[paired.aggregation.eq("mean_fold") & paired.metric.eq("average_precision")
                               & paired.comparison.str.startswith("b0_raw minus")]
        if b0 > best_metadata:
            conclusion = f"Clean B0 has the highest mean-fold AP among the three predeclared comparators ({b0:.4f}). "
            conclusion += ("Both paired AP intervals exclude zero in its favour. " if len(contrasts) == 2 and contrasts.ci_lower.gt(0).all()
                           else "At least one paired AP interval includes zero, so the advantage remains uncertain. ")
            conclusion += "This supports reviewing B0 as the ranking choice on these development patients."
        else:
            conclusion = f"Clean B0 mean-fold AP ({b0:.4f}) does not exceed the strongest predeclared metadata comparator ({best_metadata:.4f}); revisit the previous model-choice rationale."
        conclusion += " Secondary tuned models and inner pilots do not automatically change the selected procedure. No reserved-test or clinical claim follows."
    # Clean core comparators have a stable order and directly comparable scales.
    plotted = main[main.model.isin(LABELS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    names = [LABELS[name] for name in plotted.model]
    for ax, metric, title in zip(axes, ("average_precision", "roc_auc"), ("Average precision", "ROC-AUC")):
        ax.barh(names, plotted[metric], color="#276b91")
        ax.set(xlabel=title, title=f"Mean outer-fold {title}")
        ax.set_xlim(0, 1 if metric == "roc_auc" else max(.10, float(plotted[metric].max()) * 1.3))
        for index, value in enumerate(plotted[metric]):
            ax.text(value, index, f"  {value:.3f}", va="center", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Clean development comparison: {len(frame):,} images, {int(frame.target.sum())} malignant")
    fig.savefig(output / "core_model_comparison.png", dpi=150)
    plt.close(fig)
    artifacts = {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    record = {"status": "complete" if complete else "partial", "comparison_complete": complete,
        "dataset_manifest_sha256": bundle["identity"]["dataset_manifest_sha256"],
        "dataset_identity": bundle["identity"], "input_sha256": inputs, "artifact_sha256": artifacts,
        "source_sha256": {Path(__file__).relative_to(root).as_posix(): sha256_file(Path(__file__))},
        "stages": bundle["stages"], "model_names": list(scores), "rows": len(frame),
        "primary_metric": "unweighted mean-fold AP; ROC-AUC also reported",
        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
        "output_directory": output.relative_to(root).as_posix(), "reserved_test_loaded": False,
        "limitation": "Development OOF; intervals condition on fitted models and omit training/selection variability. "
                      "Secondary model/ablation comparisons are exploratory. No reserved-test evidence.",
        "image_conclusion": conclusion}
    write_json(summary_path, record)
    write_json(root / DIRECTORY / "evaluation" / "latest.json", {"summary_path": summary_path.relative_to(root).as_posix()})
    return record


def latest_evaluation(project_root):
    root = Path(project_root).resolve()
    index = root / DIRECTORY / "evaluation/latest.json"
    if not index.exists():
        return {"status": "not_started", "stages": {}, "model_names": []}
    summary = read_json(project_path(root, read_json(index)["summary_path"]))
    _verify_hashes(root, summary)
    # Do not show a report for a superseded or damaged dataset.
    _, identity = load_development_dataset(root, dataset_manifest=MANIFEST)
    if summary["dataset_identity"] != identity:
        raise ValueError("Evaluation report belongs to a different dataset.")
    # A saved summary is not sufficient if a referenced score file changes
    # while its enclosing receipt stays unchanged.
    inputs = summary.get("input_sha256", {})
    if CORE_RECEIPT in inputs:
        core = read_json(root / CORE_RECEIPT)
        if core.get("dataset_identity") != identity:
            raise ValueError("Referenced metadata dataset identity changed.")
        verify_saved_receipt(root, core)
    if f"{METADATA_SUITE}/summary.json" in inputs:
        suite = read_json(root / METADATA_SUITE / "receipt.json")
        if suite.get("dataset_identity") != identity:
            raise ValueError("Referenced metadata suite dataset identity changed.")
        _verify_hashes(root, suite)
    return summary


def load_uncertainty_report(project_root):
    root = Path(project_root).resolve()
    directory = root / UNCERTAINTY
    path = directory / "summary.json"
    if not path.exists():
        return {"status": "pending"}
    summary = read_json(path)
    if summary.get("status") != "complete":
        return summary
    receipt = read_json(directory / "receipt.json")
    _, identity = load_development_dataset(root, dataset_manifest=MANIFEST)
    if receipt.get("dataset_identity") != identity or summary != receipt:
        raise ValueError("Uncertainty receipt and dataset identity disagree.")
    hashes = receipt.get("artifacts")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Uncertainty receipt lacks artifact hashes.")
    for relative, digest in hashes.items():
        artifact = (directory / relative).resolve()
        if not artifact.is_relative_to(directory.resolve()) or sha256_file(artifact) != digest:
            raise ValueError(f"Uncertainty artifact changed: {relative}")
    return summary
