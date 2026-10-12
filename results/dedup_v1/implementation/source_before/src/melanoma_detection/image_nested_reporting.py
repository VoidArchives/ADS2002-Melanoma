"""Independent reporting of complete patient-grouped five-fold image OOF scores.

No training, threshold selection, calibration fitting, or reserved-test access.
Bootstrap intervals condition on the saved fitted models and recipe choices:
they do not include the variability of retraining or prior pilot adaptivity.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .image_evaluation import classification_metrics, curve_tables, sha256
from .metadata_evaluation import probability_metrics, validate_probabilities
from .metadata_validation import validate_binary_target
from .notebook_integrity import verify_recorded_notebook_code

IDENTITY_COLUMNS = ["image_id", "patient_id", "fold", "target"]
METRICS = ("average_precision", "roc_auc", "log_loss", "brier")
CAVEAT = (
    "Development OOF evidence after earlier architecture, augmentation and recipe pilots; "
    "fold 0 also has archived exploratory results. Independent outer prediction within "
    "this run does not erase earlier adaptation to this development dataset. The locked "
    "test remains necessary after decisions are frozen."
)
BOOTSTRAP_CAVEAT = (
    "95% percentile patient-cluster bootstrap, resampling patients within each "
    "outer fold. All images of a sampled patient stay together. Single-class "
    "draws in any fold are excluded from all metrics and comparisons. "
    "Intervals are conditional on these fitted models, selected recipes, fixed folds "
    "and observed fold memberships; no retraining or recipe reselection is performed. "
    "They omit training variability and prior pilot adaptivity."
)


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError("Input must be a non-empty DataFrame with unique columns.")
    result = frame.copy()
    if "image_id" not in result and "image_name" in result:
        result = result.rename(columns={"image_name": "image_id"})
    elif "image_id" in result and "image_name" in result:
        if not result.image_id.astype(str).equals(result.image_name.astype(str)):
            raise ValueError("image_id and image_name identities disagree.")
    if not set(IDENTITY_COLUMNS).issubset(result):
        raise ValueError("Missing image_id/image_name, patient_id, fold or target.")
    for name in ("image_id", "patient_id"):
        text = result[name].astype("string")
        if result[name].isna().any() or text.str.strip().eq("").any() or text.ne(text.str.strip()).any():
            raise ValueError(f"{name} contains missing, blank or untrimmed identifiers.")
        result[name] = text.astype(str)
    if result.image_id.duplicated().any():
        raise ValueError("Each canonical image must occur exactly once.")
    folds = pd.to_numeric(result.fold, errors="raise")
    if not np.isfinite(folds).all() or not folds.isin(range(5)).all():
        raise ValueError("fold must be exactly one of 0, 1, 2, 3, 4.")
    result["fold"] = folds.astype(int)
    # This validates original targets before integer conversion.
    result["target"] = validate_binary_target(result.target)
    if result.groupby("patient_id").fold.nunique().gt(1).any():
        raise ValueError("Patient membership crosses outer folds.")
    return result.sort_values("image_id").reset_index(drop=True)


def validate_outer_predictions(predictions, expected_frame) -> pd.DataFrame:
    """Reject partial, duplicate, relabelled or extra predictions; never inner join."""
    expected = _canonical(expected_frame)
    actual = _canonical(predictions)
    if set(expected.fold.unique()) != set(range(5)):
        raise ValueError("Expected development frame must contain all five outer folds.")
    if not actual[IDENTITY_COLUMNS].equals(expected[IDENTITY_COLUMNS]):
        missing = len(set(expected.image_id) - set(actual.image_id))
        extra = len(set(actual.image_id) - set(expected.image_id))
        raise ValueError(f"Outer predictions do not exactly match canonical coverage/labels (missing={missing}, extra={extra}).")
    if "base_probability" not in actual:
        raise ValueError("Missing raw base_probability scores.")
    _, actual["base_probability"] = validate_probabilities(actual.target, actual.base_probability)
    for fold, rows in actual.groupby("fold", sort=True):
        if rows.target.nunique() != 2:
            raise ValueError(f"Outer fold {fold} requires both classes for ranking evaluation.")
        for name in ("recipe_id", "refit_epochs"):
            if name in rows and (rows[name].isna().any() or rows[name].nunique() != 1):
                raise ValueError(f"Outer fold {fold} has inconsistent {name} provenance.")
    for name in ("run_id", "model_name"):
        if name in actual and (actual[name].isna().any() or actual[name].nunique() != 1):
            raise ValueError(f"Outer predictions must have one non-missing {name}.")
    if "refit_epochs" in actual:
        epochs = pd.to_numeric(actual.refit_epochs, errors="raise")
        if not np.isfinite(epochs).all() or (epochs <= 0).any() or (epochs != np.floor(epochs)).any():
            raise ValueError("refit_epochs must be positive finite integers.")
    return actual


def fold_metric_table(predictions, *, probability_column="base_probability") -> pd.DataFrame:
    records = []
    for fold, rows in predictions.groupby("fold", sort=True):
        records.append({"fold": int(fold), "n_images": len(rows), "n_patients": rows.patient_id.nunique(),
                        "n_positive": int(rows.target.sum()), "prevalence": float(rows.target.mean()),
                        **probability_metrics(rows.target, rows[probability_column])})
    return pd.DataFrame(records)


def _cache(y, p):
    """Fixed score order makes cluster multiplicity bootstrap fast and exact."""
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {"order": order, "ends": ends, "target": y[order],
            "log_losses": -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)),
            "brier_losses": (y - p) ** 2}


def _weighted_metrics(cache, weights):
    weights = np.asarray(weights, dtype=float)
    sorted_weights = weights[cache["order"]]
    positives = np.cumsum(sorted_weights * cache["target"])[cache["ends"]]
    totals = np.cumsum(sorted_weights)[cache["ends"]]
    negatives = totals - positives
    if positives[-1] == 0 or negatives[-1] == 0:
        raise ValueError("Bootstrap draw has no examples of one class.")
    recall = positives / positives[-1]
    precision = np.divide(positives, totals, out=np.zeros_like(positives), where=totals > 0)
    fpr, tpr = np.r_[0, negatives / negatives[-1]], np.r_[0, recall]
    return np.array([np.sum(np.diff(np.r_[0, recall]) * precision),
                     np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2),
                     float(np.dot(weights, cache["log_losses"]) / weights.sum()),
                     float(np.dot(weights, cache["brier_losses"]) / weights.sum())])


def patient_fold_bootstrap(predictions, probabilities, *, n_bootstrap=2000, seed=42):
    """Paired bootstrap of mean-fold and pooled metrics on fixed saved OOF scores.

    Each fold has an independent draw of its original number of patients.
    Mixed-label patients contribute all their images together. A draw lacking
    either image-level outcome in any fold is excluded for every model/metric;
    requested, valid and rejected counts are returned with each interval.
    """
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer of at least 2.")
    frame = _canonical(predictions)
    if not predictions.image_id.astype(str).reset_index(drop=True).equals(frame.image_id):
        raise ValueError("Bootstrap predictions must be in canonical image_id order.")
    if set(frame.fold.unique()) != set(range(5)):
        raise ValueError("Bootstrap requires all five outer folds.")
    y = frame.target.to_numpy()
    codes, patients = pd.factorize(frame.patient_id, sort=True)
    patient_folds = frame.assign(patient_code=codes).groupby("patient_code", sort=True).fold.first()
    pools = [patient_folds[patient_folds == fold].index.to_numpy() for fold in range(5)]
    fold_indices = [np.flatnonzero(frame.fold.to_numpy() == fold) for fold in range(5)]
    caches, estimates = {}, {}
    for model, scores in probabilities.items():
        _, p = validate_probabilities(y, scores)
        caches[model] = (_cache(y, p), [_cache(y[idx], p[idx]) for idx in fold_indices])
        fold_values = np.array([list(probability_metrics(y[idx], p[idx]).values()) for idx in fold_indices])
        if not np.isfinite(fold_values).all():
            raise ValueError("Every bootstrap fold must contain both classes.")
        estimates[model] = {"mean_fold": fold_values.mean(axis=0), "pooled": np.array(list(probability_metrics(y, p).values()))}
    rng = np.random.default_rng(seed)
    distributions = {(model, aggregation): np.empty((n_bootstrap, len(METRICS)))
                     for model in caches for aggregation in ("mean_fold", "pooled")}
    valid_replicates = 0
    for replicate in range(n_bootstrap):
        counts = np.zeros(len(patients), dtype=float)
        for pool in pools:
            counts += np.bincount(rng.choice(pool, len(pool), replace=True), minlength=len(patients))
        weights = counts[codes]
        if any(np.dot(weights[idx], y[idx]) == 0 or np.dot(weights[idx], 1 - y[idx]) == 0 for idx in fold_indices):
            continue
        for model, (pooled_cache, fold_caches) in caches.items():
            distributions[model, "pooled"][valid_replicates] = _weighted_metrics(pooled_cache, weights)
            distributions[model, "mean_fold"][valid_replicates] = np.mean([
                _weighted_metrics(cache, weights[idx]) for idx, cache in zip(fold_indices, fold_caches)], axis=0)
        valid_replicates += 1
    if valid_replicates < 2:
        raise ValueError(f"Only {valid_replicates}/{n_bootstrap} bootstrap draws have both classes in every fold; increase replicates or inspect patient support.")
    distributions = {key: values[:valid_replicates] for key, values in distributions.items()}
    records = []
    for (model, aggregation), values in distributions.items():
        for metric_index, metric in enumerate(METRICS):
            lower, upper = np.percentile(values[:, metric_index], [2.5, 97.5])
            records.append({"model": model, "aggregation": aggregation, "metric": metric,
                            "estimate": float(estimates[model][aggregation][metric_index]),
                            "ci_lower": float(lower), "ci_upper": float(upper),
                            "bootstrap_replicates": n_bootstrap, "valid_replicates": valid_replicates,
                            "rejected_single_class_replicates": n_bootstrap - valid_replicates, "seed": seed,
                            "interval": "95% percentile, conditional patient-cluster bootstrap"})
    comparisons = []
    if "image" in caches:
        for comparator in caches:
            if comparator == "image":
                continue
            for aggregation in ("mean_fold", "pooled"):
                differences = distributions["image", aggregation] - distributions[comparator, aggregation]
                for metric_index, metric in enumerate(METRICS):
                    lower, upper = np.percentile(differences[:, metric_index], [2.5, 97.5])
                    comparisons.append({"comparison": f"image minus {comparator}", "aggregation": aggregation,
                                        "metric": metric, "difference": float(estimates["image"][aggregation][metric_index] - estimates[comparator][aggregation][metric_index]),
                                        "ci_lower": float(lower), "ci_upper": float(upper),
                                        "bootstrap_replicates": n_bootstrap, "valid_replicates": valid_replicates,
                                        "rejected_single_class_replicates": n_bootstrap - valid_replicates, "seed": seed})
    return pd.DataFrame(records), pd.DataFrame(comparisons)


def reliability_table(target, probability, *, n_bins=10):
    y, p = validate_probabilities(target, probability)
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least 2.")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return pd.DataFrame([{"bin": b, "lower": b / n_bins, "upper": (b + 1) / n_bins,
                          "n_images": int(np.sum(bins == b)), "n_positive": int(y[bins == b].sum()),
                          "mean_probability": float(p[bins == b].mean()) if np.any(bins == b) else np.nan,
                          "observed_prevalence": float(y[bins == b].mean()) if np.any(bins == b) else np.nan}
                         for b in range(n_bins)])


def _save_plots(frame, fold_metrics, reliability, figures):
    figures.mkdir(parents=True, exist_ok=True)
    colors = {0: "#235789", 1: "#C47A14", 2: "#66743A", 3: "#A45078", 4: "#333333"}
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
        for fold, rows in frame.groupby("fold", sort=True):
            curves = curve_tables(rows.target, rows.base_probability)
            label = f"Fold {fold} ({int(rows.target.sum())} positive images)"
            axes[0].plot(curves["precision_recall"].recall, curves["precision_recall"].precision,
                         color=colors[fold], label=label, linewidth=1.6, linestyle="--" if fold % 2 else "-")
            axes[1].plot(curves["roc"].false_positive_rate, curves["roc"].true_positive_rate,
                         color=colors[fold], linewidth=1.6, linestyle="--" if fold % 2 else "-")
        pooled = curve_tables(frame.target, frame.base_probability)
        axes[0].plot(pooled["precision_recall"].recall, pooled["precision_recall"].precision,
                     color="#111111", linewidth=2.5, linestyle=":", label="Pooled OOF (secondary)")
        axes[0].axhline(frame.target.mean(), color="#999999", linestyle=":", label=f"Pooled prevalence {frame.target.mean():.2%}")
        axes[1].plot([0, 1], [0, 1], color="#999999", linestyle=":")
        axes[0].set(xlabel="Recall", ylabel="Precision", title="Raw OOF precision–recall", xlim=(0, 1), ylim=(0, 1))
        axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="Raw OOF ROC by outer fold", xlim=(0, 1), ylim=(0, 1))
        axes[0].legend(fontsize=8, loc="upper right")
        fig.savefig(figures / "outer_fold_pr_roc.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for row in fold_metrics.itertuples():
            ax.scatter(row.fold, row.average_precision, s=90, color=colors[row.fold])
            ax.annotate(f"{row.average_precision:.3f}", (row.fold, row.average_precision), xytext=(0, 8), textcoords="offset points", ha="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        ax.scatter(fold_metrics.fold, fold_metrics.prevalence, marker="x", color="#666666", label="Fold prevalence")
        ax.axhline(fold_metrics.average_precision.mean(), color="#235789", linestyle="--", label="Mean fold AP")
        ax.set(xlabel="Outer fold", ylabel="Average precision / prevalence", xticks=range(5),
               title="Outer-fold variation in image ranking", ylim=(0, min(1, max(.1, fold_metrics.average_precision.max() * 1.3))))
        ax.legend(loc="upper right")
        fig.savefig(figures / "outer_fold_average_precision.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        nonempty = reliability[reliability.n_images > 0]
        axes[0].plot([0, 1], [0, 1], color="#666666", linestyle="--", label="Ideal agreement")
        axes[0].plot(nonempty.mean_probability, nonempty.observed_prevalence, "o-", color="#235789", label="Raw image scores")
        for row in nonempty.itertuples():
            axes[0].annotate(str(row.n_images), (row.mean_probability, row.observed_prevalence), xytext=(3, 5), textcoords="offset points", fontsize=8)
        axes[0].set(xlabel="Mean raw probability in bin", ylabel="Observed melanoma fraction", xlim=(0, 1), ylim=(0, 1), title="Pooled raw reliability (bin counts shown)")
        axes[0].legend(fontsize=9)
        axes[1].hist(frame.base_probability, bins=np.linspace(0, 1, 11), color="#235789", edgecolor="white")
        axes[1].set(xlabel="Raw image probability", ylabel="Number of images", title="OOF score distribution", xlim=(0, 1))
        fig.savefig(figures / "raw_reliability.png", dpi=160)
        plt.close(fig)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def build_evaluation(predictions, expected_frame, output_dir, bootstrap_replicates=2000, seed=42, *, comparators=None):
    """Write final evaluation artifacts only after exact complete OOF validation.

    comparators maps a display name to a matching raw-score prediction frame.
    Files are restricted to evaluation/ and figures/ inside output_dir. The
    runner owns its top-level manifest, summary and progress files.
    """
    frame = validate_outer_predictions(predictions, expected_frame)
    probabilities = {"image": frame.base_probability.to_numpy()}
    for name, comparator in (comparators or {}).items():
        if name == "image":
            raise ValueError("Comparator name 'image' is reserved.")
        paired = validate_outer_predictions(comparator, expected_frame)
        probabilities[name] = paired.base_probability.to_numpy()
    # Evaluate before any output is written, so invalid comparisons leave no receipt.
    intervals, comparisons = patient_fold_bootstrap(frame, probabilities, n_bootstrap=bootstrap_replicates, seed=seed)
    folds = fold_metric_table(frame)
    reliability = reliability_table(frame.target, frame.base_probability)
    output_dir = Path(output_dir)
    tables, figures = output_dir / "evaluation", output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    folds.to_csv(tables / "fold_metrics.csv", index=False)
    intervals.to_csv(tables / "bootstrap_intervals.csv", index=False)
    reliability.to_csv(tables / "raw_reliability_bins.csv", index=False)
    threshold = classification_metrics(frame.target, frame.base_probability, .5)
    pd.DataFrame([threshold]).to_csv(tables / "raw_threshold_05.csv", index=False)
    if not comparisons.empty:
        comparisons.to_csv(tables / "paired_metadata_differences.csv", index=False)
    _save_plots(frame, folds, reliability, figures)
    mean_ap = intervals[(intervals.model == "image") & (intervals.aggregation == "mean_fold") & (intervals.metric == "average_precision")].iloc[0]
    summary = {"schema_version": 1, "status": "complete_five_fold_evaluation",
               "n_images": len(frame), "n_patients": frame.patient_id.nunique(), "n_positive": int(frame.target.sum()),
               "prevalence": float(frame.target.mean()), "outer_folds": list(range(5)),
               "primary_metric": "unweighted mean outer-fold average precision",
               "primary_mean_fold_average_precision": float(mean_ap.estimate),
               "primary_mean_fold_ap_ci95": [float(mean_ap.ci_lower), float(mean_ap.ci_upper)],
               "mean_fold_metrics": folds[list(METRICS)].mean().to_dict(),
               "fold_metric_standard_deviation": folds[list(METRICS)].std(ddof=1).to_dict(),
               "pooled_metrics_secondary": probability_metrics(frame.target, frame.base_probability),
               "raw_threshold_05": threshold, "bootstrap_replicates": int(bootstrap_replicates),
               "valid_bootstrap_replicates": int(mean_ap.valid_replicates),
               "rejected_single_class_bootstrap_replicates": int(mean_ap.rejected_single_class_replicates), "bootstrap_seed": seed,
               "bootstrap_method": BOOTSTRAP_CAVEAT, "development_caveat": CAVEAT,
               "score_interpretation": "Raw sigmoid scores; class weighting can distort probability scale. No fitted calibration or tuned decision cutoff.",
               "pooled_caveat": "Pooled AP mixes scores from independently fitted models; mean fold AP is primary because fold score scales may differ.",
               "comparator_models": list(probabilities.keys())[1:],
               "reserved_test_loaded": False}
    summary = _json_safe(summary)
    (tables / "metrics_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Only artifacts emitted by this call enter the receipt. A stale optional
    # comparison from an earlier attempt must not silently re-enter the report.
    emitted = [tables / name for name in ("metrics_summary.json", "fold_metrics.csv", "bootstrap_intervals.csv",
        "raw_threshold_05.csv", "raw_reliability_bins.csv")]
    emitted += [figures / name for name in ("outer_fold_pr_roc.png", "outer_fold_average_precision.png", "raw_reliability.png")]
    if not comparisons.empty:
        emitted.append(tables / "paired_metadata_differences.csv")
    summary["evaluation_artifact_sha256"] = {path.relative_to(output_dir).as_posix(): sha256(path) for path in emitted}
    return summary


def load_matched_metadata(project_root, expected_frame):
    """Verify the existing hash-recorded metadata OOF bundle, then pair all rows.

    The MLP mean and raw spline scores share this verified prediction file.
    Comparisons describe this development population after earlier diagnostics.
    """
    root = Path(project_root).resolve()
    config_path = root / "results/metrics/metadata_nn_run_config_d1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = (root / config["input_path"]).resolve()
    prediction_path = (root / config["prediction_path"]).resolve()
    if input_path != (root / "data/subset/processed/train_folds.csv").resolve():
        raise ValueError("Metadata comparator must refer to the canonical development CSV.")
    if not prediction_path.is_relative_to(root) or sha256(input_path) != config["input_sha256"] or sha256(prediction_path) != config["prediction_sha256"]:
        raise ValueError("Metadata comparator input or prediction hash does not match its saved receipt.")
    recorded_notebook = config["notebook_path"]
    current_notebook = recorded_notebook
    registry_path = root / "notebooks/filename_map.json"
    if registry_path.is_file():
        for rename in json.loads(registry_path.read_text(encoding="utf-8")).get("renames", []):
            if rename["old_path"] == recorded_notebook:
                current_notebook = rename["new_path"]
    verify_recorded_notebook_code(root, current_notebook, recorded_notebook, config["code_sha256"])
    source = pd.read_csv(prediction_path)
    output = {}
    for name, column in (("metadata_mlp_mean", "p_nn_mean"), ("metadata_spline_raw", "p_spline")):
        comparator = source[["image_name", "patient_id", "fold", "target", column]].rename(columns={column: "base_probability"})
        output[name] = validate_outer_predictions(comparator, expected_frame)
    return output


def load_saved_nested_run(project_root, run_id="b0_nested_5fold_5h_v1"):
    """Read a run's current progress; expose final results only after verification.

    Incomplete runs return status without loading partial OOF metrics. This
    prevents accidental promotion of a subset of folds to full evaluation.
    """
    root = Path(project_root).resolve()
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one directory name.")
    run_dir = root / "results/image_d1_runs" / run_id
    manifest_path, summary_path = run_dir / "run_manifest.json", run_dir / "summary.json"
    if not manifest_path.is_file():
        return {"status": "not_started", "complete": False, "run_dir": run_dir, "manifest": {}, "summary": {}}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    bundle = {"status": summary.get("status", "running"), "complete": False, "run_dir": run_dir,
              "manifest": manifest, "summary": summary}
    if summary.get("status") != "complete":
        return bundle
    relative = manifest.get("development_csv")
    development_path = (root / relative).resolve() if isinstance(relative, str) else None
    if development_path != (root / "data/subset/processed/train_folds.csv").resolve():
        raise ValueError("Saved run must refer to the canonical d1 development CSV.")
    if sha256(development_path) != manifest.get("development_csv_sha256"):
        raise ValueError("Canonical development CSV changed since this run.")
    prediction_path = run_dir / "outer_oof_predictions.csv"
    if sha256(prediction_path) != summary.get("outer_oof_predictions_sha256"):
        raise ValueError("Saved outer OOF prediction hash changed or is missing.")
    expected = pd.read_csv(development_path)
    predictions = validate_outer_predictions(pd.read_csv(prediction_path), expected)
    evaluation = summary.get("evaluation", summary)
    hashes = evaluation.get("evaluation_artifact_sha256", {})
    if not isinstance(hashes, dict) or "evaluation/metrics_summary.json" not in hashes:
        raise ValueError("Completed run is missing its final evaluation integrity receipt.")
    for relative, digest in hashes.items():
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir.resolve()) or path.parent not in ((run_dir / "evaluation").resolve(), (run_dir / "figures").resolve()) or sha256(path) != digest:
            raise ValueError(f"Saved final evaluation artifact changed: {relative}")
    reported = json.loads((run_dir / "evaluation/metrics_summary.json").read_text(encoding="utf-8"))
    recomputed = fold_metric_table(predictions)
    stored_folds = pd.read_csv(run_dir / "evaluation/fold_metrics.csv")
    if not recomputed.fold.equals(stored_folds.fold) or not np.allclose(recomputed[list(METRICS)], stored_folds[list(METRICS)], rtol=1e-10, atol=1e-12):
        raise ValueError("Saved fold metrics disagree with independently recomputed predictions.")
    if not np.isclose(recomputed.average_precision.mean(), reported["primary_mean_fold_average_precision"], rtol=1e-10, atol=1e-12):
        raise ValueError("Saved mean-fold AP disagrees with independently recomputed predictions.")
    bundle.update({"complete": True, "predictions": predictions, "expected": expected, "evaluation": reported,
                   "fold_metrics": stored_folds,
                   "intervals": pd.read_csv(run_dir / "evaluation/bootstrap_intervals.csv"),
                   "threshold_metrics": pd.read_csv(run_dir / "evaluation/raw_threshold_05.csv"),
                   "reliability": pd.read_csv(run_dir / "evaluation/raw_reliability_bins.csv"),
                   "verification": "canonical hash, prediction hash, evaluation hashes, exact five-fold coverage and recomputed fold metrics"})
    comparison_path = run_dir / "evaluation/paired_metadata_differences.csv"
    bundle["metadata_comparison"] = pd.read_csv(comparison_path) if comparison_path.is_file() and "evaluation/paired_metadata_differences.csv" in hashes else pd.DataFrame()
    return bundle
