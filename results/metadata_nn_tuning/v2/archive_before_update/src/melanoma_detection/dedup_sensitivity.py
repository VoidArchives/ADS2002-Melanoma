"""Development-only sensitivity of fixed saved predictions to exact-copy retention.

This module never trains, refits calibration, chooses cutoffs, or opens a test
dataset. Historical receipt readers retain their original integrity checks.
Whole-patient draws are shared across models AND original/retained populations.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .image_evaluation import sha256
from .image_nested_reporting import (
    METRICS, _cache, _canonical, _json_safe, _weighted_metrics,
    load_matched_metadata, load_saved_nested_run, validate_outer_predictions,
)
from .image_nested_calibration_reporting import (
    load_saved_calibration_run, validate_calibration_predictions,
)
from .metadata_evaluation import decision_metrics, probability_metrics, validate_probabilities

MODELS = ("b0_raw", "spline_raw", "metadata_nn_mean")
PAIRS = (("b0_raw", "spline_raw"), ("b0_raw", "metadata_nn_mean"),
         ("metadata_nn_mean", "spline_raw"))
LIMITATION = (
    "Filtered historical predictions check duplicate influence on evaluation weighting only. "
    "Duplicates still influenced historical training, preprocessing, recipe/epoch selection "
    "and calibration fitting. Models, maps and cutoffs stay fixed. Patient bootstrap intervals "
    "condition on saved fits and omit retraining/selection variability and earlier pilot adaptivity. "
    "These are development results, not reserved-test or clinical evidence."
)


def align_predictions(predictions, expected, score_columns):
    """Canonicalise order, require exact coverage and validate every score."""
    canonical = _canonical(expected)
    actual = _canonical(predictions)
    if not actual[["image_id", "patient_id", "fold", "target"]].equals(
            canonical[["image_id", "patient_id", "fold", "target"]]):
        raise ValueError("Prediction coverage/patient/fold/target does not exactly match the population.")
    for column in score_columns:
        if column not in actual:
            raise ValueError(f"Missing probability column {column}.")
        _, actual[column] = validate_probabilities(actual.target, actual[column])
    return actual


def filter_saved_predictions(predictions, original, retained, score_columns):
    """Filter one frozen global population; never pick predictions within groups."""
    source = align_predictions(predictions, original, score_columns)
    expected = _canonical(retained)
    membership = _canonical(original)
    subset = membership[membership.image_id.isin(expected.image_id)].reset_index(drop=True)
    if not subset[["image_id", "patient_id", "fold", "target"]].equals(
            expected[["image_id", "patient_id", "fold", "target"]]):
        raise ValueError("Retained population changed original patient/fold/target identities.")
    filtered = align_predictions(source[source.image_id.isin(expected.image_id)], expected, score_columns)
    indexed = source.set_index("image_id").loc[filtered.image_id, list(score_columns)].to_numpy()
    if not np.array_equal(indexed, filtered[list(score_columns)].to_numpy()):
        raise ValueError("Filtering changed retained saved scores.")
    return filtered


def paired_population_bootstrap(original, retained_ids, scores, *, n_bootstrap=2000,
                                seed=42, clean_scores=None):
    """Joint paired draws with patient multiplicities, including unequal row masks.

    ``scores`` are in canonical original order; optional clean scores are in
    canonical retained order. A single-class fold in either original/retained
    population invalidates that draw for ALL estimates/comparisons.
    """
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer of at least 2.")
    frame = _canonical(original)
    supplied = original.rename(columns={"image_name": "image_id"}) if "image_id" not in original else original
    if not supplied.image_id.astype(str).reset_index(drop=True).equals(frame.image_id):
        raise ValueError("Bootstrap input must use canonical image_id order.")
    ids = list(retained_ids)
    if len(ids) != len(set(ids)) or not set(ids).issubset(set(frame.image_id)):
        raise ValueError("Retained IDs must be unique members of the original population.")
    mask = frame.image_id.isin(ids).to_numpy()
    if not mask.any() or set(frame.fold.unique()) != set(range(5)):
        raise ValueError("Both populations require all five folds.")
    if set(frame.loc[mask, "patient_id"]) != set(frame.patient_id):
        raise ValueError("Retention must preserve the original patient population for paired draws.")
    y = frame.target.to_numpy()
    fold = frame.fold.to_numpy()
    codes, patients = pd.factorize(frame.patient_id, sort=True)
    patient_folds = frame.assign(code=codes).groupby("code").fold.first()
    pools = [patient_folds[patient_folds == f].index.to_numpy() for f in range(5)]
    indices = {"original": np.arange(len(frame)), "retained": np.flatnonzero(mask)}
    inputs = {}
    for model, p in scores.items():
        _, p = validate_probabilities(y, p)
        inputs["original", model] = p
        inputs["retained", model] = p[mask]
    if clean_scores is not None:
        indices["clean"] = indices["retained"]
        for model, p in clean_scores.items():
            _, inputs["clean", model] = validate_probabilities(y[mask], p)
    caches, estimates = {}, {}
    for key, p in inputs.items():
        idx = indices[key[0]]
        fold_idx = [np.flatnonzero(fold[idx] == f) for f in range(5)]
        if any(len(part) == 0 or len(np.unique(y[idx][part])) != 2 for part in fold_idx):
            raise ValueError("Each population requires both outcomes in every fold.")
        caches[key] = (_cache(y[idx], p), [_cache(y[idx][part], p[part]) for part in fold_idx], fold_idx)
        estimates[key, "pooled"] = _weighted_metrics(caches[key][0], np.ones(len(idx)))
        estimates[key, "mean_fold"] = np.mean([
            _weighted_metrics(cache, np.ones(len(part))) for cache, part in zip(caches[key][1], fold_idx)], axis=0)
    distributions = {(key, agg): np.empty((n_bootstrap, len(METRICS)))
                     for key in caches for agg in ("mean_fold", "pooled")}
    rng = np.random.default_rng(seed)
    valid = 0
    for _ in range(n_bootstrap):
        counts = np.zeros(len(patients), dtype=float)
        for pool in pools:
            counts += np.bincount(rng.choice(pool, len(pool), replace=True), minlength=len(patients))
        all_weights = counts[codes]
        invalid = False
        for population, idx in indices.items():
            for f in range(5):
                local = idx[fold[idx] == f]
                if np.dot(all_weights[local], y[local]) == 0 or np.dot(all_weights[local], 1-y[local]) == 0:
                    invalid = True
                    break
            if invalid:
                break
        if invalid:
            continue
        for key, (pooled_cache, fold_caches, fold_idx) in caches.items():
            weights = all_weights[indices[key[0]]]
            distributions[key, "pooled"][valid] = _weighted_metrics(pooled_cache, weights)
            distributions[key, "mean_fold"][valid] = np.mean([
                _weighted_metrics(cache, weights[part]) for cache, part in zip(fold_caches, fold_idx)], axis=0)
        valid += 1
    if valid < 2:
        raise ValueError(f"Only {valid}/{n_bootstrap} valid paired patient draws.")
    draws = {key: values[:valid] for key, values in distributions.items()}
    common = {"bootstrap_replicates": n_bootstrap, "valid_replicates": valid,
              "rejected_single_class_replicates": n_bootstrap-valid, "seed": seed,
              "interval": "95% percentile, paired whole patients within outer fold; fixed saved models"}
    intervals, comparisons = [], []
    for (key, agg), values in draws.items():
        for k, metric in enumerate(METRICS):
            lo, hi = np.percentile(values[:, k], [2.5, 97.5])
            intervals.append({"population": key[0], "model": key[1], "aggregation": agg, "metric": metric,
                              "estimate": estimates[key, agg][k], "ci_lower": lo, "ci_upper": hi, **common})
    differences = []
    for population in indices:
        for left, right in PAIRS:
            if (population, left) in caches and (population, right) in caches:
                differences.append(("model_comparison", (population, left), (population, right)))
    for model in scores:
        differences.append(("retained_minus_original", ("retained", model), ("original", model)))
    for model in clean_scores or {}:
        if ("retained", model) in caches:
            differences.append(("clean_minus_filtered_historical", ("clean", model), ("retained", model)))
    for kind, left, right in differences:
        for agg in ("mean_fold", "pooled"):
            values = draws[left, agg]-draws[right, agg]
            point = estimates[left, agg]-estimates[right, agg]
            for k, metric in enumerate(METRICS):
                lo, hi = np.percentile(values[:, k], [2.5, 97.5])
                comparisons.append({"kind": kind, "population": left[0],
                    "comparison": f"{left[1]} minus {right[1]}" if kind == "model_comparison" else kind,
                    "model": left[1] if kind != "model_comparison" else "",
                    "aggregation": agg, "metric": metric, "difference": point[k],
                    "ci_lower": lo, "ci_upper": hi, **common})
    return pd.DataFrame(intervals), pd.DataFrame(comparisons)


def _support(rows):
    return {"n_images": len(rows), "n_patients": rows.patient_id.nunique(),
            "n_malignant_images": int(rows.target.sum()),
            "n_malignant_patients": rows.loc[rows.target == 1, "patient_id"].nunique()}


def metric_tables(frame, scores, population):
    """Image-primary and equal-patient secondary point estimates."""
    records = []
    for aggregation, groups in (("by_fold", list(frame.groupby("fold"))), ("pooled", [("pooled", frame)])):
        patient_counts = frame.groupby("patient_id").image_id.transform("size")
        for label, rows in groups:
            for model, values in scores.items():
                p = np.asarray(values)[rows.index]
                for weighting in ("image", "equal_patient"):
                    weights = np.ones(len(rows)) if weighting == "image" else 1/patient_counts.loc[rows.index].to_numpy()
                    values_out = _weighted_metrics(_cache(rows.target.to_numpy(), p), weights)
                    records.append({"population": population, "model": model, "aggregation": aggregation,
                        "fold": label, "weighting": weighting, **_support(rows), **dict(zip(METRICS, values_out))})
    table = pd.DataFrame(records)
    mean = table[table.aggregation == "by_fold"].groupby(["population", "model", "weighting"], as_index=False)[list(METRICS)].mean()
    mean["aggregation"], mean["fold"] = "mean_fold", "mean_fold"
    return pd.concat([table, mean], ignore_index=True)


def subgroup_table(frame, scores, population):
    """Prespecified support-first diagnostics, never substitute zero for undefined."""
    def missing(values):
        normalised = values.astype("string").str.strip().str.casefold()
        return values.isna() | normalised.isin(["", "unknown", "missing", "nan", "none", "null", "n/a", "na"])

    unknown_site = missing(frame.anatom_site_general_challenge)
    sites = frame.anatom_site_general_challenge.astype("string").str.strip().mask(unknown_site, "Missing")
    age = pd.to_numeric(frame.age_approx.mask(missing(frame.age_approx)), errors="raise")
    groups = {
        "site_knownness": pd.Series(np.where(unknown_site, "unknown", "known"), index=frame.index),
        "anatomical_site": sites, "sex": frame.sex.astype("string").str.strip().mask(missing(frame.sex), "Missing"),
        "age_group": pd.cut(age, [-np.inf, 40, 60, np.inf], right=False,
                            labels=["under_40", "40_to_59", "60_plus"]).astype("string").fillna("Missing"),
    }
    records = []
    for axis, labels in groups.items():
        for label in sorted(labels.unique()):
            rows = frame[labels == label]
            for model, p in scores.items():
                records.append({"population": population, "axis": axis, "subgroup": label, "model": model,
                    **_support(rows), **probability_metrics(rows.target, np.asarray(p)[rows.index]),
                    "interpretation": "descriptive pooled subgroup; support can be sparse; no subgroup tuning"})
    return pd.DataFrame(records)


def frozen_decision_table(frame, raw_column, calibrated_column, model, population):
    """Check recorded decisions against frozen cutoffs, then report both scales."""
    records = []
    for score, column, decision, threshold in (("raw", raw_column, "decision_raw", "raw_threshold"),
                                              ("calibrated", calibrated_column, "decision_calibrated", "calibrated_threshold")):
        validate_probabilities(frame.target, frame[column])
        cuts = pd.to_numeric(frame[threshold], errors="raise")
        if not np.isfinite(cuts).all() or not cuts.between(0, 1).all() or frame.groupby("fold")[threshold].nunique().ne(1).any():
            raise ValueError("Frozen cutoffs must be finite [0,1] and constant within each outer fold.")
        if not np.array_equal(frame[decision].to_numpy(), (frame[column] >= cuts).astype(int).to_numpy()):
            raise ValueError("Saved decisions changed or disagree with frozen cutoffs.")
        for f, rows in [*list(frame.groupby("fold")), ("pooled", frame)]:
            counts = decision_metrics(rows.target, rows[decision])
            records.append({"population": population, "model": model, "score_scale": score, "fold": f,
                **_support(rows), **probability_metrics(rows.target, rows[column]), **counts,
                "recall": counts["sensitivity"], "n_flagged": counts["TP"]+counts["FP"],
                "frozen_threshold": rows[threshold].iloc[0] if f != "pooled" else np.nan,
                "cutoff_source": ("clean training-only OOF; frozen during reporting" if population == "clean" else
                                  "original training-only OOF; frozen without deduplicated refitting")})
    return pd.DataFrame(records)


def _checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Receipt path must name an existing file within the project.")
    return path


def _load_original_sources(root, original):
    image = load_saved_nested_run(root)
    calibrated = load_saved_calibration_run(root)
    if not image.get("complete") or not calibrated.get("complete"):
        raise ValueError("Sensitivity requires complete, receipt-verified five-fold saved image runs.")
    metadata = load_matched_metadata(root, original)
    nn_receipt_path = root / "results/metrics/metadata_nn_run_config_d1.json"
    nn_receipt = json.loads(nn_receipt_path.read_text(encoding="utf-8"))
    nn_path = _checked_path(root, nn_receipt["prediction_path"])
    nn = align_predictions(pd.read_csv(nn_path), original,
        ["p_nn_seed_42", "p_nn_seed_142", "p_nn_seed_242", "p_nn_mean", "p_spline"])
    if not np.allclose(nn.p_nn_mean, nn[["p_nn_seed_42", "p_nn_seed_142", "p_nn_seed_242"]].mean(axis=1), rtol=0, atol=1e-14):
        raise ValueError("Metadata NN scores are not the prescribed equal three-seed mean.")
    robust_receipt_path = root / "results/metrics/metadata_robustness_run_config_d1.json"
    robust_receipt = json.loads(robust_receipt_path.read_text(encoding="utf-8"))
    preservation = root / "results/notebook_organization/20261004_subset_full_readability_v1/existing_results_before.json"
    preserved = json.loads(preservation.read_text(encoding="utf-8"))["files"]
    robust_path = root / "results/predictions/metadata_robustness_oof_d1.csv"
    for path in (robust_receipt_path, robust_path):
        if sha256(path) != preserved.get(path.relative_to(root).as_posix()):
            raise ValueError("Saved metadata robustness prediction/receipt changed since preserved historical inventory.")
    if robust_receipt.get("input_sha256") != sha256(root / "data/subset/processed/train_folds.csv"):
        raise ValueError("Robustness receipt refers to a changed development source.")
    robust = align_predictions(pd.read_csv(robust_path), original, ["p_spline_raw", "p_spline_calibrated"])
    if not np.allclose(robust.p_spline_raw, nn.p_spline, rtol=0, atol=1e-14):
        raise ValueError("Historical raw spline sources disagree.")
    # Validate original metrics BEFORE any filtering, not just file identities.
    archived_nn_metrics = pd.read_csv(root / "results/metrics/metadata_nn_metrics_d1.csv").set_index("model")
    for title, column in (("Spline logistic", "p_spline"), ("NN mean of all three", "p_nn_mean")):
        m = probability_metrics(nn.target, nn[column])
        if not np.allclose([m["average_precision"], m["roc_auc"], m["brier"]],
                archived_nn_metrics.loc[title, ["AP", "ROC_AUC", "Brier"]].to_numpy(dtype=float), rtol=1e-10, atol=1e-12):
            raise ValueError("Original metadata metrics do not reproduce saved results.")
    robust_metrics = pd.read_csv(root / "results/metrics/metadata_robustness_fold_metrics_d1.csv")
    for f, rows in robust.groupby("fold"):
        for title, column in (("Raw spline", "p_spline_raw"), ("Recalibrated spline", "p_spline_calibrated")):
            values = robust_metrics[(robust_metrics.fold == f) & (robust_metrics.procedure == title)]
            if len(values) != 1 or not np.allclose(list(probability_metrics(rows.target, rows[column]).values()),
                    values[list(METRICS)].iloc[0].to_numpy(dtype=float), rtol=1e-10, atol=1e-12):
                raise ValueError("Original spline fold metrics do not reproduce saved results.")
    originals = {"image_raw": image["predictions"], "image_calibrated": calibrated["predictions"],
                 "metadata_nn": nn, "metadata_robustness": robust}
    source_paths = [nn_path, nn_receipt_path, robust_path, robust_receipt_path, preservation,
                   root / "results/metrics/metadata_nn_metrics_d1.csv",
                   root / "results/metrics/metadata_robustness_fold_metrics_d1.csv"]
    for bundle in (image, calibrated):
        folder = bundle["run_dir"]
        source_paths.extend([folder / "run_manifest.json", folder / "summary.json", folder / "outer_oof_predictions.csv"])
    receipt = {p.relative_to(root).as_posix(): sha256(p) for p in source_paths}
    scores = {"b0_raw": image["predictions"].base_probability.to_numpy(),
              "spline_raw": metadata["metadata_spline_raw"].base_probability.to_numpy(),
              "metadata_nn_mean": metadata["metadata_mlp_mean"].base_probability.to_numpy()}
    return originals, scores, receipt


def _figure(comparisons, directory):
    rows = comparisons[(comparisons.kind == "model_comparison") &
                       (comparisons.aggregation == "mean_fold") &
                       (comparisons.metric == "average_precision") & comparisons.population.isin(["original", "retained"])]
    labels = [f"{r.comparison} ({r.population})" for r in rows.itertuples()]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.hlines(y, rows.ci_lower, rows.ci_upper, color="#235789", linewidth=1.5)
    ax.scatter(rows.difference, y, color="#235789", zorder=3)
    ax.axvline(0, color="#666666", linestyle="--")
    ax.set(yticks=y, yticklabels=labels, xlabel="Difference in mean-fold AP (95% paired patient interval)",
           title="Saved-model duplicate sensitivity; training influence remains")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(directory / "paired_mean_fold_ap.png", dpi=160)
    plt.close(fig)


def run_sensitivity(project_root, dataset_manifest="data/subset/processed/dedup_v1/dataset_manifest.json",
                    *, run_id="saved_oof_v2", n_bootstrap=2000, seed=42,
                    clean_image_run=None, clean_metadata_receipt=None, clean_calibration_run=None):
    """Emit a new immutable run; a repeat verifies and returns its saved receipt."""
    from .development_dataset import load_development_dataset
    root = Path(project_root).resolve()
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one directory name.")
    original, original_identity = load_development_dataset(root)
    retained, identity = load_development_dataset(root, dataset_manifest)
    original, retained = _canonical(original), _canonical(retained)
    originals, scores, input_hashes = _load_original_sources(root, original)
    manifest_path = _checked_path(root, dataset_manifest)
    input_hashes[manifest_path.relative_to(root).as_posix()] = sha256(manifest_path)
    input_hashes["data/subset/processed/train_folds.csv"] = original_identity["canonical_source_sha256"]
    for relative in ("src/melanoma_detection/dedup_sensitivity.py", "scripts/run_dedup_sensitivity.py"):
        input_hashes[relative] = sha256(root / relative)
    filtered = {}
    columns = {"image_raw": ["base_probability"], "image_calibrated": ["base_probability", "calibrated_probability"],
               "metadata_nn": ["p_nn_seed_42", "p_nn_seed_142", "p_nn_seed_242", "p_nn_mean", "p_spline"],
               "metadata_robustness": ["p_spline_raw", "p_spline_calibrated"]}
    for name, source in originals.items():
        filtered[name] = filter_saved_predictions(source, original, retained, columns[name])
    clean_scores = None
    clean_frames = {}
    if bool(clean_image_run) != bool(clean_metadata_receipt):
        raise ValueError("Clean comparison requires both an explicit image run and metadata receipt.")
    if clean_image_run:
        # New variant receipts are explicit; legacy readers are NOT loosened.
        clean_image = load_saved_nested_run(root, clean_image_run)
        if not clean_image.get("complete"):
            raise ValueError("Clean image comparison requires complete verified five-fold reporting.")
        folder = clean_image["run_dir"]
        manifest = clean_image["manifest"]
        declared = manifest.get("dataset_identity", {})
        if isinstance(declared, str):
            declared = {"dataset_identity": declared}
        if (declared.get("dataset_identity") != identity["dataset_identity"] or
                declared.get("dataset_manifest_sha256") != identity["dataset_manifest_sha256"]):
            raise ValueError("Clean image run has a different dataset identity.")
        pred_path = folder / "outer_oof_predictions.csv"
        clean_frames["image_raw"] = validate_outer_predictions(clean_image["predictions"], retained)
        receipt_path = _checked_path(root, clean_metadata_receipt)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        declared = receipt.get("dataset_identity", receipt.get("dataset", {}))
        identity_text = declared.get("dataset_identity") if isinstance(declared, dict) else declared
        if identity_text != identity["dataset_identity"]:
            raise ValueError("Clean metadata receipt has a different dataset identity.")
        if (receipt.get("status") != "complete" or receipt.get("input_sha256") != identity["canonical_source_sha256"]
                or receipt.get("dataset_manifest_sha256") != identity["dataset_manifest_sha256"]):
            raise ValueError("Clean metadata receipt is incomplete or bound to a different source/manifest.")
        metadata_path = _checked_path(root, receipt["prediction_path"])
        if sha256(metadata_path) != receipt.get("prediction_sha256"):
            raise ValueError("Clean metadata prediction hash changed.")
        clean_frames["metadata_nn"] = align_predictions(pd.read_csv(metadata_path), retained,
            ["p_nn_seed_42", "p_nn_seed_142", "p_nn_seed_242", "p_nn_mean", "p_spline"])
        nn_clean = clean_frames["metadata_nn"]
        if not np.allclose(nn_clean.p_nn_mean, nn_clean[["p_nn_seed_42", "p_nn_seed_142", "p_nn_seed_242"]].mean(axis=1), rtol=0, atol=1e-14):
            raise ValueError("Clean metadata NN changed the equal three-seed ensemble.")
        clean_scores = {"b0_raw": clean_frames["image_raw"].base_probability.to_numpy(),
                        "spline_raw": clean_frames["metadata_nn"].p_spline.to_numpy(),
                        "metadata_nn_mean": clean_frames["metadata_nn"].p_nn_mean.to_numpy()}
        for path in (folder / "run_manifest.json", folder / "summary.json", pred_path, receipt_path, metadata_path):
            input_hashes[path.relative_to(root).as_posix()] = sha256(path)
        if "calibration_prediction_path" in receipt:
            path = _checked_path(root, receipt["calibration_prediction_path"])
            if sha256(path) != receipt.get("calibration_prediction_sha256"):
                raise ValueError("Clean metadata calibration prediction hash changed.")
            clean_frames["metadata_robustness"] = align_predictions(pd.read_csv(path), retained,
                ["p_spline_raw", "p_spline_calibrated"])
            if not np.allclose(clean_frames["metadata_robustness"].p_spline_raw,
                               nn_clean.p_spline, rtol=0, atol=1e-14):
                raise ValueError("Clean metadata calibration changed raw spline predictions.")
            input_hashes[path.relative_to(root).as_posix()] = sha256(path)
    if clean_calibration_run:
        if not clean_image_run:
            raise ValueError("Clean calibration requires the explicit matching clean raw image run.")
        calibration = load_saved_calibration_run(root, clean_calibration_run)
        if not calibration.get("complete") or calibration["manifest"].get("reference_run_id") != clean_image_run:
            raise ValueError("Clean calibration requires complete reporting and the declared clean reference.")
        clean_frames["image_calibrated"] = validate_calibration_predictions(
            calibration["predictions"], clean_frames["image_raw"], retained)
        folder = calibration["run_dir"]
        for path in (folder / "run_manifest.json", folder / "summary.json", folder / "outer_oof_predictions.csv"):
            input_hashes[path.relative_to(root).as_posix()] = sha256(path)
    output = root / "results/dedup_v1/sensitivity" / run_id
    request = {"n_bootstrap": n_bootstrap, "seed": seed, "clean_image_run": clean_image_run,
               "clean_metadata_receipt": clean_metadata_receipt, "clean_calibration_run": clean_calibration_run,
               "dataset_identity": identity["dataset_identity"]}
    completion = output / "completion_receipt.json"
    if output.exists():
        if not completion.is_file():
            raise ValueError("Sensitivity output exists without a completion receipt; use a fresh run_id.")
        saved = json.loads(completion.read_text(encoding="utf-8"))
        if saved.get("input_sha256") != input_hashes or saved.get("request") != request:
            raise ValueError("Immutable sensitivity run input/settings changed; use a fresh run_id.")
        for name, digest in saved["artifact_sha256"].items():
            path = (output / name).resolve()
            if not path.is_relative_to(output.resolve()) or sha256(path) != digest:
                raise ValueError("Saved sensitivity artifact changed.")
        return saved
    intervals, comparisons = paired_population_bootstrap(original, retained.image_id, scores,
        n_bootstrap=n_bootstrap, seed=seed, clean_scores=clean_scores)
    retained_mask = original.image_id.isin(retained.image_id).to_numpy()
    filtered_scores = {key: p[retained_mask] for key, p in scores.items()}
    tables = [metric_tables(original, scores, "original"), metric_tables(retained, filtered_scores, "retained")]
    groups = [subgroup_table(original, scores, "original"), subgroup_table(retained, filtered_scores, "retained")]
    if clean_scores is not None:
        tables.append(metric_tables(retained, clean_scores, "clean"))
        groups.append(subgroup_table(retained, clean_scores, "clean"))
    decisions = []
    for population, source in (("original", originals), ("retained", filtered)):
        decisions.extend([
            frozen_decision_table(source["image_calibrated"], "base_probability", "calibrated_probability", "b0", population),
            frozen_decision_table(source["metadata_robustness"], "p_spline_raw", "p_spline_calibrated", "spline", population)])
    if "image_calibrated" in clean_frames:
        decisions.append(frozen_decision_table(clean_frames["image_calibrated"], "base_probability",
            "calibrated_probability", "b0", "clean"))
    if "metadata_robustness" in clean_frames:
        decisions.append(frozen_decision_table(clean_frames["metadata_robustness"], "p_spline_raw",
            "p_spline_calibrated", "spline", "clean"))
    output.mkdir(parents=True, exist_ok=False)
    for subdir in ("predictions", "tables", "figures"):
        (output / subdir).mkdir()
    for name, frame in {**filtered, **{f"clean_{k}": v for k, v in clean_frames.items()}}.items():
        frame.to_csv(output / "predictions" / f"{name}.csv", index=False)
    emitted = {"metrics.csv": pd.concat(tables, ignore_index=True), "subgroups.csv": pd.concat(groups, ignore_index=True),
               "bootstrap_intervals.csv": intervals, "paired_comparisons.csv": comparisons,
               "frozen_calibration_cutoffs.csv": pd.concat(decisions, ignore_index=True)}
    for name, table in emitted.items():
        table.to_csv(output / "tables" / name, index=False)
    _figure(comparisons, output / "figures")
    primary = emitted["metrics.csv"]
    primary = primary[(primary.aggregation == "mean_fold") & (primary.weighting == "image")]
    summary = {"status": "complete", "dataset": identity,
        "counts": {"original": _support(original), "retained": _support(retained)},
        "primary_mean_fold_metrics": primary[["population", "model", *METRICS]].to_dict("records"),
        "bootstrap_valid": int(intervals.valid_replicates.iloc[0]),
        "bootstrap_rejected_single_class": int(intervals.rejected_single_class_replicates.iloc[0]),
        "training_started": False, "reserved_test_loaded": False, "limitation": LIMITATION,
        "original_metrics_reproduced": True, "retained_scores_unchanged": True}
    (output / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    text = ["# Exact-copy retention sensitivity", "", LIMITATION, "",
            f"Original: {len(original):,} images, {int(original.target.sum())} malignant; retained: {len(retained):,} images, {int(retained.target.sum())} malignant.",
            "Both populations contain the same patients and preserve outer and inner patient assignments.", "",
            "## Primary mean-fold metrics", "", "| Population | Model | AP | ROC-AUC |", "|---|---|---:|---:|"]
    for row in primary.itertuples():
        text.append(f"| {row.population} | {row.model} | {row.average_precision:.6f} | {row.roc_auc:.6f} |")
    text += ["", f"Bootstrap: {summary['bootstrap_valid']}/{n_bootstrap} valid paired patient draws, seed {seed}; invalid draws are shared across all comparisons.",
             "", "See tables/paired_comparisons.csv for the three predeclared model comparisons and retained-minus-original changes.",
             "Equal-patient metrics are secondary: each patient's retained image weights sum to one. Sparse subgroup metrics remain missing when undefined.",
             "Site-knownness treats blank, Unknown, Missing and null sentinels case-insensitively as unknown; feature records are not changed.",
             "Calibration maps and training-selected cutoffs were held fixed; no cutoff or calibrator was fitted using development OOF labels.",
             "", "Clean retraining is a separate stage. The three-condition report can be extended only with explicit matching clean run and metadata receipts."]
    (output / "report.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    for name, digest in input_hashes.items():
        if sha256(root / name) != digest:
            raise ValueError("Historical input changed during sensitivity execution; output is incomplete.")
    receipt = {"schema_version": 1, "status": "complete", "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "request": request,
        "input_sha256": input_hashes, "dataset": identity,
        "artifact_sha256": {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "summary": _json_safe(summary), "reserved_test_loaded": False, "training_started": False}
    completion.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return receipt
