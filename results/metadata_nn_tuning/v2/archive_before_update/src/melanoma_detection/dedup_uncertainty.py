"""Clean five-fold B0 uncertainty inference, with no fitting or test access.

Each checkpoint is a completed fixed-duration outer-training refit. Its own
held-out records receive deterministic/MC/TTA inference. Real held-out targets
are joined only after inference, and frozen deterministic scores drive referral.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch
import yaml

from . import image_model as core
from . import image_pilot as pilot
from . import image_uncertainty as inference
from . import image_uncertainty_evaluation as evaluation
from .development_dataset import load_development_dataset, sha256_file, stable_sha256
from .metadata_evaluation import probability_metrics

STATUS = "deduplicated_development_outer_oof_descriptive_uncertainty"
EXPECTED_COUNTS = {"images": 6706, "patients": 422, "malignant_images": 127, "malignant_patients": 88}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _within(path, directory):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Recorded artifact escapes its declared directory.")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def load_config(root, config):
    if not isinstance(config, dict):
        config = yaml.safe_load((Path(root) / config).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported uncertainty configuration.")
    run, settings, assess = config["run"], config["inference"], config["evaluation"]
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", run[key]) for key in ("run_id", "source_run_id")):
        raise ValueError("Unsafe uncertainty/source run ID.")
    if run["output_dir"] != "results/dedup_v1/uncertainty":
        raise ValueError("Uncertainty outputs must have a separate dedup_v1 identity.")
    fixed = {"mc_passes": 30, "batch_size": 64, "amp": False, "seed": 20261004,
             "tta_passes": list(inference.TTA_PASSES), "deterministic_tolerance": 1e-6}
    if settings != fixed:
        raise ValueError("Preserve the declared 30-pass/4-view full-precision uncertainty protocol.")
    expected = {"referral_fractions": list(evaluation.DEFAULT_FRACTIONS), "primary_referral_fraction": .2,
                "referral_seed": 17042, "bootstrap_seed": 20242, "bootstrap_replicates": 2000,
                "random_repeats": 100, "score_source": "base", "classification_cutoff": .5,
                "reserved_test_allowed": False}
    if assess != expected:
        raise ValueError("Preserve the predeclared descriptive referral/bootstrap protocol.")
    return config


def verify_coverage(frame, predictions):
    required = {"image_id", "patient_id", "target", "fold", "base_probability"}
    if not required.issubset(predictions) or not predictions.image_id.is_unique:
        raise ValueError("Saved outer predictions need unique image keys and full identities.")
    if set(predictions.image_id) != set(frame.image_name):
        raise ValueError("Saved predictions must cover exactly the retained image IDs.")
    ordered = predictions.set_index("image_id").loc[frame.image_name].reset_index()
    for name in ("patient_id", "target", "fold"):
        if ordered[name].tolist() != frame[name].tolist():
            raise ValueError(f"Saved outer {name} does not match retained records.")
    inference.compare_saved_predictions(frame, predictions)
    return ordered


def verify_fold_checkpoint(frame, fold, identity, fold_manifest, checkpoint, selection, result, record):
    """Reject wrong folds, historical data, early epochs, and fitting overlap."""
    if (fold_manifest.get("outer_fold") != fold or
            fold_manifest.get("compatibility", {}).get("dataset_identity") != identity):
        raise ValueError("Fold manifest belongs to a different fold or dataset identity.")
    if core.canonical_hash(fold_manifest["compatibility"]) != fold_manifest["compatibility_sha256"]:
        raise ValueError("Fold compatibility identity is corrupt.")
    manifest = checkpoint["manifest"]
    if (manifest.get("purpose") != "outer_training_fixed_duration_refit" or
            manifest.get("outer_fold") != fold or
            manifest.get("fold_compatibility_sha256") != fold_manifest["compatibility_sha256"] or
            manifest.get("run_id") != fold_manifest["run_id"]):
        raise ValueError("Checkpoint must be this clean fold's outer-training refit.")
    duration = selection["refit_epochs"]
    if (selection.get("selection_uses_outer_labels") is not False or
            checkpoint["epoch"] != duration or record["refit_epochs"] != duration or
            result.get("status") != "complete" or result.get("final_epoch") != duration or
            len(checkpoint["history"]) != duration or len(result["history"]) != duration):
        raise ValueError("Checkpoint is incomplete or differs from the fixed selected duration.")
    if core.canonical_hash(checkpoint["history"]) != core.canonical_hash(result["history"]):
        raise ValueError("Refit result history differs from the final checkpoint.")
    if (checkpoint["identity"].get("training_purpose") != "fixed_duration_refit" or
            checkpoint["identity"].get("validation") is not None):
        raise ValueError("Outer checkpoint contains validation-based stopping.")
    for key in ("spec", "training", "recipe", "manifest"):
        if core.canonical_hash(checkpoint[key]) != core.canonical_hash(checkpoint["identity"][key]):
            raise ValueError(f"Detached checkpoint {key} differs from its identity.")
    spec = checkpoint["spec"]
    if (spec["architecture"] != "efficientnet_b0" or spec["weights"] != "IMAGENET1K_V1" or
            spec["preprocessing"] != "none" or spec["augmentation"] != "weak" or
            spec["image_size"] != 224):
        raise ValueError("Uncertainty requires the preserved clean B0 preparation.")
    pretrained = checkpoint.get("pretrained_identity", {})
    if pretrained.get("kind") != "torchvision" or "IMAGENET1K_V1" not in pretrained.get("weight_enum", ""):
        raise ValueError("Outer fit did not record original pinned pretrained initialisation.")
    if core.canonical_hash(checkpoint["identity"]["architecture_policy"]) != core.canonical_hash(core.architecture_policy(spec)):
        raise ValueError("Checkpoint preparation policy differs from current inference.")
    pool = frame.loc[frame.fold.ne(fold), ["image_name", "patient_id", "fold", "target", "image_path"]].reset_index(drop=True)
    held = frame.loc[frame.fold.eq(fold), ["image_name", "patient_id", "fold", "target", "image_path"]].reset_index(drop=True)
    if set(pool.patient_id) & set(held.patient_id):
        raise ValueError("Fitting and held-out patients overlap.")
    if core._frame_identity(pool) != checkpoint["identity"]["fitting"]:
        raise ValueError("Checkpoint fitting identities differ from the retained training pool.")
    if pilot._memberships(held, targets=False) != fold_manifest["held_out_memberships"]:
        raise ValueError("Held-out memberships differ from this checkpoint's fold.")
    members = fold_manifest["compatibility"]["training_pool_memberships"]
    recorded = pd.DataFrame(members).rename(columns={"image_id": "image_name"})
    for column in ("image_name", "patient_id", "fold", "target"):
        if recorded[column].tolist() != pool[column].tolist():
            raise ValueError("Fold fitting memberships differ from retained development records.")
    if (checkpoint["recipe"]["id"] != selection["recipe"]["id"] or
            checkpoint["recipe"]["id"] != record["selected_recipe"]):
        raise ValueError("Checkpoint recipe differs from recorded inner selection.")
    return pool, held


def prepare_uncertainty(root, config, *, smoke=False):
    """Read and verify all required upstream records before writing or inference."""
    root, config = Path(root).resolve(), load_config(root, config)
    frame, identity = load_development_dataset(root, config["run"]["dataset_manifest"])
    counts = {"images": len(frame), "patients": int(frame.patient_id.nunique()),
              "malignant_images": int(frame.target.sum()),
              "malignant_patients": int(frame.loc[frame.target.eq(1), "patient_id"].nunique())}
    if identity.get("variant") != "dedup_v1" or counts != EXPECTED_COUNTS:
        raise ValueError("Uncertainty requires the frozen 6706-image development variant.")
    frame = frame.sort_values("image_name").reset_index(drop=True)
    frame["image_path"] = frame.image_name.map(lambda image: root / "data/subset/train" / f"{image}.jpg")
    source = root / "results/image_d1_runs" / config["run"]["source_run_id"]
    manifest, summary = _json(source / "run_manifest.json"), _json(source / "summary.json")
    if (manifest.get("dataset_identity") != identity or manifest.get("test_set_loaded") is not False or
            summary.get("test_set_loaded") is not False or manifest["run_id"] != config["run"]["source_run_id"]):
        raise ValueError("Source image run has a different dataset identity or test-access status.")
    if core.canonical_hash(manifest["compatibility"]) != manifest["compatibility_sha256"]:
        raise ValueError("Source run compatibility manifest is corrupt.")
    folds = [0] if smoke else list(range(5))
    records = {record["fold"]: record for record in summary["folds"] if record.get("status") == "complete"}
    if len(records) != sum(record.get("status") == "complete" for record in summary["folds"]):
        raise ValueError("Source run has duplicate completed outer-fold records.")
    if not set(folds).issubset(records) or (not smoke and summary.get("status") != "complete"):
        raise ValueError("Clean raw B0 upstream is incomplete; full uncertainty waits for all five completed outer refits.")
    protected = {str(source / name): sha256_file(source / name) for name in
                 (("run_manifest.json",) if smoke else ("run_manifest.json", "summary.json"))}
    combined = None
    if not smoke:
        combined_path = _within(source / "outer_oof_predictions.csv", source)
        if sha256_file(combined_path) != summary["outer_oof_predictions_sha256"]:
            raise ValueError("Source OOF predictions changed since completion.")
        combined = verify_coverage(frame, pd.read_csv(combined_path))
        protected[str(combined_path)] = sha256_file(combined_path)
    fold_data = {}
    for fold in folds:
        folder = source / f"folds/{fold}"
        fold_manifest, selection, result = (_json(folder / name) for name in
                                           ("fold_manifest.json", "selection.json", "refit/fit_result.json"))
        if manifest["compatibility"]["fold_identities"][str(fold)] != fold_manifest["compatibility_sha256"]:
            raise ValueError("Fold identity differs from the full source run.")
        path = _within(records[fold]["checkpoint"], folder / "refit")
        if path.name != "latest.pt" or sha256_file(path) != records[fold]["checkpoint_sha256"]:
            raise ValueError("Recorded final outer checkpoint hash changed.")
        checkpoint = core._read_checkpoint(path)
        pool, held = verify_fold_checkpoint(frame, fold, identity, fold_manifest, checkpoint, selection, result, records[fold])
        for relative, digest in fold_manifest["compatibility"]["source_sha256"].items():
            source_path = _within(root / relative, root / "src")
            if sha256_file(source_path) != digest:
                raise ValueError(f"Training/inference source changed: {relative}")
            protected[str(source_path)] = digest
        predictions_path = _within(folder / "outer_predictions.csv", folder)
        if sha256_file(predictions_path) != records[fold]["predictions_sha256"]:
            raise ValueError("Fold's saved deterministic prediction hash changed.")
        saved = verify_coverage(held, pd.read_csv(predictions_path))
        if combined is not None:
            observed = combined.loc[combined.fold.eq(fold)].sort_values("image_id").reset_index(drop=True)
            pd.testing.assert_frame_equal(observed, saved, check_dtype=False)
        fold_data[fold] = {"checkpoint": path, "manifest": checkpoint["manifest"], "frame": held,
                           "pool": pool, "saved": saved, "folder": folder / "refit",
                           "checkpoint_sha256": records[fold]["checkpoint_sha256"], "record": records[fold]}
        for relative in ("fold_manifest.json", "selection.json", "outer_predictions.csv", "refit/fit_result.json", "refit/latest.pt", "refit/latest.pt.sha256.json"):
            protected[str(folder / relative)] = sha256_file(folder / relative)
        del checkpoint
    ledger = pd.read_csv(root / "data/subset/processed/dedup_v1/retention_ledger.csv")
    image_hashes = ledger.set_index("image_name").source_sha256.to_dict()
    checked_ids = set().union(*(set(data["frame"].image_name) for data in fold_data.values()))
    for row in frame.loc[frame.image_name.isin(checked_ids)].itertuples():
        if sha256_file(row.image_path) != image_hashes[row.image_name]:
            raise ValueError(f"Retained evaluation image bytes changed: {row.image_name}")
    own_sources = {f"src/melanoma_detection/{name}": sha256_file(Path(__file__).with_name(name)) for name in
                   ("dedup_uncertainty.py", "image_uncertainty.py", "image_uncertainty_evaluation.py")}
    receipt_identity = {"config": config, "dataset_identity": identity, "source_run_manifest_sha256": sha256_file(source / "run_manifest.json"),
                        "checkpoint_sha256": {str(fold): data["checkpoint_sha256"] for fold, data in fold_data.items()},
                        "uncertainty_sources": own_sources, "runtime": core._environment("cpu")}
    return {"root": root, "config": config, "frame": frame, "identity": identity, "counts": counts,
            "folds": fold_data, "protected": protected, "receipt_identity": receipt_identity,
            "identity_sha256": stable_sha256(receipt_identity), "source": source}


def load_fold_context(prepared, fold, *, device="cuda"):
    data = prepared["folds"][fold]
    model, checkpoint = core.load_model_checkpoint(data["checkpoint"], device=device, expected_manifest=data["manifest"])
    dropout = [{"name": name, "p": float(module.p)} for name, module in model.named_modules()
               if isinstance(module, inference.DROPOUT_TYPES)]
    if not dropout or any(item["p"] != checkpoint["spec"]["dropout"] for item in dropout):
        raise ValueError("Loaded outer checkpoint does not preserve explicit trained dropout.")
    # Context targets are placeholders; real labels are joined after inference.
    frame, saved = data["frame"].assign(target=0), data["saved"].assign(target=0)
    return {"model": model, "checkpoint": checkpoint, "spec": checkpoint["spec"],
            "training": {**checkpoint["training"], "microbatch": 64, "amp": False},
            "frame": frame, "saved_predictions": saved, "recipe_dir": data["folder"], "device": str(device),
            "provenance": {"source_run_id": prepared["config"]["run"]["source_run_id"], "outer_fold": fold,
                           "checkpoint_sha256": data["checkpoint_sha256"], "checkpoint": str(data["checkpoint"]),
                           "dataset_identity": prepared["identity"], "evaluation_status": STATUS,
                           "trained_dropout_layers": dropout, "saved_prediction_microbatch": checkpoint["microbatch"],
                           "environment": core._environment(device),
                           "real_labels_supplied_to_inference": False, "reserved_test_loaded": False}}


def stratified_patient_indices(frame, rng):
    """Draw the original patient count within every outer fold, with replacement."""
    if frame.groupby("patient_id").fold.nunique().gt(1).any():
        raise ValueError("A patient crosses outer folds.")
    indices = []
    for _, group in frame.groupby("fold", sort=True):
        patients = sorted(group.patient_id.unique())
        draws = rng.integers(len(patients), size=len(patients))
        indices.extend(group.index[group.patient_id.eq(patients[index])].to_numpy() for index in draws)
    return np.concatenate(indices)


def referral_bootstrap(summary, *, fractions=evaluation.DEFAULT_FRACTIONS, n_bootstrap=2000,
                       n_random=100, seed=20242, referral_seed=17042):
    """Paired whole-patient referral differences, resampling within outer folds."""
    frame = evaluation._validate_summary(summary)
    if "fold" not in frame or frame.groupby("patient_id").fold.nunique().gt(1).any():
        raise ValueError("Full OOF referral bootstrap requires preserved patient folds.")
    fractions = evaluation._configuration(fractions, n_random, seed)
    evaluation._positive_integer(n_bootstrap, "n_bootstrap")
    orders = np.stack([np.random.default_rng(referral_seed + i).permutation(len(frame)) for i in range(n_random)])
    absolute = evaluation._bootstrap_metrics(frame, fractions, orders, "base")
    estimate = absolute[:, :1] - absolute[:, 1:]
    draws = np.empty((n_bootstrap, 2, 3, len(fractions), 2))
    rng = np.random.default_rng(seed)
    identities = []
    for replicate in range(n_bootstrap):
        indices = stratified_patient_indices(frame, rng)
        sample = frame.iloc[indices].sort_values("image_id", kind="stable").reset_index(drop=True)
        random_rng = np.random.default_rng(np.random.SeedSequence([seed, replicate, 104729]))
        orders = random_rng.permuted(np.tile(np.arange(len(sample)), (n_random, 1)), axis=1)
        metrics = evaluation._bootstrap_metrics(sample, fractions, orders, "base")
        draws[replicate] = metrics[:, :1] - metrics[:, 1:]
        identities.append({"replicate": replicate, "n_images": len(sample),
                           "patient_multiplicities_sha256": stable_sha256(indices.tolist()),
                           "single_class": sample.target.nunique() < 2})
    rows = []
    for method_index, method in enumerate(("mc", "tta")):
        for comparator_index, comparator in enumerate(("prediction_score", "predictive_entropy", "random")):
            for fraction_index, fraction in enumerate(fractions):
                for metric_index, metric in enumerate(("retained_error", "malignant_referred_fraction")):
                    values = draws[:, method_index, comparator_index, fraction_index, metric_index]
                    valid = values[np.isfinite(values)]
                    lower, upper = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
                    rows.append({"method": method, "ranking": "uncertainty_sd", "comparator": comparator,
                                 "score_column": "base_score", "referral_fraction": fraction, "metric": metric,
                                 "estimate": estimate[method_index, comparator_index, fraction_index, metric_index],
                                 "ci_lower": lower, "ci_upper": upper, "n_requested": n_bootstrap,
                                 "n_valid": len(valid), "n_undefined": n_bootstrap - len(valid),
                                 "n_random_per_replicate": n_random, "bootstrap_seed": seed,
                                 "referral_seed": referral_seed, "resampling": "whole_patients_within_outer_folds"})
    return pd.DataFrame(rows), pd.DataFrame(identities)


def _join_targets(result, held):
    labels = held.set_index("image_name").target
    for name in ("deterministic", "mc_raw", "tta_raw"):
        result[name]["target"] = result[name].image_id.map(labels)
        if result[name].target.isna().any():
            raise ValueError("Inference output has an unexpected held-out image ID.")
    return result


def _fold_summary(data, result):
    expected = data["frame"].rename(columns={"image_name": "image_id"})
    mc = evaluation.aggregate_raw(result["mc_raw"], expected, range(1, 31)).rename(columns={"mean_score": "mc_mean_score", "score_sd": "mc_score_sd", "n_passes": "mc_passes"})
    tta = evaluation.aggregate_raw(result["tta_raw"], expected, inference.TTA_PASSES).rename(columns={"mean_score": "tta_mean_score", "score_sd": "tta_score_sd", "n_passes": "tta_passes"})
    base = data["saved"].rename(columns={"base_probability": "base_score"})[["image_id", "patient_id", "target", "fold", "base_score"]]
    return base.merge(mc.drop(columns=["patient_id", "target"]), on="image_id", validate="one_to_one").merge(
        tta.drop(columns=["patient_id", "target"]), on="image_id", validate="one_to_one")


def _figures(summary, referral, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, method in enumerate(("mc", "tta")):
        selected = referral.loc[referral.method.eq(method)]
        for ranking, group in selected.groupby("ranking"):
            curve = group.groupby("referral_fraction").retained_error.mean()
            axes[index].plot(curve.index, curve.values, marker="o", label=ranking)
        axes[index].set(title=f"{method.upper()} ranking; frozen deterministic scores", xlabel="Fraction of images referred", ylabel="Retained error at descriptive cutoff 0.5")
        axes[index].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "referral_comparison.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, method in zip(axes, ("mc", "tta")):
        axis.hist(summary[f"{method}_score_sd"], bins=40)
        axis.set(title=f"{method.upper()} score variation", xlabel="Uncalibrated score SD", ylabel="Held-out images")
    figure.tight_layout()
    figure.savefig(folder / "score_sd.png", dpi=160)
    plt.close(figure)


def run_uncertainty(root, config="configs/image_uncertainty_dedup.yaml", *, device="cuda", smoke=False, resume=False):
    prepared = prepare_uncertainty(root, config, smoke=smoke)
    root, config = prepared["root"], prepared["config"]
    output = root / config["run"]["output_dir"] / config["run"]["run_id"]
    if smoke:
        context = load_fold_context(prepared, 0, device=device)
        checked = inference.run_smoke(context, n_images=16, seed=config["inference"]["seed"])
        # These inference-only smoke rows used placeholder outcomes.
        checked.pop("malignant_images", None)
        checked.update({"identity_sha256": prepared["identity_sha256"], "evaluation_status": "clean_outer_fold_0_smoke_not_full_oof",
                        "reserved_test_loaded": False, "source": context["provenance"],
                        "preservation": inference.assert_preserved_artifacts(prepared["protected"])})
        destination = output / "smoke" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        _write_json(destination / "receipt.json", checked)
        return {"status": "smoke_complete", "path": str(destination), "images": checked["images"]}
    identity_path = output / "run_identity.json"
    if identity_path.exists():
        if _json(identity_path) != prepared["receipt_identity"]:
            raise ValueError("Uncertainty identity changed; choose a new run identity.")
        if not resume:
            raise FileExistsError("Uncertainty run already exists; use --resume or a new run ID.")
    elif output.exists() and any(item.name != "smoke" for item in output.iterdir()):
        raise ValueError("Nonempty uncertainty directory has no matching identity receipt.")
    _write_json(identity_path, prepared["receipt_identity"])
    start = time.monotonic()
    _write_json(output / "summary.json", {"status": "running", "run_id": config["run"]["run_id"],
                 "dataset_manifest_sha256": prepared["identity"]["dataset_manifest_sha256"], "test_set_loaded": False, "updated_utc": _stamp()})
    summaries, all_mc, all_tta, convergence, checks = [], [], [], [], []
    try:
        for fold, data in prepared["folds"].items():
            folder = output / f"folds/{fold}"
            receipt_path = folder / "receipt.json"
            if receipt_path.exists():
                receipt = _json(receipt_path)
                if receipt["identity_sha256"] != prepared["identity_sha256"] or receipt["status"] != "complete":
                    raise ValueError("Completed uncertainty fold belongs to another identity.")
                for name, digest in receipt["artifacts"].items():
                    if sha256_file(_within(folder / name, folder)) != digest:
                        raise ValueError("Saved uncertainty fold artifact changed.")
            else:
                context = load_fold_context(prepared, fold, device=device)
                print(f"Clean outer fold {fold}: 30 MC passes and four TTA views", flush=True)
                result = inference.run_inference(context, mc_passes=30, seed=config["inference"]["seed"],
                                                 batch_size=64, progress=lambda message: print(f"Fold {fold}: {message}", flush=True))
                result = _join_targets(result, data["frame"])
                summary = _fold_summary(data, result)
                _csv(folder / "prediction_summary.csv", summary)
                _csv(folder / "mc_raw_predictions.csv", result["mc_raw"])
                _csv(folder / "tta_raw_predictions.csv", result["tta_raw"])
                _csv(folder / "deterministic_predictions.csv", result["deterministic"])
                _write_json(folder / "inference_checks.json", result["checks"])
                _write_json(folder / "inference_provenance.json", result["provenance"])
                _write_json(receipt_path, {"status": "complete", "identity_sha256": prepared["identity_sha256"], "outer_fold": fold,
                    "images": len(summary), "artifacts": {path.name: sha256_file(path) for path in folder.iterdir() if path.is_file() and path.name != "receipt.json"}})
                del context, result
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            summary = pd.read_csv(folder / "prediction_summary.csv")
            mc, tta = pd.read_csv(folder / "mc_raw_predictions.csv"), pd.read_csv(folder / "tta_raw_predictions.csv")
            expected = data["frame"].rename(columns={"image_name": "image_id"})
            evaluation.validate_raw(mc, expected, range(1, 31))
            evaluation.validate_raw(tta, expected, inference.TTA_PASSES)
            verify_coverage(data["frame"], summary.rename(columns={"base_score": "base_probability"}))
            aligned = summary.set_index("image_id").loc[data["frame"].image_name, "base_score"].to_numpy()
            inference.compare_saved_predictions(data["frame"], data["saved"], aligned, atol=1e-12)
            summaries.append(summary)
            all_mc.append(mc)
            all_tta.append(tta)
            convergence.append(evaluation.convergence_summary(mc, expected, range(1, 31)).assign(fold=fold))
            checks.append({"fold": fold, **_json(folder / "inference_checks.json")})
        summary = pd.concat(summaries, ignore_index=True).sort_values("image_id").reset_index(drop=True)
        verify_coverage(prepared["frame"], summary.rename(columns={"base_score": "base_probability"}))
        referral = evaluation.evaluate_referral(summary, n_random=100, seed=17042, score_source="base")
        referral_folds = pd.concat([evaluation.evaluate_referral(group, n_random=100, seed=17042, score_source="base").assign(fold=fold)
                                    for fold, group in summary.groupby("fold")], ignore_index=True)
        bootstrap, draws = referral_bootstrap(summary)
        quality = []
        for fold, group in summary.groupby("fold"):
            for mode, column in (("deterministic", "base_score"), ("mc", "mc_mean_score"), ("tta", "tta_mean_score")):
                quality.append({"fold": fold, "mode": mode, "images": len(group), "patients": group.patient_id.nunique(),
                                "malignant_images": int(group.target.sum()), **probability_metrics(group.target, group[column])})
        metrics = pd.DataFrame(quality)
        for name, frame in {"prediction_summary.csv": summary, "mc_raw_predictions.csv": pd.concat(all_mc, ignore_index=True),
                            "tta_raw_predictions.csv": pd.concat(all_tta, ignore_index=True), "fold_metrics.csv": metrics,
                            "mean_fold_metrics.csv": metrics.groupby("mode")[["average_precision", "roc_auc", "log_loss", "brier"]].mean().reset_index(),
                            "mc_convergence.csv": pd.concat(convergence, ignore_index=True), "referral_rankings.csv": referral,
                            "referral_per_fold.csv": referral_folds,
                            "paired_patient_bootstrap_differences.csv": bootstrap, "patient_bootstrap_draws.csv": draws}.items():
            _csv(output / name, frame)
        _figures(summary, referral, output / "figures")
        preservation = inference.assert_preserved_artifacts(prepared["protected"])
        _write_json(output / "inference_checks.json", {"folds": checks, "exact_retained_coverage": True, **preservation})
        artifacts = {path.relative_to(output).as_posix(): sha256_file(path) for path in output.rglob("*") if path.is_file() and
                     path not in (output / "summary.json", output / "receipt.json") and "smoke" not in path.relative_to(output).parts}
        receipt = {"status": "complete", "run_id": config["run"]["run_id"], "evaluation_status": STATUS,
                   "identity_sha256": prepared["identity_sha256"], "dataset_identity": prepared["identity"],
                   "dataset_manifest_sha256": prepared["identity"]["dataset_manifest_sha256"], **prepared["counts"],
                   "completed_outer_folds": list(range(5)), "mc_passes": 30, "tta_passes": 4, "training_fits": 0,
                   "elapsed_seconds": time.monotonic() - start, "test_set_loaded": False, "calibration_fitted": False,
                   "cutoff_fitted": False, "referral_score": "frozen_saved_deterministic_base_score",
                   "primary_referral_fraction": .2, "bootstrap_sampling": "whole_patients_within_outer_folds",
                   "referral_aggregation": "pooled_image_referral_with_per_fold_secondary_diagnostics",
                   "qualification": "Descriptive uncalibrated uncertainty; intervals condition on clean fitted checkpoints, selected procedures and fixed inference draws. No clinical referral validation.",
                   "paths": {"predictions": str(output / "prediction_summary.csv"), "figures": str(output / "figures")},
                   "artifacts": artifacts, "updated_utc": _stamp()}
        _write_json(output / "receipt.json", receipt)
        _write_json(output / "summary.json", receipt)
        return {key: receipt[key] for key in ("status", "run_id", "images", "patients", "elapsed_seconds", "paths")}
    except Exception as error:
        _write_json(output / "summary.json", {"status": "incomplete", "run_id": config["run"]["run_id"],
                    "reason": f"{type(error).__name__}: {error}", "test_set_loaded": False,
                    "dataset_manifest_sha256": prepared["identity"]["dataset_manifest_sha256"], "updated_utc": _stamp()})
        raise
