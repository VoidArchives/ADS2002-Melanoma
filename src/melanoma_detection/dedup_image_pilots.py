"""Bounded duration/ConvNeXt pilots on clean preserved inner patient folds.

Requires the completed clean raw and calibrated baseline. Never predicts outer
patients, promotes an architecture or extends the fixed 24-epoch cap.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import yaml

from .development_dataset import (load_development_dataset, load_preserved_inner_memberships,
                                  sha256_file, stable_sha256)
from .image_pilot import _atomic_csv, _atomic_json, _check_predictions
from .metadata_evaluation import probability_metrics, validate_probabilities

CONVNEXT_RECIPES = [{"id": "low_none", "backbone_lr": 1e-5, "loss_weight": "none"},
                   {"id": "low_sqrt", "backbone_lr": 1e-5, "loss_weight": "sqrt_ratio"},
                   {"id": "high_none", "backbone_lr": 3e-5, "loss_weight": "none"},
                   {"id": "high_sqrt", "backbone_lr": 3e-5, "loss_weight": "sqrt_ratio"}]
QUALIFICATION = "Exploratory inner validation; selected best-in-cap scores reuse epoch/recipe-selection patients. Conditional patient bootstrap omits retraining variability and prior development adaptation. No final winner is established."


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported deduplicated pilot schema.")
    protocol = config["protocol"]
    expected = {"outer_training_fold": 0, "inner_folds": [0, 1, 2], "endpoints": [12, 24],
                "screening_epochs": 6, "schedule_epochs": 30, "bootstrap_replicates": 2000, "bootstrap_seed": 42,
                "reporting_reserve_minutes": 10, "initial_epoch_seconds": 70, "epoch_safety_factor": 1.25,
                "reserved_test_allowed": False, "outer_predictions_allowed": False}
    if protocol != expected:
        raise ValueError("Pilot protocol must retain its fixed splits, endpoints, scheduler, bootstrap and evaluation limits.")
    if config["challenger"]["recipes"] != CONVNEXT_RECIPES or config["challenger"]["microbatch"] != 16:
        raise ValueError("ConvNeXt requires the predefined four recipes and microbatch 16.")
    for stage, maximum in [("duration", 90), ("challenger", 120)]:
        run_id = config[stage]["run_id"]
        if not isinstance(run_id, str) or not run_id.endswith("_dedup_v1") or Path(run_id).name != run_id or any(c in run_id for c in "/\\"):
            raise ValueError("Pilot run identities must be safe separate dedup_v1 directories.")
        if config[stage]["session_minutes"] != maximum:
            raise ValueError("Pilot budgets are fixed at 90/120 minutes.")
    if config["duration"]["run_id"] == config["challenger"]["run_id"]:
        raise ValueError("Pilot stages require separate run identities.")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_project_path(root, relative, directory):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to((root / directory).resolve()):
        raise ValueError(f"Pilot path must remain within {directory}.")
    return path


def prerequisite_baseline(root, config, identity):
    from .image_nested_reporting import load_saved_nested_run
    from .image_nested_calibration_reporting import load_saved_calibration_run
    raw = load_saved_nested_run(root, config["run"]["raw_run_id"])
    calibrated = load_saved_calibration_run(root, config["run"]["calibration_run_id"])
    if not raw.get("complete") or not calibrated.get("complete"):
        raise ValueError("Both clean five-fold raw and calibrated baseline receipts must pass before optimisation.")
    for bundle in [raw, calibrated]:
        if bundle["manifest"].get("dataset_identity") != identity:
            raise ValueError("Baseline and pilot must use identical clean development records.")
    if calibrated["manifest"].get("reference_run_id") != config["run"]["raw_run_id"]:
        raise ValueError("Calibration prerequisite must refer to this clean raw baseline.")
    return raw


def prepare_pilot(project_root, config, stage):
    validate_config(config)
    if stage not in {"duration", "challenger"}:
        raise ValueError("Unknown deduplicated pilot stage.")
    root = Path(project_root).resolve()
    frame, identity = load_development_dataset(root, config["run"]["dataset_manifest"])
    raw = prerequisite_baseline(root, config, identity)
    pool = load_preserved_inner_memberships(root, identity, 0).sort_values("image_name").reset_index(drop=True)
    pool["image_path"] = pool.image_name.map(lambda name: root / "data/subset/train" / f"{name}.jpg")
    if pool.fold.eq(0).any():
        raise ValueError("Pilot pool contains excluded outer-fold patients.")
    ledger = pd.read_csv(root / "data/subset/processed/dedup_v1/retention_ledger.csv")
    ledger = ledger[ledger.image_name.isin(pool.image_name)].sort_values("image_name")
    for row in ledger.itertuples(index=False):
        if sha256_file(root / "data/subset/train" / f"{row.image_name}.jpg") != row.source_sha256:
            raise ValueError("Clean pilot image changed after audit freezing.")
    raw_config_path = _safe_project_path(root, config["run"]["raw_config"], "configs")
    raw_config = yaml.safe_load(raw_config_path.read_text())
    if raw_config != raw["manifest"]["config"]:
        raise ValueError("Clean baseline settings changed after fitting.")
    model, training = copy.deepcopy(raw_config["model"]), copy.deepcopy(raw_config["training"])
    if model != {"architecture": "efficientnet_b0", "weights": "IMAGENET1K_V1", "image_size": 224,
                 "dropout": 0.2, "augmentation": "weak", "preprocessing": "none"}:
        raise ValueError("Duration control must match the clean 224px weak B0 baseline.")
    if training["schedule_epochs"] != 30 or training["effective_batch"] != 64:
        raise ValueError("Pilot requires the original fixed 30-epoch schedule and effective batch 64.")
    selection_path = raw["run_dir"] / "folds/0/selection.json"
    selection = _read(selection_path)
    recipe = selection["recipe"]
    if recipe not in raw_config["selection"]["recipes"] or len(selection["confirmation"]) != 3:
        raise ValueError("Baseline recipe must be selected on all three contained inner folds.")
    if stage == "challenger":
        model["architecture"] = "convnext_tiny"
        training["microbatch"] = config["challenger"]["microbatch"]
        duration_dir = root / "results/image_d1_runs" / config["duration"]["run_id"]
        load_complete_pilot(duration_dir)
        duration_manifest = _read(duration_dir / "run_manifest.json")
        if (duration_manifest.get("dataset_identity") != identity
                or duration_manifest.get("stage") != "duration"
                or duration_manifest["compatibility"].get("selected_clean_b0_recipe") != recipe):
            raise ValueError("Matching duration pilot must use this clean dataset and selected B0 recipe.")
    source_files = [Path(__file__), Path(__file__).with_name("image_model.py"), Path(__file__).with_name("image_transforms.py"),
                    Path(__file__).with_name("development_dataset.py")]
    compatibility = {"protocol": "clean_inner_duration_24_v1", "stage": stage,
                     "settings": {"model": model, "training": training, "protocol": config["protocol"],
                                  "screen_recipes": config["challenger"]["recipes"] if stage == "challenger" else None},
                     "inner_memberships": pool.drop(columns="image_path").to_dict("records"),
                     "training_pool_images": ledger[["image_name", "source_sha256", "pixel_sha256"]].to_dict("records"),
                     "source_sha256": {path.relative_to(root).as_posix(): sha256_file(path) for path in source_files},
                     "selected_clean_b0_recipe": recipe, "selection_sha256": sha256_file(selection_path)}
    run_id = config[stage]["run_id"]
    manifest = {"schema_version": 1, "run_id": run_id, "stage": stage, "compatibility": compatibility,
                "compatibility_sha256": stable_sha256(compatibility), "dataset_identity": identity,
                "outer_labels_used": False, "outer_predictions_generated": False, "reserved_test_loaded": False,
                "qualification": QUALIFICATION}
    return {"root": root, "config": config, "stage": stage, "pool": pool, "model": model,
            "training": training, "recipe": recipe, "manifest": manifest,
            "run_dir": root / "results/image_d1_runs" / run_id}


def _job_manifest(prepared, recipe, inner_fold):
    return {"schema_version": 1, "run_id": prepared["manifest"]["run_id"],
            "compatibility_sha256": prepared["manifest"]["compatibility_sha256"], "recipe": recipe,
            "inner_fold": inner_fold, "purpose": "fresh_clean_inner_trajectory",
            "outer_labels_used": False, "epoch_cap": 24}


def _fit(prepared, recipe, inner_fold, epochs, deadline, device):
    from . import image_model as core
    pool = prepared["pool"]
    fitting = pool[pool.inner_fold.ne(inner_fold)].reset_index(drop=True)
    validation = pool[pool.inner_fold.eq(inner_fold)].reset_index(drop=True)
    folder = prepared["run_dir"] / f"inner/{recipe['id']}/{inner_fold}"
    manifest = _job_manifest(prepared, recipe, inner_fold)
    if (folder / "latest.pt").exists() or (folder / "latest.previous.pt").exists():
        core.recover_interrupted_checkpoint(folder / "latest.pt", expected_manifest=manifest)
    protocol = prepared["config"]["protocol"]
    result = core.train_recipe(fitting, validation, prepared["model"], prepared["training"], recipe, folder, manifest,
                               epochs=epochs, device=device, resume=True, deadline=deadline,
                               epoch_admission_seconds=protocol["initial_epoch_seconds"],
                               epoch_admission_safety_factor=protocol["epoch_safety_factor"])
    _atomic_json(folder / "fit_result.json", result)
    return result


def snapshot_endpoint(folder, result, manifest, cap):
    """Freeze final and selected-within-cap checkpoint pairs before continuation."""
    from . import image_model as core
    folder = Path(folder)
    if result["status"] != "complete" or len(result["history"]) != cap:
        raise ValueError("Endpoint requires exactly its completed cap, never a later/partial checkpoint.")
    receipt_path = folder / f"endpoint_receipt_{cap}.json"
    committed = _read(receipt_path) if receipt_path.exists() else None
    if committed and (committed.get("cap") != cap or committed.get("manifest") != manifest):
        raise ValueError("Frozen endpoint receipt identity changed.")
    paths = {}
    for kind, original in [("endpoint", Path(result["latest_checkpoint"])), ("best_in_cap", Path(result["best_checkpoint"]))]:
        checkpoint = core._read_checkpoint(original)
        if checkpoint["manifest"] != manifest or (kind == "endpoint" and checkpoint["epoch"] != cap) or not 1 <= checkpoint["epoch"] <= cap:
            raise ValueError("Snapshot checkpoint epoch or fresh-trajectory identity differs.")
        destination = folder / f"{kind}_{cap}.pt"
        if committed:
            item = committed["checkpoints"][kind]
            if not destination.exists() or sha256_file(destination) != item["sha256"] or sha256_file(original) != item["sha256"]:
                raise ValueError("Immutable endpoint snapshot differs from its frozen receipt.")
        elif not destination.exists() or sha256_file(destination) != sha256_file(original):
            # Only an uncommitted snapshot may be repaired. Verified latest/best
            # are the sources, and atomic replacement avoids partial final files.
            temporary = destination.with_suffix(".pt.copy.tmp")
            shutil.copyfile(original, temporary)
            if sha256_file(temporary) != sha256_file(original):
                raise ValueError("Snapshot copy verification failed.")
            os.replace(temporary, destination)
        sidecar = destination.with_suffix(".pt.sha256.json")
        original_sidecar = original.with_suffix(".pt.sha256.json")
        if committed and (not sidecar.exists() or sidecar.read_bytes() != original_sidecar.read_bytes()):
            raise ValueError("Immutable snapshot sidecar differs from its original checkpoint.")
        if not committed and (not sidecar.exists() or sidecar.read_bytes() != original_sidecar.read_bytes()):
            temporary = sidecar.with_suffix(".json.copy.tmp")
            shutil.copyfile(original_sidecar, temporary)
            os.replace(temporary, sidecar)
        core._read_checkpoint(destination)
        paths[kind] = {"path": str(destination), "sha256": sha256_file(destination), "epoch": checkpoint["epoch"]}
    if not committed:
        _atomic_json(receipt_path, {"cap": cap, "manifest": manifest, "checkpoints": paths})
    return paths


def _predict_snapshots(prepared, recipe, inner_fold, cap, device):
    from . import image_model as core
    folder = prepared["run_dir"] / f"inner/{recipe['id']}/{inner_fold}"
    receipt = _read(folder / f"endpoint_receipt_{cap}.json")
    manifest = _job_manifest(prepared, recipe, inner_fold)
    if receipt["cap"] != cap or receipt["manifest"] != manifest:
        raise ValueError("Saved endpoint receipt identity changed.")
    if set(receipt.get("checkpoints", {})) != {"endpoint", "best_in_cap"}:
        raise ValueError("Endpoint receipt must contain both final and best-in-cap checkpoints.")
    validation = prepared["pool"][prepared["pool"].inner_fold.eq(inner_fold)].reset_index(drop=True)
    exports = {}
    for kind, item in receipt["checkpoints"].items():
        path = Path(item["path"])
        if not path.resolve().is_relative_to(folder.resolve()) or sha256_file(path) != item["sha256"]:
            raise ValueError("Endpoint snapshot path/hash changed.")
        output = folder / f"predictions_{kind}_{cap}.csv"
        previous = receipt.get("predictions", {}).get(kind)
        if output.exists() and previous:
            if sha256_file(output) != previous["sha256"]:
                raise ValueError("Frozen endpoint predictions changed.")
            predictions = pd.read_csv(output)
            _check_predictions(predictions, validation)
        else:
            # A CSV without its committed prediction receipt may follow a kill
            # between CSV and receipt commits. Recreate it from the frozen model.
            model, checkpoint = core.load_model_checkpoint(path, expected_manifest=manifest, device=device)
            if checkpoint["epoch"] != item["epoch"]:
                raise ValueError("Snapshot's epoch differs from its endpoint receipt.")
            predictions = core.predict_frame(model, validation, checkpoint["spec"],
                                              {**checkpoint["training"], "microbatch": checkpoint["microbatch"]}, device=device)
            _check_predictions(predictions, validation)
            predictions["endpoint_cap"], predictions["selection_kind"] = cap, kind
            predictions["run_id"] = prepared["manifest"]["run_id"]
            _atomic_csv(output, predictions)
        if (not predictions.inner_fold.eq(inner_fold).all() or not predictions.endpoint_cap.eq(cap).all()
                or not predictions.selection_kind.eq(kind).all() or not predictions.run_id.eq(prepared["manifest"]["run_id"]).all()):
            raise ValueError("Endpoint prediction inner/cap/selection/run annotations changed.")
        exports[kind] = {"path": str(output), "sha256": sha256_file(output),
                         "metrics": probability_metrics(predictions.target, predictions.base_probability)}
    receipt["predictions"] = exports
    _atomic_json(folder / f"endpoint_receipt_{cap}.json", receipt)
    return exports


def _trajectory(prepared, recipe, inner_fold, deadline, device):
    folder = prepared["run_dir"] / f"inner/{recipe['id']}/{inner_fold}"
    manifest = _job_manifest(prepared, recipe, inner_fold)
    for cap in (12, 24):
        if not (folder / f"endpoint_receipt_{cap}.json").exists():
            result = _fit(prepared, recipe, inner_fold, cap, deadline, device)
            if result["status"] != "complete":
                return None
            snapshot_endpoint(folder, result, manifest, cap)
        _predict_snapshots(prepared, recipe, inner_fold, cap, device)
    return {"inner_fold": inner_fold, "recipe": recipe, "status": "complete",
            "receipts": {str(cap): str(folder / f"endpoint_receipt_{cap}.json") for cap in (12, 24)}}


def paired_inner_bootstrap(left, right, comparison, *, n_bootstrap=2000, seed=42):
    """Shared whole-patient draws within three inner folds, conditional on fits."""
    from .image_nested_reporting import _cache, _weighted_metrics
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 2:
        raise ValueError("Pilot bootstrap requires at least two integer draws.")
    left, right = [frame.sort_values("image_id").reset_index(drop=True) for frame in (left, right)]
    identity = ["image_id", "patient_id", "target", "inner_fold"]
    if not left[identity].equals(right[identity]) or not left.image_id.is_unique or set(left.inner_fold) != {0, 1, 2}:
        raise ValueError("Paired pilot predictions must cover matching retained records across all three inner folds.")
    if not left.groupby("patient_id").inner_fold.nunique().eq(1).all():
        raise ValueError("Pilot patients cross inner folds.")
    y, left_scores = validate_probabilities(left.target, left.base_probability)
    _, right_scores = validate_probabilities(right.target, right.base_probability)
    codes, patients = pd.factorize(left.patient_id, sort=True)
    patient_folds = left.assign(code=codes).groupby("code").inner_fold.first()
    indices = [np.flatnonzero(left.inner_fold.to_numpy() == fold) for fold in range(3)]
    pools = [patient_folds[patient_folds.eq(fold)].index.to_numpy() for fold in range(3)]
    models = [left_scores, right_scores]
    caches = [[_cache(y[index], scores[index]) for index in indices] for scores in models]
    estimates = [np.mean([_weighted_metrics(cache, np.ones(len(index))) for cache, index in zip(model, indices)], axis=0) for model in caches]
    draws = []
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        counts = np.zeros(len(patients))
        for pool in pools:
            counts += np.bincount(rng.choice(pool, len(pool), replace=True), minlength=len(patients))
        weights = counts[codes]
        if any(np.dot(weights[index], y[index]) == 0 or np.dot(weights[index], 1 - y[index]) == 0 for index in indices):
            continue
        values = [np.mean([_weighted_metrics(cache, weights[index]) for cache, index in zip(model, indices)], axis=0) for model in caches]
        draws.append(values[0] - values[1])
    if len(draws) < 2:
        raise ValueError("Insufficient valid whole-patient bootstrap draws.")
    draws = np.asarray(draws)
    return [{"comparison": comparison, "aggregation": "unweighted_mean_inner_fold", "metric": metric,
             "difference": float(estimates[0][index] - estimates[1][index]),
             "ci_lower": float(np.percentile(draws[:, index], 2.5)), "ci_upper": float(np.percentile(draws[:, index], 97.5)),
             "bootstrap_replicates": n_bootstrap, "valid_replicates": len(draws),
             "rejected_single_class_replicates": n_bootstrap - len(draws), "seed": seed, "qualification": QUALIFICATION}
            for index, metric in enumerate(("average_precision", "roc_auc"))]


def load_complete_pilot(run_dir):
    """Require a complete reporting receipt and unchanged endpoint outputs."""
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise ValueError("Matching clean B0 duration pilot has not completed.")
    summary = _read(summary_path)
    if summary.get("status") != "complete" or summary.get("completed_inner_folds") != [0, 1, 2]:
        raise ValueError("Matching clean B0 duration pilot is incomplete.")
    artifacts = summary.get("artifact_sha256", {})
    required = {"run_manifest.json", "evaluation/inner_metrics.csv", "evaluation/paired_patient_comparisons.csv",
                "evaluation/learning_curves.csv", "figures/inner_learning_curves.png",
                *[f"evaluation/predictions_{kind}_{cap}.csv" for kind in ("endpoint", "best_in_cap") for cap in (12, 24)]}
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise ValueError("Complete pilot requires its full nonempty endpoint/report artifact inventory.")
    for relative, digest in artifacts.items():
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir) or sha256_file(path) != digest:
            raise ValueError("Completed pilot artifact changed.")
    manifest = _read(run_dir / "run_manifest.json")
    if (manifest.get("run_id") != run_dir.name or manifest.get("stage") not in {"duration", "challenger"}
            or manifest.get("compatibility_sha256") != stable_sha256(manifest.get("compatibility"))
            or manifest.get("outer_labels_used") is not False or manifest.get("reserved_test_loaded") is not False):
        raise ValueError("Completed pilot manifest identity or evaluation scope is invalid.")
    identity = manifest.get("dataset_identity", {})
    _, current_identity = load_development_dataset(run_dir.parents[2], identity.get("dataset_manifest"))
    if current_identity != identity or identity.get("variant") != "dedup_v1":
        raise ValueError("Completed pilot development dataset identity changed.")
    records = summary.get("records", [])
    if len(records) != 3 or {record.get("inner_fold") for record in records} != {0, 1, 2} or any(record.get("status") != "complete" for record in records):
        raise ValueError("Complete pilot requires exactly three complete inner trajectories.")
    pool = pd.DataFrame(manifest["compatibility"]["inner_memberships"])
    if pool.fold.eq(0).any() or not pool.image_name.is_unique or not pool.groupby("patient_id").inner_fold.nunique().eq(1).all():
        raise ValueError("Completed pilot inner memberships are unsafe.")
    preserved = load_preserved_inner_memberships(run_dir.parents[2], current_identity, 0)
    try:
        if set(pool.columns) != set(preserved.columns):
            raise AssertionError("membership columns differ")
        pd.testing.assert_frame_equal(pool[preserved.columns].sort_values("image_name").reset_index(drop=True),
                                      preserved.sort_values("image_name").reset_index(drop=True), check_dtype=False)
    except AssertionError as error:
        raise ValueError("Completed pilot pool differs from the frozen preserved inner memberships.") from error
    from . import image_model as core
    metric_rows, verified_predictions = [], {}
    for record in records:
        inner, recipe = record["inner_fold"], record["recipe"]
        folder = run_dir / f"inner/{recipe['id']}/{inner}"
        expected = pool[pool.inner_fold.eq(inner)].reset_index(drop=True)
        expected_manifest = _job_manifest({"manifest": manifest}, recipe, inner)
        for cap in (12, 24):
            receipt_path = folder / f"endpoint_receipt_{cap}.json"
            relative = receipt_path.relative_to(run_dir).as_posix()
            if relative not in artifacts:
                raise ValueError("Complete pilot inventory is missing an inner endpoint receipt.")
            receipt = _read(receipt_path)
            if (receipt.get("cap") != cap or receipt.get("manifest") != expected_manifest
                    or set(receipt.get("predictions", {})) != {"endpoint", "best_in_cap"}
                    or set(receipt.get("checkpoints", {})) != {"endpoint", "best_in_cap"}):
                raise ValueError("Complete endpoint receipt identity or prediction inventory differs.")
            for kind in ("endpoint", "best_in_cap"):
                item = receipt["checkpoints"][kind]
                checkpoint_path = Path(item["path"]).resolve()
                expected_path = (folder / f"{kind}_{cap}.pt").resolve()
                if checkpoint_path != expected_path or artifacts.get(expected_path.relative_to(run_dir).as_posix()) != item["sha256"]:
                    raise ValueError("Endpoint checkpoint is outside its declared trajectory or lacks an integrity receipt.")
                checkpoint = core._read_checkpoint(checkpoint_path)
                if (checkpoint["manifest"] != expected_manifest or checkpoint["epoch"] != item["epoch"]
                        or not 1 <= checkpoint["epoch"] <= cap or (kind == "endpoint" and checkpoint["epoch"] != cap)):
                    raise ValueError("Endpoint checkpoint identity/epoch differs from the declared cap.")
                export = receipt["predictions"][kind]
                prediction_path = Path(export["path"]).resolve()
                expected_path = (folder / f"predictions_{kind}_{cap}.csv").resolve()
                if prediction_path != expected_path or artifacts.get(expected_path.relative_to(run_dir).as_posix()) != export["sha256"]:
                    raise ValueError("Endpoint prediction is outside its declared trajectory or lacks an integrity receipt.")
                predictions = pd.read_csv(prediction_path)
                _check_predictions(predictions, expected)
                if (not predictions.inner_fold.eq(inner).all() or not predictions.endpoint_cap.eq(cap).all()
                        or not predictions.selection_kind.eq(kind).all() or not predictions.run_id.eq(manifest["run_id"]).all()):
                    raise ValueError("Endpoint predictions have incorrect inner/cap/selection annotations.")
                verified_predictions.setdefault((kind, cap), []).append(predictions)
                metrics = probability_metrics(predictions.target, predictions.base_probability)
                if any(not np.isclose(metrics[key], export["metrics"][key], rtol=1e-10, atol=1e-12) for key in metrics):
                    raise ValueError("Endpoint metrics disagree with saved prediction scores.")
                metric_rows.append({"inner_fold": inner, "cap": cap, "selection_kind": kind, **metrics})
    stored = pd.read_csv(run_dir / "evaluation/inner_metrics.csv").sort_values(["inner_fold", "cap", "selection_kind"]).reset_index(drop=True)
    computed = pd.DataFrame(metric_rows).sort_values(["inner_fold", "cap", "selection_kind"]).reset_index(drop=True)
    identity_columns = ["inner_fold", "cap", "selection_kind"]
    if not computed[identity_columns].equals(stored[identity_columns]) or not np.allclose(computed[list(metrics)], stored[list(metrics)], rtol=1e-10, atol=1e-12):
        raise ValueError("Pilot metric table disagrees with independently recomputed endpoint predictions.")
    for cap in (12, 24):
        for kind in ("endpoint", "best_in_cap"):
            combined = pd.read_csv(run_dir / f"evaluation/predictions_{kind}_{cap}.csv")
            _check_predictions(combined, pool)
            assigned = combined.set_index("image_id").inner_fold
            if not assigned.reindex(pool.image_name).to_numpy().tolist() == pool.inner_fold.tolist():
                raise ValueError("Combined endpoint predictions changed preserved inner memberships.")
            expected_combined = pd.concat(verified_predictions[kind, cap]).sort_values("image_id").reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(combined.sort_values("image_id").reset_index(drop=True),
                                              expected_combined, check_dtype=False, rtol=1e-12, atol=1e-14)
            except AssertionError as error:
                raise ValueError("Combined endpoint scores/annotations differ from the verified inner predictions.") from error
    return summary


def _finish_report(prepared, records, screens):
    from . import image_model as core
    run_dir = prepared["run_dir"]
    metric_rows, combined, history_rows, artifacts = [], {}, [], {}
    for record in records:
        inner = record["inner_fold"]
        folder = run_dir / f"inner/{record['recipe']['id']}/{inner}"
        checkpoint = core._read_checkpoint(folder / "latest.pt")
        if checkpoint["epoch"] != 24 or len(checkpoint["history"]) != 24:
            raise ValueError("Completed trajectory must stop exactly at epoch 24.")
        history_rows.extend({"inner_fold": inner, **row} for row in checkpoint["history"])
        for cap in (12, 24):
            receipt_path = folder / f"endpoint_receipt_{cap}.json"
            receipt = _read(receipt_path)
            artifacts[receipt_path.relative_to(run_dir).as_posix()] = sha256_file(receipt_path)
            for kind, export in receipt["predictions"].items():
                path = Path(export["path"])
                frame = pd.read_csv(path)
                combined.setdefault((kind, cap), []).append(frame)
                metric_rows.append({"inner_fold": inner, "cap": cap, "selection_kind": kind,
                                    "qualification": QUALIFICATION if kind == "best_in_cap" else "Fixed endpoint inner validation; recipe remains selected.", **export["metrics"]})
                artifacts[path.relative_to(run_dir).as_posix()] = sha256_file(path)
            for item in receipt["checkpoints"].values():
                path = Path(item["path"])
                artifacts[path.relative_to(run_dir).as_posix()] = sha256_file(path)
                artifacts[path.with_suffix(".pt.sha256.json").relative_to(run_dir).as_posix()] = sha256_file(path.with_suffix(".pt.sha256.json"))
    combined = {key: pd.concat(frames).sort_values("image_id").reset_index(drop=True) for key, frames in combined.items()}
    comparisons = []
    for kind in ("endpoint", "best_in_cap"):
        comparisons.extend(paired_inner_bootstrap(combined[kind, 24], combined[kind, 12], f"{kind}:24_minus_12"))
        if prepared["stage"] == "challenger":
            baseline = prepared["root"] / "results/image_d1_runs" / prepared["config"]["duration"]["run_id"]
            load_complete_pilot(baseline)
            b0 = pd.read_csv(baseline / f"evaluation/predictions_{kind}_24.csv")
            comparisons.extend(paired_inner_bootstrap(combined[kind, 24], b0, f"{kind}:convnext_minus_b0_at_24"))
    tables = run_dir / "evaluation"
    _atomic_csv(tables / "inner_metrics.csv", pd.DataFrame(metric_rows))
    _atomic_csv(tables / "paired_patient_comparisons.csv", pd.DataFrame(comparisons))
    _atomic_csv(tables / "learning_curves.csv", pd.DataFrame(history_rows))
    for (kind, cap), frame in combined.items():
        _atomic_csv(tables / f"predictions_{kind}_{cap}.csv", frame)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    histories = pd.DataFrame(history_rows)
    for inner, group in histories.groupby("inner_fold"):
        for axis, metric in zip(axes, ("average_precision", "roc_auc")):
            axis.plot(group.epoch, group[metric], label=f"Inner fold {inner}")
            axis.axvline(12, color="#666666", linestyle="--", linewidth=0.7)
            axis.set(xlabel="Training epoch", ylabel=metric.replace("_", " "))
    axes[0].legend()
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "inner_learning_curves.png", dpi=150)
    plt.close(fig)
    for path in [*tables.glob("*.csv"), figures / "inner_learning_curves.png", run_dir / "run_manifest.json"]:
        artifacts[path.relative_to(run_dir).as_posix()] = sha256_file(path)
    distinct_fit_histories = list(history_rows)
    if prepared["stage"] == "challenger":
        selected_id = records[0]["recipe"]["id"]
        for screen in screens:
            if screen["recipe"]["id"] != selected_id:
                saved = core._read_checkpoint(run_dir / f"inner/{screen['recipe']['id']}/0/latest.pt")
                distinct_fit_histories.extend(saved["history"])
    summary = {"status": "complete", "stage": prepared["stage"], "run_id": prepared["manifest"]["run_id"],
               "completed_inner_folds": [0, 1, 2], "fits": 3 if prepared["stage"] == "duration" else 6,
               "training_epochs": len(distinct_fit_histories), "records": records, "screening": screens,
               "recorded_training_seconds": float(sum(row["epoch_seconds"] for row in distinct_fit_histories)),
               "peak_memory_bytes": int(max(row["peak_memory_bytes"] for row in distinct_fit_histories)),
               "peak_reserved_memory_bytes": int(max(row["peak_reserved_memory_bytes"] for row in distinct_fit_histories)),
               "artifact_sha256": artifacts, "qualification": QUALIFICATION,
               "outer_predictions_generated": False, "reserved_test_loaded": False, "final_winner_selected": False}
    _atomic_json(run_dir / "summary.json", summary)
    return summary


def run_pilot(project_root, config, stage, *, device="cuda", session_file=None, deadline_utc=None):
    from . import image_model as core
    from .image_nested import claim_training_window, select_recipe
    prepared = prepare_pilot(project_root, config, stage)
    run_dir = prepared["run_dir"]
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists() and _read(manifest_path) != prepared["manifest"]:
        raise ValueError("Pilot source/data/settings identity changed; use a distinct experiment.")
    if not manifest_path.exists():
        _atomic_json(manifest_path, prepared["manifest"])
    if (run_dir / "summary.json").exists() and _read(run_dir / "summary.json").get("status") == "complete":
        return load_complete_pilot(run_dir)
    budget_config = {"run": {"run_id": config[stage]["run_id"]}, "budget": {"session_minutes": config[stage]["session_minutes"],
                     "reporting_reserve_minutes": config["protocol"]["reporting_reserve_minutes"]}}
    session_path = Path(session_file) if session_file else run_dir / "session.json"
    if not session_path.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Pilot session must stay within its dedicated run directory.")
    cutoff = claim_training_window(session_path, budget_config)
    if deadline_utc:
        cutoff = min(cutoff, datetime.fromisoformat(deadline_utc))
    deadline = time.monotonic() + max(0, (cutoff - datetime.now(timezone.utc)).total_seconds())
    screens = []
    recipe = prepared["recipe"]
    if stage == "challenger":
        import torch
        cache = _safe_project_path(prepared["root"], config["run"]["challenger_weights_cache"], "models")
        torch.hub.set_dir(str(cache))
        core.pretrained_weight_identity(prepared["model"], require_cached=True)
        for candidate in config["challenger"]["recipes"]:
            folder = run_dir / f"inner/{candidate['id']}/0"
            if (folder / "latest.pt").exists() or (folder / "latest.previous.pt").exists():
                core.recover_interrupted_checkpoint(folder / "latest.pt", expected_manifest=_job_manifest(prepared, candidate, 0))
                checkpoint = core._read_checkpoint(folder / "latest.pt")
                if checkpoint["manifest"] != _job_manifest(prepared, candidate, 0):
                    raise ValueError("Screening checkpoint identity changed.")
                history = checkpoint["history"][:6]
            else:
                history = []
            if len(history) < 6:
                result = _fit(prepared, candidate, 0, 6, deadline, device)
                history = result["history"][:6]
            if len(history) < 6:
                summary = {"status": "incomplete", "stage": stage, "completed_inner_folds": [], "stopping_reason": "screening_budget_exhausted"}
                _atomic_json(run_dir / "summary.json", summary)
                return summary
            best = max(history, key=lambda row: row["average_precision"])
            screens.append({"recipe": candidate, "status": "complete", "completed_epochs": 6,
                            "best_epoch": best["epoch"], "best_metrics": {key: best[key] for key in ("average_precision", "roc_auc", "log_loss", "brier")}})
        recipe = select_recipe(screens)
        _atomic_json(run_dir / "screening.json", screens)
        _atomic_json(run_dir / "selection.json", {"recipe": recipe, "screening_cap": 6,
                                                   "selection_used_only_inner_fold": 0, "outer_labels_used": False})
    records = []
    for inner in range(3):
        record = _trajectory(prepared, recipe, inner, deadline, device)
        if record is None:
            summary = {"status": "incomplete", "stage": stage, "completed_inner_folds": [row["inner_fold"] for row in records],
                       "stopping_reason": "trajectory_budget_exhausted", "qualification": QUALIFICATION,
                       "outer_predictions_generated": False, "reserved_test_loaded": False}
            _atomic_json(run_dir / "summary.json", summary)
            return summary
        records.append(record)
    _finish_report(prepared, records, screens)
    return load_complete_pilot(run_dir)
