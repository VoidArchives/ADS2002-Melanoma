"""Bounded, paired six-epoch architecture exploration on the saved B0 split.

The historical pilot is read-only. This stage never evaluates outer patients,
fits calibration, tunes a threshold, or selects a final architecture.
"""

from __future__ import annotations

import copy
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import image_pilot
from .metadata_evaluation import probability_metrics


SCHEMA_VERSION = 1
FAMILIES = ("densenet121", "convnext_tiny")
SLOTS = ("low_none", "low_sqrt", "high_none", "high_sqrt")
REFERENCE_IDS = {"low_none": "lr3e5_none", "low_sqrt": "lr3e5_sqrt",
                 "high_none": "lr1e4_none", "high_sqrt": "lr1e4_sqrt"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_atomic_json = image_pilot._atomic_json
_atomic_csv = image_pilot._atomic_csv
_hash_json = image_pilot._hash_json


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _utc_now():
    return datetime.now(timezone.utc)


def _utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Session deadlines require timezone-aware UTC timestamps.")
    return result.astimezone(timezone.utc)


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported architecture-pilot schema.")
    for key in ("run", "model", "training", "pilot", "families", "evaluation"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Architecture configuration requires {key}.")
    run = config["run"]
    if run.get("mode") != "architecture_pilot" or not _SAFE_ID.fullmatch(str(run.get("run_id", ""))):
        raise ValueError("Architecture pilot requires a safe separate run_id.")
    expected = {"weights": "IMAGENET1K_V1", "image_size": 224, "dropout": 0.2,
                "augmentation": "weak", "preprocessing": "none", "interpolation": "bilinear"}
    if config["model"] != expected:
        raise ValueError("This stage holds weak/unchanged whole-image preparation fixed.")
    if config["pilot"].get("epochs") != 6 or not 0 < config["pilot"].get("budget_minutes", 0) <= 50:
        raise ValueError("Architecture pilot is six epochs and at most 50 active minutes.")
    if config.get("recipe_order") != list(SLOTS):
        raise ValueError("Predefined matched-pair order must not change.")
    expected_families = {"densenet121": {"microbatch": 32, "backbone_lr": [3e-5, 1e-4]},
                         "convnext_tiny": {"microbatch": 16, "backbone_lr": [1e-5, 3e-5]}}
    if config["families"] != expected_families:
        raise ValueError("Family-specific recipe library must remain predefined.")
    training = config["training"]
    expected_training = {"effective_batch": 64, "min_microbatch": 4, "head_lr": 0.001,
                         "weight_decay": 0.0001, "warmup_epochs": 1, "schedule_epochs": 30,
                         "min_lr_factor": 0.1, "gradient_clip": 1.0, "amp": True,
                         "num_workers": 0, "seed": 42}
    if training != expected_training:
        raise ValueError("Shared training controls must match the declared pilot.")
    if config["evaluation"].get("outer_predictions_allowed") is not False or config["evaluation"].get("reserved_test_allowed") is not False:
        raise ValueError("Outer and reserved-test prediction are outside this stage.")


def _path(root: Path, relative: str, allowed: Path) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Run paths must be nonblank relative paths.")
    result = (root / relative).resolve()
    if not result.is_relative_to(allowed.resolve()):
        raise ValueError(f"Path must remain within {allowed}.")
    return result


def candidate_trials(config: dict) -> list[dict]:
    """Return paired order; LR rank is family-specific, not copied from B0."""
    trials = []
    for slot in SLOTS:
        rank, loss = slot.split("_")
        for family in FAMILIES:
            rate = config["families"][family]["backbone_lr"][int(rank == "high")]
            trials.append({"architecture": family, "recipe_id": slot, "recipe_slot": slot,
                           "pipeline_id": f"{family}/{slot}", "backbone_lr": rate,
                           "loss_weight": "none" if loss == "none" else "sqrt_ratio",
                           "microbatch": config["families"][family]["microbatch"]})
    return trials


def _reference_records(reference_dir, summary, validation):
    records, hashes = [], {}
    lookup = {item["recipe_id"]: item for item in summary["recipes"]}
    for slot, recipe_id in REFERENCE_IDS.items():
        item = lookup[recipe_id]
        folder = reference_dir / "recipes" / recipe_id
        if item["status"] != "complete" or len(item["history"]) != 6:
            raise ValueError("B0 reference requires all four completed six-epoch fits.")
        predictions_file = folder / "inner_validation_predictions.csv"
        predictions = pd.read_csv(predictions_file, dtype={"image_id": str, "patient_id": str})
        image_pilot._check_predictions(predictions, validation)
        metrics = probability_metrics(predictions.target, predictions.base_probability)
        if any(not np.isclose(metrics[key], item["best_metrics"][key], rtol=0, atol=1e-12) for key in metrics):
            raise ValueError("Historical prediction scores differ from recorded B0 metrics.")
        for name in ("inner_validation_predictions.csv", "history.csv", "recipe_result.json", "best.pt", "best.pt.sha256.json"):
            path = folder / name
            hashes[str(path.relative_to(reference_dir))] = image_pilot.file_sha256(path)
        if _read_json(folder / "best.pt.sha256.json")["sha256"] != hashes[str((folder / "best.pt").relative_to(reference_dir))]:
            raise ValueError("Historical B0 checkpoint hash does not match its sidecar.")
        records.append({**copy.deepcopy(item), "architecture": "efficientnet_b0", "recipe_slot": slot,
                        "pipeline_id": f"efficientnet_b0/{slot}", "completed_epochs": 6,
                        "predictions_file": str(predictions_file), "provenance": "historical_b0_pilot"})
    for name in ("run_manifest.json", "summary.json", "split_memberships.json"):
        hashes[name] = image_pilot.file_sha256(reference_dir / name)
    return records, hashes


def prepare_architecture_pilot(project_root: str | Path, config: dict, *, audit_images: bool = True) -> dict:
    """Read-only preparation with exact historical split/content verification."""
    validate_config(config)
    root = Path(project_root).resolve()
    import torch
    from .image_model import architecture_policy, pretrained_weight_identity
    weights_cache = _path(root, config["run"]["weights_cache"], root / "models")
    torch.hub.set_dir(str(weights_cache))
    pilot_path = _path(root, config["run"]["pilot_config"], root / "configs")
    reference_dir = _path(root, config["run"]["reference_run"], root / "results/image_d1_runs")
    historical = _read_json(reference_dir / "run_manifest.json")
    prepared = image_pilot.prepare_pilot(root, image_pilot.load_config(pilot_path), audit_images=audit_images)
    old, current = historical["compatibility"], prepared["manifest"]["compatibility"]
    for key in ("training_pool_memberships", "training_pool_images"):
        if _hash_json(old[key]) != _hash_json(current[key]):
            raise ValueError(f"Current {key} differs from the historical B0 reference.")
    if old["config"]["model"] != current["config"]["model"]:
        raise ValueError("Historical B0 preparation settings have changed.")
    references, reference_hashes = _reference_records(reference_dir, _read_json(reference_dir / "summary.json"), prepared["validation"])
    output = _path(root, config["run"]["output_dir"], root / "results/image_d1_runs") / config["run"]["run_id"]
    compatibility_config = copy.deepcopy(config)
    # Explicit future session allocations can resume the same training identity.
    compatibility_config["run"].pop("session_file", None)
    compatibility_config["pilot"].pop("budget_minutes", None)
    source_files = ("image_model.py", "image_architecture.py", "image_pilot.py", "metadata_evaluation.py", "metadata_validation.py")
    sources = {f"src/melanoma_detection/{name}": image_pilot.file_sha256(Path(__file__).with_name(name)) for name in source_files}
    compatibility = {"schema_version": 1, "config": compatibility_config,
                     "source_sha256": sources, "runtime": current["runtime"],
                     "training_pool_memberships": current["training_pool_memberships"],
                     "training_pool_images": current["training_pool_images"], "images_audited": audit_images,
                     "reference_artifact_sha256": reference_hashes,
                     "selection_algorithm_version": "paired_six_epoch_architecture_exploration_v1",
                     "preparation": {"order": ["decode_RGB", "whole_image_resize_224_bilinear_antialias", "unchanged",
                                                "weak_training_only", "tensor", "ImageNet_normalisation"],
                                     "evaluation_deterministic": True, "centre_crop": False},
                     "candidate_library": candidate_trials(config)}
    compatibility["architecture_policies"] = {
        family: architecture_policy({**config["model"], "architecture": family}) for family in FAMILIES}
    compatibility["pretrained_weights"] = {
        family: pretrained_weight_identity({**config["model"], "architecture": family}, require_cached=False)
        for family in FAMILIES}
    manifest = {"schema_version": 1, "run_id": config["run"]["run_id"], "mode": "architecture_pilot",
                "compatibility": compatibility, "compatibility_sha256": _hash_json(compatibility),
                "reference_run": str(reference_dir), "split_audit": prepared["split_audit"],
                "evaluation_status": "exploratory_inner_validation_not_outer_oof",
                "calibration_fitted": False, "threshold_tuned": False,
                "outer_predictions_generated": False, "test_set_loaded": False}
    reporting = Path(__file__).with_name("image_evaluation.py")
    if reporting.exists():
        # Figure/report edits do not change numerical training compatibility.
        manifest["reporting_provenance"] = {"src/melanoma_detection/image_evaluation.py": image_pilot.file_sha256(reporting)}
    return {**prepared, "manifest": manifest, "run_dir": output, "b0_reference": references,
            "config": copy.deepcopy(config)}


def _pending(trial, run_dir):
    folder = run_dir / "families" / trial["architecture"] / "recipes" / trial["recipe_id"]
    return {**trial, "status": "pending", "completed_epochs": 0, "history": [],
            "best_epoch": None, "best_metrics": None, "predictions_file": None,
            "output_dir": str(folder), "elapsed_seconds": 0.0}


def _is_complete(item):
    return item.get("status") == "complete" and item.get("completed_epochs") == 6 and bool(item.get("predictions_file"))


def _summarise(prepared, records, *, reason=None):
    matched = [slot for slot in SLOTS if all(any(item["architecture"] == family and item["recipe_slot"] == slot and _is_complete(item)
                for item in records) for family in FAMILIES)]
    unmatched = [item["pipeline_id"] for item in records if _is_complete(item) and item["recipe_slot"] not in matched]
    complete = len(records) == 8 and all(_is_complete(item) for item in records)
    timings = {}
    for family in FAMILIES:
        rows = [row for item in records if item["architecture"] == family for row in item.get("history", [])]
        seconds = [row["epoch_seconds"] for row in rows if "epoch_seconds" in row]
        timings[family] = {"measured_mean_epoch_seconds": float(np.mean(seconds)) if seconds else None,
                           "measured_max_epoch_seconds": max(seconds) if seconds else None,
                           "measured_epochs": len(seconds),
                           "peak_allocated_memory_bytes": max((row.get("peak_memory_bytes", 0) for row in rows), default=None)}
    return {"schema_version": 1, "run_id": prepared["manifest"]["run_id"],
            "status": "complete" if complete else "incomplete", "comparison_complete": complete,
            "evaluation_status": "exploratory_inner_validation_not_outer_oof",
            "selection_status": "six_epoch_exploration_only", "selected_architecture": None,
            "recipes": records, "b0_reference": prepared["b0_reference"],
            "matched_recipe_slots": matched, "unmatched_completed_trials": unmatched,
            "runtime_estimates": timings, "stopping_reason": reason,
            "recommendation": "Complete paused trials, then independently validate every family with longer training and three inner folds; short-pilot scores cannot eliminate a slower family.",
            "classification_threshold": 0.5, "calibration_fitted": False, "threshold_tuned": False,
            "outer_predictions_generated": False, "test_set_loaded": False,
            "updated_utc": _utc_now().isoformat()}


def persist_architecture_preparation(prepared: dict) -> Path:
    run_dir = Path(prepared["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_manifest.json"
    if path.exists():
        if _read_json(path).get("compatibility_sha256") != prepared["manifest"]["compatibility_sha256"]:
            raise ValueError("Run identity has incompatible source, data, reference artifacts or settings.")
    else:
        _atomic_json(path, prepared["manifest"])
        _atomic_json(run_dir / "split_memberships.json", {
            "training_pool": image_pilot._memberships(prepared["inner_memberships"]),
            "outer_excluded": image_pilot._memberships(prepared["outer_excluded"], targets=False)})
        _atomic_json(run_dir / "image_audit.json", prepared["image_audit"])
    if not (run_dir / "summary.json").exists():
        records = [_pending(trial, run_dir) for trial in candidate_trials(prepared["config"])]
        _atomic_json(run_dir / "summary.json", _summarise(prepared, records, reason="audit_only_not_started"))
    return run_dir


def _session_path(root, config, session_path=None):
    if session_path is not None:
        result = Path(session_path).resolve()
        if not result.is_relative_to((root / "results/image_d1_runs").resolve()):
            raise ValueError("Session file must remain inside image run outputs.")
        return result
    return _path(root, config["run"]["session_file"], root / "results/image_d1_runs")


def claim_training_window(session_path: str | Path, *, now=None) -> datetime:
    """A resumed invocation cannot reset its session's first-start deadline."""
    path = Path(session_path)
    session = _read_json(path)
    moment = _utc_now() if now is None else now.astimezone(timezone.utc)
    for key in ("started_utc", "session_deadline_utc", "latest_training_start_utc", "training_cutoff_utc"):
        _utc(session[key])
    if not (_utc(session["started_utc"]) <= _utc(session["latest_training_start_utc"])
            < _utc(session["training_cutoff_utc"]) <= _utc(session["session_deadline_utc"])):
        raise ValueError("Session deadlines are inconsistent.")
    budget = float(session["max_training_minutes"])
    if not 0 < budget <= 50:
        raise ValueError("Session training allowance must be positive and at most 50 minutes.")
    first = session.get("training_started_utc")
    if first is None:
        if moment >= _utc(session["latest_training_start_utc"]):
            raise TimeoutError("Latest training start has passed; finish verified reporting without training.")
        first = moment.isoformat()
        session.update({"training_started": True, "training_started_utc": first, "status": "training"})
    deadline = min(_utc(session["training_cutoff_utc"]), _utc(first) + timedelta(minutes=budget))
    if moment >= deadline:
        raise TimeoutError("This session's training allowance is exhausted; an explicit new session file is required to resume.")
    session["training_window_deadline_utc"] = deadline.isoformat()
    _atomic_json(path, session)
    return deadline


def _trial_manifest(prepared, trial):
    return {"schema_version": 1, "run_id": prepared["manifest"]["run_id"],
            "stage": "exploratory_architecture_six_epoch_pilot",
            "compatibility_sha256": prepared["manifest"]["compatibility_sha256"],
            "architecture": trial["architecture"], "recipe_slot": trial["recipe_slot"],
            "recipe_sha256": _hash_json(trial)}


def _prediction_export(prepared, config, trial, record, *, device, manifest):
    from .image_model import load_model_checkpoint, predict_frame
    checkpoint = record.get("best_checkpoint")
    if not checkpoint:
        return record
    model, _ = load_model_checkpoint(checkpoint, device=device, expected_manifest=manifest)
    spec = {**config["model"], "architecture": trial["architecture"]}
    training = {**config["training"], "microbatch": record["microbatch"]}
    predictions = predict_frame(model, prepared["validation"], spec, training, device)
    for key, value in {"inner_fold": 0, "split": "inner_validation", "run_id": config["run"]["run_id"],
                       "recipe_id": trial["recipe_id"], "recipe_slot": trial["recipe_slot"],
                       "pipeline_id": trial["pipeline_id"], "architecture": trial["architecture"],
                       "evaluation_status": "exploratory_not_outer_oof"}.items():
        predictions[key] = value
    predictions = predictions.sort_values("image_id").reset_index(drop=True)
    image_pilot._check_predictions(predictions, prepared["validation"])
    folder = Path(record["output_dir"])
    path = folder / "inner_validation_predictions.csv"
    _atomic_csv(path, predictions)
    _atomic_csv(folder / "reliability.csv", image_pilot.reliability_table(
        predictions.target, predictions.base_probability, groups=predictions.patient_id))
    del model
    return {**record, "predictions_file": str(path),
            "raw_validation_metrics": probability_metrics(predictions.target, predictions.base_probability)}


def _remaining_pair_seconds(records, slot, config):
    """Use each family's measured times; B0 timing never substitutes for them."""
    total = 0.0
    for family in FAMILIES:
        record = next(item for item in records if item["architecture"] == family and item["recipe_slot"] == slot)
        if _is_complete(record):
            continue
        measured = [row["epoch_seconds"] for item in records if item["architecture"] == family
                    for row in item.get("history", []) if "epoch_seconds" in row]
        estimate = max(measured) if measured else float(config["pilot"]["initial_epoch_reserve_seconds"])
        total += (6 - record.get("completed_epochs", 0)) * estimate + float(config["pilot"]["checkpoint_reserve_seconds"])
    return total


def run_architecture_pilot(project_root: str | Path, config: dict, *, resume=True,
                           predict_only=False, prepared=None, device="cuda", session_path=None,
                           deadline_utc=None) -> dict:
    """Run matched pairs or predict saved checkpoints; outer patients stay excluded."""
    from .image_model import train_recipe
    root = Path(project_root).resolve()
    import torch
    torch.hub.set_dir(str(_path(root, config["run"]["weights_cache"], root / "models")))
    validate_config(config)
    prepared = prepared if prepared is not None else prepare_architecture_pilot(root, config)
    if prepared["image_audit"]["status"] != "complete":
        raise ValueError("Architecture execution requires the complete image audit.")
    run_dir = persist_architecture_preparation(prepared)
    records = _read_json(run_dir / "summary.json")["recipes"]
    trials = candidate_trials(config)
    if predict_only:
        results, missing = [], []
        for trial in trials:
            record = next(item for item in records if item["pipeline_id"] == trial["pipeline_id"])
            if not record.get("best_checkpoint"):
                missing.append(trial["pipeline_id"])
                continue
            results.append(_prediction_export(prepared, config, trial, record, device=device,
                                             manifest=_trial_manifest(prepared, trial)))
        receipt = {"mode": "predict_only", "status": "complete" if not missing else "incomplete",
                   "recipes": results, "missing_checkpoints": missing, "training_history_preserved": True,
                   "outer_predictions_generated": False, "test_set_loaded": False}
        _atomic_json(run_dir / "prediction_receipt.json", receipt)
        return receipt
    path = _session_path(root, config, session_path)
    deadline_wall = claim_training_window(path) if deadline_utc is None else _utc(deadline_utc)
    # Supplied worker deadlines must still obey the persisted first-start session.
    deadline_wall = min(deadline_wall, claim_training_window(path))
    deadline = time.monotonic() + max(0.0, (deadline_wall - _utc_now()).total_seconds())
    reason = None
    for slot in SLOTS:
        pair_records = [item for item in records if item["recipe_slot"] == slot]
        if all(_is_complete(item) for item in pair_records):
            continue
        if _remaining_pair_seconds(records, slot, config) > deadline - time.monotonic():
            reason = "insufficient_measured_time_for_next_matched_pair"
            break
        for trial in (item for item in trials if item["recipe_slot"] == slot):
            index = next(index for index, item in enumerate(records) if item["pipeline_id"] == trial["pipeline_id"])
            if _is_complete(records[index]):
                continue
            folder = Path(records[index]["output_dir"])
            spec = {**config["model"], "architecture": trial["architecture"]}
            training = {**config["training"], "microbatch": trial["microbatch"]}
            recipe = {"id": trial["recipe_id"], "backbone_lr": trial["backbone_lr"], "loss_weight": trial["loss_weight"]}
            manifest = _trial_manifest(prepared, trial)
            records[index] = {**records[index], "status": "running"}
            _atomic_json(run_dir / "summary.json", _summarise(prepared, records, reason="running"))
            _atomic_json(run_dir / "work_ledger.json", {"recipes": records, "current_pipeline": trial["pipeline_id"],
                                                       "deadline_utc": deadline_wall.isoformat(), "session_file": str(path)})
            measured = [row["epoch_seconds"] for item in records if item["architecture"] == trial["architecture"]
                        for row in item.get("history", []) if "epoch_seconds" in row]
            reserve = (max(measured) + config["pilot"]["checkpoint_reserve_seconds"] if measured
                       else config["pilot"]["initial_epoch_reserve_seconds"])
            print(f"Architecture {trial['pipeline_id']}: LR={trial['backbone_lr']}; loss={trial['loss_weight']}", flush=True)
            try:
                result = train_recipe(prepared["fitting"], prepared["validation"], spec, training, recipe,
                                      folder, manifest, epochs=6, device=device, deadline=deadline,
                                      resume=resume, epoch_admission_seconds=reserve)
                record = {**records[index], **image_pilot._json_safe(result),
                          "completed_epochs": len(result["history"])}
                record = _prediction_export(prepared, config, trial, record, device=device, manifest=manifest)
            except Exception as error:
                record = {**records[index], "status": "failed", "failure": f"{type(error).__name__}: {error}"}
                records[index] = record
                _atomic_json(folder / "recipe_result.json", record)
                _atomic_json(run_dir / "summary.json", _summarise(prepared, records, reason="explicit_failure"))
                _atomic_json(run_dir / "work_ledger.json", {"recipes": records, "current_pipeline": None,
                                                           "stopping_reason": "explicit_failure"})
                raise
            records[index] = record
            _atomic_json(folder / "recipe_result.json", record)
            _atomic_json(run_dir / "summary.json", _summarise(prepared, records))
            _atomic_json(run_dir / "work_ledger.json", {"recipes": records, "current_pipeline": None,
                                                       "deadline_utc": deadline_wall.isoformat()})
            if not _is_complete(record):
                reason = record["status"]
                break
        if reason:
            break
    summary = _summarise(prepared, records, reason=reason)
    _atomic_json(run_dir / "summary.json", summary)
    _atomic_json(run_dir / "work_ledger.json", {"recipes": records, "current_pipeline": None,
                                               "deadline_utc": deadline_wall.isoformat(), "stopping_reason": reason})
    return summary


def record_watchdog_pause(run_dir: str | Path, *, stopping_reason="watchdog_deadline") -> dict:
    """Record a hard stop and verify checkpoint bytes without loading a model."""
    import torch
    run_dir = Path(run_dir)
    summary = _read_json(run_dir / "summary.json")
    checks = []
    for record in summary["recipes"]:
        if record["status"] != "running":
            continue
        folder = Path(record["output_dir"])
        latest = folder / "latest.pt"
        verified = {}
        for name in ("latest", "best"):
            checkpoint = folder / f"{name}.pt"
            sidecar = checkpoint.with_suffix(".pt.sha256.json")
            verified[name] = bool(checkpoint.exists() and sidecar.exists() and
                                  _read_json(sidecar).get("sha256") == image_pilot.file_sha256(checkpoint))
        if verified["latest"]:
            saved = torch.load(latest, map_location="cpu", weights_only=True)
            record.update({"history": saved["history"], "completed_epochs": saved["epoch"],
                           "best_epoch": saved["best_epoch"], "best_metrics": saved["best_metrics"],
                           "microbatch": saved["microbatch"], "latest_checkpoint": str(latest),
                           "best_checkpoint": str(folder / "best.pt") if verified["best"] else None})
        record["status"] = "budget_exhausted" if verified["latest"] else "checkpoint_recovery_required"
        record["stopping_reason"] = stopping_reason
        _atomic_json(folder / "recipe_result.json", record)
        checks.append({"pipeline_id": record["pipeline_id"], **verified})
    summary.update({"status": "incomplete", "comparison_complete": False, "stopping_reason": stopping_reason,
                    "checkpoint_checks": checks, "updated_utc": _utc_now().isoformat()})
    _atomic_json(run_dir / "summary.json", summary)
    _atomic_json(run_dir / "watchdog_receipt.json", {"stopping_reason": stopping_reason, "checkpoint_checks": checks})
    return summary


__all__ = ["load_config", "validate_config", "candidate_trials", "prepare_architecture_pilot",
           "persist_architecture_preparation", "run_architecture_pilot", "claim_training_window",
           "record_watchdog_pause"]
