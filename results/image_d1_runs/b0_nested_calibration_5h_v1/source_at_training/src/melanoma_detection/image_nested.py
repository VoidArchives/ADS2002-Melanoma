"""Bounded five-outer/three-inner patient-grouped B0 development evaluation.

The four recipes are screened independently inside each outer-training pool.
Only the screened recipe is confirmed for 12 epochs on all three inner folds.
The median best epoch fixes a fresh refit on all outer-training patients.
Held-out labels enter only the post-refit evaluation. This intentionally bounded
protocol is separate from the multi-day architecture/transform search.
"""
from __future__ import annotations

import copy
import csv
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from . import image_pilot as pilot
from .metadata_evaluation import probability_metrics
from .metadata_validation import validate_binary_target

PROTOCOL = "bounded_b0_screen_confirm_v1"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_atomic_json, _atomic_csv = pilot._atomic_json, pilot._atomic_csv
file_sha256, _hash_json = pilot.file_sha256, pilot._hash_json


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _utc_now():
    return datetime.now(timezone.utc)


def _utc(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Budget timestamps must include a timezone.")
    return result.astimezone(timezone.utc)


def validate_config(config):
    from .image_model import validate_spec, validate_training
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected nested-image schema_version 1.")
    for name in ("run", "model", "training", "selection", "budget", "evaluation"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"Missing {name} configuration.")
    if config["run"].get("mode") != "nested_b0" or not _SAFE.fullmatch(str(config["run"].get("run_id", ""))):
        raise ValueError("Nested B0 requires a safe run_id and mode nested_b0.")
    spec = validate_spec(config["model"])
    expected_spec = {"architecture": "efficientnet_b0", "weights": "IMAGENET1K_V1",
                     "image_size": 224, "dropout": 0.2, "augmentation": "weak", "preprocessing": "none"}
    if any(spec[key] != value for key, value in expected_spec.items()):
        raise ValueError("This bounded evaluation keeps the declared B0 weak/none preparation.")
    training = validate_training(config["training"])
    if training["schedule_epochs"] != 30:
        raise ValueError("Keep the fixed 30-epoch scheduler horizon.")
    selection = config["selection"]
    expected = {"protocol": PROTOCOL, "outer_folds": list(range(5)), "inner_folds": 3,
                "screening_inner_fold": 0, "screening_epochs": 6,
                "confirmation_epochs": 12, "duration_rule": "median_inner_best_epoch"}
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError("Selection must use the declared five-outer/three-inner screen/confirm protocol.")
    recipes = selection.get("recipes", [])
    choices = {(recipe.get("backbone_lr"), recipe.get("loss_weight")) for recipe in recipes}
    if len(recipes) != 4 or choices != {(lr, loss) for lr in (3e-5, 1e-4) for loss in ("none", "sqrt_ratio")}:
        raise ValueError("Screen all four declared B0 recipes in every outer partition.")
    identifiers = [str(recipe.get("id", "")) for recipe in recipes]
    if len(set(identifiers)) != 4 or any(not _SAFE.fullmatch(item) for item in identifiers):
        raise ValueError("Recipe identities must be unique safe identifiers.")
    tolerance = selection.get("ap_tie_tolerance")
    if isinstance(tolerance, bool) or tolerance != 0.0001:
        raise ValueError("Use the declared AP tie tolerance of 0.0001.")
    budget = config["budget"]
    if not 0 < budget["reporting_reserve_minutes"] < budget["session_minutes"] <= 300:
        raise ValueError("Session must fit five hours and preserve a reporting reserve.")
    if not np.isfinite(budget["initial_epoch_seconds"]) or budget["initial_epoch_seconds"] <= 0:
        raise ValueError("Initial epoch estimate must be finite and positive.")
    if not np.isfinite(budget["epoch_safety_factor"]) or budget["epoch_safety_factor"] < 1:
        raise ValueError("Epoch safety factor must be at least one.")
    evaluation = config["evaluation"]
    if evaluation.get("reserved_test_allowed") is not False or evaluation.get("classification_threshold") != 0.5:
        raise ValueError("The reserved test stays excluded; the raw classification cutoff is fixed at 0.5.")
    replicates = evaluation.get("bootstrap_replicates")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise ValueError("At least 100 patient-bootstrap replicates are required.")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def prepare_outer(frame, config, fold, image_audit):
    """Construct this fold's training identity without any held-out outcomes."""
    if fold not in range(5):
        raise ValueError("Outer fold must be 0 through 4.")
    pool = frame.loc[frame.fold != fold].copy().reset_index(drop=True)
    outer = frame.loc[frame.fold == fold].copy().reset_index(drop=True)
    pool["inner_fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=config["training"]["seed"])
    for inner_fold, (fit_index, valid_index) in enumerate(splitter.split(pool, pool.target, pool.patient_id)):
        fit, validation = pool.iloc[fit_index], pool.iloc[valid_index]
        if fit.target.nunique() != 2 or validation.target.nunique() != 2:
            raise ValueError(f"Outer {fold}/inner {inner_fold} requires both classes.")
        if set(fit.patient_id) & set(validation.patient_id):
            raise ValueError("Inner patient overlap.")
        pool.loc[valid_index, "inner_fold"] = inner_fold
    if pool.inner_fold.lt(0).any() or pool.groupby("patient_id").inner_fold.nunique().ne(1).any():
        raise ValueError("Incomplete inner patient folds.")
    if set(pool.patient_id) & set(outer.patient_id):
        raise ValueError("Outer patient overlap.")
    lookup = pool.set_index("image_name").target.to_dict()
    for group in image_audit.get("duplicate_groups_within_patient", []):
        labels = {int(lookup[item["image_id"]]) for item in group if item["image_id"] in lookup}
        if len(labels) > 1:
            raise ValueError("Exact duplicate image content has conflicting training labels.")
    sources = {f"src/melanoma_detection/{name}": file_sha256(Path(__file__).with_name(name))
               for name in ("image_nested.py", "image_model.py", "image_transforms.py", "image_pilot.py",
                            "metadata_evaluation.py", "metadata_validation.py")}
    ids = set(pool.image_name)
    identity = {"protocol": PROTOCOL, "config": copy.deepcopy(config), "outer_fold": fold,
                "training_pool_memberships": pilot._memberships(pool),
                "training_pool_images": [row for row in image_audit["rows"] if row["image_id"] in ids],
                "source_sha256": sources, "runtime": pilot._runtime_identity()}
    manifest = {"schema_version": 1, "run_id": config["run"]["run_id"], "protocol": PROTOCOL,
                "outer_fold": fold, "compatibility_sha256": _hash_json(identity), "compatibility": identity,
                "held_out_memberships": pilot._memberships(outer, targets=False)}
    return {"pool": pool, "outer": outer, "fold_manifest": manifest}


def prepare_nested(project_root, config, *, audit_images=True):
    """Audit canonical Subset images once; never open reserved test or Full data."""
    validate_config(config)
    root = Path(project_root).resolve()
    run = config["run"]
    csv_path = pilot._restricted_path(root, run["folds_csv"], root / "data/subset/processed", "folds_csv")
    image_dir = pilot._restricted_path(root, run["image_dir"], root / "data/subset/train", "image_dir")
    output = pilot._restricted_path(root, run["output_dir"], root / "results/image_d1_runs", "output_dir")
    if csv_path.name != "train_folds.csv":
        raise ValueError("Only canonical Subset train_folds.csv is permitted.")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        headers = next(csv.reader(handle))
    if len(set(headers)) != len(headers):
        raise ValueError("Duplicate CSV columns.")
    frame = pilot._validate_fold_frame(pd.read_csv(csv_path, usecols=["image_name", "patient_id", "target", "fold"],
                                                 dtype=str), 0)
    if not frame.groupby("fold").target.nunique().eq(2).all():
        raise ValueError("Every outer fold requires both classes.")
    frame["image_path"] = frame.image_name.map(lambda name: image_dir / f"{name}.jpg")
    # Image-content auditing cannot depend on any held-out outcomes. Training
    # duplicate-label conflicts are checked separately inside prepare_outer.
    audit = pilot._audit_images(frame.assign(target=0), decode=audit_images)
    folds = {fold: prepare_outer(frame, config, fold, audit) for fold in range(5)}
    identity = {"config": copy.deepcopy(config), "fold_identities": {
        str(fold): prepared["fold_manifest"]["compatibility_sha256"] for fold, prepared in folds.items()},
        "images_audited": audit_images}
    manifest = {"schema_version": 1, "run_id": run["run_id"], "protocol": PROTOCOL,
                "compatibility_sha256": _hash_json(identity), "compatibility": identity,
                "development_csv": run["folds_csv"], "development_csv_sha256": file_sha256(csv_path),
                "config": copy.deepcopy(config), "test_set_loaded": False,
                "evaluation_status": "development_outer_oof",
                "scope": "B0 recipe screening and duration confirmation; architecture and transformation selection excluded",
                "qualification": "Development folds influenced earlier pilots, including an archived fold-0 prototype. OOF estimates this declared bounded procedure, not untouched final-test generalisation."}
    return {"root": root, "config": config, "frame": frame, "folds": folds, "image_audit": audit,
            "audit": audit, "run_dir": output / run["run_id"], "manifest": manifest}


def persist_preparation(prepared):
    run_dir = prepared["run_dir"]
    path = run_dir / "run_manifest.json"
    if path.exists():
        if _read_json(path).get("compatibility_sha256") != prepared["manifest"]["compatibility_sha256"]:
            raise ValueError("Run identity changed; use a new run_id, never bypass compatibility.")
    elif run_dir.exists() and any(item.name not in {"source_before", "session.json", "historical_preservation.json", "training.log"} for item in run_dir.iterdir()):
        raise ValueError("Nonempty run directory has no compatible manifest.")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _atomic_json(path, prepared["manifest"])
    audit_path = run_dir / "image_audit.json"
    if audit_path.exists() and _read_json(audit_path) != prepared["image_audit"]:
        raise ValueError("Persisted image audit changed.")
    if not audit_path.exists():
        _atomic_json(audit_path, prepared["image_audit"])
    for fold, data in prepared["folds"].items():
        fold_path = run_dir / f"folds/{fold}/fold_manifest.json"
        if fold_path.exists() and _read_json(fold_path) != data["fold_manifest"]:
            raise ValueError("Persisted outer-fold identity changed.")
        if not fold_path.exists():
            _atomic_json(fold_path, data["fold_manifest"])
        membership_path = fold_path.parent / "inner_memberships.csv"
        expected = data["pool"].drop(columns="image_path").to_csv(index=False, lineterminator="\n")
        if membership_path.exists() and membership_path.read_text(encoding="utf-8") != expected:
            raise ValueError("Persisted inner membership changed.")
        if not membership_path.exists():
            _atomic_csv(membership_path, data["pool"].drop(columns="image_path"))
    archive = run_dir / "source_at_training"
    for relative, digest in prepared["folds"][0]["fold_manifest"]["compatibility"]["source_sha256"].items():
        target = archive / relative
        if target.exists() and file_sha256(target) != digest:
            raise ValueError("Archived training source changed.")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((prepared["root"] / relative).read_bytes())
    return run_dir


def claim_training_window(session_path, config, *, now=None, started_utc=None):
    """A resume cannot reset the persisted session start or five-hour limit."""
    path = Path(session_path)
    moment = now or _utc_now()
    if path.exists():
        session = _read_json(path)
        if session.get("run_id") != config["run"]["run_id"]:
            raise ValueError("Session belongs to a different run.")
        if session.get("budget") != config["budget"]:
            raise ValueError("Existing session budget cannot be silently changed.")
    else:
        start = _utc(started_utc) if started_utc else moment
        if start > moment:
            raise ValueError("Session start cannot be in the future.")
        budget = config["budget"]
        final = start + timedelta(minutes=budget["session_minutes"])
        session = {"run_id": config["run"]["run_id"], "budget": config["budget"],
                   "started_utc": start.isoformat(), "session_deadline_utc": final.isoformat(),
                   "training_cutoff_utc": (final - timedelta(minutes=budget["reporting_reserve_minutes"])).isoformat()}
        _atomic_json(path, session)
    start = _utc(session["started_utc"])
    final = start + timedelta(minutes=config["budget"]["session_minutes"])
    cutoff = final - timedelta(minutes=config["budget"]["reporting_reserve_minutes"])
    if _utc(session["session_deadline_utc"]) != final or _utc(session["training_cutoff_utc"]) != cutoff:
        raise ValueError("Session deadlines disagree with the declared budget.")
    if moment >= cutoff:
        raise TimeoutError("Session training budget is exhausted; checkpoints remain resumable with an explicit new session file.")
    return cutoff


def select_recipe(records, tolerance=0.0001):
    """Select only after every equally long screening fit finishes."""
    if len(records) != 4 or any(row.get("status") != "complete" or type(row.get("completed_epochs")) is not int
                              or row["completed_epochs"] != 6 for row in records):
        raise ValueError("All four six-epoch screening recipes must finish before selection.")
    scores = [row["best_metrics"]["average_precision"] for row in records]
    if not all(np.isfinite(scores)):
        raise ValueError("Screening AP must be finite.")
    eligible = [row for row in records if row["best_metrics"]["average_precision"] >= max(scores) - tolerance]
    winner = min(eligible, key=lambda row: (row["recipe"]["loss_weight"] != "none", row["recipe"]["backbone_lr"], row["recipe"]["id"]))
    return copy.deepcopy(winner["recipe"])


def select_duration(best_epochs):
    if len(best_epochs) != 3 or any(isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or not 1 <= epoch <= 12 for epoch in best_epochs):
        raise ValueError("Duration requires three completed inner best epochs in 1 through 12.")
    return int(np.median(best_epochs))


def _job_manifest(fold_manifest, recipe, inner_fold):
    # Epoch cap is deliberately absent: an exactly compatible six-epoch screen
    # can continue to twelve epochs for confirmation, without changing schedule.
    return {"schema_version": 1, "run_id": fold_manifest["run_id"], "protocol": PROTOCOL,
            "outer_fold": fold_manifest["outer_fold"], "fold_compatibility_sha256": fold_manifest["compatibility_sha256"],
            "recipe": recipe, "inner_fold": inner_fold, "purpose": "inner_screen_and_confirmation"}


def _fit(prepared, fitting, validation, recipe, folder, manifest, epochs, device, deadline, *, refit=False):
    from .image_model import train_recipe, recover_interrupted_checkpoint
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "latest.pt").exists() or (folder / "latest.previous.pt").exists():
        recover_interrupted_checkpoint(folder / "latest.pt", expected_manifest=manifest)
    result = train_recipe(fitting, validation, prepared["config"]["model"], prepared["config"]["training"],
                          recipe, folder, manifest, epochs=epochs, device=device, deadline=deadline, resume=True,
                          refit=refit, epoch_admission_seconds=prepared["config"]["budget"]["initial_epoch_seconds"],
                          epoch_admission_safety_factor=prepared["config"]["budget"]["epoch_safety_factor"])
    _atomic_json(folder / "fit_result.json", result)
    return result


def _fixed_prediction(prepared, fold, recipe, duration, result, refit_manifest, device):
    from .image_model import load_model_checkpoint, predict_frame, seed_everything
    data = prepared["folds"][fold]
    if result["status"] != "complete" or len(result["history"]) != duration:
        raise ValueError("Never predict an outer fold from a partial refit.")
    seed_everything(prepared["config"]["training"]["seed"])
    model, checkpoint = load_model_checkpoint(result["latest_checkpoint"], device=device, expected_manifest=refit_manifest)
    if checkpoint["epoch"] != duration:
        raise ValueError("The refit checkpoint is not the declared final epoch.")
    # The prediction API receives placeholder outcomes. Real outer labels are
    # joined by checked ordered identity only after deterministic inference.
    predictions = predict_frame(model, data["outer"].assign(target=0), checkpoint["spec"],
                                {**checkpoint["training"], "microbatch": result["microbatch"]}, device=device)
    if predictions.image_id.tolist() != data["outer"].image_name.tolist():
        raise ValueError("Prediction coverage/order changed.")
    predictions["target"] = data["outer"].target.to_numpy()
    predictions["run_id"], predictions["recipe_id"], predictions["refit_epochs"] = prepared["manifest"]["run_id"], recipe["id"], duration
    return predictions


def _progress(prepared, fold_records, reason=None, *, final=False):
    complete = final and len(fold_records) == 5 and all(row["status"] == "complete" for row in fold_records)
    summary = {"schema_version": 1, "run_id": prepared["manifest"]["run_id"], "protocol": PROTOCOL,
               "status": "complete" if complete else "incomplete", "folds": fold_records,
               "completed_outer_folds": [row["fold"] for row in fold_records if row["status"] == "complete"],
               "test_set_loaded": False, "calibration_fitted": False, "threshold_fitted": False,
               "stopping_reason": reason, "qualification": prepared["manifest"]["qualification"],
               "updated_utc": _utc_now().isoformat()}
    _atomic_json(prepared["run_dir"] / "summary.json", summary)
    return summary


def run_nested(project_root, config, *, prepared=None, device="cuda", session_path=None, deadline_utc=None, started_utc=None):
    """Run five folds, pausing with durable checkpoints when the budget expires."""
    validate_config(config)
    prepared = prepared or prepare_nested(project_root, config)
    if prepared["config"] != config:
        raise ValueError("Prepared inputs do not match the requested configuration.")
    if prepared["image_audit"]["status"] != "complete":
        raise ValueError("Training requires decoded, hashed image audit.")
    run_dir = persist_preparation(prepared)
    session_path = Path(session_path) if session_path else run_dir / "session.json"
    if not session_path.resolve().is_relative_to((prepared["root"] / "results/image_d1_runs").resolve()):
        raise ValueError("Session files must remain in image run outputs.")
    cutoff = claim_training_window(session_path, config, started_utc=started_utc)
    if deadline_utc:
        cutoff = min(cutoff, _utc(deadline_utc))
    deadline = time.monotonic() + max(0, (cutoff - _utc_now()).total_seconds())
    fold_records, outer_predictions = [], []
    for fold in range(5):
        data = prepared["folds"][fold]
        folder = run_dir / f"folds/{fold}"
        screen_path, selection_path = folder / "screening.json", folder / "selection.json"
        if screen_path.exists():
            screens = _read_json(screen_path)
        else:
            screens = []
        by_recipe = {row["recipe"]["id"]: row for row in screens}
        for recipe in config["selection"]["recipes"]:
            if by_recipe.get(recipe["id"], {}).get("status") == "complete":
                from .image_model import _read_checkpoint, canonical_hash, recover_interrupted_checkpoint
                checkpoint_path = folder / f"inner/{recipe['id']}/0/latest.pt"
                job_manifest = _job_manifest(data["fold_manifest"], recipe, 0)
                recover_interrupted_checkpoint(checkpoint_path, expected_manifest=job_manifest)
                saved = _read_checkpoint(checkpoint_path)
                if canonical_hash(saved["manifest"]) != canonical_hash(job_manifest) or len(saved["history"]) < 6:
                    raise ValueError("Screening checkpoint does not match its completed record.")
                best = max(saved["history"][:6], key=lambda row: row["average_precision"])
                expected = {"recipe": recipe, "status": "complete", "completed_epochs": 6,
                            "best_epoch": best["epoch"], "best_metrics": {key: best[key] for key in
                            ("average_precision", "roc_auc", "log_loss", "brier")}}
                if by_recipe[recipe["id"]] != expected:
                    raise ValueError("Persisted screening scores disagree with checkpoint history.")
                continue
            fit = data["pool"].loc[data["pool"].inner_fold != 0].copy().reset_index(drop=True)
            validation = data["pool"].loc[data["pool"].inner_fold == 0].copy().reset_index(drop=True)
            job_manifest = _job_manifest(data["fold_manifest"], recipe, 0)
            print(f"Outer {fold}: screening {recipe['id']}", flush=True)
            result = _fit(prepared, fit, validation, recipe, folder / f"inner/{recipe['id']}/0", job_manifest,
                          config["selection"]["screening_epochs"], device, deadline)
            history = result["history"][:6]
            best = max(history, key=lambda row: row["average_precision"]) if history else None
            record = {"recipe": recipe, "status": "complete" if len(history) == 6 else result["status"],
                      "completed_epochs": len(history), "best_epoch": best["epoch"] if best else None,
                      "best_metrics": {key: best[key] for key in ("average_precision", "roc_auc", "log_loss", "brier")} if best else None}
            by_recipe[recipe["id"]] = record
            screens = list(by_recipe.values())
            _atomic_json(screen_path, screens)
            if record["status"] != "complete":
                fold_records.append({"fold": fold, "status": "incomplete", "stage": "screening"})
                return _progress(prepared, fold_records, result["status"])
        recipe = select_recipe(screens, config["selection"]["ap_tie_tolerance"])
        confirmation = []
        for inner_fold in range(3):
            fit = data["pool"].loc[data["pool"].inner_fold != inner_fold].copy().reset_index(drop=True)
            validation = data["pool"].loc[data["pool"].inner_fold == inner_fold].copy().reset_index(drop=True)
            job_manifest = _job_manifest(data["fold_manifest"], recipe, inner_fold)
            print(f"Outer {fold}: confirm {recipe['id']}, inner {inner_fold}", flush=True)
            result = _fit(prepared, fit, validation, recipe, folder / f"inner/{recipe['id']}/{inner_fold}", job_manifest,
                          config["selection"]["confirmation_epochs"], device, deadline)
            if result["status"] != "complete" or len(result["history"]) != 12:
                fold_records.append({"fold": fold, "status": "incomplete", "stage": "confirmation"})
                return _progress(prepared, fold_records, result["status"])
            confirmation.append({"inner_fold": inner_fold, "best_epoch": result["best_epoch"], "best_metrics": result["best_metrics"]})
        duration = select_duration([row["best_epoch"] for row in confirmation])
        selection = {"fold": fold, "recipe": recipe, "confirmation": confirmation, "refit_epochs": duration,
                     "mean_inner_average_precision": float(np.mean([row["best_metrics"]["average_precision"] for row in confirmation])),
                     "selection_uses_outer_labels": False}
        if selection_path.exists() and _read_json(selection_path) != selection:
            raise ValueError("A completed inner selection changed on resume.")
        _atomic_json(selection_path, selection)
        refit_manifest = {"schema_version": 1, "run_id": config["run"]["run_id"], "protocol": PROTOCOL,
                          "outer_fold": fold, "fold_compatibility_sha256": data["fold_manifest"]["compatibility_sha256"],
                          "recipe": recipe, "purpose": "outer_training_fixed_duration_refit", "refit_epochs": duration}
        print(f"Outer {fold}: refit all {len(data['pool'])} training images for {duration} fixed epochs", flush=True)
        result = _fit(prepared, data["pool"], None, recipe, folder / "refit", refit_manifest,
                      duration, device, deadline, refit=True)
        if result["status"] != "complete":
            fold_records.append({"fold": fold, "status": "incomplete", "stage": "refit"})
            return _progress(prepared, fold_records, result["status"])
        predictions = _fixed_prediction(prepared, fold, recipe, duration, result, refit_manifest, device)
        _atomic_csv(folder / "outer_predictions.csv", predictions)
        outer_predictions.append(predictions)
        fold_records.append({"fold": fold, "status": "complete", "selected_recipe": recipe["id"],
                             "refit_epochs": duration, "checkpoint": result["latest_checkpoint"],
                             "checkpoint_sha256": file_sha256(result["latest_checkpoint"]),
                             "predictions_sha256": file_sha256(folder / "outer_predictions.csv"),
                             **probability_metrics(predictions.target, predictions.base_probability)})
        _progress(prepared, fold_records)
        print(f"Outer {fold} complete. Held-out AP={fold_records[-1]['average_precision']:.5f}", flush=True)
    combined = pd.concat(outer_predictions, ignore_index=True).sort_values("image_id").reset_index(drop=True)
    if len(combined) != len(prepared["frame"]) or not combined.image_id.is_unique:
        raise ValueError("Outer OOF must cover each canonical image exactly once.")
    path = run_dir / "outer_oof_predictions.csv"
    _atomic_csv(path, combined)
    _atomic_csv(run_dir / "fold_metrics.csv", pd.DataFrame(fold_records))
    summary = _progress(prepared, fold_records, final=True)
    summary.update({"outer_oof_predictions_sha256": file_sha256(path),
                    "fold_metrics_sha256": file_sha256(run_dir / "fold_metrics.csv"),
                    "n_images": len(combined), "n_patients": int(combined.patient_id.nunique()),
                    "mean_fold_average_precision": float(np.mean([row["average_precision"] for row in fold_records])),
                    "pooled_metrics": probability_metrics(combined.target, combined.base_probability)})
    _atomic_json(run_dir / "summary.json", summary)
    return summary
