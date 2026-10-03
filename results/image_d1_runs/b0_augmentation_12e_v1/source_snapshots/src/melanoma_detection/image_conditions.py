"""Fresh, bounded B0 augmentation conditions sharing the existing recipe trainer.

The historical pilot runners remain strict. This stage reuses their saved patient
memberships, never uses excluded outcomes, and requires an explicit quality review.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import yaml

from . import image_pilot
from .metadata_evaluation import probability_metrics
from .metadata_validation import validate_binary_target

SCHEMA_VERSION = 1
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_atomic_json = image_pilot._atomic_json
_atomic_csv = image_pilot._atomic_csv
_hash = image_pilot._hash_json
_read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))


def _utc(value):
    moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("Session times must include their UTC offset.")
    return moment.astimezone(timezone.utc)


def _now():
    return datetime.now(timezone.utc)


def _path(root, value, allowed):
    return image_pilot._restricted_path(root, value, allowed, "experiment path")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported condition configuration.")
    for name in ("run", "model", "training", "recipe", "experiment", "conditions"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"Missing {name} mapping.")
    from .image_model import DEFAULT_TRAINING, validate_training
    expected_model = {"architecture": "efficientnet_b0", "weights": "IMAGENET1K_V1",
                      "image_size": 224, "dropout": 0.2}
    if config["model"] != expected_model:
        raise ValueError("This controlled stage supports only the fixed B0 preparation.")
    if validate_training(config["training"]) != DEFAULT_TRAINING:
        raise ValueError("The matched comparison must preserve every declared training control.")
    if config["recipe"] != {"id": "lr3e5_sqrt", "backbone_lr": 3e-5, "loss_weight": "sqrt_ratio"}:
        raise ValueError("The exploratory B0 recipe is a fixed experimental control.")
    experiment = config["experiment"]
    for key, value in {"outer_fold": 0, "inner_fold": 0, "epochs": 12,
                       "early_stopping": False, "minimum_ap_gain": 0.005,
                       "classification_threshold": 0.5, "quality_images": 24,
                       "variants_per_image": 4, "session_minutes": 120,
                       "final_reporting_minutes": 20, "latest_optional_admission_minutes": 70,
                       "optional_setup_minutes": 10, "optional_margin_minutes": 5}.items():
        if experiment.get(key) != value:
            raise ValueError(f"Fixed experiment.{key} must be {value!r}.")
    if set(config["conditions"]) != {"A", "B", "C"}:
        raise ValueError("Declare exactly A, B and optional C.")
    for condition, expected in {"A": ("weak", "none"), "B": ("stronger", "none"),
                                "C": ("weak", "clahe")}.items():
        entry = config["conditions"][condition]
        if (entry.get("augmentation"), entry.get("preprocessing")) != expected:
            raise ValueError(f"Condition {condition} differs from the frozen library.")
        if not _SAFE.fullmatch(str(entry.get("pipeline_id", ""))):
            raise ValueError("Unsafe pipeline identity.")
    for name in ("augmentation_run_id", "lighting_run_id"):
        if not _SAFE.fullmatch(str(config["run"].get(name, ""))):
            raise ValueError("Unsafe run identity.")
    _utc(config["run"]["session_start_utc"])


def _inventory_frame(frame):
    """Validate global identities/folds without interpreting any outcomes."""
    required = {"image_name", "patient_id", "target", "fold"}
    if frame.empty or not required.issubset(frame):
        raise ValueError("Required d1 fold columns are absent.")
    result = frame[list(sorted(required))].copy()
    for column in ("image_name", "patient_id"):
        values = result[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any() or values.ne(values.str.strip()).any():
            raise ValueError(f"Invalid {column} identifiers.")
        result[column] = values.astype(str)
    if result.image_name.duplicated().any() or not result.image_name.map(lambda v: bool(_SAFE.fullmatch(v))).all():
        raise ValueError("Duplicate or unsafe image identifiers.")
    fold = pd.to_numeric(result.fold, errors="raise")
    if not np.isfinite(fold).all() or not fold.mod(1).eq(0).all() or set(fold) != set(range(5)):
        raise ValueError("Expected all five integer outer folds.")
    result["fold"] = fold.astype(int)
    if result.groupby("patient_id").fold.nunique().gt(1).any():
        raise ValueError("A patient crosses outer folds.")
    return result.sort_values("image_name").reset_index(drop=True)


def _source_hashes(root):
    source_root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), Path(__file__).with_name("image_model.py"),
               Path(__file__).with_name("image_pilot.py"),
               Path(__file__).with_name("metadata_evaluation.py"),
               Path(__file__).with_name("metadata_validation.py"),
               source_root / "scripts/run_image_conditions.py"]
    # Keep the transformation implementation in numerical compatibility, even
    # when image_model delegates to its own dedicated module.
    for name in ("image_transforms.py", "image_preparation.py", "image_transformations.py"):
        path = Path(__file__).with_name(name)
        if path.exists():
            sources.append(path)
    return {str(path.relative_to(source_root)).replace("\\", "/"): image_pilot.file_sha256(path)
            for path in sources}


def prepare_conditions(project_root, config, *, audit_images=True):
    """Read only d1 and reuse exact historical memberships; outer targets ignored."""
    validate_config(config)
    root = Path(project_root).resolve()
    run = config["run"]
    folds = _path(root, run["folds_csv"], root / "data/subset/processed")
    if folds.name != "train_folds.csv":
        raise ValueError("Only the d1 development folds may be loaded.")
    image_dir = _path(root, run["image_dir"], root / "data/subset/train")
    reference = _path(root, run["reference_memberships"], root / "results/image_d1_runs")
    with folds.open(newline="", encoding="utf-8-sig") as handle:
        headers = next(csv.reader(handle))
    if len(headers) != len(set(headers)):
        raise ValueError("Input CSV contains duplicate column names.")
    frame = _inventory_frame(pd.read_csv(folds, dtype=str))
    frame["image_path"] = frame.image_name.map(lambda name: image_dir / f"{name}.jpg")
    excluded = frame[frame.fold == 0].drop(columns="target").copy().reset_index(drop=True)
    pool = frame[frame.fold != 0].copy().reset_index(drop=True)
    pool["target"] = validate_binary_target(pool.target)
    if not pool.groupby("fold").target.nunique().eq(2).all():
        raise ValueError("Each training-pool outer fold needs both classes.")
    # Dummy constants are solely an adapter to the old image auditor. Real
    # outcomes are never passed into its global duplicate checks.
    audit = image_pilot._audit_images(frame.drop(columns="target").assign(target=0), decode=audit_images)
    label_lookup = pool.set_index("image_name").target.to_dict()
    for group in audit["duplicate_groups_within_patient"]:
        outcomes = {int(label_lookup[item["image_id"]]) for item in group if item["image_id"] in label_lookup}
        if len(outcomes) > 1:
            raise ValueError("Exact duplicate fitting-pool content has conflicting labels.")
    audit["duplicate_policy"] = "Global content identity is label-free; outcome consistency is checked within outer-training patients only."
    audit["excluded_labels_read"] = False
    saved = _read(reference)
    membership = pd.DataFrame(saved["training_pool"]).rename(columns={"image_id": "image_name"})
    required = ["image_name", "patient_id", "fold", "target", "inner_fold"]
    if not set(required).issubset(membership) or membership.image_name.duplicated().any():
        raise ValueError("Historical membership specification is invalid.")
    identity = pool[["image_name", "patient_id", "fold", "target"]].sort_values("image_name").reset_index(drop=True)
    old_identity = membership[identity.columns].sort_values("image_name").reset_index(drop=True)
    if identity.to_dict("records") != old_identity.to_dict("records"):
        raise ValueError("Current training-pool memberships/outcomes differ from the preserved B0 split.")
    excluded_identity = image_pilot._memberships(excluded, targets=False)
    if excluded_identity != sorted(saved["outer_excluded"], key=lambda row: row["image_id"]):
        raise ValueError("Excluded patient/image identities differ from the preserved split.")
    pool = pool.merge(membership[["image_name", "inner_fold"]], on="image_name", validate="one_to_one")
    if set(pool.inner_fold) != {0, 1, 2} or pool.groupby("patient_id").inner_fold.nunique().gt(1).any():
        raise ValueError("Historical inner folds are invalid or split a patient.")
    fitting = pool[pool.inner_fold != 0].copy().reset_index(drop=True)
    validation = pool[pool.inner_fold == 0].copy().reset_index(drop=True)
    from .image_model import validate_partitions
    validate_partitions(fitting, validation)
    if set(excluded.patient_id) & (set(fitting.patient_id) | set(validation.patient_id)):
        raise ValueError("Excluded patients overlap development partitions.")
    split_audit = {"patient_overlap": 0, "outer_fold": 0, "inner_fold": 0,
                   "outer_predictions_allowed": False, "test_set_loaded": False,
                   "evaluation_status": "exploratory_inner_validation_not_outer_oof",
                   "partitions": [{"partition": name, "images": len(part), "patients": int(part.patient_id.nunique()),
                                   "malignant_images": int(part.target.sum()) if "target" in part else None}
                                  for name, part in (("fitting", fitting), ("inner_validation", validation), ("outer_excluded", excluded))]}
    numerical_config = {key: copy.deepcopy(config[key]) for key in ("model", "training", "recipe", "conditions")}
    from .image_model import transformation_profile
    numerical_config["transformation_profiles"] = {key: transformation_profile(
        {**config["model"], "augmentation": entry["augmentation"], "preprocessing": entry["preprocessing"]})
        for key, entry in config["conditions"].items()}
    numerical_config["experiment"] = {key: config["experiment"][key] for key in
        ("selection_algorithm_version", "outer_fold", "inner_fold", "epochs", "early_stopping", "minimum_ap_gain", "classification_threshold", "quality_images", "variants_per_image")}
    image_ids = set(pool.image_name)
    compatibility = {"schema_version": 1, "config": numerical_config, "source_sha256": _source_hashes(root),
                     "runtime": image_pilot._runtime_identity(), "training_pool_memberships": image_pilot._memberships(pool),
                     "training_pool_images": [{key: row[key] for key in ("image_id", "source_sha256", "pixel_sha256")}
                                              for row in audit["rows"] if row["image_id"] in image_ids],
                     "images_audited": audit_images}
    manifest = {"schema_version": 1, "compatibility": compatibility, "compatibility_sha256": _hash(compatibility),
                "provenance": {"full_csv_sha256": image_pilot.file_sha256(folds),
                               "reference_memberships_sha256": image_pilot.file_sha256(reference),
                               "outer_excluded_memberships": excluded_identity}, "split_audit": split_audit,
                "threshold_tuned": False, "calibration_fitted": False, "test_set_loaded": False}
    output = _path(root, run["output_dir"], root / "results/image_d1_runs")
    return {"config": copy.deepcopy(config), "root": root, "run_dir": output / run["augmentation_run_id"],
            "lighting_run_dir": output / run["lighting_run_id"], "manifest": manifest,
            "fitting": fitting, "validation": validation, "outer_excluded": excluded,
            "inner_memberships": pool, "split_audit": split_audit, "image_audit": audit,
            "fitting_membership_sha256": _hash(fitting[["image_name", "patient_id", "target"]]
                .rename(columns={"image_name": "image_id"}).sort_values("image_id").to_dict("records"))}


def _pending(prepared, condition_id):
    entry = prepared["config"]["conditions"][condition_id]
    parent = prepared["lighting_run_dir"] if condition_id == "C" else prepared["run_dir"]
    return {"condition_id": condition_id, "pipeline_id": entry["pipeline_id"], "label": entry["label"],
            "spec": {**prepared["config"]["model"], "augmentation": entry["augmentation"], "preprocessing": entry["preprocessing"]},
            "status": "deferred" if condition_id == "C" else "pending_quality_review",
            "completed_epochs": 0, "history": [], "best_epoch": None, "best_metrics": None,
            "predictions_file": None, "epoch12_predictions_file": None, "epoch12_metrics": None,
            "output_dir": str(parent / "conditions" / entry["pipeline_id"])}


def _complete(record):
    return record.get("status") == "complete" and record.get("completed_epochs") == 12 and bool(record.get("predictions_file")) and bool(record.get("epoch12_predictions_file"))


def _summary(prepared, records, *, reason=None):
    complete = all(_complete(records[key]) for key in ("A", "B"))
    comparisons = {}
    for candidate in ("B", "C"):
        if _complete(records["A"]) and _complete(records[candidate]):
            difference = records[candidate]["best_metrics"]["average_precision"] - records["A"]["best_metrics"]["average_precision"]
            comparisons[f"{candidate}_minus_A"] = {"average_precision_difference": difference,
                "minimum_ap_gain": 0.005, "promising_b0_hypothesis": difference >= 0.005,
                "interpretation": "Exploratory screening, not statistical proof or an architecture-wide selection."}
    return {"schema_version": 1, "run_id": prepared["config"]["run"]["augmentation_run_id"],
            "status": "complete" if complete else "incomplete", "comparison_complete": complete,
            "conditions": records, "comparisons": comparisons, "stopping_reason": reason,
            "classification_threshold": 0.5, "evaluation_status": "exploratory_inner_validation_not_outer_oof",
            "selected_final_pipeline": None, "threshold_tuned": False, "calibration_fitted": False,
            "outer_predictions_generated": False, "test_set_loaded": False,
            "updated_utc": _now().isoformat()}


def persist_preparation(prepared):
    run_dir = Path(prepared["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        if _read(manifest_path)["compatibility_sha256"] != prepared["manifest"]["compatibility_sha256"]:
            raise ValueError("This run ID has incompatible numerical code, data or controls.")
        for relative, digest in prepared["manifest"]["compatibility"]["source_sha256"].items():
            snapshot = run_dir / "source_snapshots" / relative
            if not snapshot.is_file() or image_pilot.file_sha256(snapshot) != digest:
                raise ValueError("A frozen numerical source snapshot is missing or changed.")
    else:
        _atomic_json(manifest_path, {**prepared["manifest"], "run_id": prepared["config"]["run"]["augmentation_run_id"]})
        _atomic_json(run_dir / "split_memberships.json", {"training_pool": image_pilot._memberships(prepared["inner_memberships"]),
                                                           "outer_excluded": image_pilot._memberships(prepared["outer_excluded"], targets=False)})
        _atomic_json(run_dir / "image_audit.json", prepared["image_audit"])
        for relative, digest in prepared["manifest"]["compatibility"]["source_sha256"].items():
            source = Path(__file__).resolve().parents[2] / relative
            destination = run_dir / "source_snapshots" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            if image_pilot.file_sha256(destination) != digest:
                raise ValueError("Source snapshot does not match the frozen numerical source.")
        _atomic_json(run_dir / "frozen_candidate_library.json", prepared["manifest"]["compatibility"]["config"])
        _atomic_json(run_dir / "experiment_configuration.json", prepared["config"])
    if not (run_dir / "summary.json").exists():
        _atomic_json(run_dir / "summary.json", _summary(prepared, {key: _pending(prepared, key) for key in ("A", "B", "C")}, reason="audit_only_not_started"))
    return run_dir


def validate_quality_review(receipt, condition_id, *, prepared=None, verify_files=True):
    """Require a content-signed, explicitly reviewed receipt; never autoapprove."""
    if isinstance(receipt, (str, Path)):
        receipt = _read(receipt)
    signature = receipt.get("review_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "review_sha256"}
    if not signature or signature != _hash(unsigned):
        raise ValueError("Quality review is unsigned or has changed since explicit review.")
    if not receipt.get("reviewed_by") or not receipt.get("reviewed_at_utc"):
        raise ValueError("An explicit reviewer and review time are required.")
    _utc(receipt["reviewed_at_utc"])
    examples = receipt.get("selected_examples", [])
    ids = [item["image_id"] for item in examples]
    if len(ids) != 24 or len(set(ids)) != 24 or len({item["patient_id"] for item in examples}) != 24:
        raise ValueError("Quality review must cover 24 distinct fitting images/patients.")
    if prepared is not None:
        if receipt.get("fitting_membership_sha256") != prepared["fitting_membership_sha256"]:
            raise ValueError("Quality review belongs to a different fitting partition.")
        fit = prepared["fitting"].set_index("image_name")
        if not set(ids).issubset(fit.index) or any(str(fit.loc[item["image_id"], "patient_id"]) != item["patient_id"] for item in examples):
            raise ValueError("Quality previews contain non-fitting patients.")
        for item in examples:
            row = fit.loc[item["image_id"]]
            if int(row.target) != item["target"] or image_pilot.file_sha256(row.image_path) != item["source_sha256"]:
                raise ValueError("A reviewed fitting image or its fitting outcome changed.")
    if sum(item["target"] for item in examples) != 12:
        raise ValueError("The declared quality sample must contain 12 positive and 12 negative images.")
    condition = receipt.get("conditions", {}).get(condition_id, {})
    if condition.get("verdict") != "passed":
        raise ValueError(f"Condition {condition_id} has not passed explicit quality review.")
    if set(condition.get("reviewed_image_ids", [])) != set(ids):
        raise ValueError("Reviewed image coverage is incomplete.")
    rows = condition.get("reviews", [])
    variants = 1 if condition_id == "C" else 4
    expected = {(image, variant) for image in ids for variant in range(variants)}
    actual = {(item.get("image_id"), item.get("variant")) for item in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("Quality reviews do not cover every declared seeded variant.")
    for item in rows:
        for category in ("clipping", "boundary", "texture", "colour", "reflection"):
            value = item.get(category)
            if type(value) is not int or value not in range(4) or value >= 2:
                raise ValueError(f"Unsafe/unreviewed {category} in condition {condition_id}.")
        if condition_id != "C":
            expected_seed = 4200 + 4 * ids.index(item["image_id"]) + item["variant"]
            if item.get("seed") != expected_seed:
                raise ValueError("Reviewed variant seed differs from the frozen preview protocol.")
    files = condition.get("preview_sha256", receipt.get("preview_sha256", {}))
    if not files:
        raise ValueError("No reviewed preview files were recorded.")
    if verify_files:
        for name, digest in files.items():
            path = Path(name)
            if not path.is_file() or image_pilot.file_sha256(path) != digest:
                raise ValueError("A reviewed preview file is missing or changed.")
    return signature


def ensure_session(prepared, session_path=None):
    root, config = prepared["root"], prepared["config"]
    path = Path(session_path).resolve() if session_path else _path(root, config["run"]["session_file"], root / "results/image_d1_runs")
    if not path.is_relative_to((root / "results/image_d1_runs").resolve()):
        raise ValueError("Session record must remain within image outputs.")
    if not path.exists():
        start = _utc(config["run"]["session_start_utc"])
        _atomic_json(path, {"schema_version": 1, "started_utc": start.isoformat(),
            "training_cutoff_utc": (start + timedelta(minutes=100)).isoformat(),
            "session_deadline_utc": (start + timedelta(minutes=120)).isoformat(),
            "optional_admission_utc": (start + timedelta(minutes=70)).isoformat(), "status": "prepared"})
    session = _read(path)
    start, cutoff, deadline = (_utc(session[key]) for key in ("started_utc", "training_cutoff_utc", "session_deadline_utc"))
    if cutoff != start + timedelta(minutes=100) or deadline != start + timedelta(minutes=120):
        raise ValueError("Session must protect its fixed two-hour budget and 20-minute reporting reserve.")
    if _utc(session["optional_admission_utc"]) != start + timedelta(minutes=70):
        raise ValueError("Optional setup must retain its declared minute-70 admission gate.")
    return path, session


def begin_optional_setup(prepared, session_path=None):
    """Claim the bounded optional dependency/QA window, without installing anything."""
    path, session = ensure_session(prepared, session_path)
    summary = _read(Path(prepared["run_dir"]) / "summary.json")
    moment = _now()
    complete = session.get("ab_completed_utc")
    if not summary.get("comparison_complete") or not complete or _utc(complete) > _utc(session["optional_admission_utc"]):
        raise TimeoutError("A/B must complete before minute 70 to admit optional lighting setup.")
    if not session.get("optional_setup_started_utc"):
        if moment > _utc(session["optional_admission_utc"]):
            raise TimeoutError("The optional setup admission deadline has passed.")
        session["optional_setup_started_utc"] = moment.isoformat()
        session["optional_setup_deadline_utc"] = (moment + timedelta(minutes=10)).isoformat()
        _atomic_json(path, session)
    if moment > _utc(session["optional_setup_deadline_utc"]):
        raise TimeoutError("The optional dependency/quality setup allowance is exhausted.")
    return session


def _condition_manifest(prepared, record, quality_sha256):
    from .image_model import transformation_profile
    return {"schema_version": 1, "run_id": prepared["config"]["run"]["lighting_run_id" if record["condition_id"] == "C" else "augmentation_run_id"],
            "stage": "exploratory_fresh_matched_12_epoch_condition", "condition_id": record["condition_id"],
            "compatibility_sha256": prepared["manifest"]["compatibility_sha256"],
            "quality_review_sha256": quality_sha256, "transformation_profile": transformation_profile(record["spec"]),
            "candidate_library": prepared["manifest"]["compatibility"]["config"],
            "fitting_membership_sha256": prepared["fitting_membership_sha256"]}


def _prediction_export(prepared, record, manifest, *, device):
    from .image_model import load_model_checkpoint, predict_frame
    folder = Path(record["output_dir"])
    result = dict(record)
    for checkpoint_key, prediction_key, metrics_key, filename in (
        ("best_checkpoint", "predictions_file", "best_metrics", "inner_validation_predictions.csv"),
        ("latest_checkpoint", "epoch12_predictions_file", "epoch12_metrics", "epoch12_inner_validation_predictions.csv")):
        if checkpoint_key == "latest_checkpoint" and record.get("completed_epochs") != 12:
            continue
        checkpoint = record.get(checkpoint_key)
        if not checkpoint:
            continue
        model, saved = load_model_checkpoint(checkpoint, device=device, expected_manifest=manifest)
        training = {**prepared["config"]["training"], "microbatch": record["microbatch"]}
        predictions = predict_frame(model, prepared["validation"], record["spec"], training, device)
        for key, value in {"condition_id": record["condition_id"], "pipeline_id": record["pipeline_id"],
                           "run_id": manifest["run_id"], "inner_fold": 0, "split": "inner_validation",
                           "evaluation_status": "exploratory_not_outer_oof", "selected_epoch": saved["epoch"]}.items():
            predictions[key] = value
        predictions = predictions.sort_values("image_id").reset_index(drop=True)
        image_pilot._check_predictions(predictions, prepared["validation"])
        path = folder / filename
        _atomic_csv(path, predictions)
        result[prediction_key] = str(path)
        result[metrics_key] = probability_metrics(predictions.target, predictions.base_probability)
        result[prediction_key + "_sha256"] = image_pilot.file_sha256(path)
        del model
    return result


def _remaining_seconds(config, records, ids):
    total = 0.0
    for key in ids:
        record = records[key]
        if _complete(record):
            continue
        history = [row["epoch_seconds"] for row in record.get("history", [])]
        estimate = max(history) if history else config["experiment"]["initial_epoch_seconds"][key]
        total += (12 - record.get("completed_epochs", 0)) * estimate * 1.25 + config["experiment"]["prediction_reserve_seconds"]
    return total


def run_conditions(project_root, config, *, prepared=None, condition_ids=("A", "B"),
                   resume=True, predict_only=False, device="cuda", session_path=None):
    """Execute only reviewed conditions; full A/B comparison precedes optional C."""
    from .image_model import pretrained_weight_identity, train_recipe
    prepared = prepared or prepare_conditions(project_root, config)
    if prepared["image_audit"]["status"] != "complete":
        raise ValueError("Execution requires a complete decoded-image audit.")
    run_dir = persist_preparation(prepared)
    records = _read(run_dir / "summary.json")["conditions"]
    ids = tuple(condition_ids)
    if ids not in (("A", "B"), ("A",), ("B",), ("C",)):
        raise ValueError("Execute A/B, one paused A/B condition, or optional C separately.")
    path, session = ensure_session(prepared, session_path)
    cutoff = _utc(session["training_cutoff_utc"])
    reason = None
    if not predict_only:
        if _now() >= cutoff:
            reason = "session_training_budget_exhausted"
        elif ids == ("C",):
            ab_completed = session.get("ab_completed_utc")
            if not all(_complete(records[key]) for key in ("A", "B")) or not ab_completed or _utc(ab_completed) > _utc(session["optional_admission_utc"]):
                reason = "optional_clahe_deferred_ab_not_complete_by_minute_70"
            elif not session.get("optional_setup_started_utc") or _now() > _utc(session["optional_setup_deadline_utc"]):
                reason = "optional_clahe_deferred_setup_window_missing_or_exhausted"
            elif _remaining_seconds(config, records, ids) + 300 > (cutoff - _now()).total_seconds():
                reason = "optional_clahe_deferred_insufficient_remaining_budget"
        elif _remaining_seconds(config, records, tuple(key for key in ("A", "B") if not _complete(records[key]))) > (cutoff - _now()).total_seconds():
            reason = "insufficient_time_for_complete_matched_pair"
    if reason:
        if ids == ("C",):
            records["C"].update(status="deferred", stopping_reason=reason)
        result = _summary(prepared, records, reason=reason)
        _atomic_json(run_dir / "summary.json", result)
        return result
    if not predict_only:
        # Do not permit torchvision to download a new baseline silently.
        pretrained_weight_identity({**config["model"], "augmentation": "weak", "preprocessing": "none"}, require_cached=True)
        if ids == ("C",):
            try:
                installed = importlib.metadata.version("opencv-python-headless")
            except importlib.metadata.PackageNotFoundError:
                installed = "unavailable"
            if installed != "4.13.0.92":
                records["C"].update(status="deferred", stopping_reason="optional_clahe_dependency_missing_or_incompatible")
                result = _summary(prepared, records, reason=records["C"]["stopping_reason"])
                _atomic_json(run_dir / "summary.json", result)
                return result
    for key in ids:
        record = records[key]
        receipt_path = (prepared["lighting_run_dir"] if key == "C" else run_dir) / "quality_review.json"
        try:
            quality = validate_quality_review(receipt_path, key, prepared=prepared)
        except (ValueError, FileNotFoundError) as error:
            if not predict_only:
                records[key] = {**record, "status": "quality_blocked", "stopping_reason": str(error)}
                _atomic_json(run_dir / "summary.json", _summary(prepared, records, reason="quality_gate_blocked"))
            continue
        manifest = _condition_manifest(prepared, record, quality)
        if predict_only:
            # Numerical/history files are read-only; only predictions and a
            # separate receipt are refreshed.
            records[key] = _prediction_export(prepared, record, manifest, device=device)
            continue
        if _complete(record):
            continue
        initial = records["A"].get("initial_model_state_sha256") if key != "A" else None
        if key != "A" and not initial:
            records[key].update(status="reference_blocked", stopping_reason="A_verified_initialisation_unavailable")
            reason = "A_verified_initialisation_unavailable"
            break
        records[key] = {**record, "status": "running"}
        _atomic_json(run_dir / "summary.json", _summary(prepared, records, reason="running"))
        _atomic_json(run_dir / "work_ledger.json", {"condition_id": key, "output_dir": record["output_dir"],
                                                    "training_cutoff_utc": cutoff.isoformat(), "session_file": str(path)})
        deadline = time.monotonic() + max(0, (cutoff - _now()).total_seconds())
        estimate = max((row["epoch_seconds"] for row in record.get("history", [])),
                       default=config["experiment"]["initial_epoch_seconds"][key])
        try:
            fit = train_recipe(prepared["fitting"], prepared["validation"], record["spec"], config["training"],
                               config["recipe"], record["output_dir"], manifest, epochs=12, device=device,
                               deadline=deadline, resume=resume, epoch_admission_seconds=estimate,
                               epoch_admission_safety_factor=1.25,
                               expected_initial_model_state_sha256=initial)
            record = {**record, **image_pilot._json_safe(fit), "completed_epochs": len(fit["history"]),
                      "quality_review_sha256": quality}
            record = _prediction_export(prepared, record, manifest, device=device)
        except Exception as error:
            records[key] = {**record, "status": "failed", "failure": f"{type(error).__name__}: {error}"}
            _atomic_json(run_dir / "summary.json", _summary(prepared, records, reason="explicit_failure"))
            raise
        records[key] = record
        _atomic_json(Path(record["output_dir"]) / "condition_result.json", record)
        _atomic_json(run_dir / "summary.json", _summary(prepared, records))
        if key == "C":
            lighting = Path(prepared["lighting_run_dir"])
            _atomic_json(lighting / "run_manifest.json", {**prepared["manifest"], "run_id": config["run"]["lighting_run_id"],
                                                          "reference_A_predictions_sha256": records["A"]["predictions_file_sha256"],
                                                          "reference_A_manifest": str(run_dir / "run_manifest.json")})
            _atomic_json(lighting / "summary.json", _summary(prepared, records))
        if not _complete(record):
            reason = record["status"]
            break
    result = _summary(prepared, records, reason=reason)
    if predict_only:
        _atomic_json(run_dir / "prediction_receipt.json", {**result, "mode": "predict_only", "training_history_preserved": True})
    else:
        _atomic_json(run_dir / "summary.json", result)
        if result["comparison_complete"] and not session.get("ab_completed_utc"):
            session["ab_completed_utc"] = _now().isoformat()
            _atomic_json(path, session)
        _atomic_json(run_dir / "work_ledger.json", {"condition_id": None, "stopping_reason": reason,
                                                    "training_cutoff_utc": cutoff.isoformat()})
    return result


def record_watchdog_pause(run_dir, *, stopping_reason="watchdog_deadline"):
    """Recover only a committed, hash-verified checkpoint after a hard stop."""
    from .image_model import _read_checkpoint, recover_interrupted_checkpoint
    run_dir = Path(run_dir)
    summary = _read(run_dir / "summary.json")
    checks = []
    for key, record in summary["conditions"].items():
        if record["status"] != "running":
            continue
        folder = Path(record["output_dir"])
        try:
            recovery = recover_interrupted_checkpoint(folder / "latest.pt")
            latest = _read_checkpoint(folder / "latest.pt")
            record.update(history=latest["history"], completed_epochs=latest["epoch"],
                          best_epoch=latest["best_epoch"], best_metrics=latest["best_metrics"],
                          microbatch=latest["microbatch"], latest_checkpoint=str(folder / "latest.pt"),
                          initial_model_state_sha256=latest.get("initial_model_state_sha256"))
            if (folder / "best.pt").exists():
                _read_checkpoint(folder / "best.pt")
                record["best_checkpoint"] = str(folder / "best.pt")
            record["status"] = "budget_exhausted"
            checks.append({"condition_id": key, "verified_latest": True, "completed_epochs": latest["epoch"],
                           "recovery_events": recovery["recovery_events"], "candidate_failures": recovery["candidate_failures"]})
        except (ValueError, FileNotFoundError) as error:
            record["status"] = "checkpoint_recovery_required"
            checks.append({"condition_id": key, "verified_latest": False, "error": str(error)})
        record["stopping_reason"] = stopping_reason
    summary.update(status="incomplete", comparison_complete=False, comparisons={}, stopping_reason=stopping_reason)
    _atomic_json(run_dir / "summary.json", summary)
    _atomic_json(run_dir / "watchdog_receipt.json", {"checkpoint_checks": checks, "stopping_reason": stopping_reason})
    return summary


__all__ = ["load_config", "validate_config", "prepare_conditions", "persist_preparation", "validate_quality_review",
           "ensure_session", "begin_optional_setup", "run_conditions", "record_watchdog_pause"]
