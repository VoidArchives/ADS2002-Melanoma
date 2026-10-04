"""Training-only sigmoid recalibration of the bounded five-fold image model.

Each outer training pool supplies three fresh fixed-duration inner refits.
Their held-out scores fit the predeclared C=1 sigmoid and 80% recall scenario.
Outer scores are transformed only after fitting both rules. Raw image runs and
their checkpoints are read-only. The allocation is the original raw session.
"""
from __future__ import annotations

import copy
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
import yaml

from . import image_nested as raw_stage
from . import image_model as core
from . import metadata_model
from .image_evaluation import workspace_relative_path
from .metadata_evaluation import probability_metrics, select_recall_threshold, validate_probabilities

PROTOCOL = "fixed_duration_training_oof_sigmoid_v1"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_atomic_json, _atomic_csv = raw_stage._atomic_json, raw_stage._atomic_csv
file_sha256, _hash_json = raw_stage.file_sha256, raw_stage._hash_json
_read_json, _utc, _utc_now = raw_stage._read_json, raw_stage._utc, raw_stage._utc_now


def validate_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Calibration requires schema_version 1.")
    for key in ("run", "calibration", "budget", "evaluation"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Missing calibration {key} configuration.")
    run = config["run"]
    for key in ("run_id", "reference_run_id"):
        if not _SAFE.fullmatch(str(run.get(key, ""))):
            raise ValueError("Calibration run identities must be safe names.")
    if run["run_id"] == run["reference_run_id"]:
        raise ValueError("Calibration requires separate outputs from the raw run.")
    expected = {"protocol": PROTOCOL, "outer_folds": list(range(5)), "inner_folds": 3,
                "duration_source": "median_three_inner_best_epochs", "method": "regularised_sigmoid_on_logits",
                "C": 1.0, "logit_clip": 1e-7, "reject_nonpositive_slope": True, "target_recall": 0.8,
                "threshold_source": "training_only_fixed_duration_oof", "threshold_scores": ["raw", "calibrated"],
                "failure_policy": "preserve_raw_and_do_not_promote_calibrated",
                "threshold_purpose": "academic_recall_scenario"}
    if any(config["calibration"].get(key) != value for key, value in expected.items()):
        raise ValueError("Use the predeclared fixed-duration OOF, C=1, positive-slope, 80%-recall procedure.")
    if config["budget"] != {"source": "reference_session_only", "allow_deadline_extension": False}:
        raise ValueError("Calibration shares the existing raw session without extending its cutoff.")
    evaluation = config["evaluation"]
    if evaluation.get("reserved_test_allowed") is not False or evaluation.get("complete_outer_folds_required") != 5:
        raise ValueError("All five outer folds are required; reserved test access is prohibited.")
    if type(evaluation.get("bootstrap_replicates")) is not int or evaluation["bootstrap_replicates"] < 100:
        raise ValueError("At least 100 patient-bootstrap replicates are required.")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def _paths(root, config):
    root = Path(root).resolve()
    output = workspace_relative_path(root, config["run"]["output_dir"])
    if output != root / "results/image_d1_runs":
        raise ValueError("Calibration output must remain in results/image_d1_runs.")
    raw_dir = output / config["run"]["reference_run_id"]
    run_dir = output / config["run"]["run_id"]
    reference_config = workspace_relative_path(root, config["run"]["reference_config"])
    session = workspace_relative_path(root, config["run"]["reference_session"])
    if not reference_config.is_relative_to(root / "configs") or session != raw_dir / "session.json":
        raise ValueError("Use the declared raw configuration and original session.json.")
    return root, raw_dir, run_dir, reference_config, session


def declare_protocol(project_root, config, config_path=None):
    """Record the fixed algorithm before any raw outer evaluation is available."""
    validate_config(config)
    root, raw_dir, run_dir, _, _ = _paths(project_root, config)
    config_path = Path(config_path) if config_path else root / "configs/image_nested_calibration.yaml"
    if _hash_json(load_config(config_path)) != _hash_json(config):
        raise ValueError("Declaration configuration disagrees with the saved YAML.")
    declaration = run_dir / "protocol_declaration.json"
    if declaration.exists():
        record = _read_json(declaration)
        if record["config_sha256"] != file_sha256(config_path) or record.get("declared_before_outer_fold0_prediction") is not True:
            raise ValueError("The predeclared calibration configuration changed.")
        return record
    seen = [fold for fold in range(5) if (raw_dir / f"folds/{fold}/outer_predictions.csv").exists()]
    if seen:
        raise ValueError("Calibration must be declared before inspecting raw outer predictions.")
    record = {"schema_version": 1, "declared_utc": _utc_now().isoformat(),
              "config_sha256": file_sha256(config_path), "reference_run_id": raw_dir.name,
              "existing_outer_prediction_folds": [], "declared_before_outer_fold0_prediction": True}
    _atomic_json(declaration, record)
    return record


def prepare_calibration(project_root, config, *, raw_prepared=None, audit_images=True):
    """Verify raw training identities and create a separate calibration manifest."""
    validate_config(config)
    root, raw_dir, run_dir, reference_config, session = _paths(project_root, config)
    declaration_path = run_dir / "protocol_declaration.json"
    if not declaration_path.exists():
        raise ValueError("Missing predeclared calibration protocol; declare it before raw outer inference.")
    declaration = declare_protocol(root, config)
    if not session.is_file():
        raise ValueError("The original raw session must already exist; calibration cannot create a new allocation.")
    raw_config = raw_stage.load_config(reference_config)
    if raw_config["run"]["run_id"] != config["run"]["reference_run_id"]:
        raise ValueError("Raw reference configuration belongs to a different run.")
    reference = _read_json(raw_dir / "run_manifest.json")
    if _hash_json(reference["config"]) != _hash_json(raw_config):
        raise ValueError("Raw configuration differs from its training manifest.")
    prepared_raw = raw_prepared or raw_stage.prepare_nested(root, raw_config, audit_images=audit_images)
    if reference["compatibility_sha256"] != prepared_raw["manifest"]["compatibility_sha256"]:
        raise ValueError("Raw source/data identity differs from the reference run.")
    source_names = ("image_nested_calibration.py", "image_nested.py", "image_model.py", "image_transforms.py",
                    "image_pilot.py", "metadata_model.py", "metadata_evaluation.py", "metadata_validation.py")
    sources = {f"src/melanoma_detection/{name}": file_sha256(Path(__file__).with_name(name)) for name in source_names}
    identity = {"protocol": PROTOCOL, "config": copy.deepcopy(config), "source_sha256": sources,
                "runtime": raw_stage.pilot._runtime_identity(),
                "reference_run_compatibility_sha256": reference["compatibility_sha256"]}
    manifest = {"schema_version": 1, "run_id": config["run"]["run_id"], "reference_run_id": raw_dir.name,
                "protocol": PROTOCOL, "compatibility_sha256": _hash_json(identity), "compatibility": identity,
                "config": copy.deepcopy(config), "source_sha256": sources,
                "protocol_declaration_sha256": file_sha256(declaration_path),
                "development_csv": reference["development_csv"],
                "development_csv_sha256": reference["development_csv_sha256"],
                "reference_session": config["run"]["reference_session"], "test_set_loaded": False,
                "qualification": reference["qualification"],
                "evaluation_status": "development_outer_oof_training_only_recalibration"}
    return {"root": root, "config": config, "raw_config": raw_config, "raw_dir": raw_dir, "run_dir": run_dir,
            "session_path": session, "raw_prepared": prepared_raw, "manifest": manifest,
            "declaration": declaration, "source_sha256": sources}


def persist_preparation(prepared):
    path = prepared["run_dir"] / "run_manifest.json"
    if path.exists() and _read_json(path) != prepared["manifest"]:
        raise ValueError("Calibration run identity changed; use a separate run, never bypass compatibility.")
    if not path.exists():
        _atomic_json(path, prepared["manifest"])
    for relative, digest in prepared["source_sha256"].items():
        target = prepared["run_dir"] / "source_at_training" / relative
        if target.exists() and file_sha256(target) != digest:
            raise ValueError("Archived calibration source changed.")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((prepared["root"] / relative).read_bytes())


def shared_deadline(prepared, deadline_utc=None):
    if not prepared["session_path"].is_file():
        raise ValueError("Calibration cannot create or replace the original session.")
    cutoff = raw_stage.claim_training_window(prepared["session_path"], prepared["raw_config"])
    if deadline_utc is not None:
        cutoff = min(cutoff, _utc(deadline_utc))
    return cutoff


def validate_oof_coverage(predictions, pool):
    columns = ["image_id", "patient_id", "fold", "target", "inner_fold"]
    if not set(columns + ["base_probability"]).issubset(predictions):
        raise ValueError("Training OOF is missing identity, fold or score columns.")
    if not predictions.image_id.is_unique or len(predictions) != len(pool):
        raise ValueError("Training OOF must cover every training image exactly once.")
    expected = pool.rename(columns={"image_name": "image_id"}) if "image_id" not in pool else pool
    actual = predictions[columns].sort_values("image_id").reset_index(drop=True)
    desired = expected[columns].sort_values("image_id").reset_index(drop=True)
    # CSV loading widens canonical int8 targets to int64. Validate before
    # normalising so fractional labels cannot be hidden by integer conversion.
    actual["target"] = validate_probabilities(actual.target, np.zeros(len(actual)))[0]
    desired["target"] = validate_probabilities(desired.target, np.zeros(len(desired)))[0]
    if not actual.equals(desired):
        raise ValueError("Training OOF membership, labels or fold assignment changed.")
    validate_probabilities(predictions.target, predictions.base_probability)
    if not predictions.groupby("patient_id").inner_fold.nunique().eq(1).all():
        raise ValueError("Training OOF patient occurs across inner folds.")


def calibrated_scores(calibrator, probabilities):
    slope, intercept = calibrator.get("slope"), calibrator.get("intercept")
    if calibrator.get("status") != "complete" or not np.isfinite([slope, intercept]).all() or slope <= 0:
        raise ValueError("A complete finite positive-slope calibrator is required.")
    logits = metadata_model.probability_logits(probabilities).reshape(-1)
    return expit(slope * logits + intercept)


def fit_calibration(training_oof, config):
    """Fit solely on fixed-duration training OOF scores; no outer input accepted."""
    validate_config(config)
    y, p = validate_probabilities(training_oof.target, training_oof.base_probability)
    base = {"protocol": PROTOCOL, "method": "regularised_sigmoid_on_logits", "C": 1.0,
            "logit_clip": 1e-7, "target_recall": 0.8, "threshold_source": "training_only_fixed_duration_oof",
            "training_images": len(training_oof), "training_patients": int(training_oof.patient_id.nunique()),
            "training_malignant_images": int(y.sum()), "outer_labels_used": False}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fitted = metadata_model.fit_recalibrator(y, p)
        result = {**base, "status": "complete", "slope": float(fitted.coef_[0, 0]),
                  "intercept": float(fitted.intercept_[0])}
        scores = calibrated_scores(result, p)
        result["raw_threshold"] = select_recall_threshold(y, p, 0.8)
        result["calibrated_threshold"] = select_recall_threshold(y, scores, 0.8)
        return result
    except (ValueError, FloatingPointError, ConvergenceWarning) as error:
        return {**base, "status": "failed", "failure_reason": f"{type(error).__name__}: {error}",
                "slope": None, "intercept": None, "raw_threshold": None, "calibrated_threshold": None}


def apply_calibration(raw_outer, calibrator):
    """Transform frozen outer scores without consulting their outcomes."""
    scores = calibrated_scores(calibrator, raw_outer.base_probability)
    result = raw_outer.copy()
    result["calibrated_probability"] = scores
    for key in ("raw_threshold", "calibrated_threshold"):
        value = calibrator[key]
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Training-selected thresholds must be finite scores in [0, 1].")
        result[key] = float(value)
    result["decision_raw"] = (result.base_probability >= calibrator["raw_threshold"]).astype(int)
    result["decision_calibrated"] = (scores >= calibrator["calibrated_threshold"]).astype(int)
    result["calibration_status"] = "complete"
    result["calibration_protocol"] = PROTOCOL
    return result


def _selection(prepared, fold):
    data = prepared["raw_prepared"]["folds"][fold]
    folder = prepared["raw_dir"] / f"folds/{fold}"
    if _read_json(folder / "fold_manifest.json") != data["fold_manifest"]:
        raise ValueError("Raw outer-training membership changed.")
    selection = _read_json(folder / "selection.json")
    if selection.get("fold") != fold or selection.get("selection_uses_outer_labels") is not False:
        raise ValueError("Reference selection must exclude outer labels.")
    recipe = selection["recipe"]
    if recipe not in prepared["raw_config"]["selection"]["recipes"]:
        raise ValueError("Reference recipe is outside the frozen library.")
    confirmations = selection["confirmation"]
    if [row["inner_fold"] for row in confirmations] != list(range(3)):
        raise ValueError("Reference selection lacks all three confirmation folds.")
    duration = raw_stage.select_duration([row["best_epoch"] for row in confirmations])
    if duration != selection["refit_epochs"]:
        raise ValueError("Reference duration differs from the declared median.")
    for row in confirmations:
        checkpoint = core._read_checkpoint(folder / f"inner/{recipe['id']}/{row['inner_fold']}/latest.pt")
        if checkpoint["epoch"] != 12 or checkpoint["best_epoch"] != row["best_epoch"] or checkpoint["best_metrics"] != row["best_metrics"]:
            raise ValueError("Reference confirmation does not match its checkpoint.")
        expected_manifest = raw_stage._job_manifest(data["fold_manifest"], recipe, row["inner_fold"])
        if checkpoint["manifest"] != expected_manifest:
            raise ValueError("Reference confirmation checkpoint belongs to a different split.")
    return selection, file_sha256(folder / "selection.json")


def inner_job_manifest(prepared, fold, inner_fold, selection):
    """Identity deliberately excludes global raw OOF hashes and outer labels."""
    data = prepared["raw_prepared"]["folds"][fold]
    return {"schema_version": 1, "run_id": prepared["config"]["run"]["run_id"], "protocol": PROTOCOL,
            "reference_run_id": prepared["raw_dir"].name, "outer_fold": fold, "inner_fold": inner_fold,
            "raw_fold_training_compatibility_sha256": data["fold_manifest"]["compatibility_sha256"],
            "calibration_config_sha256": _hash_json(prepared["config"]),
            "calibration_source_sha256": prepared["source_sha256"], "recipe": selection["recipe"],
            "fixed_epochs": selection["refit_epochs"], "purpose": "training_only_fixed_duration_oof_refit"}


def _inner_fit_predict(prepared, fold, inner_fold, selection, *, device, deadline):
    data = prepared["raw_prepared"]["folds"][fold]
    fitting = data["pool"].loc[data["pool"].inner_fold != inner_fold].copy().reset_index(drop=True)
    validation = data["pool"].loc[data["pool"].inner_fold == inner_fold].copy().reset_index(drop=True)
    core.validate_partitions(fitting, validation)
    if (set(fitting.patient_id) | set(validation.patient_id)) & set(data["outer"].patient_id):
        raise ValueError("Calibration fitting or validation includes outer patients.")
    folder = prepared["run_dir"] / f"folds/{fold}/inner/{inner_fold}"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = inner_job_manifest(prepared, fold, inner_fold, selection)
    path = folder / "job_manifest.json"
    if path.exists() and _read_json(path) != manifest:
        raise ValueError("Calibration inner refit identity changed.")
    _atomic_json(path, manifest)
    if (folder / "latest.pt").exists() or (folder / "latest.previous.pt").exists():
        core.recover_interrupted_checkpoint(folder / "latest.pt", expected_manifest=manifest)
    print(f"Calibration outer {fold}, inner {inner_fold}: {selection['recipe']['id']} for {selection['refit_epochs']} fixed epochs", flush=True)
    budget = prepared["raw_config"]["budget"]
    result = core.train_recipe(fitting, None, prepared["raw_config"]["model"], prepared["raw_config"]["training"],
                               selection["recipe"], folder, manifest, epochs=selection["refit_epochs"],
                               refit=True, resume=True, device=device, deadline=deadline,
                               epoch_admission_seconds=budget["initial_epoch_seconds"],
                               epoch_admission_safety_factor=budget["epoch_safety_factor"])
    _atomic_json(folder / "fit_result.json", result)
    if result["status"] != "complete":
        return None, result
    if result["final_epoch"] != selection["refit_epochs"] or result["best_checkpoint"] is not None:
        raise ValueError("Calibration requires the complete final epoch without validation checkpoint selection.")
    core.seed_everything(prepared["raw_config"]["training"]["seed"])
    model, checkpoint = core.load_model_checkpoint(result["final_checkpoint"], device=device, expected_manifest=manifest)
    if checkpoint["epoch"] != selection["refit_epochs"]:
        raise ValueError("Calibration refit checkpoint has the wrong final epoch.")
    predictions = core.predict_frame(model, validation.assign(target=0), checkpoint["spec"],
                                     {**checkpoint["training"], "microbatch": result["microbatch"]}, device=device)
    if predictions.image_id.tolist() != validation.image_name.tolist() or predictions.patient_id.tolist() != validation.patient_id.tolist():
        raise ValueError("Inner inference identity/order changed.")
    predictions["target"] = validation.target.to_numpy()
    predictions["inner_fold"] = inner_fold
    predictions["calibration_run_id"] = prepared["config"]["run"]["run_id"]
    predictions["refit_epochs"] = selection["refit_epochs"]
    _atomic_csv(folder / "predictions.csv", predictions)
    receipt = {"inner_fold": inner_fold, "fixed_epochs": selection["refit_epochs"],
               "checkpoint_sha256": file_sha256(result["final_checkpoint"]),
               "job_manifest_sha256": file_sha256(path), "history_sha256": file_sha256(folder / "history.csv"),
               "predictions_sha256": file_sha256(folder / "predictions.csv")}
    _atomic_json(folder / "prediction_receipt.json", receipt)
    return predictions, receipt


def _read_predictions(path):
    return pd.read_csv(path, dtype={"image_id": str, "image_name": str, "patient_id": str},
                       float_precision="round_trip")


def _check_raw_outer(prepared, fold, raw_outer):
    outer = prepared["raw_prepared"]["folds"][fold]["outer"]
    expected = outer.assign(image_id=outer.image_name)
    if len(raw_outer) != len(outer) or not raw_outer.image_id.is_unique:
        raise ValueError("Raw outer predictions lack complete unique image coverage.")
    for key in ("image_id", "patient_id", "fold", "target"):
        left = raw_outer.sort_values("image_id")[key].reset_index(drop=True)
        right = expected.sort_values("image_id")[key].reset_index(drop=True)
        if key == "target":
            left = pd.Series(validate_probabilities(left, np.zeros(len(left)))[0])
            right = pd.Series(validate_probabilities(right, np.zeros(len(right)))[0])
        if not left.equals(right):
            raise ValueError(f"Raw outer prediction {key} differs from canonical membership.")
    validate_probabilities(raw_outer.target, raw_outer.base_probability)


def _verify_completed_fold(prepared, fold, selection_sha256):
    folder = prepared["run_dir"] / f"folds/{fold}"
    receipt = _read_json(folder / "fold_result.json")
    if receipt["status"] != "complete" or receipt["training_selection_sha256"] != selection_sha256:
        raise ValueError("Saved calibration completion has a different selection.")
    files = {"training_oof_sha256": "training_oof_predictions.csv", "calibrator_sha256": "calibrator.json",
             "predictions_sha256": "outer_predictions.csv"}
    for key, name in files.items():
        if file_sha256(folder / name) != receipt[key]:
            raise ValueError(f"Saved calibration artifact changed: {name}")
    for inner in receipt["inner_refits"]:
        inner_folder = folder / f"inner/{inner['inner_fold']}"
        for key, name in {"checkpoint_sha256": "latest.pt", "job_manifest_sha256": "job_manifest.json",
                          "history_sha256": "history.csv", "predictions_sha256": "predictions.csv"}.items():
            if file_sha256(inner_folder / name) != inner[key]:
                raise ValueError("Saved calibration inner refit artifact changed.")
    raw_path = prepared["raw_dir"] / f"folds/{fold}/outer_predictions.csv"
    if file_sha256(raw_path) != receipt["reference_outer_predictions_sha256"]:
        raise ValueError("Raw outer predictions changed after calibration.")
    training_oof = _read_predictions(folder / "training_oof_predictions.csv")
    validate_oof_coverage(training_oof, prepared["raw_prepared"]["folds"][fold]["pool"])
    calibrator = _read_json(folder / "calibrator.json")
    predictions = _read_predictions(folder / "outer_predictions.csv")
    raw = _read_predictions(raw_path)
    _check_raw_outer(prepared, fold, raw)
    expected = apply_calibration(raw, calibrator)
    if not predictions.image_id.equals(expected.image_id) or not np.allclose(predictions.calibrated_probability, expected.calibrated_probability, rtol=0, atol=1e-15):
        raise ValueError("Saved outer calibration does not reproduce the frozen transform.")
    return predictions, receipt


def _progress(prepared, records, reason=None, *, final=False):
    complete = final and len(records) == 5 and all(item["status"] == "complete" for item in records)
    summary = {"schema_version": 1, "run_id": prepared["config"]["run"]["run_id"],
               "reference_run_id": prepared["raw_dir"].name, "protocol": PROTOCOL,
               "status": "complete" if complete else "incomplete", "folds": records,
               "completed_outer_folds": [item["fold"] for item in records if item["status"] == "complete"],
               "calibration_fitted": complete, "threshold_fitted": complete, "test_set_loaded": False,
               "stopping_reason": reason, "qualification": prepared["manifest"]["qualification"],
               "updated_utc": _utc_now().isoformat()}
    _atomic_json(prepared["run_dir"] / "summary.json", summary)
    return summary


def run_calibration(project_root, config, *, prepared=None, device="cuda", deadline_utc=None):
    """Complete five calibrated folds within the existing raw session's cutoff."""
    validate_config(config)
    prepared = prepared or prepare_calibration(project_root, config)
    if _hash_json(prepared["config"]) != _hash_json(config):
        raise ValueError("Injected calibration preparation disagrees with the requested configuration.")
    if prepared["raw_prepared"]["image_audit"]["status"] != "complete":
        raise ValueError("Calibration training requires decoded and hashed image audit.")
    persist_preparation(prepared)
    try:
        cutoff = shared_deadline(prepared, deadline_utc)
        deadline = time.monotonic() + max(0, (cutoff - _utc_now()).total_seconds())
    except TimeoutError:
        # Completed folds remain verifiable after the allocation expires.
        # A new refit is never admitted; preserve previously completed work.
        deadline = time.monotonic() - 1
    records, outer_predictions = [], []
    for fold in range(5):
        raw_path = prepared["raw_dir"] / f"folds/{fold}/outer_predictions.csv"
        if not raw_path.exists():
            records.append({"fold": fold, "status": "incomplete", "stage": "awaiting_raw_outer_refit"})
            return _progress(prepared, records, "awaiting_raw_outer_predictions")
        selection, selection_sha256 = _selection(prepared, fold)
        folder = prepared["run_dir"] / f"folds/{fold}"
        saved_path = folder / "fold_result.json"
        if saved_path.exists() and _read_json(saved_path)["status"] == "complete":
            predictions, receipt = _verify_completed_fold(prepared, fold, selection_sha256)
            records.append(receipt)
            outer_predictions.append(predictions)
            continue
        if time.monotonic() >= deadline:
            records.append({"fold": fold, "status": "incomplete", "stage": "shared_session_budget"})
            return _progress(prepared, records, "shared_session_budget_exhausted")
        data = prepared["raw_prepared"]["folds"][fold]
        membership = {"fold": fold, "raw_fold_training_compatibility_sha256": data["fold_manifest"]["compatibility_sha256"],
                      "training_selection_sha256": selection_sha256, "selection": selection,
                      "training_pool": raw_stage.pilot._memberships(data["pool"]),
                      "held_out": raw_stage.pilot._memberships(data["outer"], targets=False)}
        manifest_path = folder / "fold_manifest.json"
        if manifest_path.exists() and _read_json(manifest_path) != membership:
            raise ValueError("Calibration fold membership or fixed duration changed.")
        _atomic_json(manifest_path, membership)
        inner_predictions, inner_refits = [], []
        for inner_fold in range(3):
            predictions, receipt = _inner_fit_predict(prepared, fold, inner_fold, selection, device=device, deadline=deadline)
            if predictions is None:
                records.append({"fold": fold, "status": "incomplete", "stage": "fixed_duration_inner_refits", "inner_fold": inner_fold})
                return _progress(prepared, records, receipt["status"])
            inner_predictions.append(predictions)
            inner_refits.append(receipt)
        training_oof = pd.concat(inner_predictions, ignore_index=True).sort_values("image_id").reset_index(drop=True)
        validate_oof_coverage(training_oof, data["pool"])
        oof_path = folder / "training_oof_predictions.csv"
        _atomic_csv(oof_path, training_oof)
        calibrator = fit_calibration(training_oof, config)
        calibrator.update({"fold": fold, "fixed_epochs": selection["refit_epochs"], "recipe": selection["recipe"],
                           "training_oof_sha256": file_sha256(oof_path), "training_selection_sha256": selection_sha256})
        _atomic_json(folder / "calibrator.json", calibrator)
        if calibrator["status"] != "complete":
            records.append({"fold": fold, "status": "failed", "stage": "sigmoid_fit", "failure_reason": calibrator["failure_reason"]})
            return _progress(prepared, records, "calibration_failed_no_fallback")
        raw_outer = _read_predictions(raw_path)
        _check_raw_outer(prepared, fold, raw_outer)
        predictions = apply_calibration(raw_outer.assign(target=0), calibrator)
        predictions["target"] = raw_outer.target.to_numpy()
        predictions["calibration_run_id"] = config["run"]["run_id"]
        _atomic_csv(folder / "outer_predictions.csv", predictions)
        receipt = {"fold": fold, "status": "complete", "selected_recipe": selection["recipe"]["id"],
                   "fixed_epochs": selection["refit_epochs"], "slope": calibrator["slope"], "intercept": calibrator["intercept"],
                   "raw_threshold": calibrator["raw_threshold"], "calibrated_threshold": calibrator["calibrated_threshold"],
                   "training_oof_sha256": file_sha256(oof_path), "training_selection_sha256": selection_sha256,
                   "calibrator_sha256": file_sha256(folder / "calibrator.json"),
                   "predictions_sha256": file_sha256(folder / "outer_predictions.csv"),
                   "reference_outer_predictions_sha256": file_sha256(raw_path), "inner_refits": inner_refits}
        _atomic_json(saved_path, receipt)
        records.append(receipt)
        outer_predictions.append(predictions)
        _progress(prepared, records)
        print(f"Calibration outer {fold} complete; training-only sigmoid slope={calibrator['slope']:.5f}.", flush=True)
    combined = pd.concat(outer_predictions, ignore_index=True).sort_values("image_id").reset_index(drop=True)
    expected = prepared["raw_prepared"]["frame"].assign(image_id=lambda frame: frame.image_name)
    if len(combined) != len(expected) or not combined.image_id.is_unique or set(combined.image_id) != set(expected.image_id):
        raise ValueError("Calibrated OOF must cover all canonical images exactly once.")
    path = prepared["run_dir"] / "outer_oof_predictions.csv"
    _atomic_csv(path, combined)
    summary = _progress(prepared, records, final=True)
    summary.update({"outer_oof_predictions_sha256": file_sha256(path), "n_images": len(combined),
                    "n_patients": int(combined.patient_id.nunique()),
                    "pooled_raw_metrics": probability_metrics(combined.target, combined.base_probability),
                    "pooled_calibrated_metrics": probability_metrics(combined.target, combined.calibrated_probability)})
    _atomic_json(prepared["run_dir"] / "summary.json", summary)
    return summary
