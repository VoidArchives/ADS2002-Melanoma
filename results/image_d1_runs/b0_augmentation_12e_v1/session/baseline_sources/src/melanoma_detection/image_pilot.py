"""Bounded, patient-safe B0 pilot orchestration; no final-test or outer scoring.

Later candidate procedures are declarations only. These entry points deliberately
refuse to turn the first-stage pilot into an augmentation/architecture search.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from .metadata_evaluation import calibration_bins, probability_metrics, validate_probabilities
from .metadata_validation import validate_binary_target


SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_config(path: str | Path) -> dict:
    """Read safe YAML, rejecting requests outside the implemented pilot stage."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported image-pilot configuration schema.")
    for key in ("run", "model", "training", "pilot", "candidates"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Configuration requires a {key} mapping.")
    run = config["run"]
    if run.get("mode") != "pilot":
        raise NotImplementedError("Only the B0 pilot is implemented; later stages are declarations.")
    if not isinstance(run.get("run_id"), str) or not _SAFE_ID.fullmatch(run["run_id"]):
        raise ValueError("run_id must be a safe nonblank identifier without path separators.")
    expected = {"architecture": "efficientnet_b0", "weights": "IMAGENET1K_V1",
                "image_size": 224, "dropout": 0.2, "augmentation": "weak", "preprocessing": "none"}
    if any(config["model"].get(key) != value for key, value in expected.items()):
        raise NotImplementedError("First implementation supports B0/weak/none with the declared preparation only.")
    pilot = config["pilot"]
    if pilot.get("outer_fold") not in range(5) or pilot.get("inner_folds") != 3 or pilot.get("inner_fold") != 0:
        raise ValueError("Pilot requires one of the five outer folds and first of three inner folds.")
    if pilot.get("epochs") != 6:
        raise ValueError("This pilot compares all four recipes at six epochs.")
    if not 0 < float(pilot.get("budget_minutes", 0)) <= 60:
        raise ValueError("Pilot budget must be positive and at most 60 minutes.")
    if pilot.get("ap_tie_tolerance") != 0.0001:
        raise ValueError("Pilot AP tie tolerance must match the predefined 0.0001 rule.")
    recipes = config["candidates"].get("pilot_recipes")
    if not isinstance(recipes, list) or len(recipes) != 4:
        raise ValueError("The pilot requires the four predefined B0 recipes.")
    identities = []
    for recipe in recipes:
        if not isinstance(recipe, dict) or not _SAFE_ID.fullmatch(str(recipe.get("id", ""))):
            raise ValueError("Each recipe needs a safe id.")
        identities.append(recipe["id"])
    if len(set(identities)) != 4:
        raise ValueError("Recipe IDs must be unique.")
    choices = {(item.get("backbone_lr"), item.get("loss_weight")) for item in recipes}
    if choices != {(lr, loss) for lr in (3e-5, 1e-4) for loss in ("none", "sqrt_ratio")}:
        raise ValueError("Recipe library must cross the two specified LRs with none/sqrt_ratio loss weighting.")
    training = config["training"]
    for key in ("microbatch", "effective_batch", "min_microbatch", "warmup_epochs", "schedule_epochs", "seed", "num_workers"):
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"training.{key} must be an integer.")
    if not 0 < training["min_microbatch"] <= training["microbatch"] <= training["effective_batch"]:
        raise ValueError("Invalid microbatch/effective-batch bounds.")
    if training["effective_batch"] % training["microbatch"]:
        raise ValueError("Effective batch must be divisible by microbatch.")
    if training["warmup_epochs"] != 1 or training["schedule_epochs"] != 30:
        raise ValueError("Pilot must retain one warmup epoch and its fixed 30-epoch scheduler horizon.")


def _restricted_path(root: Path, value: str, permitted: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonblank relative path.")
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(permitted.resolve()):
        raise ValueError(f"{name} must remain within {permitted.relative_to(root)}.")
    return path


def _validate_fold_frame(frame: pd.DataFrame, outer_fold: int) -> pd.DataFrame:
    required = {"image_name", "patient_id", "target", "fold"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Image folds require nonempty image_name, patient_id, target and fold columns.")
    frame = frame[list(sorted(required))].copy()
    for column in ("image_name", "patient_id"):
        identifiers = frame[column].astype("string")
        if identifiers.isna().any() or identifiers.str.strip().eq("").any() or identifiers.ne(identifiers.str.strip()).any():
            raise ValueError(f"{column} contains missing, blank or untrimmed identifiers.")
        frame[column] = identifiers.astype(str)
    if not frame.image_name.is_unique or not frame.image_name.map(lambda item: bool(_SAFE_ID.fullmatch(item))).all():
        raise ValueError("image_name contains duplicate or unsafe identifiers.")
    frame["target"] = validate_binary_target(frame.target)
    try:
        exact_folds = [Decimal(str(value)) for value in frame.fold]
    except InvalidOperation as error:
        raise ValueError("fold must contain integer labels 0 through 4.") from error
    if any(not value.is_finite() or value not in {Decimal(fold) for fold in range(5)} for value in exact_folds):
        raise ValueError("fold must contain exactly integer labels 0 through 4 before numeric conversion.")
    folds = pd.to_numeric(frame.fold, errors="raise")
    if not np.isfinite(folds).all() or not folds.mod(1).eq(0).all() or set(folds) != set(range(5)):
        raise ValueError("Expected exactly five integer folds labelled 0 through 4.")
    frame["fold"] = folds.astype(int)
    if frame.groupby("patient_id").fold.nunique().gt(1).any():
        raise ValueError("A patient occurs in more than one outer fold.")
    # Excluded outer labels must not affect preparation or recipe decisions.
    fitting_pool = frame[frame.fold != outer_fold]
    if not fitting_pool.groupby("fold").target.nunique().eq(2).all():
        raise ValueError("Every outer-training fold needs both binary classes.")
    return frame.sort_values("image_name").reset_index(drop=True)


def _audit_images(frame: pd.DataFrame, *, decode: bool) -> dict:
    rows = []
    by_pixels = {}
    for row in frame.itertuples(index=False):
        path = Path(row.image_path)
        item = {"image_id": row.image_name, "patient_id": row.patient_id, "fold": int(row.fold)}
        if decode:
            if not path.is_file():
                raise FileNotFoundError(f"Missing d1 JPEG: {path}")
            try:
                with Image.open(path) as image:
                    if image.format != "JPEG":
                        raise ValueError("Expected JPEG encoding.")
                    pixels = image.convert("RGB")
                    pixels.load()
                    item["size"] = list(pixels.size)
                    pixel_hash = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
            except Exception as error:
                raise ValueError(f"Corrupt or invalid JPEG {path.name}: {error}") from error
            item.update({"source_sha256": file_sha256(path), "pixel_sha256": pixel_hash})
            by_pixels.setdefault(pixel_hash, []).append(item)
        else:
            item.update({"source_sha256": None, "pixel_sha256": None})
        rows.append(item)
    duplicate_groups = [members for members in by_pixels.values() if len(members) > 1]
    unsafe = [members for members in duplicate_groups if len({item["patient_id"] for item in members}) > 1
              or len({item["fold"] for item in members}) > 1]
    if unsafe:
        example = [item["image_id"] for item in unsafe[0]]
        raise ValueError(f"Exact duplicate image content crosses patients or folds: {example}")
    label_lookup = frame.set_index("image_name").target.to_dict()
    conflicts = [members for members in duplicate_groups
                 if len({int(label_lookup[item["image_id"]]) for item in members}) > 1]
    if conflicts:
        details = [{"image_id": item["image_id"], "target": int(label_lookup[item["image_id"]])}
                   for item in conflicts[0]]
        raise ValueError(f"Exact duplicate image content has conflicting target labels: {details}")
    return {"status": "complete" if decode else "skipped_for_synthetic_preparation",
            "images": len(rows), "patients": int(frame.patient_id.nunique()),
            "duplicate_groups_within_patient": duplicate_groups,
            "crosspatient_duplicate_groups": 0, "label_conflicts": 0 if decode else None,
            "duplicate_policy": "Retain label-consistent duplicates within the same patient/fold for canonical-fold comparability.",
            "rows": rows}


def _memberships(frame: pd.DataFrame, *, targets: bool = True) -> list[dict]:
    columns = ["image_name", "patient_id", "fold"]
    if "inner_fold" in frame:
        columns.append("inner_fold")
    if targets:
        columns.append("target")
    output = frame[columns].sort_values("image_name").rename(columns={"image_name": "image_id"})
    return output.to_dict("records")


def _runtime_identity() -> dict:
    versions = {}
    for name in ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "Pillow", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


def prepare_pilot(project_root: str | Path, config: dict, *, audit_images: bool = True) -> dict:
    """Read d1 only, audit images and prepare one grouped inner pilot split.

    This is read-only. Full CSV hash is provenance, whereas the compatibility
    hash excludes outer labels so changing excluded outcomes cannot select a run.
    ``audit_images=False`` is a synthetic-test seam and cannot launch training.
    """
    validate_config(config)
    root = Path(project_root).resolve()
    run, pilot = config["run"], config["pilot"]
    folds_csv = _restricted_path(root, run["folds_csv"], root / "data/subset/processed", "folds_csv")
    image_dir = _restricted_path(root, run["image_dir"], root / "data/subset/train", "image_dir")
    if folds_csv.name != "train_folds.csv":
        raise ValueError("The first-stage pilot may read only the d1 train_folds.csv input.")
    output = _restricted_path(root, run["output_dir"], root / "results/image_d1_runs", "output_dir")
    with folds_csv.open(newline="", encoding="utf-8-sig") as handle:
        headers = next(csv.reader(handle))
    if len(set(headers)) != len(headers):
        raise ValueError("Input CSV has duplicate column names.")
    frame = _validate_fold_frame(pd.read_csv(
        folds_csv, usecols=["image_name", "patient_id", "target", "fold"],
        dtype={"image_name": str, "patient_id": str, "target": str, "fold": str}), pilot["outer_fold"])
    frame["image_path"] = frame.image_name.map(lambda name: image_dir / f"{name}.jpg")
    audit = _audit_images(frame, decode=audit_images)
    outer_excluded = frame[frame.fold == pilot["outer_fold"]].copy().reset_index(drop=True)
    pool = frame[frame.fold != pilot["outer_fold"]].copy().reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=config["training"]["seed"])
    pool["inner_fold"] = -1
    for fold, (fit_index, validation_index) in enumerate(inner.split(pool, pool.target, groups=pool.patient_id)):
        fit, validation = pool.iloc[fit_index], pool.iloc[validation_index]
        if fit.target.nunique() != 2 or validation.target.nunique() != 2:
            raise ValueError(f"Inner fold {fold} needs both binary classes in fitting and validation.")
        if set(fit.patient_id) & set(validation.patient_id):
            raise ValueError("An inner patient occurs in fitting and validation.")
        pool.loc[validation_index, "inner_fold"] = fold
    if not pool.groupby("patient_id").inner_fold.nunique().eq(1).all() or (pool.inner_fold < 0).any():
        raise ValueError("Incomplete or inconsistent inner patient folds.")
    fitting = pool[pool.inner_fold != 0].copy().reset_index(drop=True)
    validation = pool[pool.inner_fold == 0].copy().reset_index(drop=True)
    excluded_patients = set(outer_excluded.patient_id)
    if excluded_patients & (set(fitting.patient_id) | set(validation.patient_id)):
        raise ValueError("Outer patients overlap the pilot training pool.")
    audit_columns = ["partition", "images", "patients", "malignant_images"]
    split_rows = []
    for label, subset in (("fitting", fitting), ("inner_validation", validation)):
        split_rows.append(dict(zip(audit_columns, [label, len(subset), int(subset.patient_id.nunique()), int(subset.target.sum())])))
    # Excluded outcomes are never used even for pilot summary class counts.
    split_rows.append({"partition": "outer_excluded", "images": len(outer_excluded),
                       "patients": int(outer_excluded.patient_id.nunique()), "malignant_images": None})
    split_audit = {"patient_overlap": 0, "outer_fold": pilot["outer_fold"], "inner_folds": 3,
                   "pilot_validation_inner_fold": 0, "partitions": split_rows,
                   "outer_predictions_allowed": False, "test_set_loaded": False,
                   "evaluation_status": "exploratory_inner_validation_not_outer_oof"}
    source_hashes = {}
    for name in ("image_model.py", "image_pilot.py", "metadata_evaluation.py", "metadata_validation.py"):
        path = Path(__file__).with_name(name)
        source_hashes[f"src/melanoma_detection/{name}"] = file_sha256(path)
    membership = _memberships(pool)
    train_ids = set(pool.image_name)
    image_identity = [{key: row[key] for key in ("image_id", "source_sha256", "pixel_sha256")}
                      for row in audit["rows"] if row["image_id"] in train_ids]
    compatibility = {"schema_version": 1, "config": copy.deepcopy(config),
                     "source_sha256": source_hashes, "runtime": _runtime_identity(),
                     "training_pool_memberships": membership, "training_pool_images": image_identity,
                     "images_audited": audit_images,
                     "selection_algorithm_version": "b0_four_recipe_inner_pilot_v1"}
    manifest = {"schema_version": 1, "run_id": run["run_id"], "mode": "pilot",
                "compatibility_sha256": _hash_json(compatibility), "compatibility": compatibility,
                "provenance": {"full_csv_sha256": file_sha256(folds_csv), "folds_csv": run["folds_csv"],
                               "image_dir": run["image_dir"], "outer_excluded_memberships": _memberships(outer_excluded, targets=False)},
                "split_audit": split_audit,
                "calibration": {"status": "deferred", "reason": "Requires training-only OOF scores in a later stage; epoch-selection validation is not calibration fitting data."}}
    return {"fitting": fitting, "validation": validation, "outer_excluded": outer_excluded,
            "inner_memberships": pool, "split_audit": split_audit, "image_audit": audit,
            "manifest": manifest, "run_dir": output / run["run_id"]}


def persist_preparation(prepared: dict) -> Path:
    """Write initial audit artifacts; refuse incompatible reuse of a run ID."""
    run_dir = Path(prepared["run_dir"])
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("compatibility_sha256") != prepared["manifest"]["compatibility_sha256"]:
            raise ValueError("Run ID belongs to incompatible inputs/code/settings/runtime; use a new run_id.")
        return run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Nonempty output directory lacks a compatible manifest; use a new run_id.")
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest_path, prepared["manifest"])
    _atomic_json(run_dir / "image_audit.json", prepared["image_audit"])
    _atomic_json(run_dir / "split_memberships.json", {
        "training_pool": _memberships(prepared["inner_memberships"]),
        "outer_excluded": _memberships(prepared["outer_excluded"], targets=False),
        "fitting_image_ids": prepared["fitting"].image_name.tolist(),
        "validation_image_ids": prepared["validation"].image_name.tolist(),
    })
    return run_dir


def reliability_table(target, probabilities, *, bins: int = 5, groups=None) -> pd.DataFrame:
    """Reuse shared equal-count reliability bins, optionally with patient support."""
    return calibration_bins(target, probabilities, n_bins=bins, groups=groups)


def _recipe_manifest(prepared: dict, recipe: dict) -> dict:
    return {"schema_version": 1, "compatibility_sha256": prepared["manifest"]["compatibility_sha256"],
            "run_id": prepared["manifest"]["run_id"], "recipe_id": recipe["id"],
            "recipe_sha256": _hash_json(recipe), "stage": "exploratory_b0_inner_pilot"}


def _check_predictions(predictions: pd.DataFrame, validation: pd.DataFrame) -> None:
    if len(predictions) != len(validation) or not predictions.image_id.is_unique:
        raise ValueError("Pilot prediction coverage is incomplete or duplicated.")
    expected = validation.rename(columns={"image_name": "image_id"}).set_index("image_id")
    observed = predictions.set_index("image_id")
    if set(expected.index) != set(observed.index):
        raise ValueError("Pilot predictions do not match the inner validation split.")
    observed = observed.loc[expected.index]
    for column in ("patient_id", "fold", "target"):
        if not np.array_equal(expected[column].to_numpy(), observed[column].to_numpy()):
            raise ValueError(f"Prediction {column} differs from split membership.")
    validate_probabilities(observed.target, observed.base_probability)


def _summary(config: dict, results: list[dict], elapsed: float) -> dict:
    epochs = config["pilot"]["epochs"]
    complete = len(results) == 4 and all(item["status"] == "complete" and len(item["history"]) == epochs for item in results)
    selected = None
    if complete:
        maximum = max(item["best_metrics"]["average_precision"] for item in results)
        contenders = [item for item in results if maximum - item["best_metrics"]["average_precision"] <= config["pilot"]["ap_tie_tolerance"]]
        by_id = {item["id"]: item for item in config["candidates"]["pilot_recipes"]}
        selected = min(contenders, key=lambda item: (by_id[item["recipe_id"]]["loss_weight"] != "none",
                                                   by_id[item["recipe_id"]]["backbone_lr"]))
    epoch_seconds = [float(row["epoch_seconds"]) for item in results for row in item["history"] if "epoch_seconds" in row]
    mean_seconds = float(np.mean(epoch_seconds)) if epoch_seconds else None
    recommendation = "Comparison incomplete: resume unfinished recipes before selecting a provisional setting."
    if selected:
        if selected["best_epoch"] >= epochs - 1:
            recommendation = "Best inner AP is near the pilot cap; extend all four candidates on inner patients to a 30-epoch cap before stronger claims."
        else:
            recommendation = "Best inner AP occurred earlier; inspect the learning curves for plateau/overfitting. Extra epochs are not guaranteed to help."
    return {"schema_version": 1, "run_id": config["run"]["run_id"],
            "status": "complete" if complete else "incomplete", "comparison_complete": complete,
            "evaluation_status": "exploratory_inner_validation_not_outer_oof",
            "selected_recipe": selected["recipe_id"] if selected else None,
            "selection_status": "provisional_inner_screening" if selected else "no_winner",
            "recommendation": recommendation, "recipes": results, "elapsed_seconds_this_invocation": elapsed,
            "calibration": {"status": "deferred_training_only_oof", "fitted": False,
                            "checks": "Raw inner-validation log loss, Brier and up to five equal-count reliability bins with patient counts only; epoch-selection optimism applies."},
            "runtime_estimates": {"measured_b0_mean_epoch_seconds": mean_seconds,
                                  "four_30_epoch_b0_fit_hours_same_split": mean_seconds * 120 / 3600 if mean_seconds else None,
                                  "future_nested_maximum_logical_fits": 180, "future_nested_maximum_unequal_training_epochs": 4380,
                                  "optional_swin_added_fits": 35, "optional_stability_added_fits": 10,
                                  "future_hours": None,
                                  "qualification": "B0 pilot timing cannot estimate unequal unmeasured architectures; later stage budgets are allocations, not completion predictions."},
            "outer_predictions_generated": False, "test_set_loaded": False,
            "updated_utc": datetime.now(timezone.utc).isoformat()}


def run_pilot(project_root: str | Path, config: dict, *, resume: bool = True,
              predict_only: bool = False, prepared: dict | None = None, device: str = "cuda") -> dict:
    """Run four recipes within one budget, or re-predict compatible checkpoints.

    Prediction-only never rewrites training histories, latest checkpoints or
    comparison summaries. Its own receipt records raw probability checks.
    """
    from .image_model import load_model_checkpoint, predict_frame, train_recipe

    validate_config(config)
    prepared = prepared if prepared is not None else prepare_pilot(project_root, config)
    if prepared["image_audit"]["status"] != "complete":
        raise ValueError("Actual pilot execution requires the complete JPEG audit.")
    run_dir = persist_preparation(prepared)
    start = time.monotonic()
    deadline = start + float(config["pilot"]["budget_minutes"]) * 60
    results = []
    missing = []
    for recipe in config["candidates"]["pilot_recipes"]:
        folder = run_dir / "recipes" / recipe["id"]
        manifest = _recipe_manifest(prepared, recipe)
        result_path = folder / "recipe_result.json"
        if predict_only:
            if not result_path.exists():
                missing.append(recipe["id"])
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            print(f"Pilot recipe {recipe['id']}: backbone LR {recipe['backbone_lr']}, loss {recipe['loss_weight']}", flush=True)
            result = train_recipe(prepared["fitting"], prepared["validation"], config["model"],
                                  config["training"], recipe, folder, manifest,
                                  epochs=config["pilot"]["epochs"], device=device, deadline=deadline, resume=resume)
            result = {**_json_safe(result), "recipe_id": recipe["id"]}
            _atomic_json(result_path, result)
        best_checkpoint = result.get("best_checkpoint")
        if best_checkpoint and Path(best_checkpoint).exists():
            model, _ = load_model_checkpoint(best_checkpoint, device=device, expected_manifest=manifest)
            evaluation_training = {**config["training"], "microbatch": result["microbatch"]}
            predictions = predict_frame(model, prepared["validation"], config["model"], evaluation_training, device)
            predictions["inner_fold"] = 0
            predictions["split"] = "inner_validation"
            predictions["evaluation_status"] = "exploratory_not_outer_oof"
            predictions["run_id"] = config["run"]["run_id"]
            predictions["recipe_id"] = recipe["id"]
            predictions = predictions.sort_values("image_id").reset_index(drop=True)
            _check_predictions(predictions, prepared["validation"])
            _atomic_csv(folder / "inner_validation_predictions.csv", predictions)
            _atomic_csv(folder / "reliability.csv", reliability_table(
                predictions.target, predictions.base_probability, groups=predictions.patient_id))
            result["raw_validation_metrics"] = probability_metrics(predictions.target, predictions.base_probability)
            del model
        else:
            if predict_only:
                missing.append(recipe["id"])
        results.append(result)
        if not predict_only:
            _atomic_json(run_dir / "summary.json", _summary(config, results, time.monotonic() - start))
    if predict_only:
        receipt = {"status": "complete" if not missing else "incomplete", "mode": "predict_only",
                   "run_id": config["run"]["run_id"], "missing_recipe_checkpoints": missing,
                   "recipes": results, "training_history_preserved": True,
                   "outer_predictions_generated": False, "calibration_fitted": False}
        _atomic_json(run_dir / "prediction_receipt.json", receipt)
        return receipt
    summary = _summary(config, results, time.monotonic() - start)
    _atomic_json(run_dir / "summary.json", summary)
    return summary


__all__ = ["load_config", "validate_config", "prepare_pilot", "persist_preparation", "run_pilot", "reliability_table"]
