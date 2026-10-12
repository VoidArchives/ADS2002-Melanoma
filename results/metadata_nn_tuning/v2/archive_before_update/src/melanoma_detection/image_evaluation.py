"""Saved image-experiment verification and exploratory evaluation.

These helpers never train, choose thresholds, fit calibration or read the test.
Historical reporting verifies recorded identities, not current training source.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .metadata_evaluation import decision_metrics, probability_metrics, validate_probabilities


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


UNCERTAINTY_ORIGINAL = "notebooks/05a_image_uncertainty_d1_wip.ipynb"
UNCERTAINTY_ARCHIVE = "notebooks/archive/05a_image_uncertainty_d1_historical_three_epoch_fold0.ipynb"


def workspace_relative_path(root, relative):
    """Resolve a canonical repository path without escaping the workspace."""
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).is_absolute()
            or any(part in ("..", ".") or ":" in part for part in relative.split("/"))
            or PurePosixPath(relative).as_posix() != relative):
        raise ValueError(f"Invalid workspace-relative preservation path: {relative}")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Preservation path escapes workspace: {relative}")
    return path


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _snapshot_path(root, relative, snapshot, category):
    path = workspace_relative_path(root, snapshot)
    parts = PurePosixPath(snapshot).parts
    # Exact path suffixes bind snapshots to their original repository key.
    expected_suffix = (category, *PurePosixPath(relative).parts)
    if (parts[:2] != ("results", "notebook_organization") or len(parts) != 3 + len(expected_suffix)
            or parts[3:] != expected_suffix):
        raise ValueError(f"Snapshot is not the exact original path for {relative}.")
    return path


def notebook_rename_registry(root):
    """Read and validate the explicit rename/snapshot registry, if present.

    This registry never changes immutable historical receipts. Renamed files
    and original snapshots must still match each receipt's ORIGINAL digest.
    It authorises only exact old-to-new notebook keys and explicitly recorded
    supporting-source paths; there is no search for a similarly named file.
    """
    registry_path = workspace_relative_path(root, "notebooks/filename_map.json")
    if not registry_path.is_file():
        return {"schema_version": 1, "renames": [], "authorized_source_snapshots": {}}
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Notebook filename registry cannot be read.") from error
    if (not isinstance(registry, dict) or type(registry.get("schema_version")) is not int
            or registry["schema_version"] != 1 or not isinstance(registry.get("renames"), list)):
        raise ValueError("Notebook filename registry must use schema_version 1 and a renames list.")
    old_paths, new_paths = set(), set()
    for record in registry["renames"]:
        if not isinstance(record, dict) or not {"old_path", "new_path", "original_sha256", "snapshot_path"}.issubset(record):
            raise ValueError("Notebook filename registry has an incomplete rename record.")
        old, new = record["old_path"], record["new_path"]
        for relative in (old, new):
            workspace_relative_path(root, relative)
            if not relative.startswith("notebooks/") or not relative.endswith(".ipynb"):
                raise ValueError("Notebook rename paths must be .ipynb files under notebooks/.")
        if old in old_paths or new in new_paths:
            raise ValueError("Notebook filename registry old and new paths must each be unique.")
        old_paths.add(old)
        new_paths.add(new)
        if not _valid_sha256(record["original_sha256"]):
            raise ValueError("Notebook rename record has an invalid original SHA-256.")
        _snapshot_path(root, old, record["snapshot_path"], "originals")
    sources = registry.get("authorized_source_snapshots", {})
    if not isinstance(sources, dict):
        raise ValueError("authorized_source_snapshots must be a mapping of exact source paths.")
    for relative, record in sources.items():
        workspace_relative_path(root, relative)
        if (not relative.endswith((".py", ".md")) or not isinstance(record, dict)
                or not {"snapshot_path", "original_sha256"}.issubset(record)
                or not _valid_sha256(record["original_sha256"])):
            raise ValueError("Authorised source snapshot has an invalid path, record or original SHA-256.")
        _snapshot_path(root, relative, record["snapshot_path"], "original_sources")
    return registry


def _renamed_original(root, relative, expected, registry):
    for record in registry["renames"]:
        if record["old_path"] != relative:
            continue
        renamed = workspace_relative_path(root, record["new_path"])
        if renamed.is_file() and sha256(renamed) == expected:
            return renamed
        if record["original_sha256"] == expected:
            snapshot = _snapshot_path(root, relative, record["snapshot_path"], "originals")
            if snapshot.is_file() and sha256(snapshot) == expected:
                return snapshot
    return None


def preserved_artifact_path(root, relative, expected):
    """Locate exact historical bytes after an explicitly registered relocation.

    Try the original file, its exact renamed path, and the original snapshot
    recorded for that old key. The earlier three-epoch/fold-0 uncertainty
    notebook remains a named exception because its historical digest differs
    from the later current-workflow notebook copied during organisation.
    No output or receipt is changed and no mismatched snapshot is accepted.
    """
    path = workspace_relative_path(root, relative)
    if not _valid_sha256(expected):
        raise ValueError("Expected preservation SHA-256 is invalid.")
    if path.is_file() and sha256(path) == expected:
        return path
    registry = notebook_rename_registry(root)
    renamed = _renamed_original(root, relative, expected, registry)
    if renamed is not None:
        return renamed
    if relative == UNCERTAINTY_ORIGINAL:
        archive = workspace_relative_path(root, UNCERTAINTY_ARCHIVE)
        if archive.is_file() and sha256(archive) == expected:
            return archive
        archive = _renamed_original(root, UNCERTAINTY_ARCHIVE, expected, registry)
        if archive is not None:
            return archive
    raise ValueError(f"Preserved historical artifact changed: {relative}")


def classification_metrics(target, probability, threshold=0.5):
    """Raw score >= a declared fixed cutoff; undefined rates stay missing."""
    y, p = validate_probabilities(target, probability)
    if isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite score in [0, 1].")
    metrics = decision_metrics(y, p >= threshold)
    denominator = 2 * metrics["TP"] + metrics["FP"] + metrics["FN"]
    return {"threshold": float(threshold), "score_type": "raw_base_probability",
            "decision_rule": "score >= threshold", **metrics,
            "recall": metrics["sensitivity"],
            "f1": 2 * metrics["TP"] / denominator if denominator else np.nan}


def curve_tables(target, probability):
    """Actual ROC/PR coordinates; AP remains sklearn average precision."""
    y, p = validate_probabilities(target, probability)
    if np.unique(y).size != 2:
        raise ValueError("ROC/PR comparison requires both classes.")
    fpr, tpr, thresholds = roc_curve(y, p)
    precision, recall, pr_thresholds = precision_recall_curve(y, p)
    return {"roc": pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr,
                                 "threshold": thresholds}),
            "precision_recall": pd.DataFrame({"recall": recall, "precision": precision,
                                               "threshold": np.r_[pr_thresholds, np.nan]}),
            "average_precision": probability_metrics(y, p)["average_precision"]}


def assert_matching_predictions(frames):
    """Require exact paired image/patient/target/fold coverage, never inner join."""
    reference = None
    for identity, frame in frames.items():
        required = ["image_id", "patient_id", "target", "fold", "base_probability"]
        if not set(required).issubset(frame):
            raise ValueError(f"{identity}: missing prediction columns.")
        if frame.image_id.isna().any() or frame.image_id.duplicated().any():
            raise ValueError(f"{identity}: duplicate or missing image identifiers.")
        validate_probabilities(frame.target, frame.base_probability)
        paired = frame[["image_id", "patient_id", "target", "fold"]].sort_values("image_id").reset_index(drop=True)
        if reference is None:
            reference = paired
        elif not reference.equals(paired):
            raise ValueError(f"{identity}: prediction membership/labels differ.")


def load_historical_pilot(project_root, *, verify_checkpoints=True, verify_images=True):
    """Load the preserved pilot against its archive, artifacts and development data.

    Training resume remains governed by the trainer's current-source guard.
    No image, checkpoint or historical output is rewritten by this loader.
    """
    root = Path(project_root).resolve()
    run_dir = root / "results/image_d1_runs/b0_pilot_v1"
    report_dir = root / "results/image_d1_runs/b0_pilot_reporting_v2"
    receipt = json.loads((report_dir / "preservation_receipt.json").read_text(encoding="utf-8"))
    for relative, expected in receipt["protected_files"].items():
        path = root / relative
        if not verify_checkpoints and path.suffix == ".pt":
            continue
        preserved_artifact_path(root, relative, expected)
    for relative, expected in receipt["archived_sources"].items():
        path = report_dir / "archive" / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Historical source archive changed: {relative}")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["compatibility"]["source_sha256"].items():
        if sha256(report_dir / "archive" / relative) != expected:
            raise ValueError("Historical source identity disagrees with training manifest.")
    provenance = manifest["provenance"]
    csv_path = root / provenance["folds_csv"]
    if csv_path.resolve() != (root / "data/subset/processed/train_folds.csv").resolve():
        raise ValueError("Historical report is restricted to the d1 development CSV.")
    if sha256(csv_path) != provenance["full_csv_sha256"]:
        raise ValueError("Development CSV changed since the preserved pilot.")
    memberships = json.loads((run_dir / "split_memberships.json").read_text(encoding="utf-8"))
    pool = pd.DataFrame(memberships["training_pool"])
    excluded = pd.DataFrame(memberships["outer_excluded"])
    current = pd.read_csv(csv_path, usecols=["image_name", "patient_id", "fold", "target"],
                          dtype={"image_name": str, "patient_id": str})
    current = current.rename(columns={"image_name": "image_id"})
    active = current.loc[current.image_id.isin(pool.image_id), ["image_id", "patient_id", "fold", "target"]]
    expected = pool[["image_id", "patient_id", "fold", "target"]]
    if not active.sort_values("image_id").reset_index(drop=True).equals(expected.sort_values("image_id").reset_index(drop=True)):
        raise ValueError("Historical fitting/validation memberships changed.")
    image_dir = root / provenance["image_dir"]
    if image_dir.resolve() != (root / "data/subset/train").resolve():
        raise ValueError("Historical image paths must remain within d1 development images.")
    if verify_images:
        for row in manifest["compatibility"]["training_pool_images"]:
            if sha256(image_dir / f"{row['image_id']}.jpg") != row["source_sha256"]:
                raise ValueError(f"Historical development image changed: {row['image_id']}")
    pool["image_name"] = pool.image_id
    pool["image_path"] = pool.image_id.map(lambda value: image_dir / f"{value}.jpg")
    fitting = pool.loc[pool.image_id.isin(memberships["fitting_image_ids"])].reset_index(drop=True)
    validation = pool.loc[pool.image_id.isin(memberships["validation_image_ids"])].reset_index(drop=True)
    if set(fitting.patient_id) & set(validation.patient_id) or set(pool.patient_id) & set(excluded.patient_id):
        raise ValueError("Historical patient partitions overlap.")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    predictions, histories, records = {}, [], []
    recipes = manifest["compatibility"]["config"]["candidates"]["pilot_recipes"]
    for recipe in recipes:
        folder = run_dir / "recipes" / recipe["id"]
        result = json.loads((folder / "recipe_result.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(folder / "inner_validation_predictions.csv")
        identity = f"efficientnet_b0/{recipe['id']}"
        frame["pipeline_id"] = identity
        frame["architecture"] = "efficientnet_b0"
        assert_matching_predictions({identity: frame, "expected": validation.assign(base_probability=.5)})
        metrics = probability_metrics(frame.target, frame.base_probability)
        for key, value in metrics.items():
            if not np.isclose(value, result["best_metrics"][key], rtol=0, atol=1e-12):
                raise ValueError(f"Historical prediction metric mismatch: {identity}/{key}")
        history = pd.read_csv(folder / "history.csv").assign(pipeline_id=identity,
            recipe_id=recipe["id"], architecture="efficientnet_b0", loss_weight=recipe["loss_weight"],
            backbone_lr=recipe["backbone_lr"], selected_epoch=result["best_epoch"])
        predictions[identity] = frame
        histories.append(history)
        records.append({"pipeline_id": identity, "architecture": "efficientnet_b0", **recipe,
                        "recipe_id": recipe["id"], "best_epoch": result["best_epoch"],
                        "status": result["status"], "completed_epochs": int(history.epoch.max()),
                        **metrics})
    assert_matching_predictions(predictions)
    return {"run_dir": run_dir, "report_dir": report_dir, "manifest": manifest, "summary": summary,
            "fitting": fitting, "validation": validation, "outer_excluded": excluded,
            "predictions": predictions, "histories": pd.concat(histories, ignore_index=True),
            "records": pd.DataFrame(records), "image_audit": json.loads((run_dir / "image_audit.json").read_text()),
            "verification": {"archive_verified": True, "artifacts_verified": True,
                             "checkpoints_verified": verify_checkpoints, "images_verified": verify_images,
                             "reserved_test_loaded": False}}


def metric_table(frames, *, threshold=.5, epochs=None):
    assert_matching_predictions(frames)
    rows = []
    for identity, frame in frames.items():
        rows.append({"pipeline_id": identity, "selected_epoch": (epochs or {}).get(identity),
                     "n_images": len(frame), "n_patients": frame.patient_id.nunique(),
                     "n_melanoma": int(frame.target.sum()),
                     "evaluation_status": "exploratory_inner_validation_not_outer_oof",
                     **probability_metrics(frame.target, frame.base_probability),
                     **classification_metrics(frame.target, frame.base_probability, threshold)})
    return pd.DataFrame(rows)


def load_architecture_outputs(project_root, summary):
    """Read executed trial outputs; unfinished trials are never comparison inputs."""
    root = Path(project_root).resolve()
    predictions, histories = {}, []
    records = summary.get("recipes", [])
    records = list(records.values()) if isinstance(records, dict) else records
    reference = summary.get("b0_reference", [])
    reference = list(reference.values()) if isinstance(reference, dict) else reference
    for record in [*reference, *records]:
        rows = record.get("history", [])
        if rows:
            identity = record["pipeline_id"]
            architecture = record.get("architecture", record.get("family"))
            hist = pd.DataFrame(rows).assign(pipeline_id=identity, architecture=architecture,
                recipe_id=record["recipe_id"], recipe_slot=record["recipe_slot"],
                selected_epoch=record.get("best_epoch"), run_status=record.get("status"))
            hist["loss_weight"] = "sqrt_ratio" if "sqrt" in record["recipe_slot"] else "none"
            histories.append(hist)
        if record.get("status") != "complete" or not record.get("predictions_file"):
            continue
        if record.get("completed_epochs", 6) != 6:
            raise ValueError("A completed exploratory comparison must contain six epochs.")
        path = Path(record["predictions_file"])
        path = path if path.is_absolute() else root / path
        frame = pd.read_csv(path)
        identity = record["pipeline_id"]
        frame["pipeline_id"] = identity
        frame["architecture"] = record.get("architecture", record.get("family"))
        frame["recipe_slot"] = record["recipe_slot"]
        computed = probability_metrics(frame.target, frame.base_probability)
        if record.get("best_metrics") and any(not np.isclose(value, record["best_metrics"][name],
                rtol=0, atol=1e-12) for name, value in computed.items()):
            raise ValueError(f"Saved prediction metrics disagree with best checkpoint: {identity}")
        predictions[identity] = frame
    assert_matching_predictions(predictions)
    slots = set(summary.get("matched_recipe_slots", []))
    matched = {identity: frame for identity, frame in predictions.items()
               if frame.recipe_slot.iloc[0] in slots}
    for slot in slots:
        families = {frame.architecture.iloc[0] for frame in matched.values() if frame.recipe_slot.iloc[0] == slot}
        if families != {"efficientnet_b0", "densenet121", "convnext_tiny"}:
            raise ValueError("Matched comparison slot does not contain all three completed families.")
    return {"predictions": predictions, "matched_predictions": matched,
            "histories": pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()}
