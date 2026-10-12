"""Read-only uncertainty inference for explicitly selected image pilot checkpoints.

The saved training split and preparation are the authority. Nothing in this
module trains, calibrates, chooses an architecture, or reads the reserved test.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import image_model as core


DEFAULT_RUN = "b0_augmentation_12e_v1"
DEFAULT_RECIPE = "lr3e5_sqrt"
TTA_PASSES = ("identity", "hflip", "vflip", "hvflip")
DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                 nn.AlphaDropout, nn.FeatureAlphaDropout)
EVALUATION_STATUS = "exploratory_inner_validation_not_outer_oof_or_final_test"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _within(path, directory):
    result = Path(path).resolve()
    if not result.is_relative_to(Path(directory).resolve()):
        raise ValueError("Recorded artifact path is outside the selected run directory.")
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def verify_checkpoint_compatibility(checkpoint, recipe_manifest, result):
    """Check checkpoint metadata against its hashed identity and recorded history."""
    for key in ("spec", "training", "recipe", "manifest"):
        if core.canonical_hash(checkpoint[key]) != core.canonical_hash(checkpoint["identity"][key]):
            raise ValueError(f"Checkpoint {key} differs from its compatibility identity.")
        if core.canonical_hash(checkpoint[key]) != core.canonical_hash(recipe_manifest[key]):
            raise ValueError(f"Checkpoint {key} differs from the saved recipe manifest.")
    policy = checkpoint["identity"].get("architecture_policy")
    current_policy = core.architecture_policy(checkpoint["spec"])
    if policy is not None and "transformation_profile" not in policy:
        if checkpoint["spec"].get("augmentation") != "weak" or checkpoint["spec"].get("preprocessing") != "none":
            raise ValueError("A checkpoint without an explicit preparation profile must use historical weak/none preparation.")
        # The six-epoch architecture pilot predates named transformation
        # profiles. Retain its exact policy contract; never infer CLAHE/new QA.
        current_policy.pop("transformation_profile")
    if policy is None or core.canonical_hash(policy) != core.canonical_hash(current_policy):
        raise ValueError("Checkpoint architecture/preprocessing policy is incompatible with current preparation.")
    if checkpoint["epoch"] != result["best_epoch"] or checkpoint["best_epoch"] != result["best_epoch"]:
        raise ValueError("Checkpoint epoch differs from the recorded best epoch.")
    history = result.get("history", [])
    if not history or not all(np.isfinite(row.get("average_precision", np.nan)) for row in history):
        raise ValueError("Recorded training history is absent or contains invalid selection metrics.")
    # Training keeps the first epoch when AP ties (strict > improvement).
    selected = max(history, key=lambda row: row["average_precision"])
    if selected["epoch"] != result["best_epoch"]:
        raise ValueError("Recorded best epoch disagrees with the training-history selection rule.")
    for metric, value in checkpoint["best_metrics"].items():
        if not np.isclose(value, result["best_metrics"][metric], atol=1e-12, rtol=0):
            raise ValueError("Checkpoint best metrics differ from the recorded result.")
    return True


def verify_patient_separation(fitting, validation, excluded):
    """Fail on image leakage, patient leakage, or a split crossing a patient."""
    partitions = (fitting, validation, excluded)
    for frame in partitions:
        if frame.empty or frame.image_name.duplicated().any():
            raise ValueError("A partition is empty or contains duplicate images.")
        if frame.patient_id.isna().any() or frame.image_name.isna().any():
            raise ValueError("Missing patient/image identity in a partition.")
    for index, left in enumerate(partitions):
        for right in partitions[index + 1:]:
            if set(left.patient_id.astype(str)) & set(right.patient_id.astype(str)):
                raise ValueError("Evaluation, fitting, and excluded patients must be separate.")
            if set(left.image_name.astype(str)) & set(right.image_name.astype(str)):
                raise ValueError("Images overlap between partitions.")
    if fitting.fold.eq(0).any() or validation.fold.eq(0).any() or not excluded.fold.eq(0).all():
        raise ValueError("Outer fold 0 must remain excluded from fitting and diagnostics.")
    return {"patient_overlap": 0, "image_overlap": 0,
            "fitting_images": len(fitting), "fitting_patients": int(fitting.patient_id.nunique()),
            "validation_images": len(validation), "validation_patients": int(validation.patient_id.nunique()),
            "validation_malignant_images": int(validation.target.sum()),
            "outer_excluded_images": len(excluded), "outer_excluded_patients": int(excluded.patient_id.nunique())}


def reconstruct_partitions(root, run_manifest, memberships, checkpoint):
    """Reconstruct recorded inner memberships without rerunning a split algorithm.

    Outer targets are dropped immediately and never passed to evaluation. The
    exact ordered training identities, including absolute image paths, must
    match those embedded in the checkpoint.
    """
    root = Path(root).resolve()
    csv_path = root / "data/subset/processed/train_folds.csv"
    expected_csv = run_manifest.get("provenance", {}).get("full_csv_sha256")
    if expected_csv and core.file_sha256(csv_path) != expected_csv:
        raise ValueError("Current d1 train_folds.csv differs from the recorded input hash.")
    inventory = pd.read_csv(csv_path, dtype={"image_name": str, "patient_id": str})
    required = {"image_name", "patient_id", "fold", "target"}
    if not required.issubset(inventory) or inventory.image_name.duplicated().any():
        raise ValueError("The d1 fold inventory is missing columns or contains duplicate images.")
    inventory = inventory[sorted(required)].sort_values("image_name").reset_index(drop=True)
    for column in ("image_name", "patient_id"):
        if inventory[column].isna().any() or inventory[column].astype(str).str.strip().eq("").any():
            raise ValueError("Missing image or patient identifier in d1 inventory.")
    fold_values = pd.to_numeric(inventory.fold, errors="raise")
    if not np.isfinite(fold_values).all() or not fold_values.mod(1).eq(0).all():
        raise ValueError("Invalid outer fold values.")
    inventory["fold"] = fold_values.astype(int)
    if set(inventory.fold) != set(range(5)) or inventory.groupby("patient_id").fold.nunique().gt(1).any():
        raise ValueError("Outer folds are invalid or divide a patient.")
    # Do not allow identifiers to escape the fixed d1 image directory.
    if not inventory.image_name.str.fullmatch(r"[A-Za-z0-9_-]+").all():
        raise ValueError("Unsafe image identifiers in d1 inventory.")
    inventory["image_path"] = inventory.image_name.map(lambda name: root / "data/subset/train" / f"{name}.jpg")
    excluded = inventory[inventory.fold == 0].drop(columns="target").reset_index(drop=True)
    pool = inventory[inventory.fold != 0].copy().reset_index(drop=True)
    pool["target"] = pd.to_numeric(pool.target, errors="raise")
    if not pool.target.isin([0, 1]).all():
        raise ValueError("Training pool targets must be binary.")
    pool["target"] = pool.target.astype(int)
    recorded = pd.DataFrame(memberships["training_pool"]).rename(columns={"image_id": "image_name"})
    required_members = ["image_name", "patient_id", "fold", "target", "inner_fold"]
    if not set(required_members).issubset(recorded) or recorded.image_name.duplicated().any():
        raise ValueError("Recorded training memberships are invalid.")
    columns = ["image_name", "patient_id", "fold", "target"]
    current = pool[columns].to_dict("records")
    saved = recorded[columns].sort_values("image_name").to_dict("records")
    if current != saved:
        raise ValueError("Current d1 training identities/labels differ from recorded memberships.")
    compat_members = run_manifest["compatibility"]["training_pool_memberships"]
    if core.canonical_hash(memberships["training_pool"]) != core.canonical_hash(compat_members):
        raise ValueError("Split file differs from memberships bound into the run manifest.")
    expected_excluded = sorted(memberships["outer_excluded"], key=lambda item: item["image_id"])
    actual_excluded = excluded.rename(columns={"image_name": "image_id"})[["image_id", "patient_id", "fold"]].to_dict("records")
    if actual_excluded != expected_excluded:
        raise ValueError("Outer excluded image/patient identities differ from the saved split.")
    pool = pool.merge(recorded[["image_name", "inner_fold"]], on="image_name", validate="one_to_one")
    if set(pool.inner_fold) != {0, 1, 2} or pool.groupby("patient_id").inner_fold.nunique().gt(1).any():
        raise ValueError("Inner membership divides a patient or has invalid folds.")
    fitting = pool[pool.inner_fold != 0].reset_index(drop=True)
    validation = pool[pool.inner_fold == 0].reset_index(drop=True)
    separation = verify_patient_separation(fitting, validation, excluded)
    for name, frame in (("fitting", fitting), ("validation", validation)):
        if core._frame_identity(frame) != checkpoint["identity"][name]:
            raise ValueError(f"Reconstructed {name} does not match checkpoint frame identity.")
    core.validate_frame(validation)
    # Bind evaluation image bytes, not just identifiers. Fitting membership is
    # verified above; inference never needs to open fitting or outer images.
    image_hashes = {row["image_id"]: row["source_sha256"]
                    for row in run_manifest["compatibility"]["training_pool_images"]}
    for row in validation.itertuples():
        if image_hashes.get(row.image_name) != core.file_sha256(row.image_path):
            raise ValueError(f"Evaluation image bytes differ from the run audit: {row.image_name}.")
    separation["evaluation_image_hashes_verified"] = len(validation)
    separation["train_folds_sha256"] = core.file_sha256(csv_path)
    return fitting, validation, excluded, separation


def load_selected_checkpoint(root, run_id=DEFAULT_RUN, recipe_id=DEFAULT_RECIPE,
                             architecture="efficientnet_b0", condition_id="A", device="cuda"):
    """Load an explicitly requested recorded best checkpoint via the shared loader.

    Supported selections are B0 conditions A/B/C in the matched augmentation run
    and explicit DenseNet121/ConvNeXt-Tiny recipes in the architecture pilot.
    There is no score-based default selection or search for another checkpoint.
    """
    root = Path(root).resolve()
    runs = root / "results/image_d1_runs"
    if run_id == DEFAULT_RUN and architecture == "efficientnet_b0":
        conditions = {"A": "A_weak_none", "B": "B_stronger_none", "C": "C_weak_clahe"}
        if condition_id not in conditions or recipe_id != DEFAULT_RECIPE:
            raise ValueError("Select an existing B0 condition and its explicit lr3e5_sqrt recipe.")
        recipe_dir = runs / run_id / "conditions" / conditions[condition_id]
        result_name = "condition_result.json"
    elif run_id == "architecture_pilot_3h_v1" and architecture in ("densenet121", "convnext_tiny"):
        if recipe_id not in ("low_none", "low_sqrt", "high_none", "high_sqrt"):
            raise ValueError("Architecture checkpoint selection requires an explicit supported recipe slot.")
        recipe_dir = runs / run_id / "families" / architecture / "recipes" / recipe_id
        result_name = "recipe_result.json"
    else:
        raise ValueError("Unsupported run/architecture selection; no automatic pilot winner is selected.")
    run_dir = runs / run_id
    result = _read_json(recipe_dir / result_name)
    if result.get("status") != "complete" or not result.get("best_checkpoint"):
        raise ValueError("The selected recipe has no complete recorded best checkpoint.")
    checkpoint_path = _within(result["best_checkpoint"], recipe_dir)
    if checkpoint_path.name != "best.pt":
        raise ValueError("The recipe result must explicitly identify its recorded best.pt checkpoint.")
    recipe_manifest = _read_json(recipe_dir / "manifest.json")
    expected_manifest = recipe_manifest["manifest"]
    model, checkpoint = core.load_model_checkpoint(checkpoint_path, device=device,
                                                    expected_manifest=expected_manifest)
    verify_checkpoint_compatibility(checkpoint, recipe_manifest, result)
    if checkpoint["spec"]["architecture"] != architecture or checkpoint["recipe"]["id"] != recipe_id:
        raise ValueError("Checkpoint architecture/recipe differs from explicit selection.")
    if expected_manifest.get("run_id") != run_id:
        raise ValueError("Checkpoint belongs to a different run.")
    if architecture == "efficientnet_b0" and expected_manifest.get("condition_id") != condition_id:
        raise ValueError("Checkpoint condition differs from explicit selection.")
    run_manifest = _read_json(run_dir / "run_manifest.json")
    if core.canonical_hash(run_manifest["compatibility"]) != run_manifest["compatibility_sha256"]:
        raise ValueError("The run compatibility manifest is corrupt.")
    if expected_manifest["compatibility_sha256"] != run_manifest["compatibility_sha256"]:
        raise ValueError("Checkpoint is incompatible with the recorded data run.")
    memberships_path = run_dir / "split_memberships.json"
    fitting, validation, excluded, separation = reconstruct_partitions(
        root, run_manifest, _read_json(memberships_path), checkpoint)
    predictions_path = _within(result["predictions_file"], recipe_dir)
    if (result.get("predictions_file_sha256") and
            core.file_sha256(predictions_path) != result["predictions_file_sha256"]):
        raise ValueError("Saved deterministic predictions differ from their recorded hash.")
    saved = pd.read_csv(predictions_path, dtype={"image_id": str, "patient_id": str})
    if set(saved.image_id) != set(validation.image_name):
        raise ValueError("Saved predictions do not have exact inner-validation image coverage.")
    compare_saved_predictions(validation, saved)
    dropout = [{"name": name, "p": float(module.p)} for name, module in model.named_modules()
               if isinstance(module, DROPOUT_TYPES)]
    if not dropout or any(item["p"] != checkpoint["spec"]["dropout"] for item in dropout):
        raise ValueError("Explicit dropout layers do not preserve the recorded trained dropout rate.")
    provenance = {"run_id": run_id, "recipe_id": recipe_id, "architecture": architecture,
                  "condition_id": condition_id if architecture == "efficientnet_b0" else None,
                  "checkpoint": str(checkpoint_path), "checkpoint_sha256": core.file_sha256(checkpoint_path),
                  "checkpoint_compatibility_hash": checkpoint["compatibility_hash"],
                  "best_epoch": int(checkpoint["epoch"]), "training_seed": checkpoint["training"]["seed"],
                  "run_compatibility_sha256": run_manifest["compatibility_sha256"],
                  "split_memberships": str(memberships_path),
                  "split_memberships_sha256": core.file_sha256(memberships_path),
                  "fitting_frame_sha256": checkpoint["identity"]["fitting"],
                  "validation_frame_sha256": checkpoint["identity"]["validation"],
                  "fitting_image_ids": fitting.image_name.tolist(),
                  "fitting_patient_ids": sorted(fitting.patient_id.unique().tolist()),
                  "evaluation_image_ids": validation.image_name.tolist(),
                  "evaluation_patient_ids": sorted(validation.patient_id.unique().tolist()),
                  "outer_excluded_patient_ids": sorted(excluded.patient_id.unique().tolist()),
                  "evaluation_status": EVALUATION_STATUS, "outer_fold": 0, "inner_fold": 0,
                  "outer_fold_evaluated": False, "reserved_test_loaded": False,
                  "calibrator_fitted": False, "referral_threshold_selected": False,
                  "spec": checkpoint["spec"], "training": checkpoint["training"],
                  "preprocessing": core.transformation_profile(checkpoint["spec"]),
                  "trained_dropout_layers": dropout, "patient_separation": separation,
                  "saved_predictions": str(predictions_path),
                  "saved_predictions_sha256": core.file_sha256(predictions_path),
                  "environment": core._environment(device)}
    return {"model": model, "checkpoint": checkpoint, "spec": checkpoint["spec"],
            "training": {**checkpoint["training"], "microbatch": checkpoint["microbatch"]}, "frame": validation,
            "fitting_frame": fitting, "excluded_frame": excluded,
            "saved_predictions": saved, "recipe_dir": recipe_dir,
            "device": str(device), "provenance": provenance}


def compare_saved_predictions(frame, saved, scores=None, *, atol=1e-6):
    """Align by identifiers and check labels plus finite saved score values."""
    if not {"image_id", "patient_id", "target", "base_probability"}.issubset(saved):
        raise ValueError("Saved deterministic predictions lack expected fields.")
    if saved.image_id.duplicated().any() or saved.image_id.isna().any():
        raise ValueError("Saved deterministic predictions have duplicate/missing image keys.")
    expected = frame.rename(columns={"image_name": "image_id"})
    if not set(expected.image_id).issubset(set(saved.image_id)):
        raise ValueError("Saved deterministic predictions omit expected images.")
    observed = saved.set_index("image_id").loc[expected.image_id]
    if (observed.patient_id.tolist() != expected.patient_id.tolist() or
            observed.target.tolist() != expected.target.tolist()):
        raise ValueError("Saved prediction patient/label identities differ from evaluation.")
    actual = observed.base_probability.to_numpy(dtype=float)
    if not np.isfinite(actual).all() or ((actual < 0) | (actual > 1)).any():
        raise ValueError("Saved deterministic scores are non-finite or outside [0, 1].")
    if scores is None:
        return 0.0
    scores = np.asarray(scores, dtype=float)
    if scores.shape != actual.shape or not np.allclose(scores, actual, atol=atol, rtol=0):
        raise ValueError("Reconstructed deterministic predictions disagree with the recorded checkpoint predictions.")
    return float(np.max(np.abs(scores - actual)))


@contextmanager
def explicit_dropout_only(model, *, active=True):
    """Only explicit dropout is stochastic; restore all prior module modes/p.

    Calling model.train() would also activate native stochastic depth and
    BatchNorm updates. Set dropout children individually after model.eval().
    """
    modes = {module: module.training for module in model.modules()}
    dropout_p = {module: module.p for module in model.modules() if isinstance(module, DROPOUT_TYPES)}
    model.eval()
    for module in dropout_p:
        module.train(active)
    try:
        if any(module.training for module in model.modules() if not isinstance(module, DROPOUT_TYPES)):
            raise AssertionError("A non-dropout module is active during inference.")
        yield
        if any(module.p != value for module, value in dropout_p.items()):
            raise AssertionError("Inference altered a trained dropout rate.")
    finally:
        for module, value in dropout_p.items():
            module.p = value
        for module, training in modes.items():
            module.training = training


def batchnorm_buffers(model):
    return {f"{name}.{key}": value.detach().cpu().clone()
            for name, module in model.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            for key, value in module.named_buffers(recurse=False)}


def assert_unchanged_batchnorm(model, before):
    after = batchnorm_buffers(model)
    if set(before) != set(after) or any(not torch.equal(before[key], after[key]) for key in before):
        raise AssertionError("BatchNorm running buffers changed during uncertainty inference.")
    return len(before)


def prepare_tensor_cache(context, frame=None):
    """Decode deterministic shared preparation once, retaining exact float32 values."""
    frame = context["frame"] if frame is None else frame
    core.validate_frame(frame, require_both_classes=False)
    size = context["spec"]["image_size"]
    tensors = torch.empty((len(frame), 3, size, size), dtype=torch.float32)
    dataset = core.ImageFrameDataset(frame, core.make_transform(context["spec"], training=False))
    offset = 0
    for images, _ in DataLoader(dataset, batch_size=context["training"]["microbatch"],
                                shuffle=False, num_workers=0):
        tensors[offset:offset + len(images)] = images
        offset += len(images)
    if offset != len(frame) or not torch.isfinite(tensors).all():
        raise ValueError("Deterministic prepared cache is incomplete or non-finite.")
    return tensors


def tta_transform(images, pass_id):
    """Whole-image tensor flips after recorded preparation; no interpolation/padding."""
    if pass_id == "identity":
        return images
    if pass_id == "hflip":
        return torch.flip(images, (-1,))
    if pass_id == "vflip":
        return torch.flip(images, (-2,))
    if pass_id == "hvflip":
        return torch.flip(images, (-2, -1))
    raise ValueError(f"Unsupported conservative TTA transformation: {pass_id}.")


@torch.inference_mode()
def predict_cached(model, tensors, *, device="cpu", batch_size=64, pass_id="identity"):
    """Infer current modes with full precision; reject invalid scores immediately."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if len(tensors) == 0:
        raise ValueError("Prediction cache must not be empty.")
    scores = []
    for offset in range(0, len(tensors), batch_size):
        images = tta_transform(tensors[offset:offset + batch_size], pass_id).to(device)
        logits = model(images).reshape(-1)
        if logits.numel() != len(images) or not torch.isfinite(logits).all():
            raise ValueError("Raw prediction logits are non-finite or have the wrong shape.")
        values = torch.sigmoid(logits.float()).cpu().numpy()
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Raw prediction scores must be finite values in [0, 1].")
        scores.extend(values.tolist())
    return np.asarray(scores, dtype=float)


def _prediction_frame(frame, values, score_name="score"):
    result = frame.drop(columns="image_path").reset_index(drop=True).copy()
    result["image_id"] = result["image_name"]
    result[score_name] = values
    return result


def run_smoke(context, n_images=16, seed=20261004):
    """Short, read-only check before the full 30-pass inner-validation run."""
    if n_images < 2:
        raise ValueError("Smoke check needs at least two images.")
    core.seed_everything(seed)
    frame = context["frame"]
    positive = frame[frame.target == 1].head(min(2, n_images // 2))
    negative = frame[frame.target == 0].head(n_images - len(positive))
    sample = pd.concat([negative, positive]).sort_values("image_name").reset_index(drop=True)
    tensors = prepare_tensor_cache(context, sample)
    repeated = prepare_tensor_cache(context, sample)
    if not torch.equal(tensors, repeated):
        raise AssertionError("Deterministic preprocessing is not repeatable.")
    model, device = context["model"], context["device"]
    state_before = core.model_state_sha256(model)
    bn_before = batchnorm_buffers(model)
    batch = context["training"]["microbatch"]
    with explicit_dropout_only(model, active=False):
        deterministic = predict_cached(model, tensors, device=device, batch_size=batch)
        repeated_scores = predict_cached(model, repeated, device=device, batch_size=batch)
        identity = predict_cached(model, tensors, device=device, batch_size=batch, pass_id="identity")
    if not np.array_equal(deterministic, repeated_scores) or not np.array_equal(deterministic, identity):
        raise AssertionError("Deterministic repeat/identity TTA scores differ.")
    # Reproduce saved scores with their original microbatch composition. CUDA
    # convolution algorithms can differ numerically for a 16-image smoke batch
    # versus the original 64-image export, even with deterministic algorithms.
    positions = frame.index[frame.image_name.isin(sample.image_name)].to_numpy()
    block_starts = sorted(set(int(index // batch) * batch for index in positions))
    reference = pd.concat([frame.iloc[start:start + batch] for start in block_starts], ignore_index=True)
    reference_tensors = prepare_tensor_cache(context, reference)
    with explicit_dropout_only(model, active=False):
        reference_scores = predict_cached(model, reference_tensors, device=device, batch_size=batch)
    saved_delta = compare_saved_predictions(reference, context["saved_predictions"], reference_scores)
    activation_calls = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, DROPOUT_TYPES):
            activation_calls[name] = 0
            def record_activation(module, arguments, output, module_name=name):
                if not module.training:
                    raise AssertionError("Explicit dropout forward hook found inactive dropout.")
                activation_calls[module_name] += 1
            handles.append(module.register_forward_hook(record_activation))
    try:
        with explicit_dropout_only(model):
            core.seed_everything(seed)
            first = predict_cached(model, tensors, device=device, batch_size=batch)
            core.seed_everything(seed)
            repeated_mc = predict_cached(model, tensors, device=device, batch_size=batch)
            core.seed_everything(seed + 1)
            second = predict_cached(model, tensors, device=device, batch_size=batch)
    finally:
        for handle in handles:
            handle.remove()
    if not activation_calls or not all(activation_calls.values()):
        raise AssertionError("No explicit dropout layer actually ran during MC inference.")
    if not np.array_equal(first, repeated_mc):
        raise AssertionError("Seeded MC dropout is not repeatable.")
    if np.array_equal(first, second):
        raise AssertionError("Changing MC seed did not change any score; dropout activation needs review.")
    buffers = assert_unchanged_batchnorm(model, bn_before)
    if core.model_state_sha256(model) != state_before:
        raise AssertionError("Smoke inference changed model parameters or buffers.")
    return {"passed": True, "images": len(sample), "malignant_images": int(sample.target.sum()),
            "saved_comparison_original_batch_images": len(reference),
            "saved_comparison_original_microbatch": batch,
            "seed": seed, "preprocessing_repeatable": True, "deterministic_repeatable": True,
            "identity_tta_agreement": True, "saved_prediction_max_abs_difference": saved_delta,
            "saved_prediction_tolerance": 1e-6, "seeded_mc_repeatable": True,
            "mc_seed_changes_predictions": True, "dropout_forward_calls": activation_calls,
            "only_explicit_dropout_active": True, "trained_dropout_rate_preserved": True,
            "batchnorm_buffers_checked": buffers, "batchnorm_buffers_unchanged": True,
            "model_state_unchanged": True}


def snapshot_preserved_artifacts(root, archive_path=None):
    """Fingerprint existing checkpoints, histories and historical prototype outputs."""
    root = Path(root).resolve()
    paths = set()
    for base in (root / "results/image_d1_runs", root / "models"):
        if base.exists():
            for pattern in ("*.pt", "*.pt.sha256.json", "*.pth", "*.pth.sha256.json", "*history*.csv", "*history*.json",
                            "condition_result.json", "recipe_result.json"):
                paths.update(path for path in base.rglob(pattern) if "source_snapshots" not in path.parts)
    for base in (root / "results/predictions", root / "results/metrics", root / "results/figures"):
        if base.exists():
            paths.update(base.glob("*prototype_fold0_d1*"))
    if archive_path is not None:
        paths.add(Path(archive_path).resolve())
    return {str(path): core.file_sha256(path) for path in sorted(paths) if path.is_file()}


def assert_preserved_artifacts(snapshot):
    changed = [path for path, digest in snapshot.items()
               if not Path(path).is_file() or core.file_sha256(path) != digest]
    if changed:
        raise AssertionError(f"Inference modified protected artifacts: {changed}.")
    return {"protected_files_checked": len(snapshot), "all_protected_files_unchanged": True}


def run_inference(context, mc_passes=30, seed=20261004, batch_size=None,
                  progress: Callable | None = None):
    """Separate deterministic, MC-dropout, and conservative TTA predictions.

    Returns raw passes before any aggregation. Each MC pass has a separately
    recorded seed. TTA is deterministic and runs with dropout disabled.
    """
    if not isinstance(mc_passes, int) or isinstance(mc_passes, bool) or mc_passes < 2:
        raise ValueError("MC inference requires at least two complete passes.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Inference seed must be a nonnegative integer.")
    core.seed_everything(seed)
    model, frame = context["model"], context["frame"]
    device = context["device"]
    batch_size = batch_size or context["training"]["microbatch"]
    if batch_size != context["training"]["microbatch"]:
        raise ValueError("Use the recorded prediction microbatch to reproduce saved deterministic scores.")
    state_before = core.model_state_sha256(model)
    bn_before = batchnorm_buffers(model)
    protected = {str(path): core.file_sha256(path) for path in context["recipe_dir"].iterdir()
                 if path.is_file() and (path.suffix in (".pt", ".csv", ".json"))}
    tensors = prepare_tensor_cache(context)
    if progress:
        progress(f"Prepared {len(frame)} exact deterministic image tensors ({tensors.numel() * 4 / 2**20:.0f} MiB).")
    with explicit_dropout_only(model, active=False):
        deterministic = predict_cached(model, tensors, device=device, batch_size=batch_size)
    saved_delta = compare_saved_predictions(frame, context["saved_predictions"], deterministic)
    mc_rows = []
    with explicit_dropout_only(model):
        for pass_id in range(1, mc_passes + 1):
            pass_seed = seed + pass_id - 1
            core.seed_everything(pass_seed)
            values = predict_cached(model, tensors, device=device, batch_size=batch_size)
            raw = _prediction_frame(frame, values)
            raw["pass_id"], raw["seed"] = pass_id, pass_seed
            mc_rows.append(raw)
            if progress:
                progress(f"MC dropout pass {pass_id}/{mc_passes} complete.")
    tta_rows = []
    with explicit_dropout_only(model, active=False):
        for pass_id in TTA_PASSES:
            values = predict_cached(model, tensors, device=device, batch_size=batch_size, pass_id=pass_id)
            raw = _prediction_frame(frame, values)
            raw["pass_id"], raw["seed"] = pass_id, seed
            tta_rows.append(raw)
            if pass_id == "identity" and not np.array_equal(deterministic, values):
                raise AssertionError("Identity TTA disagrees with deterministic predictions.")
            if progress:
                progress(f"Deterministic TTA {pass_id} complete.")
    buffers_checked = assert_unchanged_batchnorm(model, bn_before)
    if core.model_state_sha256(model) != state_before:
        raise AssertionError("Inference altered model parameters or buffers.")
    preservation = assert_preserved_artifacts(protected)
    checks = {"checkpoint_compatibility": True, "patient_separation": True,
              "evaluation_image_hashes_verified": len(frame),
              "saved_predictions_agree": True, "saved_prediction_max_abs_difference": saved_delta,
              "identity_tta_agreement": True, "only_explicit_dropout_active": True,
              "trained_dropout_rate_preserved": True, "batchnorm_buffers_checked": buffers_checked,
              "batchnorm_buffers_unchanged": True, "model_state_unchanged": True, **preservation}
    provenance = {**context["provenance"], "inference": {
        "mc_passes": mc_passes, "mc_seeds": list(range(seed, seed + mc_passes)),
        "base_seed": seed, "batch_size": batch_size, "amp": False,
        "score": "sigmoid_of_one_logit_uncalibrated_prediction_score",
        "mc_dropout": "explicit_dropout_only_at_trained_rate",
        "batchnorm": "evaluation_mode_buffers_unchanged", "stochastic_depth": "evaluation_mode",
        "tta_passes": list(TTA_PASSES), "tta_application": "after_exact_deterministic_preparation",
        "tta_seed": seed, "tta_stochastic": False, "tta_dropout": False,
        "tta_geometry": "whole_image_tensor_flips_no_crop_padding_rotation_or_interpolation",
        "prepared_cache_dtype": "float32", "prepared_cache_device": "cpu",
        "deterministic_algorithms": True, "cudnn_benchmark": False,
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "model_state_sha256_before_and_after": state_before}}
    return {"deterministic": _prediction_frame(frame, deterministic, "base_score"),
            "mc_raw": pd.concat(mc_rows, ignore_index=True),
            "tta_raw": pd.concat(tta_rows, ignore_index=True),
            "checks": checks, "provenance": provenance}
