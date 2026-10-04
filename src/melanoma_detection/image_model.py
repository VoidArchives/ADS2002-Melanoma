"""Reusable, patient-safe image training for the staged d1 experiments.

EfficientNet-B0, DenseNet121 and ConvNeXt-Tiny share versioned transformation
profiles. Unsupported adapters fail explicitly. The pilot
orchestrator owns data audits, patient folds, budgets and candidate selection;
this module owns one reproducible recipe fit. It never reads a reserved test.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import time
from typing import Callable, Mapping
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from .metadata_evaluation import probability_metrics
from . import image_transforms as transform_profiles


CHECKPOINT_VERSION = 1
DEFAULT_SPEC = {
    "architecture": "efficientnet_b0", "weights": "IMAGENET1K_V1",
    "image_size": 224, "dropout": 0.2, "augmentation": "weak",
    "preprocessing": "none",
}
DEFAULT_TRAINING = {
    "microbatch": 64, "effective_batch": 64, "min_microbatch": 4,
    "head_lr": 1e-3, "weight_decay": 1e-4, "warmup_epochs": 1,
    "schedule_epochs": 30, "min_lr_factor": 0.1,
    "gradient_clip": 1.0, "amp": True, "num_workers": 0, "seed": 42,
}

# Architecture identities are explicit, rather than torchvision's changing
# DEFAULT aliases. Native centre crops are deliberately replaced by the same
# whole-image preparation for this controlled architecture pilot.
ARCHITECTURES = {
    "efficientnet_b0": {
        "constructor": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "weight_enum": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "normalisation_policy": "batchnorm_statistics_and_affines_frozen",
        "native_stochastic_depth": True,
    },
    "densenet121": {
        "constructor": models.densenet121,
        "weights": models.DenseNet121_Weights.IMAGENET1K_V1,
        "weight_enum": "DenseNet121_Weights.IMAGENET1K_V1",
        "normalisation_policy": "batchnorm_statistics_and_affines_frozen",
        "native_stochastic_depth": False,
    },
    "convnext_tiny": {
        "constructor": models.convnext_tiny,
        "weights": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "weight_enum": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "normalisation_policy": "layernorm_frozen_in_warmup_trainable_in_finetuning",
        "native_stochastic_depth": True,
    },
}


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_hash(value) -> str:
    """Hash configuration without relying on dictionary insertion order."""
    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True,
                                   allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    """Seed all supported RNGs and request reproducible torch operations.

    Device/package versions remain part of the manifest: bitwise equivalence
    across different CUDA builds or machines is not promised.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    # DataLoader assigns an explicit torch seed to each worker.
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"kind": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["kind"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"],
                         numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA RNG checkpoint cannot resume training on CPU.")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def validate_spec(spec: Mapping) -> dict:
    spec = {**DEFAULT_SPEC, **dict(spec)}
    if spec["architecture"] not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {spec['architecture']!r}; "
                         f"registered adapters are {tuple(ARCHITECTURES)}.")
    if spec["weights"] != "IMAGENET1K_V1":
        raise ValueError(f"Unsupported weights: {spec['weights']!r}; adapter supports 'IMAGENET1K_V1'.")
    transform_profiles.validate_profiles(spec)
    if not isinstance(spec["image_size"], int) or spec["image_size"] < 16:
        raise ValueError("image_size must be an integer of at least 16.")
    if not 0 <= spec["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1).")
    return spec


def validate_training(training: Mapping) -> dict:
    training = {**DEFAULT_TRAINING, **dict(training)}
    for key in ("microbatch", "effective_batch", "min_microbatch", "schedule_epochs"):
        if not isinstance(training[key], int) or isinstance(training[key], bool) or training[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if training["microbatch"] > training["effective_batch"]:
        raise ValueError("microbatch must not exceed effective_batch.")
    if training["min_microbatch"] > training["microbatch"]:
        raise ValueError("min_microbatch must not exceed microbatch.")
    if training["effective_batch"] % training["microbatch"]:
        raise ValueError("effective_batch must be divisible by microbatch.")
    if training["warmup_epochs"] != 1 or training["schedule_epochs"] < 3:
        raise ValueError("The initial schedule requires one warmup and at least two fine-tuning epochs.")
    for key in ("head_lr", "gradient_clip"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if not math.isfinite(training["weight_decay"]) or training["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    if not 0 < training["min_lr_factor"] <= 1:
        raise ValueError("min_lr_factor must be in (0, 1].")
    if not isinstance(training["num_workers"], int) or training["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer.")
    if not isinstance(training["seed"], int) or training["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer.")
    return training


def validate_frame(frame: pd.DataFrame, *, require_both_classes: bool = True) -> None:
    required = {"patient_id", "target", "image_path"}
    missing = required - set(frame.columns)
    if missing or not ({"image_name", "image_id"} & set(frame.columns)):
        raise ValueError(f"Image frame requires patient_id/target/image_path/image_name or image_id; missing {missing}.")
    if frame.empty:
        raise ValueError("Image frame must not be empty.")
    id_column = "image_id" if "image_id" in frame else "image_name"
    for column in (id_column, "patient_id", "image_path"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {column} values.")
    if frame[id_column].duplicated().any():
        raise ValueError("Duplicated image identifiers in image frame.")
    if not frame["target"].isin([0, 1]).all():
        raise ValueError("target must contain exactly binary labels 0 and 1.")
    if require_both_classes and frame["target"].nunique() != 2:
        raise ValueError("Both classes must be present in the fitting/validation frame.")
    for path in frame["image_path"]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing image: {path}")


def validate_partitions(fitting: pd.DataFrame, validation: pd.DataFrame) -> None:
    validate_frame(fitting)
    validate_frame(validation)
    if set(fitting["patient_id"]) & set(validation["patient_id"]):
        raise ValueError("Fitting and validation patients overlap.")
    fitting_ids = fitting.get("image_id", fitting.get("image_name"))
    validation_ids = validation.get("image_id", validation.get("image_name"))
    if set(fitting_ids) & set(validation_ids):
        raise ValueError("Fitting and validation image identifiers overlap.")


def make_transform(spec: Mapping, training: bool = False):
    """Build versioned preparation, preserving the WIP weak profile exactly."""
    return transform_profiles.make_transform(validate_spec(spec), training)


def transformation_profile(spec: Mapping) -> dict:
    """Return the complete immutable-identity profile for checkpoint manifests."""
    return transform_profiles.profile_identity(validate_spec(spec))


def deterministic_prepare(spec: Mapping):
    """PIL image preparation shared by training, previews and prediction."""
    return transform_profiles.deterministic_prepare(validate_spec(spec))


def stochastic_augmentation(spec: Mapping):
    return transform_profiles.stochastic_augmentation(validate_spec(spec))


def make_preview_transform(spec: Mapping, training: bool = True):
    """PIL preview before tensor conversion and pretrained normalisation."""
    return transform_profiles.make_preview_transform(validate_spec(spec), training)


class ImageFrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as original:
            image = self.transform(original.convert("RGB"))
        return image, torch.tensor(float(row["target"]), dtype=torch.float32)


def build_model(spec: Mapping, pretrained: bool = True, device="cpu") -> nn.Module:
    spec = validate_spec(spec)
    adapter = ARCHITECTURES[spec["architecture"]]
    model = adapter["constructor"](weights=adapter["weights"] if pretrained else None)
    if spec["architecture"] == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(p=spec["dropout"]), nn.Linear(in_features, 1))
    elif spec["architecture"] == "convnext_tiny":
        # Keep pretrained classifier LayerNorm and flattening; only the output
        # linear is new. LayerNorm belongs to the backbone LR group.
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(model.classifier[0], model.classifier[1],
                                         nn.Dropout(p=spec["dropout"]), nn.Linear(in_features, 1))
    else:
        # Preserve the B0 state-dict head layout for historical pilot loading.
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=spec["dropout"]), nn.Linear(in_features, 1))
    model._image_architecture = spec["architecture"]
    freeze_batchnorm(model)
    return model.to(device)


def architecture_policy(spec: Mapping) -> dict:
    """Serializable construction/preparation policy recorded in new manifests."""
    spec = validate_spec(spec)
    adapter = ARCHITECTURES[spec["architecture"]]
    native = adapter["weights"].transforms()
    return {"architecture": spec["architecture"], "weight_enum": adapter["weight_enum"],
            "output_head": "dropout_then_one_logit_linear", "dropout": spec["dropout"],
            "head_lr_parameters": "new_output_linear_only",
            "normalisation_policy": adapter["normalisation_policy"],
            "native_stochastic_depth_retained": adapter["native_stochastic_depth"],
            "preparation": ("whole_image_bilinear_resize_imagenet_normalisation"
                            if spec["preprocessing"] == "none" else
                            "whole_image_bilinear_resize_clahe_lab_l_imagenet_normalisation"),
            "transformation_profile": transformation_profile(spec),
            "image_size": spec["image_size"], "centre_crop": False,
            "mean": list(native.mean), "std": list(native.std),
            "native_pretrained_preparation": {
                "crop_size": list(native.crop_size), "resize_size": list(native.resize_size),
                "interpolation": native.interpolation.value}}


def architecture_adapter(spec: Mapping) -> dict:
    """Return a copy of the registered native constructor and pinned weights."""
    return dict(ARCHITECTURES[validate_spec(spec)["architecture"]])


def pretrained_weight_identity(spec: Mapping, *, require_cached: bool = True) -> dict:
    """Pinned enum/URL identity without download; optionally verify cached bytes.

    The non-cached form is stable before and after weight downloads, so stage
    manifests do not change merely because a cache became available.
    """
    spec = validate_spec(spec)
    if require_cached:
        return _pretrained_identity(None, spec)
    adapter = architecture_adapter(spec)
    url = adapter["weights"].url
    filename = Path(urlparse(url).path).name
    return {"kind": "torchvision", "weight_enum": adapter["weight_enum"], "url": url,
            "expected_sha256_prefix": filename.rsplit("-", 1)[-1].split(".")[0]}


def training_parameter_groups(model: nn.Module) -> tuple[list, list]:
    """Return the new output linear and every remaining pretrained parameter.

    Do not put ConvNeXt's pretrained classifier LayerNorm at the head LR or
    unfreeze it during classifier-only warmup. Tiny test models follow this
    same classifier/linear contract without a production factory substitute.
    """
    linear = next((module for module in reversed(list(model.classifier.modules()))
                   if isinstance(module, nn.Linear)), None)
    if linear is None or linear.out_features != 1:
        raise ValueError("Image adapter must expose a one-logit output linear in classifier.")
    head = list(linear.parameters())
    head_ids = {id(parameter) for parameter in head}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    return head, backbone


def freeze_batchnorm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def configure_training_stage(model: nn.Module, warmup: bool) -> None:
    """Keep pretrained features deterministic during classifier-only warmup."""
    model.train()
    head, _ = training_parameter_groups(model)
    head_ids = {id(parameter) for parameter in head}
    for parameter in model.parameters():
        parameter.requires_grad_(not warmup or id(parameter) in head_ids)
    if warmup:
        model.eval()
        # Train explicit head dropout, but leave pretrained classifier
        # normalisation and native backbone stochastic depth in eval mode.
        for module in model.classifier.modules():
            if isinstance(module, (nn.Dropout, nn.Linear)):
                module.train()
    freeze_batchnorm(model)


def _loader(frame, spec, training, *, training_mode=False, microbatch=None, seed=None,
            device="cpu"):
    generator = torch.Generator()
    generator.manual_seed(training["seed"] if seed is None else seed)
    return DataLoader(ImageFrameDataset(frame, make_transform(spec, training_mode)),
                      batch_size=microbatch or training["microbatch"],
                      shuffle=training_mode, drop_last=False,
                      num_workers=training["num_workers"], worker_init_fn=seed_worker,
                      generator=generator, pin_memory=torch.device(device).type == "cuda",
                      persistent_workers=False)


@torch.inference_mode()
def predict_frame(model, frame: pd.DataFrame, spec: Mapping, training: Mapping,
                  device="cpu") -> pd.DataFrame:
    """Ordered, deterministic probabilities; never mutate checkpoints/history."""
    validate_frame(frame, require_both_classes=False)
    spec, training = validate_spec(spec), validate_training(training)
    model.eval()
    freeze_batchnorm(model)
    probabilities = []
    for images, _ in _loader(frame, spec, training, device=device):
        logits = model(images.to(device, non_blocking=True)).reshape(-1)
        if logits.numel() != images.shape[0] or not torch.isfinite(logits).all():
            raise FloatingPointError("Prediction logits are non-finite or have the wrong shape.")
        probabilities.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
    result = frame.drop(columns=["image_path"], errors="ignore").reset_index(drop=True).copy()
    if "image_id" not in result:
        result["image_id"] = result["image_name"]
    if "fold" not in result:
        result["fold"] = -1
    result["model_name"] = spec["architecture"]
    result["base_probability"] = probabilities
    return result


class EpochCosineSchedule:
    """A fixed horizon independent of pilot cap or later resume duration."""
    def __init__(self, optimizer, head_lr, backbone_lr, epochs, minimum):
        self.optimizer = optimizer
        self.head_lr, self.backbone_lr = head_lr, backbone_lr
        self.epochs, self.minimum, self.epoch = epochs, minimum, 0

    def set_epoch(self, epoch):
        if not 1 <= epoch <= self.epochs:
            raise ValueError("Epoch outside the fixed scheduler horizon.")
        fraction = 0 if epoch <= 2 else (epoch - 2) / (self.epochs - 2)
        factor = self.minimum + (1 - self.minimum) * (1 + math.cos(math.pi * fraction)) / 2
        self.optimizer.param_groups[0]["lr"] = self.head_lr * factor
        self.optimizer.param_groups[1]["lr"] = self.backbone_lr * factor
        self.epoch = epoch

    def state_dict(self):
        return {"head_lr": self.head_lr, "backbone_lr": self.backbone_lr,
                "epochs": self.epochs, "minimum": self.minimum, "epoch": self.epoch}

    def load_state_dict(self, state):
        expected = {key: value for key, value in self.state_dict().items() if key != "epoch"}
        if expected != {key: value for key, value in state.items() if key != "epoch"}:
            raise ValueError("Incompatible scheduler state.")
        self.epoch = state["epoch"]
        if state["epoch"]:
            self.set_epoch(state["epoch"])


def train_epoch(model, loader, optimizer, criterion, scaler, device,
                *, effective_batch: int, gradient_clip: float, amp: bool) -> float:
    """Accumulate sum losses divided by actual samples in each update window.

    A partial final window uses its actual sample count, so neither a small
    microbatch nor a short final batch changes its intended gradient scale.
    """
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_count, window_count = 0.0, 0, 0
    skipped_updates, successful_updates = 0, 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=torch.device(device).type,
                                enabled=amp and torch.device(device).type == "cuda"):
            logits = model(images).reshape(-1)
            if logits.numel() != targets.numel():
                raise ValueError("Model must return one output logit per image.")
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")
        scaler.scale(loss).backward()
        window_count += targets.numel()
        total_count += targets.numel()
        total_loss += float(loss.detach())
        if window_count >= effective_batch or total_count == len(loader.dataset):
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window_count)
            gradients_finite = torch.stack([
                torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                if parameter.grad is not None]).all().item()
            if not gradients_finite and not scaler.is_enabled():
                raise FloatingPointError("Non-finite training gradients without AMP scaling.")
            if gradients_finite:
                norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip,
                                                error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite training gradient norm.")
                successful_updates += 1
            else:
                # GradScaler recorded the overflow during unscale_; step skips
                # this update and update() lowers its scale. Initial overflows
                # are expected with float16 and do not invalidate finite losses.
                skipped_updates += 1
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            window_count = 0
    model._last_skipped_optimizer_updates = skipped_updates
    model._last_successful_optimizer_updates = successful_updates
    if successful_updates == 0:
        raise FloatingPointError("Every optimizer update was skipped; inspect AMP scale and gradients.")
    return total_loss / total_count


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def model_state_sha256(model: nn.Module) -> str:
    """Hash named initial parameters/buffers without constructing another model."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)],
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _environment(device) -> dict:
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": np.__version__,
            "pandas": pd.__version__, "device": str(device),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else "CPU"}


def _frame_identity(frame) -> str:
    # Caller manifest contains the audited image hashes. This identity additionally
    # binds ordered labels/patients/paths even for independently used recipe fits.
    columns = [key for key in ("image_name", "image_id", "patient_id", "fold", "target", "image_path")
               if key in frame]
    return canonical_hash(frame[columns].to_dict("records"))


def _pretrained_identity(model_factory, spec=None) -> dict:
    if model_factory is not None:
        return {"kind": "test_factory", "factory": model_factory.__qualname__}
    adapter = ARCHITECTURES[validate_spec(spec or DEFAULT_SPEC)["architecture"]]
    url = adapter["weights"].url
    filename = Path(urlparse(url).path).name
    cached = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not cached.is_file():
        raise FileNotFoundError("Pinned pretrained cache is missing after model initialisation.")
    digest = file_sha256(cached)
    expected_prefix = filename.rsplit("-", 1)[-1].split(".")[0]
    if not digest.startswith(expected_prefix):
        raise ValueError("Cached pretrained weight hash does not match the pinned torchvision weight.")
    return {"kind": "torchvision", "weight_enum": adapter["weight_enum"],
            "url": url, "file_sha256": digest}


def _save_checkpoint(path: Path, checkpoint) -> None:
    if path.exists():
        # A process kill between replacing a tensor file and its SHA sidecar
        # must not destroy the previous committed epoch. Keep verified pairs;
        # ordinary reads remain strict and never silently recover from backup.
        sidecar = path.with_suffix(path.suffix + ".sha256.json")
        if not sidecar.is_file():
            raise ValueError("Existing checkpoint has no hash sidecar; explicit supervisor recovery is required.")
        previous_digest = file_sha256(path)
        if json.loads(sidecar.read_text(encoding="utf-8")).get("sha256") != previous_digest:
            raise ValueError("Existing checkpoint hash differs; explicit supervisor recovery is required.")
        previous = path.with_name(path.stem + ".previous" + path.suffix)
        previous_temporary = previous.with_suffix(previous.suffix + ".tmp")
        shutil.copyfile(path, previous_temporary)
        os.replace(previous_temporary, previous)
        _atomic_json(previous.with_suffix(previous.suffix + ".sha256.json"), {"sha256": previous_digest})
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)
    _atomic_json(path.with_suffix(path.suffix + ".sha256.json"), {"sha256": file_sha256(path)})


def _read_checkpoint(path: Path, device="cpu") -> dict:
    # All checkpoint RNG state is tensor/plain-container data; no unrestricted
    # pickle loading is needed, including for NumPy RNG state.
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    if not sidecar.is_file() or json.loads(sidecar.read_text(encoding="utf-8"))["sha256"] != file_sha256(path):
        raise ValueError("Checkpoint hash sidecar is missing or does not match.")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version; use the separate legacy WIP loader.")
    if canonical_hash(checkpoint["identity"]) != checkpoint["compatibility_hash"]:
        raise ValueError("Checkpoint compatibility identity is corrupt.")
    return checkpoint


def recover_interrupted_checkpoint(latest_path, *, expected_manifest=None,
                                   expected_compatibility_hash=None) -> dict:
    """Explicit watchdog recovery after the training child has terminated.

    Validate file hashes, payload identities and best/latest epoch consistency.
    Unverified or displaced files are retained for inspection. This helper is
    never called by normal checkpoint reads or ordinary recipe resume.
    """
    latest_path = Path(latest_path)
    if latest_path.name != "latest.pt":
        raise ValueError("Supervisor recovery requires the run's latest.pt path.")
    previous_latest = latest_path.with_name("latest.previous.pt")
    best_path = latest_path.with_name("best.pt")
    previous_best = latest_path.with_name("best.previous.pt")
    failures, events = [], []

    def validated(path):
        state = _read_checkpoint(path)
        if (expected_manifest is not None and canonical_hash(state["manifest"]) !=
                canonical_hash(expected_manifest)):
            raise ValueError("Recovery candidate belongs to a different manifest.")
        if (expected_compatibility_hash is not None and
                state["compatibility_hash"] != expected_compatibility_hash):
            raise ValueError("Recovery candidate belongs to a different compatibility identity.")
        if [row["epoch"] for row in state["history"]] != list(range(1, state["epoch"] + 1)):
            raise ValueError("Recovery candidate history does not match its committed epoch.")
        if state["best_epoch"] is not None and not 1 <= state["best_epoch"] <= state["epoch"]:
            raise ValueError("Recovery candidate's best epoch is outside its committed history.")
        return state

    selected_latest, latest = None, None
    for candidate in (latest_path, previous_latest):
        try:
            latest = validated(candidate)
            selected_latest = candidate
            break
        except Exception as error:
            failures.append({"path": str(candidate), "error": f"{type(error).__name__}: {error}"})
    if latest is None:
        raise ValueError(f"No verified latest checkpoint is recoverable: {failures}")

    selected_best = None
    if latest["best_epoch"] is not None:
        for candidate in (best_path, previous_best):
            try:
                best = validated(candidate)
                if (best["compatibility_hash"] != latest["compatibility_hash"] or
                        best["epoch"] != latest["best_epoch"] or best["best_epoch"] != best["epoch"]):
                    raise ValueError("Best checkpoint does not match the latest committed selection.")
                selected_best = candidate
                break
            except Exception as error:
                failures.append({"path": str(candidate), "error": f"{type(error).__name__}: {error}"})
        if selected_best is None:
            if latest["epoch"] == latest["best_epoch"]:
                selected_best = selected_latest
            else:
                raise ValueError(f"No verified checkpoint contains the selected best epoch: {failures}")

    def preserve_displaced(path):
        if not path.exists():
            return
        retained = path.with_name(f"{path.stem}.interrupted.{time.time_ns()}{path.suffix}")
        shutil.copyfile(path, retained)
        sidecar = path.with_suffix(path.suffix + ".sha256.json")
        if sidecar.exists():
            shutil.copyfile(sidecar, retained.with_suffix(retained.suffix + ".sha256.json"))
        events.append({"event": "interrupted_checkpoint_retained", "path": str(retained)})

    def install(source, destination):
        if source == destination:
            return
        preserve_displaced(destination)
        state = validated(source)
        temporary = destination.with_suffix(destination.suffix + ".recovery.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        _atomic_json(destination.with_suffix(destination.suffix + ".sha256.json"),
                     {"sha256": file_sha256(destination)})
        repaired = validated(destination)
        if repaired["compatibility_hash"] != state["compatibility_hash"]:
            raise ValueError("Recovered checkpoint identity changed during copy.")
        events.append({"event": "checkpoint_recovered", "source": str(source),
                       "destination": str(destination), "epoch": repaired["epoch"]})

    install(selected_latest, latest_path)
    if selected_best is not None:
        install(selected_best, best_path)
    elif best_path.exists():
        # A best file committed before a killed first epoch is not a selection
        # in epoch-zero history. Retain it, then clear that uncommitted alias.
        preserve_displaced(best_path)
        best_path.unlink()
        best_sidecar = best_path.with_suffix(best_path.suffix + ".sha256.json")
        if best_sidecar.exists():
            best_sidecar.unlink()
        events.append({"event": "uncommitted_best_alias_cleared"})
    return {"latest_checkpoint": str(latest_path),
            "best_checkpoint": str(best_path) if selected_best is not None else None,
            "epoch": latest["epoch"], "best_epoch": latest["best_epoch"],
            "compatibility_hash": latest["compatibility_hash"],
            "recovery_events": events, "candidate_failures": failures}


def load_model_checkpoint(path, *, device="cpu", expected_manifest=None,
                          model_factory: Callable | None = None):
    checkpoint = _read_checkpoint(Path(path), device="cpu")
    if expected_manifest is not None and canonical_hash(expected_manifest) != canonical_hash(checkpoint["manifest"]):
        raise ValueError("Checkpoint manifest does not match the expected run.")
    # Old weak/unchanged checkpoints predate explicit profiles and keep their
    # separate historical loader contract. New profiles must reconstruct the
    # exact declared preparation, including their optional dependency identity.
    recorded_policy = checkpoint["identity"].get("architecture_policy", {})
    recorded_profile = recorded_policy.get("transformation_profile")
    if (recorded_profile is not None and canonical_hash(recorded_profile) !=
            canonical_hash(transformation_profile(checkpoint["spec"]))):
        raise ValueError("Checkpoint transformation profile differs from the registered version.")
    if checkpoint["spec"].get("preprocessing", "none") == "clahe":
        actual_opencv = str(transform_profiles.require_opencv().__version__)
        if checkpoint["identity"].get("preprocessing_environment") != {"opencv": actual_opencv}:
            raise ValueError("Checkpoint CLAHE dependency version differs from the prediction environment.")
    factory = model_factory or build_model
    model = factory(checkpoint["spec"], pretrained=False, device=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    freeze_batchnorm(model)
    return model, checkpoint


def train_recipe(fitting, validation, spec, training, recipe, output_dir, manifest,
                 *, epochs=6, device="cuda", deadline=None, resume=True,
                 refit: bool = False,
                 model_factory: Callable | None = None,
                 epoch_admission_seconds: float | None = None,
                 epoch_admission_safety_factor: float = 1.0,
                 expected_initial_model_state_sha256: str | None = None) -> dict:
    """Fit one recipe; deadline is an absolute ``time.monotonic()`` value.

    Budget stops occur only between complete epochs. CUDA OOM restores the
    whole pre-epoch model/optimizer/scaler/RNG state before retrying with a
    smaller microbatch. This routine never silently switches to CPU.
    Optional epoch admission guards reserve 30 seconds for checkpointing and
    use the worst measured epoch, or the supplied first-epoch estimate. The
    caller still owns a hard process deadline for unexpectedly slow epochs.

    Set ``refit=True`` and ``validation=None`` for a fixed-duration fit whose
    epoch count was selected on separate inner patients. This mode never reads
    validation labels or selects a checkpoint by performance. A completed fit
    returns its final epoch as ``final_checkpoint``; incomplete fits return no
    final checkpoint. ``latest_checkpoint`` remains available for resume.
    """
    spec, training = validate_spec(spec), validate_training(training)
    if not isinstance(refit, bool):
        raise ValueError("refit must be a boolean.")
    if refit:
        if validation is not None:
            raise ValueError("Fixed-duration refits require validation=None; held-out labels are not accepted.")
        validate_frame(fitting)
    else:
        if validation is None:
            raise ValueError("Checkpoint selection requires validation; use refit=True for a fixed-duration fit.")
        validate_partitions(fitting, validation)
    recipe = dict(recipe)
    if recipe.get("loss_weight") not in ("none", "sqrt_ratio"):
        raise ValueError("loss_weight must be 'none' or 'sqrt_ratio'.")
    if not recipe.get("id") or not math.isfinite(recipe.get("backbone_lr", float("nan"))) or recipe["backbone_lr"] <= 0:
        raise ValueError("Recipe requires an id and finite positive backbone_lr.")
    if not isinstance(epochs, int) or not 1 <= epochs <= training["schedule_epochs"]:
        raise ValueError("epochs must be within the fixed schedule horizon.")
    if epoch_admission_seconds is not None and (not math.isfinite(epoch_admission_seconds)
                                                or epoch_admission_seconds <= 0):
        raise ValueError("epoch_admission_seconds must be finite and positive.")
    if not math.isfinite(epoch_admission_safety_factor) or epoch_admission_safety_factor < 1:
        raise ValueError("epoch_admission_safety_factor must be finite and at least one.")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select CPU explicitly for tests only.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest, best = output_dir / "latest.pt", output_dir / "best.pt"
    source_hash = file_sha256(__file__)
    identity = {"spec": spec, "training": training, "recipe": recipe,
                "manifest": _json_value(manifest), "fitting": _frame_identity(fitting),
                "validation": None if refit else _frame_identity(validation), "environment": _environment(device),
                "source_sha256": source_hash,
                "transformation_source_sha256": file_sha256(transform_profiles.__file__),
                "preprocessing_environment": ({"opencv": str(transform_profiles.require_opencv().__version__)}
                                              if spec["preprocessing"] == "clahe" else {"opencv": "not_used"}),
                "architecture_policy": architecture_policy(spec),
                "model_factory": spec["architecture"] if model_factory is None else model_factory.__qualname__}
    if refit:
        identity["training_purpose"] = "fixed_duration_refit"
    compatibility_hash = canonical_hash(identity)
    if latest.exists() and not resume:
        raise FileExistsError("Run already has a checkpoint; use resume or a new output directory.")
    saved = _read_checkpoint(latest) if latest.exists() else None
    if saved is not None and saved["compatibility_hash"] != compatibility_hash:
        raise ValueError("Checkpoint is incompatible with data/configuration/source/environment/manifest.")
    if refit and saved is not None and saved["epoch"] > epochs:
        raise ValueError("Refit checkpoint exceeds the requested fixed duration; use a new output directory.")
    seed_everything(training["seed"])
    factory = model_factory or build_model
    model = factory(spec, pretrained=saved is None, device=device)
    initial_model_state_sha256 = (saved["initial_model_state_sha256"] if saved is not None
                                 else model_state_sha256(model))
    if (expected_initial_model_state_sha256 is not None and
            initial_model_state_sha256 != expected_initial_model_state_sha256):
        raise ValueError("Initial model parameters differ from the matched experimental control.")
    pretrained_identity = _pretrained_identity(model_factory, spec)
    if saved is not None and saved["pretrained_identity"] != pretrained_identity:
        raise ValueError("Pretrained weight identity changed since this run began.")
    head, backbone = training_parameter_groups(model)
    optimizer = torch.optim.AdamW([
        {"params": head, "lr": training["head_lr"]},
        {"params": backbone, "lr": recipe["backbone_lr"]}], weight_decay=training["weight_decay"])
    schedule = EpochCosineSchedule(optimizer, training["head_lr"], recipe["backbone_lr"],
                                  training["schedule_epochs"], training["min_lr_factor"])
    scaler = torch.amp.GradScaler(device.type, enabled=training["amp"] and device.type == "cuda")
    count_positive = int(fitting["target"].sum())
    count_negative = len(fitting) - count_positive
    weight = math.sqrt(count_negative / count_positive) if recipe["loss_weight"] == "sqrt_ratio" else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device), reduction="sum")
    history, events, best_epoch, best_metrics, completed = [], [], None, None, 0
    microbatch = training["microbatch"]
    elapsed_previous = 0.0
    if saved:
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        schedule.load_state_dict(saved["scheduler_state"])
        scaler.load_state_dict(saved["scaler_state"])
        restore_rng_state(saved["rng_state"])
        history, events = saved["history"], saved["events"]
        completed, microbatch = saved["epoch"], saved["microbatch"]
        best_epoch, best_metrics = saved["best_epoch"], saved["best_metrics"]
        elapsed_previous = saved.get("elapsed_seconds", 0.0)
        if best_epoch is not None and not best.is_file():
            raise ValueError("Best checkpoint is missing for this resumable run.")
        if best_epoch is not None and _read_checkpoint(best)["compatibility_hash"] != compatibility_hash:
            raise ValueError("Best checkpoint belongs to a different run.")
    start = time.monotonic()
    status = "complete" if completed >= epochs else "running"

    def checkpoint_payload(stopping_reason):
        return {"checkpoint_version": CHECKPOINT_VERSION, "spec": spec, "training": training,
                "recipe": recipe, "manifest": _json_value(manifest), "identity": identity,
                "pretrained_identity": pretrained_identity,
                "compatibility_hash": compatibility_hash, "model_state": _cpu_tree(model.state_dict()),
                "initial_model_state_sha256": initial_model_state_sha256,
                "initial_state_sha256": initial_model_state_sha256,
                "optimizer_state": _cpu_tree(optimizer.state_dict()), "scheduler_state": schedule.state_dict(),
                "scaler_state": scaler.state_dict(), "rng_state": capture_rng_state(),
                "epoch": completed, "microbatch": microbatch, "effective_batch": training["effective_batch"],
                "best_epoch": best_epoch, "best_metrics": best_metrics, "history": history,
                "events": events, "positive_weight": weight, "fit_class_counts": {
                    "negative": count_negative, "positive": count_positive},
                "stopping_reason": stopping_reason,
                "elapsed_seconds": elapsed_previous + time.monotonic() - start}

    # An epoch-zero checkpoint also makes a budget stop before epoch 1 resumable.
    if saved is None:
        _save_checkpoint(latest, checkpoint_payload("initialised"))
    while completed < epochs:
        if deadline is not None and time.monotonic() >= deadline:
            status = "budget_exhausted"
            break
        if deadline is not None and epoch_admission_seconds is not None:
            estimate = epoch_admission_safety_factor * max(
                (row["epoch_seconds"] for row in history), default=epoch_admission_seconds) + 30.0
            if deadline - time.monotonic() < estimate:
                events.append({"event": "epoch_admission_budget_pause", "epoch": completed + 1,
                               "required_seconds": estimate})
                status = "budget_exhausted"
                break
        epoch = completed + 1
        snapshot = _read_checkpoint(latest)
        epoch_start = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            seed_everything(training["seed"] + epoch)
            configure_training_stage(model, warmup=epoch <= training["warmup_epochs"])
            schedule.set_epoch(epoch)
            loader = _loader(fitting, spec, training, training_mode=True,
                             microbatch=microbatch, seed=training["seed"] + epoch, device=device)
            train_loss = train_epoch(model, loader, optimizer, criterion, scaler, device,
                                     effective_batch=training["effective_batch"],
                                     gradient_clip=training["gradient_clip"], amp=training["amp"])
            metrics = {}
            if not refit:
                eval_training = {**training, "microbatch": microbatch}
                predictions = predict_frame(model, validation, spec, eval_training, device)
                metrics = probability_metrics(predictions["target"], predictions["base_probability"])
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        except torch.cuda.OutOfMemoryError:
            # The failed function's locals have unwound. Remove gradients before
            # restoring CPU snapshots so no failed update survives the retry.
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model.load_state_dict(snapshot["model_state"])
            optimizer.load_state_dict(snapshot["optimizer_state"])
            schedule.load_state_dict(snapshot["scheduler_state"])
            scaler.load_state_dict(snapshot["scaler_state"])
            restore_rng_state(snapshot["rng_state"])
            reduced = microbatch // 2
            events.append({"epoch": epoch, "event": "cuda_oom_epoch_rollback",
                           "microbatch": microbatch, "next_microbatch": reduced})
            if reduced < training["min_microbatch"] or training["effective_batch"] % reduced:
                status = "infeasible"
                break
            microbatch = reduced
            _save_checkpoint(latest, checkpoint_payload("oom_retry"))
            continue
        row = {"epoch": epoch, "train_loss": train_loss, **metrics,
               "head_lr": optimizer.param_groups[0]["lr"],
               "backbone_lr": optimizer.param_groups[1]["lr"],
               "epoch_seconds": time.monotonic() - epoch_start,
               "peak_memory_bytes": peak, "peak_reserved_memory_bytes": reserved,
               "microbatch": microbatch,
               "skipped_optimizer_updates": model._last_skipped_optimizer_updates,
               "successful_optimizer_updates": model._last_successful_optimizer_updates,
               "effective_batch": training["effective_batch"]}
        history.append(row)
        completed = epoch
        improved = not refit and (best_metrics is None or metrics["average_precision"] > best_metrics["average_precision"])
        if improved:
            best_epoch, best_metrics = epoch, dict(metrics)
            _save_checkpoint(best, checkpoint_payload("best_ap"))
        status = "complete" if completed >= epochs else "running"
        _save_checkpoint(latest, checkpoint_payload(status))
        _atomic_json(output_dir / "history.json", history)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        evaluation = "fixed-duration refit" if refit else f"AP={metrics['average_precision']:.5f}"
        print(f"{recipe['id']}: epoch {epoch}/{epochs}; {evaluation}; "
              f"loss={train_loss:.5f}; {row['epoch_seconds']:.1f}s", flush=True)
    _save_checkpoint(latest, checkpoint_payload(status))
    _atomic_json(output_dir / "history.json", history)
    temporary_history = output_dir / "history.csv.tmp"
    pd.DataFrame(history).to_csv(temporary_history, index=False)
    os.replace(temporary_history, output_dir / "history.csv")
    _atomic_json(output_dir / "manifest.json", {**identity, "compatibility_hash": compatibility_hash,
                                             "initial_model_state_sha256": initial_model_state_sha256,
                                             "initial_state_sha256": initial_model_state_sha256,
                                             "stopping_reason": status, "microbatch": microbatch,
                                             "positive_weight": weight,
                                             "pretrained_identity": pretrained_identity})
    result = {"status": status, "best_epoch": best_epoch, "best_metrics": best_metrics,
            "initial_model_state_sha256": initial_model_state_sha256,
            "initial_state_sha256": initial_model_state_sha256,
            "history": history, "latest_checkpoint": str(latest),
            "best_checkpoint": str(best) if not refit and best.is_file() else None,
            "microbatch": microbatch, "elapsed_seconds": elapsed_previous + time.monotonic() - start}
    if refit:
        result.update({"final_checkpoint": str(latest) if status == "complete" else None,
                       "final_epoch": completed if status == "complete" else None,
                       "completed_epochs": completed, "training_purpose": "fixed_duration_refit"})
    return result


__all__ = ["DEFAULT_SPEC", "DEFAULT_TRAINING", "build_model", "make_transform",
           "transformation_profile", "deterministic_prepare", "stochastic_augmentation",
           "make_preview_transform", "model_state_sha256",
           "ARCHITECTURES", "architecture_adapter", "architecture_policy",
           "pretrained_weight_identity", "training_parameter_groups",
           "freeze_batchnorm", "configure_training_stage", "predict_frame", "train_recipe",
           "train_epoch", "load_model_checkpoint", "seed_everything", "seed_worker",
           "recover_interrupted_checkpoint",
           "capture_rng_state", "restore_rng_state", "canonical_hash", "file_sha256",
           "validate_frame", "validate_partitions", "validate_spec", "validate_training",
           "ImageFrameDataset", "EpochCosineSchedule"]
