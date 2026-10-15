"""Shared CNN training: warmup, full fine-tuning, frozen BN and AUC checkpoints."""
from __future__ import annotations

import gc
import logging
import math
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch import nn

from .evaluation import compute_metrics, confusion_counts, plot_confusion_pair, select_threshold
from .model_registry import (MODEL_REGISTRY, SUPPORTED_MODELS, get_model_spec,
    TARGET_EFFECTIVE_BATCH_SIZE, NUMBER_OF_EPOCHS, WEIGHT_DECAY,
    GRADIENT_CLIP_NORM, MICROBATCH_CANDIDATES)
from .utils import RANDOM_STATE, get_device, set_seed, save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


def class_weight_from_training(train_targets):
    """Only accepts training labels; never receives validation or test labels."""
    targets = np.asarray(train_targets)
    if targets.ndim != 1 or not targets.size or not np.isin(targets, [0, 1]).all():
        raise ValueError("Training labels must be a non-empty 1D binary sequence.")
    positive, negative = int((targets == 1).sum()), int((targets == 0).sum())
    if not positive or not negative:
        raise ValueError("Training data must contain both classes.")
    ratio = negative / positive
    return dict(train_positive=positive, train_negative=negative,
                raw_imbalance_ratio=ratio, sqrt_pos_weight=math.sqrt(ratio))


def classifier_parameters(model):
    path = getattr(model, "_image_classifier_path", None)
    if path is None:
        raise ValueError("Model needs a classifier path; build it through the model registry.")
    return list(model.get_submodule(path).parameters())


def freeze_batchnorm(model):
    """Freeze affine parameters AND running statistics; LayerNorm is unaffected."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)


def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in classifier_parameters(model):
        p.requires_grad_(True)
    freeze_batchnorm(model)


def unfreeze_backbone(model):
    for p in model.parameters():
        p.requires_grad_(True)
    freeze_batchnorm(model)


def _train_mode_with_frozen_batchnorm(model):
    model.train()
    freeze_batchnorm(model)


def parameter_groups(model, backbone_lr, head_lr):
    head = classifier_parameters(model)
    head_ids = {id(p) for p in head}
    return [dict(params=[p for p in model.parameters() if id(p) not in head_ids],
                 lr=backbone_lr, name="backbone"),
            dict(params=head, lr=head_lr, name="head")]


def setup_image_model(model_name, train_targets, seed=RANDOM_STATE,
                      feature_lr=None, classifier_lr=None, weight_decay=WEIGHT_DECAY,
                      device=None, loader_generator=None, pretrained=True):
    """Legacy tuple API; feature_lr/classifier_lr override registry backbone/head LRs."""
    spec = get_model_spec(model_name)
    counts = class_weight_from_training(train_targets)
    set_seed(seed, loader_generator)
    device = get_device(device)
    model = spec.build(pretrained).to(device)
    freeze_backbone(model)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        [counts["sqrt_pos_weight"]], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(parameter_groups(model,
        spec.backbone_lr if feature_lr is None else feature_lr,
        spec.head_lr if classifier_lr is None else classifier_lr), weight_decay=weight_decay)
    model._image_training_counts = counts
    logger.info("Train melanoma: %d | non-melanoma: %d | raw imbalance ratio: %.4f | sqrt pos_weight: %.4f",
                counts["train_positive"], counts["train_negative"],
                counts["raw_imbalance_ratio"], counts["sqrt_pos_weight"])
    return model, criterion, optimizer, device


def load_image_state_dict(model, state_dict):
    """Load current or legacy single-linear classifier checkpoints strictly.

    Older EfficientNet-B3 exports replaced the complete classifier container.
    Remap those head keys through the registry; all backbone keys stay strict.
    This does not infer preprocessing: use the original checkpoint's config.
    """
    spec = get_model_spec(model._image_model_name)
    state = dict(state_dict)
    if spec.head_path != spec.classifier_path:
        for suffix in ("weight", "bias"):
            old = f"{spec.classifier_path}.{suffix}"
            new = f"{spec.head_path}.{suffix}"
            if old in state and new not in state:
                state[new] = state.pop(old)
    return model.load_state_dict(state, strict=True)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}" if hours else f"{minutes:02d}:{sec:02d}"


def _autocast(device):
    return torch.amp.autocast("cuda") if torch.device(device).type == "cuda" else nullcontext()


def _memory_probe(model, criterion, device, input_size, batch_size):
    # lr=0 preserves weights. Two passes include AdamW state in peak memory.
    temporary_optimizer = images = labels = loss = None
    try:
        temporary_optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0, weight_decay=0)
        images = torch.randn(batch_size, 3, input_size, input_size, device=device)
        labels = torch.zeros(batch_size, device=device)
        for _ in range(2):
            temporary_optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                loss = criterion(model(images).squeeze(1), labels)
            # Unscaled backward is sufficient for the memory probe. Zero grads
            # before the lr=0 step avoids any synthetic NaN touching weights and
            # guarantees AdamW state allocation (GradScaler could skip a step).
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.zero_()
            temporary_optimizer.step()
        synchronize(device)
    finally:
        model.zero_grad(set_to_none=True)
        del images, labels, loss, temporary_optimizer


def choose_microbatch(model, criterion, device, input_size, requested=None):
    """Probe full fine-tuning on CUDA; preserve weights, RNG, flags and modes.

    Only typed CUDA OOM exceptions trigger retry. Explicit requested microbatches
    cap the probe; CPU defaults to 16. Every microbatch must divide 128.
    """
    if requested is not None and (not isinstance(requested, int) or isinstance(requested, bool)
            or requested < 1 or TARGET_EFFECTIVE_BATCH_SIZE % requested):
        raise ValueError("batch_size must be a positive divisor of 128.")
    if torch.device(device).type != "cuda":
        batch = requested or 16
        return batch, TARGET_EFFECTIVE_BATCH_SIZE // batch
    candidates = [b for b in MICROBATCH_CANDIDATES if requested is None or b <= requested]
    if requested is not None and requested not in candidates:
        candidates.insert(0, requested)
    flags = [p.requires_grad for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    selected = None
    try:
        index = torch.device(device).index
        with torch.random.fork_rng(devices=[torch.cuda.current_device() if index is None else index]):
            unfreeze_backbone(model)
            _train_mode_with_frozen_batchnorm(model)
            for batch in candidates:
                try:
                    _memory_probe(model, criterion, device, input_size, batch)
                    selected = batch
                except torch.cuda.OutOfMemoryError:
                    logger.info("CUDA preflight OOM at microbatch %d; trying smaller batch", batch)
                finally:
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                if selected is not None:
                    break
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        for m, mode in zip(model.modules(), modes):
            m.training = mode
    if selected is None:
        raise torch.cuda.OutOfMemoryError("CUDA preflight failed at every candidate, including microbatch 8.")
    return selected, TARGET_EFFECTIVE_BATCH_SIZE // selected


class FineTuneCosine:
    """Update-based cosine, first update at base LR and last at 10%.

    Called only after successful optimizer updates (AMP overflows do not advance).
    """
    def __init__(self, optimizer, total_updates):
        self.optimizer, self.total_updates = optimizer, total_updates
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.completed_updates = 0

    def step(self):
        self.completed_updates += 1
        progress = min(self.completed_updates / max(1, self.total_updates - 1), 1)
        factor = 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * factor


def _train_one_epoch(model, loader, criterion, optimizer, device, desc="",
                     *, accumulation_steps=1, scaler=None, scheduler=None,
                     gradient_clip_norm=GRADIENT_CLIP_NORM):
    _train_mode_with_frozen_batchnorm(model)
    if criterion.reduction != "mean":
        raise ValueError("Accumulation requires a mean-reduced loss.")
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=False)
    total, seen, group_samples = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    # Backpropagate scaled sample sums, then normalise by actual group size.
    # This handles both a short last microbatch and a short accumulation group.
    nominal = (loader.batch_size or 1) * accumulation_steps
    for batch, (images, labels, _) in enumerate(loader, 1):
        images = images.to(device, non_blocking=torch.device(device).type == "cuda")
        labels = labels.to(device, non_blocking=torch.device(device).type == "cuda")
        with _autocast(device):
            loss = criterion(model(images).squeeze(1), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        count = labels.numel()
        scaler.scale(loss * count / nominal).backward()
        total += loss.detach().item() * count
        seen += count
        group_samples += count
        if batch % accumulation_steps == 0 or batch == len(loader):
            scaler.unscale_(optimizer)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(nominal / group_samples)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scaler.get_scale() >= scale_before:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            group_samples = 0
    if not seen:
        raise ValueError("Empty training loader.")
    return total / seen


@torch.no_grad()
def _validate(model, loader, criterion, device, desc=""):
    model.eval()
    total, targets, scores = 0.0, [], []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=torch.device(device).type == "cuda")
        labels = labels.to(device, non_blocking=torch.device(device).type == "cuda")
        with _autocast(device):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss.")
        total += loss.item() * labels.numel()
        targets.extend(labels.cpu().tolist())
        scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
    if not targets:
        raise ValueError("Empty validation loader.")
    return total / len(targets), np.asarray(targets), np.asarray(scores)


def validation_metrics(y_true, scores, fold, threshold=0.5):
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(f"Fold {fold}: validation labels must be binary 0/1.")
    labels, counts = np.unique(y_true, return_counts=True)
    if len(labels) < 2:
        raise ValueError(f"Fold {fold}: validation class counts {dict(zip(labels.tolist(), counts.tolist()))}; "
                         "ROC-AUC cannot be calculated with fewer than two classes.")
    m = compute_metrics(y_true, scores, threshold)
    return dict(roc_auc=m["ROC-AUC"], average_precision=m["Average precision"],
                sensitivity=m["Sensitivity / recall"], specificity=m["Specificity"],
                balanced_accuracy=m["average_recall"],
                **confusion_counts(y_true, np.asarray(scores) >= threshold))


def train_image_model(model, criterion, optimizer, train_loader, validation_loader,
                      device, max_epochs=NUMBER_OF_EPOCHS, patience=None,
                      selection_metric="validation_ROC_AUC", threshold=0.5, label="model",
                      *, fold=1, input_size=None, accumulation_steps=1,
                      gradient_clip_norm=GRADIENT_CLIP_NORM, screening=True,
                      checkpoint_path=None, checkpoint_metadata=None):
    """Three complete epochs, restore highest validation ROC-AUC; no early stopping.

    Old selection/patience arguments remain accepted only for the new policy.
    ``screening=False`` permits a different epoch budget for separate experiments.
    """
    if selection_metric != "validation_ROC_AUC" or patience is not None:
        raise ValueError("CNN training selects validation_ROC_AUC only and requires patience=None.")
    if max_epochs < 1 or (screening and max_epochs != NUMBER_OF_EPOCHS):
        raise ValueError("Architecture screening requires exactly 3 epochs.")
    if accumulation_steps < 1 or not isinstance(accumulation_steps, int):
        raise ValueError("accumulation_steps must be a positive integer.")
    if not len(train_loader) or not len(validation_loader):
        raise ValueError("Training and validation loaders must be non-empty.")
    model_name = getattr(model, "_image_model_name", label)
    microbatch = train_loader.batch_size
    if screening and microbatch * accumulation_steps != TARGET_EFFECTIVE_BATCH_SIZE:
        raise ValueError("Screening requires microbatch * accumulation_steps == 128.")
    history, best_auc, best_epoch, best_state = [], -np.inf, None, None
    selected_checkpoints = {}
    best_specificity = -np.inf
    scaler = torch.amp.GradScaler("cuda", enabled=torch.device(device).type == "cuda")
    scheduler = FineTuneCosine(optimizer, (max_epochs - 1) * math.ceil(len(train_loader) / accumulation_steps))
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        synchronize(device)
        epoch_start = time.perf_counter()
        if epoch == 1:
            freeze_backbone(model)
        elif epoch == 2:
            unfreeze_backbone(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if epoch <= 2:
            logger.info("Epoch %s trainable parameters: %s", "1" if epoch == 1 else "2+", f"{trainable:,}")
        phase = "head warmup" if epoch == 1 else "full fine-tune"
        start_lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
        train_start = time.perf_counter()
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device,
            accumulation_steps=accumulation_steps, scaler=scaler,
            scheduler=scheduler if epoch > 1 else None, gradient_clip_norm=gradient_clip_norm)
        synchronize(device)
        train_end = time.perf_counter()
        val_loss, y_true, scores = _validate(model, validation_loader, criterion, device)
        metrics = validation_metrics(y_true, scores, fold, threshold)
        recall_threshold = select_threshold(y_true, scores, method="sensitivity", min_sensitivity=0.90)
        recall_metrics = validation_metrics(y_true, scores, fold, recall_threshold)
        selection_metadata = {
            **(checkpoint_metadata or {}), "best_epoch": epoch,
            "threshold": recall_threshold, "target_recall": 0.90,
            "validation_recall": float(recall_metrics['sensitivity']),
            "validation_specificity": float(recall_metrics['specificity']),
            "validation_roc_auc": float(metrics['roc_auc']),
            "threshold_source": "validation", "fold": fold,
        }
        synchronize(device)
        validation_end = time.perf_counter()
        if metrics["roc_auc"] > best_auc:
            best_auc, best_epoch = metrics["roc_auc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            selected_checkpoints['auc'] = dict(state_dict=best_state,
                metadata={**selection_metadata, 'selection_metric': 'validation_ROC_AUC'})
            if checkpoint_path is not None:
                save_checkpoint(best_state, Path(checkpoint_path).parent / 'best_auc.pt',
                                **selected_checkpoints['auc']['metadata'])
                save_checkpoint(best_state, checkpoint_path,
                    **{**(checkpoint_metadata or {}), "best_epoch": best_epoch,
                       "best_validation_roc_auc": float(best_auc), "threshold": float(threshold)})
            logger.info("NEW BEST CHECKPOINT: ROC-AUC %.4f", best_auc)
        # Strict improvement retains the earliest epoch on equal specificity.
        if recall_metrics['specificity'] > best_specificity:
            best_specificity = recall_metrics['specificity']
            selected_checkpoints['specificity'] = dict(
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                metadata={**selection_metadata, 'selection_metric': 'specificity_at_recall_90'})
            if checkpoint_path is not None:
                chosen = selected_checkpoints['specificity']
                save_checkpoint(chosen['state_dict'],
                    Path(checkpoint_path).parent / 'best_specificity_recall90.pt', **chosen['metadata'])
            logger.info('NEW BEST RECALL-90 CHECKPOINT: specificity %.4f | threshold %.8g',
                        best_specificity, recall_threshold)
        synchronize(device)
        finished = time.perf_counter()
        row = dict(model=model_name, fold=fold, epoch=epoch, phase=phase, input_size=input_size,
            microbatch_size=microbatch, accumulation_steps=accumulation_steps,
            effective_batch_size=microbatch * accumulation_steps, train_loss=train_loss,
            val_loss=val_loss, **metrics, backbone_lr=optimizer.param_groups[0]["lr"],
            head_lr=optimizer.param_groups[1]["lr"], backbone_lr_start=start_lrs["backbone"],
            head_lr_start=start_lrs["head"], trainable_parameters=trainable,
            fine_tune_optimizer_updates=scheduler.completed_updates,
            train_seconds=train_end-train_start, validation_seconds=validation_end-train_end,
            epoch_seconds=finished-epoch_start, cumulative_seconds=finished-started)
        # Reporting aliases preserve existing plotting code, never checkpoint selection.
        row.update(validation_loss=val_loss, validation_ROC_AUC=metrics["roc_auc"],
                   validation_AP=metrics["average_precision"], average_recall=metrics["balanced_accuracy"],
                   melanoma_recall=metrics["sensitivity"])
        row.update(reporting_threshold=threshold, recall90_threshold=recall_threshold,
                   **{'recall90_' + key: value for key, value in recall_metrics.items()})
        history.append(row)
        logger.info("Epoch %d/%d [%s] | ROC-AUC %.4f | AP %.4f | sensitivity %.3f | specificity %.3f | "
                    "balanced accuracy %.3f | TP %d FN %d TN %d FP %d | train loss %.4f | val loss %.4f | "
                    "LR backbone %.2e head %.2e | Train %s | Val %s | Epoch %s | Total %s",
                    epoch, max_epochs, phase, metrics["roc_auc"], metrics["average_precision"],
                    metrics["sensitivity"], metrics["specificity"], metrics["balanced_accuracy"],
                    metrics["tp"], metrics["fn"], metrics["tn"], metrics["fp"], train_loss, val_loss,
                    row["backbone_lr"], row["head_lr"], format_duration(row["train_seconds"]),
                    format_duration(row["validation_seconds"]), format_duration(row["epoch_seconds"]),
                    format_duration(row["cumulative_seconds"]))
        logger.info('  Fixed-threshold metrics above use %.4g | Validation recall target >=90%%: '
                    'threshold %.8g | sensitivity %.3f | specificity %.3f | TP %d FN %d TN %d FP %d',
                    threshold, recall_threshold, recall_metrics['sensitivity'], recall_metrics['specificity'],
                    recall_metrics['tp'], recall_metrics['fn'], recall_metrics['tn'], recall_metrics['fp'])
    if best_state is None:
        raise RuntimeError("No finite validation ROC-AUC checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    synchronize(device)
    frame = pd.DataFrame(history)
    frame.attrs["training_wall_seconds"] = time.perf_counter() - started
    frame.attrs['selected_checkpoints'] = selected_checkpoints
    logger.info("Restored %s fold %s epoch %d | Best validation ROC-AUC: %.4f",
                model_name, fold, best_epoch, best_auc)
    return model, frame, best_epoch


def load_saved_image_model(path, device=None):
    """Load saved best weights and their deterministic transform, without training.

    Returns (model, evaluation_transform, metadata). This loads inference weights;
    it does not resume optimizer/scheduler state from an interrupted epoch.
    """
    from .preprocessing import image_transforms

    checkpoint = load_checkpoint(path, map_location="cpu")
    metadata = checkpoint["metadata"]
    config = metadata["config"]
    device = get_device(device)
    model = get_model_spec(config["model_name"]).build(pretrained=False)
    load_image_state_dict(model, checkpoint["state_dict"])
    model.to(device).eval()
    _, evaluation = image_transforms(config["image_size"], config["augmentation"],
        config["hair_removal"], config["seed"], model_name=config["model_name"],
        rotation=config.get("rotation", 0))
    return model, evaluation, metadata


# --------------------------------------------------------------------------- #
# Prediction / evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_image_model(model: nn.Module, loader, device) -> pd.DataFrame:
    """Melanoma probabilities for every image: DataFrame[image_name, target, melanoma_score]."""
    model.eval()
    rows = []
    for images, labels, image_ids in loader:
        with _autocast(device):
            logits = model(images.to(device, non_blocking=torch.device(device).type == "cuda"))
        logits = logits.float()
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if logits.ndim != 1 or logits.shape[0] != images.shape[0]:
            raise ValueError("The model must return one logit per image.")
        if not torch.isfinite(logits).all():
            raise ValueError("Model output contains NaN or infinity.")

        scores = torch.sigmoid(logits).cpu().numpy()
        batch_labels = labels.detach().cpu().reshape(-1).numpy()
        if not (len(image_ids) == len(batch_labels) == len(scores)):
            raise ValueError("Image IDs, labels and scores do not align.")
        if not np.isin(batch_labels, [0, 1]).all():
            raise ValueError("Labels must be 0 or 1.")
        rows.extend(
            {"image_name": i, "target": int(t), "melanoma_score": float(s)}
            for i, t, s in zip(image_ids, batch_labels, scores)
        )
    if not rows:
        raise ValueError("The loader contains no images.")
    return pd.DataFrame(rows)


def evaluate_image_model(
    model: nn.Module,
    loader,
    device,
    model_name: str = "Image model",
    split_name: str = "Validation",
    threshold: float = 0.5,
    show_plots: bool = True,
):
    """Predict, score and (optionally) plot one image model on one split.

    Returns ``(predictions_df, metrics_series, confusion_matrix)``; predictions gain a
    0/1 ``prediction`` column at ``threshold``. Display the Series in the notebook.
    """
    predictions = predict_image_model(model, loader, device)
    predictions["prediction"] = (predictions["melanoma_score"] >= threshold).astype(int)

    metrics = compute_metrics(predictions["target"], predictions["melanoma_score"], threshold,
                              name=f"{model_name} â€” {split_name}")
    c = confusion_counts(predictions["target"], predictions["prediction"])
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])

    if show_plots:
        plot_confusion_pair(cm, f"{model_name} | Threshold = {threshold:g}", split_name)
    logger.info("Melanoma detected: %d of %d | missed: %d | false alarms: %d of %d",
                c["tp"], c["tp"] + c["fn"], c["fn"], c["fp"], c["tn"] + c["fp"])
    return predictions, metrics, cm
