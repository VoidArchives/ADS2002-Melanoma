"""Patient-safe CNN experiments and architecture screening with pooled OOF metrics."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .data import make_loaders
from .evaluation import select_threshold
from .image_model import (setup_image_model, train_image_model, predict_image_model,
    evaluate_image_model, choose_microbatch, synchronize, format_duration, validation_metrics,
    freeze_backbone, unfreeze_backbone, class_weight_from_training)
from .model_registry import (MODEL_REGISTRY, get_model_spec, NUMBER_OF_EPOCHS,
    TARGET_EFFECTIVE_BATCH_SIZE, WEIGHT_DECAY, GRADIENT_CLIP_NORM)
from .preprocessing import image_transforms
from .splitting import (patient_level_split, iter_patient_folds, save_splits,
                        assert_no_patient_overlap)
from .utils import save_checkpoint, set_seed, RANDOM_STATE

logger = logging.getLogger(__name__)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ImageExperimentConfig:
    model_name: str = "resnet18"
    image_size: int | None = None
    augmentation: str = "basic"
    hair_removal: bool = False
    rotation: float = 0
    batch_size: int | None = None  # optional preflight cap, not effective batch
    num_workers: int = 0
    worker_timeout: float = 60
    max_epochs: int = NUMBER_OF_EPOCHS
    patience: int | None = None
    selection_metric: str = "validation_ROC_AUC"
    threshold_method: str = "fixed"
    min_sensitivity: float = 0.95
    feature_lr: float | None = None
    classifier_lr: float | None = None
    weight_decay: float = WEIGHT_DECAY
    gradient_clip_norm: float = GRADIENT_CLIP_NORM
    seed: int = RANDOM_STATE
    pretrained: bool = True
    screening: bool = True

    def __post_init__(self):
        spec = get_model_spec(self.model_name)
        object.__setattr__(self, "model_name", spec.name)
        if self.image_size is None:
            object.__setattr__(self, "image_size", spec.input_size)
        if self.patience is not None or self.selection_metric != "validation_ROC_AUC":
            raise ValueError("Use patience=None and selection_metric='validation_ROC_AUC'.")
        if self.max_epochs < 1 or self.gradient_clip_norm <= 0:
            raise ValueError("Epoch budget and gradient clip norm must be positive.")
        if self.screening and (self.max_epochs != NUMBER_OF_EPOCHS or self.image_size != spec.input_size
                or self.augmentation != "basic" or self.hair_removal or self.threshold_method != "fixed"):
            raise ValueError("Screening requires canonical size, basic augmentation, no hair removal, "
                             "3 epochs and fixed threshold 0.5; use screening=False for separate experiments.")
        if self.threshold_method not in {"fixed", "balanced_accuracy", "f1", "sensitivity"}:
            raise ValueError("Unknown threshold_method.")

    @property
    def run_name(self):
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.model_name}_{self.image_size}_{self.augmentation}_hair{int(self.hair_removal)}_{digest}"


class ImageExperiment:
    """One fold; fit and selection never read held-out test images or labels.

    evaluate('test') is an explicit final evaluation action after model selection.
    The CV runner always rejects tables containing held-out test rows.
    """
    def __init__(self, config, clean_images, device=None, *, fold=1):
        self.config, self.fold = config, fold
        self.started_at = timestamp()
        start = time.perf_counter()
        if clean_images['patient_id'].isna().any() or clean_images['image_name'].isna().any():
            raise ValueError("Patient and image IDs must not be missing.")
        assert_no_patient_overlap(clean_images)
        if not clean_images['image_name'].is_unique:
            raise ValueError("Duplicate image IDs.")
        for split in ('train', 'validation'):
            if clean_images.loc[clean_images['split'].eq(split)].empty:
                raise ValueError(f"Empty split: {split}")
        validation_targets = clean_images.loc[clean_images['split'].eq('validation'), 'target']
        validation_metrics(validation_targets, np.zeros(len(validation_targets)), fold)
        train_targets = clean_images.loc[clean_images['split'].eq('train'), 'target']
        self.model, self.criterion, self.optimizer, self.device = setup_image_model(
            config.model_name, train_targets, seed=config.seed,
            feature_lr=config.feature_lr, classifier_lr=config.classifier_lr,
            weight_decay=config.weight_decay, device=device, pretrained=config.pretrained)
        self.microbatch_size, self.accumulation_steps = choose_microbatch(
            self.model, self.criterion, self.device, config.image_size, config.batch_size)
        self.effective_batch_size = self.microbatch_size * self.accumulation_steps
        set_seed(config.seed)  # preflight must not shift training augmentation/dropout RNG
        train, evaluation = image_transforms(config.image_size, config.augmentation,
            config.hair_removal, config.seed, model_name=config.model_name, rotation=config.rotation)
        self.datasets, self.loaders, _ = make_loaders(clean_images, train, evaluation,
            self.microbatch_size, config.seed, config.num_workers,
            worker_timeout=config.worker_timeout, device=self.device)
        spec = get_model_spec(config.model_name)
        unfreeze_backbone(self.model)
        full_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        freeze_backbone(self.model)
        self.metadata = dict(model=config.model_name, fold=fold, seed=config.seed,
            device=str(self.device), weight_enum=spec.weight_enum if config.pretrained else 'None',
            canonical_weight_enum=spec.weight_enum, input_size=config.image_size,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            trainable_parameters_epoch1=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            trainable_parameters_full=full_count, microbatch_size=self.microbatch_size,
            accumulation_steps=self.accumulation_steps, effective_batch_size=self.effective_batch_size,
            training_presets={g['name'] + '_lr': g['lr'] for g in self.optimizer.param_groups},
            started_at=self.started_at, **class_weight_from_training(train_targets))
        synchronize(self.device)
        self.metadata['setup_seconds'] = time.perf_counter() - start
        self.threshold = self.history = self.best_epoch = None
        logger.info("Model: %s | Fold: %s | Input: %dx%d | Weights: %s | Device: %s\n"
                    "Target effective batch: %d | Microbatch: %d | Accumulation: %d | Effective batch: %d",
                    config.model_name, fold, config.image_size, config.image_size,
                    self.metadata['weight_enum'], self.device, TARGET_EFFECTIVE_BATCH_SIZE,
                    self.microbatch_size, self.accumulation_steps, self.effective_batch_size)

    def fit(self, checkpoint_dir=None):
        c = self.config
        self.threshold = None
        self.model, self.history, self.best_epoch = train_image_model(
            self.model, self.criterion, self.optimizer, self.loaders['train'],
            self.loaders['validation'], self.device, max_epochs=c.max_epochs,
            patience=c.patience, selection_metric=c.selection_metric, label=c.model_name,
            fold=self.fold, input_size=c.image_size, accumulation_steps=self.accumulation_steps,
            gradient_clip_norm=c.gradient_clip_norm, screening=c.screening,
            checkpoint_path=None if checkpoint_dir is None else Path(checkpoint_dir) / 'best.pt',
            checkpoint_metadata={'config': asdict(c), 'fold': self.fold})
        self.selected_checkpoints = self.history.attrs.pop('selected_checkpoints')
        self.validation_predictions = predict_image_model(self.model, self.loaders['validation'], self.device)
        self.threshold = 0.5 if c.threshold_method == 'fixed' else select_threshold(
            self.validation_predictions['target'], self.validation_predictions['melanoma_score'],
            method=c.threshold_method, min_sensitivity=c.min_sensitivity)
        self.validation_predictions['prediction'] = (
            self.validation_predictions['melanoma_score'] >= self.threshold).astype(int)
        h = self.history
        self.metadata.update(training_wall_seconds=h.attrs['training_wall_seconds'],
            total_train_seconds=float(h.train_seconds.sum()),
            total_validation_seconds=float(h.validation_seconds.sum()),
            mean_epoch_seconds=float(h.epoch_seconds.mean()),
            fastest_epoch_seconds=float(h.epoch_seconds.min()),
            slowest_epoch_seconds=float(h.epoch_seconds.max()),
            best_validation_roc_auc=float(h.roc_auc.max()), ended_at=timestamp())
        return self

    def evaluate(self, split='validation', show_plots=True, checkpoint='auc'):
        if self.threshold is None:
            raise RuntimeError('Call fit() before evaluation.')
        if split not in {'validation', 'test'}:
            raise ValueError('Evaluate validation or explicitly selected final test only.')
        if checkpoint not in self.selected_checkpoints:
            raise ValueError('checkpoint must be auc or specificity.')
        selected = self.selected_checkpoints[checkpoint]
        # Both weights and cutoff were selected on validation during fit. Test
        # labels are used for reporting only, never for threshold selection.
        self.model.load_state_dict(selected['state_dict'])
        try:
            return evaluate_image_model(self.model, self.loaders[split], self.device,
                self.config.run_name, split, selected['metadata']['threshold'], show_plots)
        finally:
            self.model.load_state_dict(self.selected_checkpoints['auc']['state_dict'])

    def save(self, output_dir):
        if self.threshold is None:
            raise RuntimeError('Call fit() before saving.')
        folder = Path(output_dir) / self.config.run_name
        folder.mkdir(parents=True, exist_ok=True)
        save_checkpoint(self.model.state_dict(), folder / 'model.pt',
                        config=asdict(self.config), best_epoch=self.best_epoch,
                        threshold=self.threshold, run_metadata=self.metadata)
        for key, filename in [('auc', 'best_auc.pt'), ('specificity', 'best_specificity_recall90.pt')]:
            chosen = self.selected_checkpoints[key]
            save_checkpoint(chosen['state_dict'], folder / filename, **chosen['metadata'])
        (folder / 'checkpoint_selection.json').write_text(json.dumps({
            key: chosen['metadata'] for key, chosen in self.selected_checkpoints.items()
        }, indent=2), encoding='utf-8')
        self.history.to_csv(folder / 'history.csv', index=False)
        self.validation_predictions.to_csv(folder / 'validation_predictions.csv', index=False)
        (folder / 'config.json').write_text(json.dumps({**asdict(self.config),
            'best_epoch': self.best_epoch, 'threshold': self.threshold,
            'run_metadata': self.metadata}, indent=2), encoding='utf-8')
        return folder


def validate_fold_assignments(clean_images, split_maps, seed=RANDOM_STATE):
    if (clean_images['patient_id'].isna().any() or clean_images['image_name'].isna().any()
            or not clean_images['image_name'].is_unique):
        raise ValueError('Images must have unique IDs and non-missing patient IDs.')
    if 'split' in clean_images and clean_images['split'].eq('test').any():
        raise ValueError('Remove the held-out test set before cross-validation.')
    if split_maps is None:
        patients = clean_images.groupby('patient_id', as_index=False)['target'].max()
        split_maps = patient_level_split(patients, seed=seed, label_col='target')
    if len(split_maps) != 3:
        raise ValueError('Exactly three fold mappings are required.')
    assignments = pd.DataFrame([clean_images['patient_id'].map(m).to_numpy() for m in split_maps])
    if (not assignments.isin(['train', 'validation']).all().all()
            or not assignments.eq('validation').sum(axis=0).eq(1).all()):
        raise ValueError('Every patient must validate exactly once and train twice.')
    # Fail all invalid folds before downloading any weights or starting training.
    for fold, rows in iter_patient_folds(clean_images, split_maps):
        train = rows.loc[rows.split.eq('train')]
        validation = rows.loc[rows.split.eq('validation')]
        assert set(train.patient_id).isdisjoint(validation.patient_id), 'Patient leakage'
        class_weight_from_training(train.target)
        validation_metrics(validation.target, np.zeros(len(validation)), fold)
    return split_maps


def _load_completed_fold(folder, config, rows, mapping, pooled_path):
    """Validate saved artifacts before reusing a completed fold (no GPU needed)."""
    required = ['model.pt', 'config.json', 'history.csv', 'validation_predictions.csv',
                'splits.csv', 'runtime.json', 'best_auc.pt', 'best_specificity_recall90.pt',
                'checkpoint_selection.json']
    if not all((folder / name).is_file() for name in required):
        return None
    saved = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if any(saved.get(key) != value for key, value in asdict(config).items()):
        raise ValueError(f'Saved configuration differs at {folder}; use another output directory or reuse="never".')
    from .splitting import load_splits
    if load_splits(folder / 'splits.csv') != mapping:
        raise ValueError(f'Saved patient splits differ at {folder}; use another output directory or reuse="never".')
    # New runs store all input labels/IDs/paths, including training rows. Legacy
    # runs can be checked against their complete pooled OOF table instead.
    manifest = folder / 'input_rows.csv'
    source = manifest if manifest.exists() else pooled_path
    if not source.exists():
        logger.info('Cannot verify legacy training rows in %s; this fold needs training.', folder)
        return None
    prior = pd.read_csv(source)
    keys = ['image_name', 'patient_id', 'target']
    if manifest.exists():
        keys += ['split'] + (['file_path'] if 'file_path' in rows else [])
    def aligned(table, columns):
        return table[columns].sort_values('image_name').reset_index(drop=True).astype(str)
    if not prior.image_name.is_unique or not aligned(prior, keys).equals(aligned(rows, keys)):
        raise ValueError(f'Saved input rows differ at {folder}; use another output directory or reuse="never".')
    table = pd.read_csv(folder / 'validation_predictions.csv')
    expected = rows.loc[rows.split.eq('validation')]
    if (not table.image_name.is_unique
            or not aligned(table, ['image_name', 'patient_id', 'target']).equals(
                aligned(expected, ['image_name', 'patient_id', 'target']))):
        raise ValueError(f'Saved validation predictions do not match this fold: {folder}')
    scores = table.melanoma_score.to_numpy()
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError(f'Invalid saved probabilities: {folder}')
    history = pd.read_csv(folder / 'history.csv')
    if history.epoch.tolist() != list(range(1, config.max_epochs + 1)):
        return None
    metadata = json.loads((folder / 'runtime.json').read_text(encoding='utf-8'))
    return SimpleNamespace(validation_predictions=table, history=history,
                           best_epoch=saved['best_epoch'], metadata=metadata)


def cross_validate_image_model(config, clean_images, output_dir, *, split_maps=None, device=None,
                               reuse='auto'):
    """Restore best AUC weights per fold, predict each image once, pool OOF metrics."""
    started_at, start = timestamp(), time.perf_counter()
    if reuse not in {'auto', 'only', 'never'}:
        raise ValueError('reuse must be auto, only, or never.')
    split_maps = validate_fold_assignments(clean_images, split_maps, config.seed)
    folder = Path(output_dir) / config.run_name
    folder.mkdir(parents=True, exist_ok=True)
    metrics_rows, predictions, histories, fold_folders, run_metadata = [], [], {}, {}, []
    for fold, rows in iter_patient_folds(clean_images, split_maps):
        fold_start = time.perf_counter()
        experiment = None
        try:
            fold_folder = folder / f'fold_{fold}' / config.run_name
            experiment = None if reuse == 'never' else _load_completed_fold(
                fold_folder, config, rows, split_maps[fold - 1], folder / 'out_of_fold_predictions.csv')
            reused = experiment is not None
            if not reused:
                if reuse == 'only':
                    raise FileNotFoundError(f'No verified completed fold at {fold_folder}. Reuse-only never trains.')
                # Remove the completion marker before any new writes. An interrupted
                # rerun must not be mistaken for an old completed fold.
                (fold_folder / 'runtime.json').unlink(missing_ok=True)
                experiment = ImageExperiment(config, rows, device=device, fold=fold).fit(
                    checkpoint_dir=fold_folder)
            else:
                logger.info('Reusing %s fold %d from %s', config.model_name, fold, fold_folder)
            table = experiment.validation_predictions.copy()
            metrics = validation_metrics(table.target, table.melanoma_score, fold)
            if 'patient_id' not in table:
                table = table.merge(rows[['image_name', 'patient_id']], on='image_name', validate='one_to_one')
            table = table.assign(fold=fold, model=config.model_name, probability=table.melanoma_score)
            if not reused:
                experiment.validation_predictions = table
                fold_folder = experiment.save(folder / f'fold_{fold}')
                save_splits(split_maps[fold - 1], fold_folder / 'splits.csv')
                columns = ['image_name', 'patient_id', 'target', 'split']
                if 'file_path' in rows:
                    columns.append('file_path')
                rows[columns].to_csv(fold_folder / 'input_rows.csv', index=False)
                synchronize(experiment.device)
            elapsed = time.perf_counter() - fold_start
            if not reused:
                experiment.metadata.update(total_seconds=elapsed, ended_at=timestamp())
                (fold_folder / 'runtime.json').write_text(json.dumps(experiment.metadata, indent=2), encoding='utf-8')
            predictions.append(table)
            histories[fold] = experiment.history.copy()
            fold_folders[fold] = fold_folder
            run_metadata.append(experiment.metadata.copy())
            metrics_rows.append({**metrics, 'ROC-AUC': metrics['roc_auc'], 'fold': fold,
                                 'best_epoch': experiment.best_epoch, 'total_seconds': elapsed,
                                 'reused': reused})
            pd.DataFrame(metrics_rows).to_csv(folder / 'fold_metrics.csv', index=False)
            logger.info('Fold %d total time: %s', fold, format_duration(elapsed))
        finally:
            del experiment
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    oof = pd.concat(predictions, ignore_index=True)
    if not oof.image_name.is_unique or set(oof.image_name) != set(clean_images.image_name):
        raise ValueError('Out-of-fold predictions must cover every image exactly once.')
    if not oof.groupby('patient_id').fold.nunique().eq(1).all():
        raise AssertionError('Each patient must appear in one OOF fold.')
    pooled = validation_metrics(oof.target, oof.probability, 'pooled OOF')
    fold_metrics = pd.DataFrame(metrics_rows).set_index('fold')
    aucs = fold_metrics.roc_auc
    first = run_metadata[0]
    summary_row = {key: first[key] for key in ('model', 'weight_enum', 'input_size', 'parameter_count',
        'trainable_parameters_epoch1', 'trainable_parameters_full', 'device')}
    for key in ('microbatch_size', 'accumulation_steps', 'effective_batch_size'):
        values = [m[key] for m in run_metadata]
        summary_row[key] = values[0] if len(set(values)) == 1 else json.dumps(values)
    summary_row.update(fold_roc_auc_values=json.dumps(aucs.tolist()),
        mean_fold_roc_auc=aucs.mean(), std_fold_roc_auc=aucs.std(ddof=1),
        min_fold_roc_auc=aucs.min(), max_fold_roc_auc=aucs.max(),
        **{'pooled_oof_' + key: value for key, value in pooled.items()},
        total_seconds=time.perf_counter()-start,
        mean_seconds_per_epoch=pd.concat(histories.values()).epoch_seconds.mean())
    summary = pd.DataFrame([summary_row])
    oof.to_csv(folder / 'out_of_fold_predictions.csv', index=False)
    summary.to_csv(folder / 'summary.csv', index=False)
    pd.concat(histories.values()).to_csv(folder / 'history.csv', index=False)
    runtime = dict(started_at=started_at, ended_at=timestamp(),
                   total_seconds=time.perf_counter()-start, folds=run_metadata)
    (folder / 'runtime.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    checkpoint_rows = []
    for fold, fold_folder in fold_folders.items():
        choices = json.loads((fold_folder / 'checkpoint_selection.json').read_text(encoding='utf-8'))
        for policy, metadata in choices.items():
            checkpoint_rows.append(dict(model=config.model_name, fold=fold, policy=policy,
                **{key: metadata[key] for key in ('best_epoch', 'threshold', 'target_recall',
                   'validation_recall', 'validation_specificity', 'validation_roc_auc')}))
    checkpoint_comparison = pd.DataFrame(checkpoint_rows)
    checkpoint_comparison.to_csv(folder / 'checkpoint_comparison.csv', index=False)
    return dict(folder=folder, fold_folders=fold_folders, histories=histories,
                fold_metrics=fold_metrics, summary=summary, predictions=oof, runtime=runtime,
                checkpoint_comparison=checkpoint_comparison)


def benchmark_image_models(clean_images, output_dir, *, configs=None, split_maps=None, device=None,
                           reuse='auto'):
    """Generate/validate folds ONCE and reuse the exact mappings for every architecture."""
    start, started_at = time.perf_counter(), timestamp()
    configs = list(configs) if configs is not None else [ImageExperimentConfig(model_name=n) for n in MODEL_REGISTRY]
    if not configs or len({c.model_name for c in configs}) != len(configs):
        raise ValueError('Provide one screening configuration per architecture.')
    fairness = lambda c: (c.seed, c.rotation, c.pretrained, c.weight_decay, c.gradient_clip_norm)
    if any(not c.screening or fairness(c) != fairness(configs[0]) for c in configs):
        raise ValueError('Architecture comparison requires equivalent screening conditions and seeds.')
    split_maps = validate_fold_assignments(clean_images, split_maps, configs[0].seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold, mapping in enumerate(split_maps, 1):
        save_splits(mapping, output_dir / f'splits_fold_{fold}.csv')
    results = {}
    for config in configs:
        results[config.model_name] = cross_validate_image_model(config, clean_images, output_dir,
            split_maps=split_maps, device=device, reuse=reuse)
        summary = pd.concat([r['summary'] for r in results.values()], ignore_index=True)
        summary = summary.sort_values('pooled_oof_roc_auc', ascending=False)
        summary.to_csv(output_dir / 'architecture_summary.csv', index=False)
        checkpoint_comparison = pd.concat([r['checkpoint_comparison'] for r in results.values()], ignore_index=True)
        checkpoint_comparison.to_csv(output_dir / 'checkpoint_comparison.csv', index=False)
    runtime = dict(started_at=started_at, ended_at=timestamp(), total_seconds=time.perf_counter()-start,
                   architectures={name: r['runtime'] for name, r in results.items()})
    (output_dir / 'runtime.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    return dict(summary=summary, results=results, split_maps=split_maps, runtime=runtime,
                checkpoint_comparison=checkpoint_comparison)
