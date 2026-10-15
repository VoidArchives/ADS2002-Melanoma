"""Fast behavioural checks; no pretrained downloads or dataset dependencies."""
import copy
from dataclasses import replace
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from melanoma_detection.model_registry import MODEL_REGISTRY, MODEL_INPUT_SIZE, TRAINING_PRESETS
from melanoma_detection.image_model import (class_weight_from_training, freeze_backbone,
    unfreeze_backbone, freeze_batchnorm, _train_mode_with_frozen_batchnorm,
    parameter_groups, _train_one_epoch, FineTuneCosine, train_image_model,
    choose_microbatch, validation_metrics)
from melanoma_detection.image_model import load_image_state_dict
from melanoma_detection.preprocessing import image_transforms
from melanoma_detection.experiments import (ImageExperimentConfig, ImageExperiment,
    benchmark_image_models, validate_fold_assignments)
from melanoma_detection.splitting import assert_no_patient_overlap
from melanoma_detection.utils import set_seed


def toy_model(device='cpu'):
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(), nn.LayerNorm(4), nn.Linear(4, 1)).to(device)
    model._image_classifier_path = '5'
    model._image_model_name = 'toy'
    return model


def toy_setup(name, targets, **kwargs):
    set_seed(kwargs.get('seed', 42))
    model = toy_model()
    optimizer = torch.optim.AdamW(parameter_groups(model, 1e-4, 1e-3))
    weight = class_weight_from_training(targets)['sqrt_pos_weight']
    return model, nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight])), optimizer, torch.device('cpu')


class ScreeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_registry_sizes_presets_and_transform_outputs(self):
        expected = dict(resnet18=224, resnet34=224, resnet50=224, efficientnet_b0=224,
                        efficientnet_b3=300, efficientnet_b4=380, densenet121=224, convnext_tiny=224)
        self.assertEqual(MODEL_INPUT_SIZE, expected)
        self.assertEqual(set(MODEL_REGISTRY), set(expected))
        image = Image.new('RGB', (65, 91), 'white')
        for name, size in expected.items():
            spec = MODEL_REGISTRY[name]
            self.assertEqual(spec.input_size, size)
            self.assertEqual(ImageExperimentConfig(model_name=name).image_size, size)
            train, validation = image_transforms(model_name=name)
            self.assertEqual(tuple(train(image).shape), (3, size, size))
            self.assertEqual(tuple(validation(image).shape), (3, size, size))
            self.assertTrue(torch.equal(validation(image), validation(image)))
        self.assertEqual(TRAINING_PRESETS['efficientnet_b4'], dict(backbone_lr=3e-5, head_lr=5e-4))
        self.assertEqual(TRAINING_PRESETS['resnet50']['backbone_lr'], 5e-5)

    def test_explicit_fallback_when_crop_metadata_absent(self):
        class Weights:
            def transforms(self):
                return object()
        self.assertEqual(replace(MODEL_REGISTRY['efficientnet_b3'], weights=Weights()).input_size, 300)

    def test_screening_guards_obsolete_configuration(self):
        for args in [dict(model_name='efficientnet_b3', image_size=224), dict(max_epochs=2),
                     dict(patience=1), dict(selection_metric='average_recall'),
                     dict(augmentation='strong'), dict(hair_removal=True)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                ImageExperimentConfig(**args)

    def test_sqrt_class_weight(self):
        result = class_weight_from_training([0] * 4482 + [1] * 87)
        self.assertAlmostEqual(result['sqrt_pos_weight'], math.sqrt(4482 / 87))
        self.assertAlmostEqual(result['raw_imbalance_ratio'], 4482 / 87)
        for values in [[], [0, 0], [1], [2, 0], [[0, 1]]]:
            with self.assertRaises(ValueError):
                class_weight_from_training(values)

    def test_legacy_b3_checkpoint_adapter(self):
        model = MODEL_REGISTRY['efficientnet_b3'].build(pretrained=False)
        state = model.state_dict()
        old = {k.replace('classifier.1.', 'classifier.'): v for k, v in state.items()}
        load_image_state_dict(model, old)
        self.assertTrue(torch.equal(old['classifier.weight'], model.classifier[1].weight))
        self.assertIn('classifier.weight', old)  # input mapping was not mutated

    def test_all_architecture_freeze_policies(self):
        for name, spec in MODEL_REGISTRY.items():
            with self.subTest(name=name):
                model = spec.build(pretrained=False)
                head = {id(p) for p in model.get_submodule(spec.classifier_path).parameters()}
                freeze_backbone(model)
                for p in model.parameters():
                    self.assertEqual(p.requires_grad, id(p) in head)
                unfreeze_backbone(model)
                bn = {id(p) for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
                      for p in m.parameters()}
                for p in model.parameters():
                    self.assertEqual(p.requires_grad, id(p) not in bn)
                if name == 'convnext_tiny':
                    norms = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
                    self.assertTrue(norms)
                    self.assertTrue(all(p.requires_grad for m in norms for p in m.parameters()))
                groups = parameter_groups(model, spec.backbone_lr, spec.head_lr)
                ids = [{id(p) for p in g['params']} for g in groups]
                self.assertFalse(ids[0] & ids[1])
                self.assertEqual(ids[0] | ids[1], {id(p) for p in model.parameters()})
                del model, groups

    def test_running_statistics_frozen_after_train_mode_and_updates(self):
        model = toy_model()
        unfreeze_backbone(model)
        before = {k: v.clone() for k, v in model.state_dict().items() if k.startswith('1.')}
        _train_mode_with_frozen_batchnorm(model)
        self.assertFalse(model[1].training)
        self.assertTrue(model[4].training)
        model(torch.randn(4, 3, 8, 8)).sum().backward()
        torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1).step()
        for k, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[k]), k)

    def test_accumulation_matches_large_batches_and_partial_group(self):
        # 11 samples: effective 8 then a final group of 3 (microbatches 2 + 1).
        torch.manual_seed(1)
        inputs, labels = torch.randn(11, 3), torch.tensor([0., 1.] * 5 + [0.])
        dataset = TensorDataset(inputs, labels, torch.arange(11))
        first, second = nn.Linear(3, 1), nn.Linear(3, 1)
        second.load_state_dict(first.state_dict())
        loss = nn.BCEWithLogitsLoss()
        optimizers = [torch.optim.SGD(m.parameters(), lr=0.1) for m in (first, second)]
        small_scheduler = FineTuneCosine(optimizers[0], 2)
        big_scheduler = FineTuneCosine(optimizers[1], 2)
        small_loss = _train_one_epoch(first, DataLoader(dataset, batch_size=2), loss,
            optimizers[0], 'cpu', accumulation_steps=4, scheduler=small_scheduler)
        big_loss = _train_one_epoch(second, DataLoader(dataset, batch_size=8), loss,
            optimizers[1], 'cpu', scheduler=big_scheduler)
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b)
        self.assertAlmostEqual(small_loss, big_loss, places=6)
        self.assertEqual(small_scheduler.completed_updates, 2)

    def test_schedule_actual_finetuning_updates_and_timing_cpu(self):
        model = toy_model()
        opt = torch.optim.AdamW(parameter_groups(model, 1e-4, 1e-3))
        # 130 -> 65 microbatches -> two updates per epoch, one partial group.
        ds = TensorDataset(torch.randn(130, 3, 4, 4), torch.tensor([0., 1.] * 65), torch.arange(130))
        loader = DataLoader(ds, batch_size=2)
        _, history, _ = train_image_model(model, nn.BCEWithLogitsLoss(), opt,
            loader, loader, 'cpu', accumulation_steps=64, input_size=4)
        self.assertEqual(history.fine_tune_optimizer_updates.tolist(), [0, 2, 4])
        self.assertAlmostEqual(history.head_lr.iloc[-1], 1e-4)
        self.assertAlmostEqual(history.backbone_lr.iloc[-1], 1e-5)
        self.assertAlmostEqual(history.head_lr.iloc[0], 1e-3)
        self.assertTrue(history.effective_batch_size.eq(128).all())
        self.assertEqual(history.phase.tolist(), ['head warmup', 'full fine-tune', 'full fine-tune'])
        for field in ('train_seconds', 'validation_seconds', 'epoch_seconds', 'cumulative_seconds'):
            self.assertTrue(history[field].gt(0).all())
        self.assertTrue(history.cumulative_seconds.is_monotonic_increasing)
        self.assertTrue((history.epoch_seconds >= history.train_seconds + history.validation_seconds).all())
        self.assertTrue(history.average_precision.notna().all())

    def test_validation_single_class_message_and_threshold_independence(self):
        with self.assertRaisesRegex(ValueError, 'Fold 7.*counts.*ROC-AUC cannot'):
            validation_metrics([0, 0], [0.1, 0.2], 7)
        a = validation_metrics([0, 1], [0.2, 0.4], 1, 0.5)
        b = validation_metrics([0, 1], [0.2, 0.4], 1, 0.3)
        self.assertEqual(a['roc_auc'], b['roc_auc'])
        self.assertEqual(a['average_precision'], b['average_precision'])
        self.assertNotEqual(a['balanced_accuracy'], b['balanced_accuracy'])

    def test_auc_wins_when_average_precision_prefers_another_epoch(self):
        model = toy_model()
        optimizer = torch.optim.AdamW(parameter_groups(model, 1e-4, 1e-3))
        labels = np.array([0] * 8 + [1, 1])
        # A very high-ranked positive favours AP in epoch 1; overall ranking
        # across BOTH positives favours AUC in epoch 2.
        scores = [list(np.linspace(0.1, 0.8, 8)) + [0.9, 0.05],
                  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.] + [0.7, 0.8],
                  [0.8] * 8 + [0.1, 0.2]]
        dataset = TensorDataset(torch.zeros(10, 3, 4, 4), torch.tensor(labels).float(), torch.arange(10))
        loader = DataLoader(dataset, batch_size=128)
        with patch('melanoma_detection.image_model._train_one_epoch', return_value=0.5), \
             patch('melanoma_detection.image_model._validate',
                   side_effect=[(0.5, labels, np.array(s)) for s in scores]):
            _, history, epoch = train_image_model(model, nn.BCEWithLogitsLoss(), optimizer,
                loader, loader, 'cpu')
        self.assertEqual(epoch, 2)
        self.assertEqual(history.average_precision.idxmax(), 0)
        self.assertEqual(history.roc_auc.idxmax(), 1)

    def test_patient_leakage_and_final_test_guards(self):
        rows = pd.DataFrame(dict(image_name=['a', 'b'], patient_id=['p', 'p'],
                                 target=[0, 1], split=['train', 'validation']))
        with self.assertRaises(AssertionError):
            assert_no_patient_overlap(rows)
        with self.assertRaises(AssertionError):
            ImageExperiment(ImageExperimentConfig(), rows)
        rows.loc[1, 'split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'held-out test'):
            validate_fold_assignments(rows, None)

    def test_microbatch_cpu_and_mocked_cuda_oom(self):
        model = toy_model()
        criterion = nn.BCEWithLogitsLoss()
        freeze_backbone(model)
        flags = [p.requires_grad for p in model.parameters()]
        with patch('torch.cuda.empty_cache') as cleanup:
            self.assertEqual(choose_microbatch(model, criterion, 'cpu', 32), (16, 8))
            cleanup.assert_not_called()
        # Only the probe is mocked; retry order and cleanup/restoration run normally.
        with patch('torch.random.fork_rng'), patch('torch.cuda.current_device', return_value=0), \
             patch('torch.cuda.empty_cache') as cleanup, \
             patch('melanoma_detection.image_model._memory_probe',
                   side_effect=[torch.cuda.OutOfMemoryError(), torch.cuda.OutOfMemoryError(), None]) as probe:
            self.assertEqual(choose_microbatch(model, criterion, 'cuda', 32), (32, 4))
            self.assertEqual([c.args[-1] for c in probe.call_args_list], [128, 64, 32])
            self.assertEqual(cleanup.call_count, 3)
        self.assertEqual([p.requires_grad for p in model.parameters()], flags)
        with patch('torch.random.fork_rng'), patch('torch.cuda.current_device', return_value=0), \
             patch('torch.cuda.empty_cache'), \
             patch('melanoma_detection.image_model._memory_probe', side_effect=RuntimeError('unrelated')):
            with self.assertRaisesRegex(RuntimeError, 'unrelated'):
                choose_microbatch(model, criterion, 'cuda', 32)
        for batch in (128, 64, 32, 16, 8):
            actual, steps = choose_microbatch(model, criterion, 'cpu', 32, batch)
            self.assertEqual(actual * steps, 128)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_real_cuda_amp_preflight_preserves_state(self):
        model = toy_model('cuda')
        freeze_backbone(model)
        initial = copy.deepcopy(model.state_dict())
        rng = torch.cuda.get_rng_state().clone()
        criterion = nn.BCEWithLogitsLoss().cuda()
        batch, steps = choose_microbatch(model, criterion, 'cuda', 32, requested=8)
        self.assertEqual((batch, steps), (8, 16))
        for k, v in initial.items():
            self.assertTrue(torch.equal(model.state_dict()[k], v), k)
        self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))
        dataset = TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0., 1., 0., 1.]), torch.arange(4))
        opt = torch.optim.AdamW(parameter_groups(model, 1e-4, 1e-3))
        _, history, _ = train_image_model(model, criterion, opt, DataLoader(dataset, batch_size=2),
            DataLoader(dataset, batch_size=2), 'cuda', accumulation_steps=64, input_size=32)
        self.assertEqual(len(history), 3)
        self.assertTrue(np.isfinite(history.train_loss).all())

    def test_training_weights_unaffected_by_validation_or_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'x.jpg'
            Image.new('RGB', (32, 32)).save(path)
            rows = pd.DataFrame(dict(image_name=list('abcdefgh'), patient_id=list('abcdefgh'),
                split=['train'] * 4 + ['validation'] * 2 + ['test'] * 2,
                target=[0, 0, 0, 1, 0, 1, 0, 1], file_path=str(path)))
            config = ImageExperimentConfig(image_size=32, screening=False, pretrained=False)
            with patch('melanoma_detection.experiments.setup_image_model', side_effect=toy_setup):
                first = ImageExperiment(config, rows, device='cpu')
                rows.loc[rows.split.ne('train'), 'target'] = [1, 0, 1, 1]
                second = ImageExperiment(config, rows, device='cpu')
            self.assertAlmostEqual(first.criterion.pos_weight.item(), math.sqrt(3), places=6)
            self.assertEqual(first.criterion.pos_weight.item(), second.criterion.pos_weight.item())

    def test_benchmark_reuses_folds_and_pooled_metrics(self):
        from sklearn.metrics import roc_auc_score, average_precision_score
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'x.jpg'
            Image.new('RGB', (32, 32), 'white').save(path)
            rows = pd.DataFrame(dict(image_name=[f'i{i}' for i in range(12)],
                patient_id=[f'p{i//2}' for i in range(12)], target=[0] * 6 + [1] * 6,
                file_path=str(path)))
            configs = [ImageExperimentConfig(model_name=n, pretrained=False) for n in ['resnet18', 'efficientnet_b3']]
            prediction_calls = [0]

            def shifted_fold_predictions(model, loader, device):
                # Perfect ranking within each fold, but different score offsets
                # across folds: pooled AUC MUST differ from mean fold AUC.
                offset = (prediction_calls[0] % 3) * 0.3
                prediction_calls[0] += 1
                table = loader.dataset.rows[['image_name', 'target']].copy()
                table['melanoma_score'] = offset + 0.1 + table.target * 0.1
                return table

            with patch('melanoma_detection.experiments.setup_image_model', side_effect=toy_setup), \
                 patch('melanoma_detection.experiments.predict_image_model', side_effect=shifted_fold_predictions), \
                 patch('melanoma_detection.experiments.patient_level_split', wraps=__import__(
                     'melanoma_detection.splitting', fromlist=['patient_level_split']).patient_level_split) as splitter:
                result = benchmark_image_models(rows, directory, configs=configs, device='cpu')
            splitter.assert_called_once()
            assignments = []
            for r in result['results'].values():
                table = r['predictions']
                self.assertEqual(len(table), len(rows))
                self.assertTrue(table.image_name.is_unique)
                self.assertTrue(table.groupby('patient_id').fold.nunique().eq(1).all())
                assignments.append(table.set_index('image_name').fold.sort_index())
                summary = r['summary'].iloc[0]
                self.assertAlmostEqual(summary.pooled_oof_roc_auc, roc_auc_score(table.target, table.probability))
                self.assertAlmostEqual(summary.pooled_oof_average_precision, average_precision_score(table.target, table.probability))
                self.assertAlmostEqual(summary.mean_fold_roc_auc, r['fold_metrics'].roc_auc.mean())
                self.assertNotAlmostEqual(summary.mean_fold_roc_auc, summary.pooled_oof_roc_auc)
                self.assertEqual(len(json.loads(summary.fold_roc_auc_values)), 3)
                self.assertTrue(summary.total_seconds > 0)
            pd.testing.assert_series_equal(*assignments)
            self.assertTrue((Path(directory) / 'architecture_summary.csv').exists())
            self.assertTrue((Path(directory) / 'runtime.json').exists())


if __name__ == '__main__':
    unittest.main()
