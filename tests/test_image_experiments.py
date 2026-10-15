import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

from melanoma_detection.evaluation import select_threshold, compute_metrics, compare_validation_thresholds
from melanoma_detection.image_model import setup_image_model, SUPPORTED_MODELS, train_image_model, parameter_groups
from melanoma_detection.preprocessing import image_transforms, DullRazor
from melanoma_detection.experiments import ImageExperiment, ImageExperimentConfig
from melanoma_detection.image_model import load_saved_image_model


class ImageExperimentTests(unittest.TestCase):
    def test_validation_threshold_comparison(self):
        table = pd.DataFrame(dict(image_name=['a', 'b', 'c', 'd'], target=[0, 0, 1, 1],
                                  score=[0.01, 0.03, 0.02, 0.04]))
        reference = table.set_index('image_name').target
        report = compare_validation_thresholds({'toy': table.iloc[::-1]}, reference)
        self.assertEqual(report.TP.tolist(), [0, 2, 2])
        self.assertEqual(report.FN.tolist(), [2, 0, 0])
        self.assertEqual(report.FP.tolist(), [0, 1, 1])
        self.assertTrue(report.AP.eq(report.AP.iloc[0]).all())
        self.assertTrue(report.ROC_AUC.eq(report.ROC_AUC.iloc[0]).all())
        self.assertEqual(report.threshold.iloc[1], 0.02)

    def test_threshold_matches_exhaustive_search(self):
        rng = np.random.default_rng(7)
        labels = np.r_[np.zeros(60), np.ones(20)]
        scores = rng.choice(np.linspace(0, 1, 25), len(labels))
        candidates = np.unique(np.r_[scores, 0, 0.5, 1])
        for method, metric in [("balanced_accuracy", "average_recall"), ("f1", "F1")]:
            threshold = select_threshold(labels, scores, method=method)
            actual = compute_metrics(labels, scores, threshold)[metric]
            best = max(compute_metrics(labels, scores, t)[metric] for t in candidates)
            self.assertAlmostEqual(actual, best)
        threshold = select_threshold(labels, scores, method="sensitivity", min_sensitivity=0.9)
        metrics = compute_metrics(labels, scores, threshold)
        self.assertGreaterEqual(metrics["Sensitivity / recall"], 0.9)
        feasible = [compute_metrics(labels, scores, t) for t in candidates]
        self.assertAlmostEqual(metrics["Specificity"], max(m["Specificity"] for m in feasible
                               if m["Sensitivity / recall"] >= 0.9))

    def test_invalid_threshold_data(self):
        for labels, scores in [([0, 0], [0.1, 0.2]), ([0, 1], [0.1, float("nan")]),
                               ([0, 1], [-0.1, 0.3]), ([0, 2], [0.1, 0.2])]:
            with self.assertRaises(ValueError):
                select_threshold(labels, scores)

    def test_transforms_and_hair_removal(self):
        image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (90, 120, 3), dtype=np.uint8))
        for size in [384, 512]:
            for augmentation in ["basic", "strong", "none"]:
                train, evaluation = image_transforms(size, augmentation, hair_removal=True)
                output = train(image)
                self.assertEqual(tuple(output.shape), (3, size, size))
                self.assertEqual(output.dtype, torch.float32)
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.equal(evaluation(image), evaluation(image)))
        blank = np.full((64, 64, 3), 180, dtype=np.uint8)
        self.assertTrue(np.array_equal(DullRazor()(blank), blank))
        hair = blank.copy()
        hair[10:54, 31:33] = 0
        self.assertGreater(DullRazor()(hair)[20:40, 31:33].mean(), hair[20:40, 31:33].mean())

    def test_all_model_heads_and_optimizer_groups(self):
        torch.set_num_threads(2)
        for name in sorted(SUPPORTED_MODELS):
            with self.subTest(model=name):
                model, _, optimizer, _ = setup_image_model(name, [0, 1], pretrained=False, device="cpu")
                model.eval()
                with torch.no_grad():
                    self.assertEqual(tuple(model(torch.zeros(1, 3, 64, 64)).shape), (1, 1))
                optimized = [id(p) for g in optimizer.param_groups for p in g["params"]]
                self.assertEqual(len(optimized), len(set(optimized)))
                self.assertEqual(set(optimized), {id(p) for p in model.parameters()})
                del model, optimizer

    def test_auc_selection_restores_best_weights(self):
        model = nn.Sequential(nn.Linear(1, 1))
        model._image_classifier_path = "0"
        optimizer = torch.optim.SGD(parameter_groups(model, 0.01, 0.1))
        targets = np.array([0, 0, 1, 1])
        # Epoch 2 has perfect ranking but every score is below 0.5.
        scores = [[0.1, 0.6, 0.4, 0.9], [0.1, 0.2, 0.3, 0.4], [0.8, 0.7, 0.2, 0.1]]
        step = [0]
        def train(*args, **kwargs):
            step[0] += 1
            with torch.no_grad():
                model[0].weight.fill_(step[0])
            return 0.5
        values = [(0.5, targets, np.array(s)) for s in scores]
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(2, 1)), batch_size=128)
        with patch("melanoma_detection.image_model._train_one_epoch", side_effect=train), \
             patch("melanoma_detection.image_model._validate", side_effect=values):
            model, history, epoch = train_image_model(model, nn.BCEWithLogitsLoss(), optimizer,
                                                       loader, loader, "cpu")
        self.assertEqual(epoch, 2)
        self.assertEqual(len(history), 3)
        self.assertEqual(model[0].weight.item(), 2)

    def test_experiment_fit_save_evaluate(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.jpg"
            Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(path)
            rows = pd.DataFrame([dict(image_name=f"{split}_{label}", patient_id=f"{split}_{label}",
                       file_path=str(path), target=label, split=split)
                       for split in ["train", "validation", "test"] for label in [0, 1]])
            cfg = ImageExperimentConfig(image_size=32, augmentation="basic", pretrained=False,
                                        batch_size=2, max_epochs=1, screening=False)
            experiment = ImageExperiment(cfg, rows, device="cpu").fit(checkpoint_dir=Path(folder) / 'during_training')
            saved = experiment.save(folder)
            self.assertTrue((saved / "model.pt").is_file())
            predictions, metrics, _ = experiment.evaluate("test", show_plots=False)
            self.assertEqual(len(predictions), 2)
            self.assertEqual(metrics["Threshold"], experiment.selected_checkpoints['auc']['metadata']['threshold'])
            for name in ['best_auc.pt', 'best_specificity_recall90.pt', 'checkpoint_selection.json']:
                self.assertTrue((saved / name).is_file())
            with patch('melanoma_detection.image_model.select_threshold', side_effect=AssertionError('test tuning')):
                _, specific_metrics, _ = experiment.evaluate('test', show_plots=False, checkpoint='specificity')
            self.assertEqual(specific_metrics['Threshold'],
                             experiment.selected_checkpoints['specificity']['metadata']['threshold'])
            self.assertEqual(experiment.best_epoch, 1)
            self.assertTrue((Path(folder) / 'during_training/best.pt').is_file())
            model, transform, metadata = load_saved_image_model(saved / 'model.pt', device='cpu')
            self.assertEqual(metadata['best_epoch'], 1)
            with Image.open(path) as image:
                tensor = transform(image.convert('RGB')).unsqueeze(0)
            with torch.no_grad():
                torch.testing.assert_close(model(tensor), experiment.model(tensor))

    def test_best_checkpoint_survives_interrupted_training(self):
        from melanoma_detection.utils import load_checkpoint
        model = nn.Sequential(nn.Linear(1, 1))
        model._image_classifier_path = '0'
        optimizer = torch.optim.SGD(parameter_groups(model, 0.01, 0.1))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.zeros(2, 1)), batch_size=128)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'best.pt'
            with patch('melanoma_detection.image_model._train_one_epoch',
                       side_effect=[0.5, RuntimeError('interrupted')]), \
                 patch('melanoma_detection.image_model._validate',
                       return_value=(0.5, np.array([0, 1]), np.array([0.1, 0.9]))):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    train_image_model(model, nn.BCEWithLogitsLoss(), optimizer, loader, loader,
                                      'cpu', checkpoint_path=checkpoint)
            self.assertEqual(load_checkpoint(checkpoint)['metadata']['best_epoch'], 1)
            self.assertTrue((Path(directory) / 'best_auc.pt').exists())
            self.assertTrue((Path(directory) / 'best_specificity_recall90.pt').exists())

    def test_separate_auc_and_recall90_checkpoints(self):
        from melanoma_detection.utils import load_checkpoint
        model = nn.Sequential(nn.Linear(1, 1))
        model._image_classifier_path = '0'
        optimizer = torch.optim.SGD(parameter_groups(model, 0.01, 0.1))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.zeros(2, 1)), batch_size=128)
        labels = np.r_[np.zeros(10), np.ones(10)]
        # Epoch 1: high AUC, but 2/10 positives lie beneath every negative.
        # Epoch 2: lower AUC, but 9/10 positives can be caught with only 5 FP.
        first = np.r_[np.full(10, 0.5), np.full(8, 0.9), [0.1, 0.1]]
        second = np.r_[np.full(5, 0.7), np.full(5, 0.1), np.full(9, 0.6), 0.05]
        step = [0]
        def train(*args, **kwargs):
            step[0] += 1
            with torch.no_grad():
                model[0].weight.fill_(step[0])
            return 0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best.pt'
            with patch('melanoma_detection.image_model._train_one_epoch', side_effect=train), \
                 patch('melanoma_detection.image_model._validate',
                       side_effect=[(0.5, labels, s) for s in [first, second, second]]):
                _, history, epoch = train_image_model(model, nn.BCEWithLogitsLoss(), optimizer,
                    loader, loader, 'cpu', checkpoint_path=path)
            auc = load_checkpoint(Path(directory) / 'best_auc.pt')
            specific = load_checkpoint(Path(directory) / 'best_specificity_recall90.pt')
            self.assertEqual(epoch, 1)
            self.assertEqual(auc['metadata']['best_epoch'], 1)
            self.assertEqual(specific['metadata']['best_epoch'], 2)  # earliest tie wins
            self.assertEqual(auc['state_dict']['0.weight'].item(), 1)
            self.assertEqual(specific['state_dict']['0.weight'].item(), 2)
            self.assertEqual(specific['metadata']['threshold'], 0.5)  # same decisions; tie closest to 0.5
            self.assertAlmostEqual(specific['metadata']['validation_recall'], 0.9)
            self.assertAlmostEqual(specific['metadata']['validation_specificity'], 0.5)
            self.assertEqual(history.recall90_fp.tolist(), [10, 5, 5])
            self.assertEqual(model[0].weight.item(), 1)  # legacy return remains best AUC


if __name__ == "__main__":
    unittest.main()
