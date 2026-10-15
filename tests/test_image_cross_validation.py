import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

from melanoma_detection.experiments import ImageExperimentConfig, cross_validate_image_model
from melanoma_detection.image_model import parameter_groups
from melanoma_detection.splitting import load_or_create_patient_folds


class CrossValidationTests(unittest.TestCase):
    def test_three_independent_training_runs_and_saved_predictions(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "image.jpg"
            Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(path)
            rows = pd.DataFrame({
                "image_name": [f"img{i}" for i in range(12)],
                "patient_id": [f"p{i // 2}" for i in range(12)],
                "target": [0] * 6 + [1] * 6,
                "file_path": str(path), "split": "train",
            })
            folds = load_or_create_patient_folds(rows, root)
            self.assertEqual(folds, load_or_create_patient_folds(rows, root))
            created = []

            def setup(*args, **kwargs):
                model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 1))
                model._image_classifier_path = "2"
                optimizer = torch.optim.SGD(parameter_groups(model, 0.001, 0.01))
                created.append((model, optimizer))
                return model, nn.BCEWithLogitsLoss(), optimizer, torch.device("cpu")

            cfg = ImageExperimentConfig(image_size=32, augmentation="none", batch_size=4,
                                        pretrained=False, screening=False)
            self.assertEqual((cfg.max_epochs, cfg.patience), (3, None))
            with patch("melanoma_detection.experiments.setup_image_model", side_effect=setup):
                result = cross_validate_image_model(cfg, rows, root / "runs",
                                                    split_maps=folds, device="cpu")
            self.assertEqual(len({id(m) for m, _ in created}), 3)
            self.assertEqual(len({id(o) for _, o in created}), 3)
            self.assertEqual(len(result["predictions"]), len(rows))
            self.assertTrue(result["predictions"].image_name.is_unique)
            for fold, folder in result["fold_folders"].items():
                self.assertTrue((folder / "model.pt").exists())
                self.assertTrue((folder / "splits.csv").exists())
                self.assertEqual(len(result["histories"][fold]), 3)
                validated = result["predictions"].query("fold == @fold").image_name
                patients = rows.set_index("image_name").loc[validated, "patient_id"]
                self.assertTrue(patients.map(folds[fold - 1]).eq("validation").all())
            self.assertTrue((result["folder"] / "summary.csv").exists())
            self.assertTrue((result["folder"] / "out_of_fold_predictions.csv").exists())
            # A second invocation must use disk artifacts without even setting up
            # a model, optimizer, pretrained download or GPU preflight.
            with patch("melanoma_detection.experiments.ImageExperiment") as experiment:
                reused = cross_validate_image_model(cfg, rows, root / "runs",
                                                     split_maps=folds, device="cpu", reuse="only")
                experiment.assert_not_called()
            self.assertTrue(reused['fold_metrics'].reused.all())
            pd.testing.assert_frame_equal(result['predictions'], reused['predictions'],
                                          check_exact=False, atol=1e-14, rtol=1e-14)
            for folder in result['fold_folders'].values():
                self.assertTrue((folder / 'best.pt').is_file())

            # Existing runs predate the manifest; full OOF labels plus the saved
            # split maps still allow them to be reused without retraining.
            first_folder = result['fold_folders'][1]
            manifest = (first_folder / 'input_rows.csv').read_bytes()
            (first_folder / 'input_rows.csv').unlink()
            with patch("melanoma_detection.experiments.ImageExperiment") as experiment:
                cross_validate_image_model(cfg, rows, root / 'runs', split_maps=folds, reuse='only')
                experiment.assert_not_called()
            (first_folder / 'input_rows.csv').write_bytes(manifest)

            # A changed training label must also invalidate a cache, not only a
            # changed validation label. The all-input manifest covers both.
            changed = rows.copy()
            train_image = rows.loc[rows.patient_id.map(folds[0]).eq('train')].index[0]
            changed.loc[train_image, 'target'] = 1 - changed.loc[train_image, 'target']
            with patch("melanoma_detection.experiments.ImageExperiment") as experiment:
                with self.assertRaisesRegex(ValueError, 'Saved input rows differ'):
                    cross_validate_image_model(cfg, changed, root / 'runs', split_maps=folds)
                experiment.assert_not_called()

            # Incomplete folds are never loaded as finished, and reuse-only never
            # silently starts an expensive training run.
            (result['fold_folders'][1] / 'runtime.json').unlink()
            with patch("melanoma_detection.experiments.ImageExperiment") as experiment:
                with self.assertRaisesRegex(FileNotFoundError, 'Reuse-only never trains'):
                    cross_validate_image_model(cfg, rows, root / 'runs', split_maps=folds, reuse='only')
                experiment.assert_not_called()
            with patch("melanoma_detection.experiments.setup_image_model", side_effect=setup):
                partial = cross_validate_image_model(cfg, rows, root / 'runs', split_maps=folds)
            self.assertEqual(partial['fold_metrics'].reused.tolist(), [False, True, True])

    def test_rejects_repeated_validation_before_training(self):
        rows = pd.DataFrame({"image_name": ["a", "b"], "patient_id": ["p", "q"],
                             "target": [0, 1]})
        with tempfile.TemporaryDirectory() as directory:
            with patch("melanoma_detection.experiments.ImageExperiment") as experiment:
                with self.assertRaises(ValueError):
                    cross_validate_image_model(ImageExperimentConfig(), rows, directory,
                        split_maps=[{"p": "train", "q": "validation"}] * 3)
                experiment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
