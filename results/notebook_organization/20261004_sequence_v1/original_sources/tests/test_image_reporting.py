"""Metrics, saved-report contracts and plots; no model training or test access."""
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from melanoma_detection.image_evaluation import (classification_metrics, curve_tables,
    assert_matching_predictions, metric_table, load_architecture_outputs)
from melanoma_detection.image_reporting import (plot_learning_curves, plot_training_losses,
    plot_roc_precision_recall, plot_confusion_matrices, plot_raw_reliability, save_figure)


class ImageReportingTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"image_id": ["a", "b", "c", "d"], "patient_id": ["p", "p", "q", "q"],
            "fold": [1] * 4, "target": [0, 1, 0, 1], "base_probability": [.1, .5, .7, .9]})

    def test_declared_threshold_and_counts(self):
        frame = self.frame()
        m = classification_metrics(frame.target, frame.base_probability)
        self.assertEqual((m["TN"], m["FP"], m["FN"], m["TP"]), (1, 1, 0, 2))
        np.testing.assert_array_equal(confusion_matrix(frame.target, frame.base_probability >= .5), [[1, 1], [0, 2]])
        self.assertEqual(m["recall"], 1)
        self.assertEqual(m["specificity"], .5)
        self.assertAlmostEqual(m["f1"], .8)
        self.assertEqual(m["threshold"], .5)

    def test_undefined_denominators_are_nan(self):
        m = classification_metrics([0, 1], [.1, .2])
        self.assertTrue(np.isnan(m["precision"]))
        self.assertEqual(m["recall"], 0)
        self.assertEqual(m["f1"], 0)
        all_benign = classification_metrics([0, 0], [.1, .2])
        self.assertTrue(np.isnan(all_benign["recall"]))
        self.assertTrue(np.isnan(all_benign["f1"]))

    def test_invalid_threshold_rejected(self):
        for threshold in [True, -1, 1.01, np.nan, np.inf]:
            with self.assertRaises(ValueError): classification_metrics([0, 1], [.2, .8], threshold)

    def test_actual_curve_and_average_precision(self):
        frame = self.frame()
        curves = curve_tables(frame.target, frame.base_probability)
        self.assertEqual(curves["average_precision"], average_precision_score(frame.target, frame.base_probability))
        self.assertEqual(curves["roc"].false_positive_rate.iloc[-1], 1)
        self.assertEqual(curves["precision_recall"].precision.iloc[-1], 1)
        self.assertTrue(np.isnan(curves["precision_recall"].threshold.iloc[-1]))
        with self.assertRaises(ValueError): curve_tables([1, 1], [.2, .8])

    def test_pairing_rejects_missing_or_changed_identity(self):
        frame = self.frame()
        assert_matching_predictions({"one": frame, "two": frame.iloc[::-1]})
        for altered in [frame.iloc[:-1], frame.assign(patient_id="other"), frame.assign(target=[1, 1, 0, 1]),
                        pd.concat([frame, frame.iloc[:1]])]:
            with self.assertRaises(ValueError): assert_matching_predictions({"one": frame, "two": altered})

    def test_metric_table_records_scale_threshold_epoch_support(self):
        table = metric_table({"x": self.frame()}, epochs={"x": 6})
        row = table.iloc[0]
        self.assertEqual(row.selected_epoch, 6)
        self.assertEqual(row.n_patients, 2)
        self.assertEqual(row.score_type, "raw_base_probability")
        self.assertEqual(row.roc_auc, roc_auc_score(self.frame().target, self.frame().base_probability))

    def test_generic_plots_save_actual_records_and_selected_markers(self):
        frame = self.frame()
        history = pd.DataFrame([{"pipeline_id": f"family_{i}/recipe", "recipe_id": "recipe",
            "loss_weight": "none" if i < 2 else "sqrt_ratio", "selected_epoch": 2,
            "epoch": epoch, "average_precision": .5, "roc_auc": .6, "log_loss": .7,
            "brier": .2, "train_loss": .8} for i in range(3) for epoch in [1, 2]])
        frames = {f"family_{i}/recipe": frame.copy() for i in range(3)}
        figs = [plot_learning_curves(history, .5), plot_training_losses(history),
                plot_roc_precision_recall(frames, epochs={key: 2 for key in frames}),
                plot_confusion_matrices(frames), plot_raw_reliability(frames)[0]]
        self.assertGreater(len(figs[0].axes[0].collections), 0)
        self.assertIn("Exploratory inner validation", figs[0]._suptitle.get_text())
        with tempfile.TemporaryDirectory() as folder:
            for index, fig in enumerate(figs):
                path = save_figure(fig, Path(folder) / f"{index}.png", dpi=60)
                self.assertGreater(path.stat().st_size, 1000)
                plt.close(fig)

    def test_paused_outputs_are_excluded_from_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            self.frame().to_csv(path, index=False)
            record = {"pipeline_id": "f/r", "architecture": "f", "recipe_id": "r",
                "recipe_slot": "low_none", "predictions_file": str(path), "status": "paused",
                "history": [{"epoch": 1, "average_precision": .4}], "best_epoch": 1}
            with self.assertRaises(ValueError):
                load_architecture_outputs(folder, {"recipes": [record], "matched_recipe_slots": ["low_none"]})
            bundle = load_architecture_outputs(folder, {"recipes": [record], "matched_recipe_slots": []})
            self.assertFalse(bundle["predictions"])
            self.assertEqual(len(bundle["histories"]), 1)
            self.assertEqual(bundle["histories"].run_status.iloc[0], "paused")
            record["status"] = "complete"
            record["best_epoch"] = 6
            bundle = load_architecture_outputs(folder, {"recipes": [record], "matched_recipe_slots": []})
            self.assertEqual(len(bundle["predictions"]), 1)
            self.assertFalse(bundle["matched_predictions"])


if __name__ == "__main__": unittest.main()
