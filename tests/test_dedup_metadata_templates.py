"""Check original plot composition and current primary-interval accounting."""
from pathlib import Path
import re
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.dedup_notebook_metadata_templates import (
    metadata_patient_intervals, metadata_template_specs,
)
from melanoma_detection.metadata_nn_tuning_reporting import (
    plot_nn_comparison, plot_tuning_bars, plot_tuning_curves,
)


class MetadataTemplateTests(unittest.TestCase):
    def test_original_figure_compositions_and_semantic_colours_are_retained(self):
        code = {filename[:2]: "\n".join(source for kind, _, source in sections if kind == "__code__")
                for filename, _, _, sections in metadata_template_specs()}
        self.assertIn('figsize=(12.5, 14.5)', code["05"])
        self.assertIn('"spline_logistic": "#C58A00"', code["05"])
        self.assertIn('"random_forest": "#6B7F2A"', code["05"])
        self.assertIn('"xgboost": "#EA580C"', code["05"])
        self.assertIn("from melanoma_detection.metadata_nn_tuning_reporting import", code["06"])
        self.assertIn("render_section(report, 'nn_metrics')", code["06"])
        self.assertIn("render_section(report, 'tuning_bars')", code["07"])
        self.assertIn("render_section(report, 'tuning_curves')", code["07"])
        rows = []
        for fold in range(5):
            for target in [0, 1]:
                row = {"fold": fold, "target": target}
                for name in ["spline", "spline_tuned", "random_forest", "random_forest_tuned",
                             "xgboost", "xgboost_tuned", "nn_mean", "nn_tuned"]:
                    row[f"p_{name}"] = .1 + .6 * target
                for seed in [42, 142, 242]:
                    row[f"p_nn_seed_{seed}"] = .1 + .6 * target
                rows.append(row)
        predictions = pd.DataFrame(rows)
        figures = [plot_nn_comparison(predictions), plot_tuning_bars(predictions), plot_tuning_curves(predictions)]
        try:
            self.assertEqual(len(figures[0].axes), 2)
            nn_colors = ["#2563EB", "#D97706", "#D97706", "#D97706", "#303030"]
            for bar, color in zip(figures[0].axes[0].patches, nn_colors):
                np.testing.assert_allclose(bar.get_facecolor(), to_rgba(color))
            self.assertEqual(len(figures[1].axes[0].patches), 8)
            for bar, color in zip(figures[1].axes[0].patches, ["#2563EB"] * 4 + ["#C58A00"] * 4):
                np.testing.assert_allclose(bar.get_facecolor(), to_rgba(color))
            self.assertEqual(len(figures[2].axes), 8)
            np.testing.assert_allclose(figures[2].get_size_inches(), [14, 17.333], atol=.001)
        finally:
            for figure in figures:
                plt.close(figure)
        self.assertIn('("Recalibrated spline", "#C58A00", "s")', code["08"])

    def test_template_execution_has_no_model_fitting_or_reserved_test_read(self):
        for filename, _, _, sections in metadata_template_specs():
            for kind, _, source in sections:
                if kind == "__code__":
                    self.assertIsNone(re.search(r"\.(?:fit|fit_transform)\s*\(", source), filename)
                    self.assertNotIn("evaluate_metadata_holdout(", source)
                    self.assertNotIn("fit_final_bundle(", source)

    def test_paired_interval_point_matches_mean_fold_difference_and_sign(self):
        frame = pd.DataFrame([
            {"image_id": f"i{fold:02}_{patient:02}_{target}", "patient_id": f"p{fold}_{patient}",
             "fold": fold, "target": target}
            for fold in range(5) for patient in range(3) for target in [0, 1]
        ]).sort_values("image_id").reset_index(drop=True)
        raw = np.array([.05 if target else .04 for target in frame.target])
        alternative = raw.copy()
        alternative[::4] = .10
        intervals = metadata_patient_intervals(frame,
            {"Raw spline": raw, "NN mean of three": alternative}, repetitions=30)
        row = intervals.query("aggregation == 'mean_fold' and metric == 'average_precision' and reference_model == 'Raw spline'").iloc[0]
        expected = np.mean([
            average_precision_score(rows.target, alternative[rows.index]) - average_precision_score(rows.target, raw[rows.index])
            for _, rows in frame.groupby("fold")
        ])
        self.assertAlmostEqual(row.estimate, expected)
        self.assertLessEqual(row.ci_lower, row.ci_upper)
        self.assertEqual(row.n_valid, 30)


if __name__ == "__main__":
    unittest.main()
