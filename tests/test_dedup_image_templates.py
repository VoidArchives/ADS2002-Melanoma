"""Regression guards against clean image reports drifting from their templates."""
import ast
import unittest
from unittest.mock import patch

from melanoma_detection.dedup_notebook_image_templates import image_template_specs
from melanoma_detection.dedup_notebook_templates import MARKDOWN_CELL, CODE_CELL, original_cells, markdown_words


class ImageTemplateConsistencyTests(unittest.TestCase):
    def test_image_reports_retain_original_narrative_depth_and_cell_types(self):
        sources = {
            "09": ["09_image_b0_pilot.ipynb", "14_image_nested_evaluation.ipynb"],
            "10": ["10_image_architecture_comparison.ipynb", "11_image_augmentation.ipynb", "12_image_lighting.ipynb"],
            "11": ["13_image_uncertainty.ipynb"],
        }
        for filename, _, _, sections in image_template_specs():
            with self.subTest(filename=filename):
                original = [(MARKDOWN_CELL, cell.source, "")
                    for name in sources[filename[:2]] for cell in original_cells(name) if cell.cell_type == "markdown"]
                self.assertGreaterEqual(markdown_words(sections), .9 * markdown_words(original))
                self.assertTrue(all(kind in {MARKDOWN_CELL, CODE_CELL} for kind, _, _ in sections))
                code = "\n".join(source for kind, _, source in sections if kind == CODE_CELL)
                ast.parse(code)
                self.assertIn(f"NOTEBOOK_STAGE = {filename[:-6]!r}", code)
                for forbidden in ("run_inference(", "run_smoke(", "load_selected_checkpoint(", ".fit("):
                    self.assertNotIn(forbidden, code)

    def test_deferred_experiments_do_not_copy_historical_performance(self):
        filename, _, _, sections = image_template_specs()[1]
        markdown = "\n".join(source for kind, source, _ in sections if kind == MARKDOWN_CELL)
        self.assertIn("DenseNet121 has no clean pilot result", markdown)
        self.assertIn("Scope change — deferred stronger augmentation", markdown)
        self.assertIn("Scope change — deferred CLAHE lighting correction", markdown)
        code = "\n".join(source for kind, _, source in sections if kind == CODE_CELL)
        self.assertIn("deferred_condition_status", code)
        self.assertNotIn("render_condition_report", code)

    def test_referral_styles_are_identical_to_original_literals(self):
        from melanoma_detection.dedup_notebook_images import REFERRAL_COLORS, REFERRAL_LINES
        original = {}
        for cell in original_cells("13_image_uncertainty.ipynb"):
            if cell.cell_type != "code":
                continue
            try:
                tree = ast.parse(cell.source)
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id in {"COLORS", "LINESTYLES"}:
                        original[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(REFERRAL_COLORS, original["COLORS"])
        self.assertEqual(REFERRAL_LINES, original["LINESTYLES"])

    def test_outer_figures_preserve_fold_and_fixed_reliability_composition(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from melanoma_detection import dedup_notebook_images as image_docs
        frame = pd.DataFrame({"fold": np.repeat(np.arange(5), 4),
            "patient_id": [f"p{i}" for i in range(20)], "target": np.tile([0, 0, 1, 1], 5),
            "base_probability": np.linspace(.02, .98, 20)})
        folds = pd.DataFrame([{"fold": i, "average_precision": .5 + i * .02,
            "roc_auc": .6 + i * .02, "prevalence": .5, "log_loss": .5, "brier": .2} for i in range(5)])
        raw = {"complete": True, "predictions": frame, "fold_metrics": folds}
        figures = {}
        def capture(figure, name):
            figure.canvas.draw()
            figures[name] = figure
        try:
            with patch.object(image_docs, "display"):
                image_docs.raw_outer_ranking(raw, capture)
                image_docs.raw_outer_reliability(raw, capture)
            curves = figures["outer_fold_pr_roc"]
            self.assertEqual(len(curves.axes), 2)
            self.assertEqual(curves.axes[0].get_xlabel(), "Recall")
            self.assertEqual(curves.axes[1].get_xlabel(), "False positive rate")
            self.assertEqual([line.get_color() for line in curves.axes[0].lines[:5]],
                ["#235789", "#C47A14", "#66743A", "#A45078", "#333333"])
            self.assertEqual([line.get_linestyle() for line in curves.axes[0].lines[:5]], ["-", "--", "-", "--", "-"])
            ap = figures["outer_fold_average_precision"]
            self.assertEqual(len(ap.axes), 1)
            self.assertEqual(len(ap.axes[0].patches), 0, "Original fold AP is a scatter, not replacement bars")
            self.assertEqual(len(ap.axes[0].collections), 6, "Five coloured fold points and prevalence crosses")
            reliability = figures["raw_reliability"]
            self.assertEqual(len(reliability.axes), 2)
            self.assertEqual(len(reliability.axes[1].patches), 10, "Original score histogram uses ten fixed-width bins")
            self.assertEqual(len(reliability.axes[0].texts), 10, "Each supported fixed bin has an image-count annotation")
            self.assertEqual(reliability.axes[0].lines[1].get_color(), "#235789")
            confusion = figures["raw_b0_confusion_counts"]
            np.testing.assert_allclose(np.asarray(confusion.axes[0].images[0].get_array()).sum(axis=1), [1., 1.])
        finally:
            for figure in figures.values():
                plt.close(figure)

    def test_generated_report_calls_cannot_train_or_run_inference(self):
        forbidden = {"fit", "fit_transform", "train", "run_inference", "run_smoke", "load_selected_checkpoint"}
        for filename, _, _, sections in image_template_specs():
            for kind, _, code in sections:
                if kind != CODE_CELL:
                    continue
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                    self.assertNotIn(name, forbidden, f"{filename}: disallowed report action {name}")

    def test_combined_extensions_keep_one_model_style_across_chart_types(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex
        import numpy as np
        import pandas as pd
        from melanoma_detection.dedup_notebook_evaluation import combined_extension_specs, COMPARISON_COLORS, COMPARISON_LINES, METRIC_LABELS
        from melanoma_detection.dedup_notebook_metadata import LABELS
        labels = {**LABELS, "b0_raw": "Raw B0"}
        names = ["b0_raw", "spline_raw", "metadata_nn_mean"]
        frame = pd.DataFrame({"target": np.tile([0, 0, 1, 1], 5)})
        scores = {name: np.linspace(.02 + i * .001, .97 + i * .001, len(frame)) for i, name in enumerate(names)}
        table = pd.DataFrame([{"model": name, "average_precision": .1 + i * .01, "roc_auc": .7 + i * .01} for i, name in enumerate(names)])
        metrics = pd.DataFrame([{"model": name, "fold": fold, "aggregation": "by_fold", "weighting": "image",
            "n_images": 4, "n_malignant_images": 2, "n_malignant_patients": 2,
            "average_precision": .1 + i * .01, "roc_auc": .7 + i * .01}
            for i, name in enumerate(names) for fold in range(5)])
        subgroups = pd.DataFrame([{"model": name, "axis": "site_knownness", "subgroup": group,
            "n_images": 10, "n_patients": 10, "n_malignant_images": 5, "n_malignant_patients": 5,
            "average_precision": .1 + i * .01, "roc_auc": .7 + i * .01}
            for i, name in enumerate(names) for group in ("known", "unknown")])
        figures = {}
        def capture(figure, name):
            figure.canvas.draw()
            figures[name] = figure
        namespace = {"np": np, "pd": pd, "plt": plt, "display": lambda *args: None,
            "comparison_table": lambda: table, "expected": frame, "scores": scores,
            "COMPARISON_COLORS": COMPARISON_COLORS, "COMPARISON_LINES": COMPARISON_LINES,
            "METRIC_LABELS": METRIC_LABELS, "COMPARISON_LABELS": labels,
            "finish_figure": capture, "evaluation": {"output_directory": "dummy"},
            "ROOT": ".", "project_path": lambda *args: __import__("pathlib").Path("dummy")}
        sections = combined_extension_specs()[0][3]
        def saved_table(path, *args, **kwargs):
            return subgroups if str(path).endswith("subgroups.csv") else metrics
        try:
            with patch.object(pd, "read_csv", side_effect=saved_table):
                for index in (1, 2, 6):
                    exec(compile(sections[index][2], "combined-extension", "exec"), namespace)
            bars = figures["core_mean_fold_ap_auc"]
            for ax in bars.axes:
                self.assertEqual([to_hex(bar.get_facecolor()) for bar in ax.patches], [COMPARISON_COLORS[name].lower() for name in names])
                self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()], [labels[name] for name in names])
            for filename in ("core_roc_precision_recall", "core_fold_variation", "known_unknown_site_core"):
                for ax in figures[filename].axes:
                    model_lines = [line for line in ax.lines if any(line.get_label().startswith(labels[name]) for name in names)]
                    self.assertEqual(len(model_lines), 3)
                    for line in model_lines:
                        name = next(name for name in names if line.get_label().startswith(labels[name]))
                        self.assertEqual(to_hex(line.get_color()), COMPARISON_COLORS[name].lower())
                        self.assertEqual(line.get_linestyle(), COMPARISON_LINES[name])
        finally:
            for figure in figures.values():
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
