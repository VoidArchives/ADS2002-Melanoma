"""Report notebooks cannot accidentally fit models or regenerate patient folds."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.dedup_notebooks import build_notebooks
from melanoma_detection.dedup_notebook_data import data_specs
from melanoma_detection.dedup_notebook_templates import original_cells, template_sections


class DedupNotebookTests(unittest.TestCase):
    def test_all_twelve_reports_are_valid_explicit_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = build_notebooks(directory)
            self.assertEqual(len(paths), 12)
            self.assertEqual([path.name[:2] for path in paths], [f"{k:02d}" for k in range(1, 13)])
            for path in paths:
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
                if path.name.startswith(("06", "07")):
                    self.assertTrue(any("load_tuning(PROJECT_ROOT, 'dedup_v1'" in source for source in sources))
                else:
                    self.assertTrue(any("dataset_manifest=MANIFEST" in source for source in sources))
                for source in sources:
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                            self.assertNotIn(name, {"fit", "fit_transform", "train", "run_training", "run_metadata", "run_suite", "split", "StratifiedGroupKFold"}, path.name)
                    self.assertNotIn("test_labels.csv", source)
                    self.assertNotIn("test_clean.csv", source)
                self.assertFalse(notebook.metadata.deduplicated_workflow.training_on_execute)
                self.assertEqual(notebook.metadata.deduplicated_workflow.depth_revision, 3)
                self.assertTrue(notebook.metadata.deduplicated_workflow.original_method_sources)
                presentation = notebook.metadata.deduplicated_workflow
                self.assertEqual(presentation.presentation_template, "original_markdown_and_cell_order")
                self.assertGreaterEqual(presentation.adapted_markdown_words, .9 * presentation.original_markdown_words)
                original_markdown_cells = sum(
                    cell.cell_type == "markdown"
                    for source in presentation.original_method_sources for cell in original_cells(source))
                self.assertGreaterEqual(sum(cell.cell_type == "markdown" for cell in notebook.cells), original_markdown_cells)

    def test_template_cannot_copy_a_historical_training_cell_implicitly(self):
        with self.assertRaisesRegex(ValueError, "No report-code replacement"):
            template_sections("04_dummy_logistic_baseline.ipynb", {})

    def test_original_typography_and_resolution_are_restored(self):
        from melanoma_detection.dedup_notebook_templates import original_export_dpi, original_plot_style
        from melanoma_detection.dedup_notebooks import notebook_specs, ORIGINAL_SOURCES
        for filename, _, _, sections in notebook_specs():
            code = "\n".join(source for kind, _, source in sections if kind == "__code__")
            for source in ORIGINAL_SOURCES[filename[:2]]:
                style = original_plot_style(source)
                if style:
                    self.assertIn(style, code, filename)
                self.assertIn(f"ORIGINAL_EXPORT_DPI = {original_export_dpi(source)}", code, filename)

    def test_original_eda_units_and_rates_are_preserved(self):
        # A patient positive in one site must not become positive in a different
        # site's table. The notebook retains the original within-category rule.
        import numpy as np
        import pandas as pd
        specs = {filename:sections for filename,_,_,sections in data_specs()}
        code = next(code for kind,_,code in specs["02_clean_data_eda.ipynb"]
            if kind == "__code__" and "def summarise_feature" in code)
        code = code.partition("features =")[0]
        namespace = {"pd":pd,"np":np,"display":lambda *_:None,"Markdown":str,
            "data":pd.DataFrame({"patient_id":["p1"],"sex":["female"],"age_approx":[45],"anatom_site_general_challenge":["torso"]})}
        exec(code, namespace)
        records = pd.DataFrame({"patient_id":["p1","p1","p2"],"site":["a","b","b"],"target":[1,0,0]})
        table = namespace["summarise_feature"](records,"site")
        self.assertEqual(table.loc["a","malignant_patients"],1)
        self.assertEqual(table.loc["b","malignant_patients"],0)
        self.assertEqual(table.loc["b","patient_count"],2)
        self.assertAlmostEqual(table.loc["b","patient_membership_share"],2/3)
        self.assertAlmostEqual(table.patient_membership_share.sum(),1)

    def test_each_stage_has_declared_method_and_figure_correspondence(self):
        with tempfile.TemporaryDirectory() as directory:
            build_notebooks(directory)
            coverage=(Path(directory)/"notebooks/Subset_dedup/METHOD_COVERAGE.md").read_text(encoding="utf-8")
            for method in ["class balance","ROC/PR","seven feature sets","three-seed stability",
                "parameter frequencies","reliability","12/24","convergence","equal-patient"]:
                self.assertIn(method,coverage)
            self.assertIn("stronger augmentation and CLAHE remain deferred",coverage)


if __name__ == "__main__":
    unittest.main()
