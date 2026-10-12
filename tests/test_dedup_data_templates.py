"""The retained-data reports preserve actual original presentation cells."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melanoma_detection.dedup_notebook_data import data_specs
from melanoma_detection.dedup_notebook_templates import original_cells


class DeduplicatedDataTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = {filename: cells for filename, _, _, cells in data_specs()}

    def test_all_original_data_explanations_are_retained_or_expanded(self):
        correspondence = {
            "01_data_cleaning.ipynb": "01_data_cleaning.ipynb",
            "02_clean_data_eda.ipynb": "02_data_exploration.ipynb",
            "03_patient_fold_verification.ipynb": "03_patient_splitting.ipynb",
        }
        for clean_name, original_name in correspondence.items():
            original = [cell.source for cell in original_cells(original_name)
                        if cell.cell_type == "markdown"]
            clean = [source for kind, source, _ in self.specs[clean_name]
                     if kind == "__markdown__"]
            self.assertGreaterEqual(sum(len(text.split()) for text in clean),
                                    sum(len(text.split()) for text in original))
            self.assertGreaterEqual(len(clean), len(original))
        eda = "\n".join(source for kind, source, _ in self.specs["02_clean_data_eda.ipynb"]
                        if kind == "__markdown__")
        for detail in ["Left panel: category share", "Right panel: malignant percentage",
                       "patient-category memberships", "1.89%", "20.85%"]:
            self.assertIn(detail, eda)

    def test_original_eda_plot_bodies_are_literal_templates(self):
        cells = self.specs["02_clean_data_eda.ipynb"]
        code = [source for kind, _, source in cells if kind == "__code__"]
        original = original_cells("02_data_exploration.ipynb")
        for number, filename in [(13, "class_balance_images_patients"),
                                 (15, "images_per_patient")]:
            expected = original[number - 1].source.replace(
                "plt.show()", f'finish_figure(fig, "{filename}")')
            self.assertIn(expected, code)
        expected = original[18].source.replace("    plt.show()",
            '    finish_figure(fig, {"sex": "sex_image_patient_share_rates",\n'
            '        "age_approx": "age_image_patient_share_rates",\n'
            '        "anatom_site_general_challenge": "site_image_patient_share_rates"}[feature])')
        self.assertIn(expected, code)

    def test_original_fold_composition_and_scope_adaptations_are_explicit(self):
        cells = self.specs["03_patient_fold_verification.ipynb"]
        source = "\n".join(code for kind, _, code in cells if kind == "__code__")
        prose = "\n".join(text for kind, text, _ in cells if kind == "__markdown__")
        for style in ['figsize=(9, 5.4)', 'color="#2563EB"',
                      'edgecolor="#1E3A8A"', 'axis.bar_label(', 'dpi=320']:
            self.assertIn(style, source)
        self.assertNotIn("assign_patient_folds(", source)
        self.assertNotIn("to_csv(", source)
        self.assertNotIn("test_clean.csv", source)
        for detail in ["Scope adaptation", "deferred", "24–26", "1.79%–2.00%"]:
            self.assertIn(detail, prose)


if __name__ == "__main__":
    unittest.main()
