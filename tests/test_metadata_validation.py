"""Regression tests for invalid labels and patient-safe metadata contracts."""

import ast
import json
import unittest
from decimal import Decimal
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from melanoma_detection.metadata_validation import (
    validate_binary_target,
    validate_metadata_frame,
)


class BinaryTargetTests(unittest.TestCase):
    def test_accepts_binary_numeric_values_and_preserves_index(self):
        original = pd.Series(["0", "1.0", 0, 1.0], index=[4, 9, 12, 20], name="target")
        result = validate_binary_target(original)
        expected = pd.Series([0, 1, 0, 1], index=original.index, name="target", dtype="int8")
        assert_series_equal(result, expected)
        self.assertEqual(original.dtype, object)

    def test_rejects_invalid_values_before_casting(self):
        bad_values = [1.5, -0.1, 2, -1, np.nan, pd.NA, None, np.inf, -np.inf, "bad", "", "NaN", 1j]
        for bad in bad_values:
            with self.subTest(value=repr(bad)), self.assertRaises(ValueError):
                validate_binary_target(pd.Series([0, bad], dtype=object))

    def test_rejects_high_precision_values_that_float_conversion_would_round(self):
        for bad in ["1.000000000000000001", "1e-400", Decimal("0.999999999999999999")]:
            with self.subTest(value=repr(bad)), self.assertRaises(ValueError):
                validate_binary_target(pd.Series([0, bad], dtype=object))

    def test_rejects_empty_or_non_series_input(self):
        with self.assertRaises(ValueError):
            validate_binary_target(pd.Series([], dtype=float))
        with self.assertRaises(TypeError):
            validate_binary_target([0, 1])


class MetadataFrameTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "image_name": [f"ISIC_{index}" for index in range(10)],
                "patient_id": [f"patient_{index}" for index in range(10)],
                "target": [0, 1] * 5,
                "fold": np.repeat(np.arange(5), 2),
                "age_approx": [35, "Missing", "Unknown", pd.NA, 65] * 2,
                "sex": ["female", "male"] * 5,
                "anatom_site_general_challenge": ["torso"] * 10,
            }
        )

    def test_returns_validated_copy_without_changing_inputs(self):
        original = self.frame.copy(deep=True)
        result = validate_metadata_frame(self.frame)
        self.assertEqual(str(result["target"].dtype), "int8")
        self.assertIsNot(result, self.frame)
        assert_frame_equal(original, self.frame)

    def test_rejects_patient_in_multiple_folds(self):
        self.frame.loc[2, "patient_id"] = self.frame.loc[0, "patient_id"]
        with self.assertRaisesRegex(ValueError, "more than one fold"):
            validate_metadata_frame(self.frame)

    def test_rejects_fold_without_positive_images(self):
        self.frame.loc[1, "target"] = 0
        with self.assertRaisesRegex(ValueError, "both benign and malignant"):
            validate_metadata_frame(self.frame)

    def test_rejects_missing_or_malformed_fold_labels(self):
        for bad in [np.nan, 0.5, -1, 5, np.inf, "bad"]:
            modified = self.frame.copy()
            modified["fold"] = modified["fold"].astype(object)
            modified.loc[0, "fold"] = bad
            with self.subTest(value=repr(bad)), self.assertRaises(ValueError):
                validate_metadata_frame(modified)

    def test_rejects_missing_required_column(self):
        for column in ["target", "patient_id", "image_name", "fold", "age_approx", "sex"]:
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "columns are missing"):
                validate_metadata_frame(self.frame.drop(columns=column))

    def test_rejects_duplicate_missing_or_unclean_identifiers(self):
        for column, bad in [("image_name", "ISIC_1"), ("patient_id", pd.NA), ("patient_id", " "), ("patient_id", " patient")]:
            modified = self.frame.copy()
            modified.loc[0, column] = bad
            with self.subTest(column=column, value=repr(bad)), self.assertRaises(ValueError):
                validate_metadata_frame(modified)

    def test_rejects_malformed_or_non_finite_age(self):
        for bad in ["thirty", "", np.inf, -1, 121]:
            modified = self.frame.copy()
            modified.loc[0, "age_approx"] = bad
            with self.subTest(value=repr(bad)), self.assertRaisesRegex(ValueError, "age_approx"):
                validate_metadata_frame(modified)

    def test_allows_cleaned_training_data_before_fold_assignment(self):
        result = validate_metadata_frame(self.frame.drop(columns="fold"), require_folds=False)
        self.assertNotIn("fold", result.columns)


class NotebookCleaningRegressionTests(unittest.TestCase):
    """Exercise the actual cleaner definitions without reading or writing data."""

    def test_notebook_csv_loads_preserve_original_target_precision(self):
        root = Path(__file__).resolve().parents[1]
        for prefix in ["00_data_cleaning", "01_data_exploration"]:
            for dataset in ["d1", "d2"]:
                path = root / "notebooks" / f"{prefix}_{dataset}.ipynb"
                notebook = json.loads(path.read_text(encoding="utf-8"))
                reads = [
                    node
                    for cell in notebook["cells"] if cell["cell_type"] == "code"
                    for node in ast.walk(ast.parse("".join(cell["source"])))
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "pd"
                    and node.func.attr == "read_csv"
                ]
                self.assertTrue(reads, f"No CSV input found in {path.name}")
                for read in reads:
                    for source_label in ["1.000000000000000001", "1e-400"]:
                        csv = StringIO(f"image_name,target,age_approx\nISIC_0,{source_label},35\n")
                        input_name = read.args[0].id
                        namespace = {"pd": pd, input_name: csv}
                        frame = eval(compile(ast.Expression(body=read), str(path), "eval"), namespace)
                        with self.subTest(notebook=path.name, input=input_name, label=source_label):
                            self.assertEqual(str(frame["target"].dtype), "string")
                            self.assertEqual(frame.loc[0, "target"], source_label)
                            with self.assertRaises(ValueError):
                                validate_binary_target(frame["target"])

    def test_both_cleaning_notebooks_reject_fractional_source_target(self):
        root = Path(__file__).resolve().parents[1]
        valid = pd.DataFrame(
            {
                "image_name": ["ISIC_0", "ISIC_1"],
                "patient_id": ["patient_0", "patient_1"],
                "sex": ["female", None],
                "age_approx": [35, None],
                "anatom_site_general_challenge": ["torso", None],
                "target": [0, 1],
            }
        )
        for dataset in ["d1", "d2"]:
            path = root / "notebooks" / f"00_data_cleaning_{dataset}.ipynb"
            notebook = json.loads(path.read_text(encoding="utf-8"))
            source = next("".join(cell["source"]) for cell in notebook["cells"] if "def clean_metadata(" in "".join(cell["source"]))
            parsed = ast.parse(source)
            definitions = [
                node for node in parsed.body
                if isinstance(node, ast.FunctionDef)
                or (isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id.startswith("EXPECTED_")
                    for target in node.targets
                ))
            ]
            namespace = {"pd": pd, "UNKNOWN_LABEL": "Unknown", "validate_binary_target": validate_binary_target}
            exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
            with self.subTest(dataset=dataset, case="valid"):
                cleaned = namespace["clean_metadata"](valid)
                self.assertEqual(cleaned["target"].tolist(), [0, 1])
                self.assertEqual(str(cleaned["target"].dtype), "int8")
            for bad in [1.5, np.nan, np.inf, "bad", 2]:
                modified = valid.copy()
                modified["target"] = modified["target"].astype(object)
                modified.loc[1, "target"] = bad
                with self.subTest(dataset=dataset, value=repr(bad)), self.assertRaises(ValueError):
                    namespace["clean_metadata"](modified)


if __name__ == "__main__":
    unittest.main()
