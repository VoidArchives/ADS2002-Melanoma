"""Independent synthetic metric and presentation checks for NN tuning reports."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melanoma_detection import metadata_nn_tuning_reporting as reporting
from melanoma_detection import metadata_nn_tuning as tuning
from melanoma_detection import development_dataset

FIXED = ["p_spline", "p_random_forest", "p_xgboost", "p_nn_mean"]
TUNED = ["p_spline_tuned", "p_random_forest_tuned", "p_xgboost_tuned", "p_nn_tuned"]


def prediction_fixture():
    """Fold-dependent score scales make pooled and mean-fold AP differ."""
    rows = []
    targets = [1, 0, 0, 1, 0, 0, 0, 0]
    for fold in range(5):
        for image, target in enumerate(targets):
            row = {"image_name": f"image_{fold}_{image}", "patient_id": f"patient_{fold}_{image // 2}",
                   "fold": fold, "target": target}
            ranking = (8 - image) / 10
            for family, column in enumerate(FIXED):
                row[column] = .02 + fold * .15 + .10 * ranking + .005 * family
            for family, column in enumerate(TUNED):
                row[column] = .02 + fold * .15 + .10 * (1 - ranking) + .005 * family
            for seed in [42, 142, 242]:
                offset = {42: -.002, 142: 0., 242: .002}[seed]
                row[f"p_nn_seed_{seed}"] = row["p_nn_mean"] + offset
                row[f"p_nn_tuned_seed_{seed}"] = row["p_nn_tuned"] + offset
            rows.append(row)
    return pd.DataFrame(rows)


def mean_fold_metric(frame, column, metric):
    return np.mean([metric(rows.target, rows[column]) for _, rows in frame.groupby("fold")])


def completed_report_fixture(root):
    """Write artificial predictions and their receipt to a new run only."""
    frame = prediction_fixture()
    directory = root / "results/metadata_nn_tuning/v1/original/synthetic_report"
    directory.mkdir(parents=True)
    files = {}
    for name in ("oof", "candidates", "selected_parameters", "epochs", "histories", "memberships", "fit_log"):
        table = frame if name == "oof" else pd.DataFrame({"fixture": ["synthetic"]})
        source = directory / f"{name}.csv"
        table.to_csv(source, index=False)
        files[source.relative_to(root).as_posix()] = reporting._hash(source)
    identity = {"dataset_manifest": None, "canonical_source_sha256": "synthetic-development"}
    receipt = {"status": "complete", "condition": "original", "run_id": directory.name,
        "output_root": directory.relative_to(root).as_posix(), "dataset_identity": identity,
        "test_set_loaded": False, "final_model_fitted": False, "rows": len(frame),
        "patients": int(frame.patient_id.nunique()), "positive_images": int(frame.target.sum()),
        "output_sha256": files}
    path = directory / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return frame, identity, directory, receipt


class MetadataNNTuningReportingTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_metric_table_recomputes_known_mean_fold_and_pooled_values(self):
        frame = prediction_fixture()
        # No persisted metric estimate can override the supplied predictions.
        frame.attrs["average_precision"] = -123
        metrics = reporting.metric_table(frame, columns=["p_spline"])
        expected_ap = mean_fold_metric(frame, "p_spline", average_precision_score)
        expected_auc = mean_fold_metric(frame, "p_spline", roc_auc_score)
        primary = metrics.loc[metrics.scope.eq("mean_fold")]
        pooled = metrics.loc[metrics.scope.eq("pooled")]
        self.assertEqual(len(primary), 1)
        self.assertEqual(len(pooled), 1)
        self.assertAlmostEqual(primary.average_precision.iloc[0], expected_ap)
        self.assertAlmostEqual(primary.roc_auc.iloc[0], expected_auc)
        self.assertAlmostEqual(pooled.average_precision.iloc[0], average_precision_score(frame.target, frame.p_spline))
        self.assertNotAlmostEqual(expected_ap, pooled.average_precision.iloc[0])

    def test_nn_plot_preserves_original_two_panels_colours_and_actual_metric_values(self):
        frame = prediction_fixture()
        figure = reporting.plot_nn_comparison(frame, scope="mean_fold")
        self.assertEqual(len(figure.axes), 2)
        np.testing.assert_allclose(figure.get_size_inches(), [11, 4.3])
        columns = ["p_spline", "p_nn_seed_42", "p_nn_seed_142", "p_nn_seed_242", "p_nn_mean"]
        colours = ["#2563EB", "#D97706", "#D97706", "#D97706", "#303030"]
        for axis, metric in zip(figure.axes, [average_precision_score, roc_auc_score]):
            self.assertGreaterEqual(len(axis.patches), 5)
            for patch, column, colour in zip(axis.patches, columns, colours):
                self.assertAlmostEqual(patch.get_height(), mean_fold_metric(frame, column, metric))
                np.testing.assert_allclose(patch.get_facecolor(), to_rgba(colour))

    def test_tuning_bars_include_all_four_families_and_pair_matching_predictions(self):
        frame = prediction_fixture()
        figure = reporting.plot_tuning_bars(frame, metric="average_precision", scope="mean_fold")
        self.assertEqual(len(figure.axes), 1)
        axis = figure.axes[0]
        self.assertEqual(len(axis.patches), 8)
        self.assertEqual(len(axis.get_xticklabels()), 4)
        for patch, column in zip(axis.patches[:4], FIXED):
            self.assertAlmostEqual(patch.get_height(), mean_fold_metric(frame, column, average_precision_score))
            np.testing.assert_allclose(patch.get_facecolor(), to_rgba("#2563EB"))
        for patch, column in zip(axis.patches[4:], TUNED):
            self.assertAlmostEqual(patch.get_height(), mean_fold_metric(frame, column, average_precision_score))
            np.testing.assert_allclose(patch.get_facecolor(), to_rgba("#C58A00"))

    def test_curves_use_four_rows_two_panels_original_fixed_and_tuned_styles(self):
        figure = reporting.plot_tuning_curves(prediction_fixture())
        self.assertEqual(len(figure.axes), 8)
        np.testing.assert_allclose(figure.get_size_inches(), [14, 17.333], atol=.001)
        for row in range(4):
            for column in range(2):
                axis = figure.axes[row * 2 + column]
                model_lines = [line for line in axis.lines if line.get_color() in {"#2563EB", "#C58A00"}]
                self.assertEqual(len(model_lines), 2)
                self.assertEqual(model_lines[0].get_color(), "#2563EB")
                self.assertEqual(model_lines[0].get_linestyle(), "--")
                self.assertEqual(model_lines[1].get_color(), "#C58A00")
                self.assertEqual(model_lines[1].get_linestyle(), "-")
                if column == 1:
                    self.assertAlmostEqual(axis.get_ylim()[1], .30)

    def test_candidate_grid_displays_every_recipe_and_actual_inner_ap_on_one_common_scale(self):
        rows = []
        for outer in range(5):
            for recipe in tuning.recipes():
                rows.append({**recipe, "outer_fold": outer, "selected": recipe["recipe_order"] == outer,
                    "mean_inner_average_precision": .05 + .01 * outer + .002 * recipe["recipe_order"]})
        candidates = pd.DataFrame(rows).sample(frac=1, random_state=42)
        figure = reporting.plot_candidate_grid(candidates)
        panels = [axis for axis in figure.axes if axis.images]
        self.assertEqual(len(panels), 15)
        self.assertEqual(len(figure.axes), 16)  # Fifteen grids plus one shared colour bar.
        self.assertEqual(sum(len(axis.texts) for axis in panels), 180)
        self.assertEqual(sum(len(axis.patches) for axis in panels), 5)
        for index, axis in enumerate(panels):
            outer, dropout_index = divmod(index, 3)
            expected = candidates.loc[(candidates.outer_fold == outer) &
                candidates.dropout.eq([0., .2, .4][dropout_index])].pivot(
                    index="hidden_units", columns="learning_rate", values="mean_inner_average_precision")
            np.testing.assert_allclose(axis.images[0].get_array(), expected.to_numpy())
            self.assertEqual(axis.images[0].get_clim(),
                (candidates.mean_inner_average_precision.min(), candidates.mean_inner_average_precision.max()))
            for border in axis.patches:
                np.testing.assert_allclose(border.get_edgecolor(), to_rgba("#C58A00"))

    def test_missing_run_is_pending_and_returns_no_invented_predictions(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            result = reporting.load_tuning(Path(directory), "original", run_id="missing")
        self.assertFalse(result["ready"])
        self.assertTrue(result.get("oof") is None or result["oof"].empty)

    def test_completed_report_checks_receipt_and_development_identity_without_any_fit(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            frame, identity, output, receipt = completed_report_fixture(root)
            with patch.object(development_dataset, "load_development_dataset", return_value=(frame, identity)) as load, \
                 patch.object(tuning, "fit_nn_pair", side_effect=AssertionError("report started a fit")):
                report = reporting.load_tuning(root, "original", run_id=output.name)
                self.assertTrue(report["ready"])
                load.assert_called_once_with(root, None)
                np.testing.assert_allclose(report["oof"]["p_nn_tuned"], frame["p_nn_tuned"])
                changed = report["oof"].copy()
                changed.loc[changed.index[0], "image_name"] = changed.image_name.iloc[1]
                changed.to_csv(output / "oof.csv", index=False)
                key = (output / "oof.csv").relative_to(root).as_posix()
                receipt["output_sha256"][key] = reporting._hash(output / "oof.csv")
                (output / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unique"):
                    reporting.load_tuning(root, "original", run_id=output.name)

    def test_report_rejects_test_or_final_fit_scope_before_development_data_is_loaded(self):
        for field in ["test_set_loaded", "final_model_fitted"]:
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                root = Path(directory)
                _, _, output, receipt = completed_report_fixture(root)
                receipt[field] = True
                (output / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
                with patch.object(development_dataset, "load_development_dataset",
                                  side_effect=AssertionError("unsafe receipt reached development data")):
                    with self.assertRaisesRegex(ValueError, "development-only"):
                        reporting.load_tuning(root, "original", run_id=output.name)

    def test_report_inputs_stay_within_the_requested_new_run_namespace(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            for run_id in ["../metadata_d1", "nested/run", "", "a" * 65]:
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    reporting.load_tuning(root, "original", run_id=run_id)
            _, _, output, receipt = completed_report_fixture(root)
            outside = root / "results/metadata_d1/historical_predictions.csv"
            outside.parent.mkdir(parents=True)
            outside.write_text("historical experiment remains separate\n", encoding="utf-8")
            receipt["output_sha256"][outside.relative_to(root).as_posix()] = reporting._hash(outside)
            (output / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"), \
                 patch.object(development_dataset, "load_development_dataset",
                              side_effect=AssertionError("out-of-run receipt reached development data")):
                reporting.load_tuning(root, "original", run_id=output.name)

    def test_pending_notebooks_execute_without_training_or_csv_data_access(self):
        with patch.object(tuning, "fit_nn_pair", side_effect=AssertionError("notebook started training")), \
             patch.object(tuning, "run_tuning", side_effect=AssertionError("notebook started the runner")), \
             patch.object(pd, "read_csv", side_effect=AssertionError("pending notebook opened data")), \
             patch("IPython.display.display"):
            for condition in ["original", "dedup_v1"]:
                for stage in [6, 7]:
                    with self.subTest(condition=condition, stage=stage):
                        notebook = reporting.build_notebook(ROOT, condition, stage, run_id="missing_test_run")
                        namespace = {"__name__": "synthetic_pending_notebook"}
                        for cell in notebook.cells:
                            if cell.cell_type == "code":
                                exec(compile(cell.source, "pending_notebook", "exec"), namespace)
                        self.assertFalse(namespace["report"]["ready"])

    def test_notebooks_keep_original_teaching_depth_and_execute_reports_only(self):
        archive = ROOT / "results/metadata_nn_tuning/v1/archive_before_update/notebooks/Subset"
        for condition in ["original", "dedup_v1"]:
            for stage, filename in [(6, "06_metadata_neural_network.ipynb"), (7, "07_metadata_tuning.ipynb")]:
                original = nbformat.read(archive / filename, as_version=4)
                notebook = reporting.build_notebook(ROOT, condition, stage, run_id="missing_test_run")
                nbformat.validate(notebook)
                original_words = sum(len(cell.source.split()) for cell in original.cells if cell.cell_type == "markdown")
                words = sum(len(cell.source.split()) for cell in notebook.cells if cell.cell_type == "markdown")
                self.assertGreaterEqual(words, original_words)
                expected_headings = []
                for cell in original.cells:
                    if cell.cell_type == "markdown":
                        for line in cell.source.splitlines():
                            if line.startswith("#"):
                                heading = line.replace("Subset data", "original Subset data" if condition == "original"
                                                       else "deduplicated Subset data")
                                if stage == 6:
                                    heading = heading.replace("before training", "in the saved experiment")
                                    heading = heading.replace("Train the prescribed runs", "Review the prescribed runs")
                                    heading = heading.replace("Save predictions and run settings", "Read predictions and run settings")
                                expected_headings.append(heading)
                actual_headings = [line for cell in notebook.cells if cell.cell_type == "markdown"
                                   for line in cell.source.splitlines() if line.startswith("#")]
                position = 0
                for heading in expected_headings:
                    position = actual_headings.index(heading, position) + 1
                self.assertFalse(notebook.metadata.metadata_nn_tuning_report.training_on_execution)
                if stage == 7:
                    self.assertIn("**Experiment-scope addition:**", "\n".join(
                        cell.source for cell in notebook.cells if cell.cell_type == "markdown"))
                for cell in notebook.cells:
                    if cell.cell_type != "code":
                        continue
                    tree = ast.parse(cell.source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                            self.assertNotIn(called, {"fit", "fit_transform", "run_tuning", "run_training", "train", "fit_final_bundle"})
                    self.assertNotIn("test_labels.csv", cell.source)
                    self.assertNotIn("test_clean.csv", cell.source)


if __name__ == "__main__":
    unittest.main()
