"""Active compatibility entrypoints cannot restore AP reports or old writes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat

from melanoma_detection import dedup_notebooks

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    path=ROOT/"scripts"/name
    spec=importlib.util.spec_from_file_location(name[:-3],path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class ROCEntrypointTests(unittest.TestCase):
    def test_clean_generation_routes_all_nine_reports_and_preserves_preparation_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            preparation_dir=root/dedup_notebooks.NOTEBOOK_DIRECTORY
            preparation_dir.mkdir(parents=True)
            for name in dedup_notebooks.PREPARATION_NOTEBOOKS:
                (preparation_dir/name).write_bytes((ROOT/dedup_notebooks.NOTEBOOK_DIRECTORY/name).read_bytes())
            # Construction reads only notebook archives, not modelling tables.
            with patch("pandas.read_csv",side_effect=AssertionError("Builder read modelling data")), patch("melanoma_detection.dedup_notebook_data.data_specs",side_effect=AssertionError("Builder rebuilt preparation")):
                paths=dedup_notebooks.build_notebooks(root)
            self.assertEqual(len(paths),12)
            for path in paths[3:]:
                book=nbformat.read(path,as_version=4)
                self.assertEqual(book.metadata.deduplicated_workflow.experiment_version,2)
                self.assertEqual(book.metadata.deduplicated_workflow.primary_metric,"mean_fold_roc_auc")
                self.assertIn("v2/archive_before_update",book.metadata.deduplicated_workflow.presentation_template_source_path)
            preparation={p.name:p.read_bytes() for p in paths[:3]}
            with patch("pandas.read_csv",side_effect=AssertionError("Builder read modelling data")), patch("melanoma_detection.dedup_notebook_data.data_specs",side_effect=AssertionError("Builder rebuilt preparation")):
                dedup_notebooks.build_notebooks(root)
            self.assertEqual(preparation,{p.name:p.read_bytes() for p in paths[:3]})
            coverage=(root/"notebooks/Subset_dedup/METHOD_COVERAGE.md").read_text(encoding="utf-8")
            self.assertIn("inner ROC-AUC selection",coverage)
            self.assertIn("Historical AP-era diagnostics",coverage)
            self.assertNotIn("inner AP selection separate",coverage)

    def test_legacy_nn_executor_defaults_to_the_eight_v2_targets(self):
        dispatcher=module("execute_metadata_nn_tuning_reports.py")
        forwarded=dispatcher.forwarded_arguments(["--timeout","900"])
        self.assertEqual(forwarded.count("--notebook"),8)
        for path in dispatcher.NOTEBOOKS:self.assertIn(path,forwarded)
        custom=["--notebook",dispatcher.NOTEBOOKS[0],"--timeout","600"]
        self.assertEqual(dispatcher.forwarded_arguments(custom),custom)
        self.assertEqual(dispatcher.forwarded_arguments(["--help"]),["--help"])

    def test_clean_executor_dispatches_only_to_new_guarded_executor(self):
        with patch.object(dedup_notebooks.subprocess,"run") as runner,patch.object(dedup_notebooks,"read_json",return_value={"status":"complete"}):
            dedup_notebooks.execute_notebooks(ROOT,notebook_names=["07_metadata_tuning.ipynb"],timeout=900)
        command=runner.call_args.args[0]
        self.assertIn(str(ROOT/"scripts/execute_metadata_roc_reports.py"),command)
        self.assertIn("notebooks/Subset_dedup/07_metadata_tuning.ipynb",command)
        self.assertNotIn("execute_metadata_nn_tuning_reports.py",command)
        with self.assertRaisesRegex(ValueError,"preparation"):
            dedup_notebooks.execute_notebooks(ROOT,notebook_names=["01_data_cleaning.ipynb"])

    def test_legacy_renderer_api_remains_available(self):
        legacy=module("build_metadata_nn_tuning_notebooks.py")
        self.assertTrue(callable(legacy.render_html))
        self.assertEqual(legacy.DEFAULT_RUN_ID,"roc_auc_grid36_v1")
        self.assertEqual(legacy.build_notebook.__module__,"melanoma_detection.metadata_tuning_v2_reporting")


if __name__=="__main__":
    unittest.main()
