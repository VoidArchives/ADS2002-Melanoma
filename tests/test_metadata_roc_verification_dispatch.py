"""Legacy verification commands dispatch to v2 without touching AP receipts."""
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_script(filename):
    spec = importlib.util.spec_from_file_location("dispatch_test_" + filename[:-3], ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stub_verifier(name, result):
    module = types.ModuleType(name)
    module.verify = Mock(return_value=result)
    return module


class VerificationDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = load_script("verify_metadata_nn_tuning_results.py")
        cls.reports = load_script("verify_metadata_nn_tuning_reports.py")

    def test_result_default_verifies_both_current_conditions_without_ap_verification(self):
        current = stub_verifier("verify_metadata_tuning_v2_results", {"experiment_version": 2, "selection_metric": "roc_auc"})
        with patch.dict(sys.modules, {current.__name__: current}), \
                patch.object(self.results, "verify", side_effect=AssertionError("Historical AP verifier invoked")), \
                patch.object(Path, "write_text", side_effect=AssertionError("Compatibility command wrote a receipt")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.results.main([]), 0)
        self.assertEqual([call.args for call in current.verify.call_args_list], [("original",), ("dedup_v1",)])

    def test_result_condition_routes_only_the_named_v2_condition(self):
        current = stub_verifier("verify_metadata_tuning_v2_results", {"status": "complete"})
        with patch.dict(sys.modules, {current.__name__: current}), contextlib.redirect_stdout(io.StringIO()):
            self.results.main(["--condition", "dedup_v1"])
        current.verify.assert_called_once_with("dedup_v1")

    def test_result_rejects_legacy_custom_roots_receipts_or_run_ids_before_dispatch(self):
        current = stub_verifier("verify_metadata_tuning_v2_results", {})
        for arguments in (["--root", str(ROOT)], ["--receipt", "results/metadata_nn_tuning/v1/receipt.json"],
                          ["--run-id", "nn_grid36_v1"], ["--selection-metric", "average_precision"]):
            with self.subTest(arguments=arguments), patch.dict(sys.modules, {current.__name__: current}), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                self.results.main(arguments)
            self.assertEqual(raised.exception.code, 2)
        current.verify.assert_not_called()

    def test_report_default_and_complete_flag_dispatch_all_current_reports(self):
        for arguments in ([], ["--require-complete"], ["--condition", "both"]):
            current = stub_verifier("verify_metadata_roc_reports", {"status": "complete", "errors": []})
            output = io.StringIO()
            with self.subTest(arguments=arguments), patch.dict(sys.modules, {current.__name__: current}), \
                    patch.object(self.reports, "verify_reports", side_effect=AssertionError("Historical AP reports invoked")), \
                    patch.object(Path, "write_text", side_effect=AssertionError("Compatibility command wrote a receipt")), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(self.reports.main(arguments), 0)
            current.verify.assert_called_once_with()
            self.assertIn("results/metadata_nn_tuning/v2/report_verification/report_acceptance.json", output.getvalue())
            self.assertNotIn("v1/report_verification", output.getvalue())

    def test_report_failure_remains_failure(self):
        current = stub_verifier("verify_metadata_roc_reports", {"status": "failed", "errors": ["incomplete visual checks"]})
        with patch.dict(sys.modules, {current.__name__: current}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.reports.main([]), 1)
        current.verify.assert_called_once_with()

    def test_report_rejects_custom_legacy_inputs_and_partial_condition_before_dispatch(self):
        current = stub_verifier("verify_metadata_roc_reports", {})
        for arguments in (["--root", str(ROOT)],
                          ["--execution-receipt", "results/metadata_nn_tuning/v1/report_verification/execution_receipt.json"],
                          ["--visual-receipt", "results/metadata_nn_tuning/v1/report_verification/visual_inspection_receipt.json"],
                          ["--condition", "original"], ["--condition", "dedup_v1"]):
            with self.subTest(arguments=arguments), patch.dict(sys.modules, {current.__name__: current}), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                self.reports.main(arguments)
            self.assertEqual(raised.exception.code, 2)
        current.verify.assert_not_called()

    def test_every_importable_historical_helper_is_unchanged_from_pre_roc_archive(self):
        archive = ROOT / "results/metadata_nn_tuning/v2/archive_before_update/scripts"
        for filename in ("verify_metadata_nn_tuning_results.py", "verify_metadata_nn_tuning_reports.py"):
            def functions(path):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != "main"}
            with self.subTest(filename=filename):
                self.assertEqual(functions(ROOT / "scripts" / filename), functions(archive / filename))


if __name__ == "__main__":
    unittest.main()
