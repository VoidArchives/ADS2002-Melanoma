"""Queue state and exclusive ownership protect active image runs."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from continue_clean_workflow import controller_lock, terminal_image_state, reports


class CleanControllerTests(unittest.TestCase):
    def test_intermediate_stage_completion_does_not_wake_queue(self):
        self.assertFalse(terminal_image_state({"phase": "raw_b0", "status": "complete"}))
        self.assertFalse(terminal_image_state({"phase": "bounded_pilots", "status": "running"}))
        self.assertTrue(terminal_image_state({"phase": "workflow", "status": "complete"}))
        self.assertTrue(terminal_image_state({"phase": "raw_b0", "status": "incomplete"}))

    def test_lock_blocks_second_controller_then_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            with controller_lock(directory):
                with self.assertRaises((RuntimeError, BlockingIOError)):
                    with controller_lock(directory):
                        self.fail("Second controller entered the queue")
            with controller_lock(directory):
                pass

    def test_report_refresh_loads_current_notebook_code_without_training(self):
        # The controller may wait for hours. Notebook edits made during that
        # wait must be loaded by a fresh process rather than stale imports.
        with patch("continue_clean_workflow.build_evaluation",return_value={"status":"partial","output_directory":"saved"}), \
             patch("continue_clean_workflow.subprocess.run") as run, \
             patch("continue_clean_workflow.update"):
            reports()
        args=run.call_args.args[0]
        self.assertIn("scripts/build_dedup_workflow_notebooks.py",args)
        self.assertIn("--execute",args)
        self.assertFalse(any("run_image" in argument or "run_metadata" in argument for argument in args))


if __name__ == "__main__":
    unittest.main()
