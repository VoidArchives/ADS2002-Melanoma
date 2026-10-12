"""Execute the new report only; never invoke a trainer or fit any model.

Uses the existing project interpreter without registering a global kernel.
Optionally waits for the implementation controller's terminal workflow status.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from jupyter_client import KernelManager
from nbclient import NotebookClient
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Subset/15_deduplicated_workflow.ipynb"
REPORT_DIRECTORY = ROOT / "results/dedup_v1/reporting"
WORKFLOW_STATUS = ROOT / "results/dedup_v1/implementation/workflow_status.json"


def wait_for_workflow(timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    previous = None
    terminal_failures = {"incomplete", "failed", "stopped", "cancelled"}
    while time.monotonic() < deadline:
        if WORKFLOW_STATUS.is_file():
            try:
                state = json.loads(WORKFLOW_STATUS.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(10)
                continue
            marker = (state.get("status"), state.get("phase"))
            if marker != previous:
                print(f"Report waiting: workflow status={marker[0]}, phase={marker[1]}", flush=True)
                previous = marker
            if state.get("status") in terminal_failures or (
                state.get("status") == "complete" and state.get("phase") == "workflow"
            ):
                return state
        time.sleep(10)
    raise TimeoutError("Report wait limit reached; no training was changed.")


def execute(*, timeout=600, export_html=True):
    if timeout < 1:
        raise ValueError("Per-cell timeout must be positive.")
    interpreter = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not interpreter.is_file():
        raise FileNotFoundError(f"Project interpreter is missing: {interpreter}")
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name="python3", connection_file=str(REPORT_DIRECTORY / "report_kernel.json"))
    manager.kernel_spec.argv = [str(interpreter), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=timeout,
                            resources={"metadata": {"path": str(NOTEBOOK.parent)}})
    environment = os.environ.copy()
    environment.update({"IPYTHONDIR": str(REPORT_DIRECTORY / ".ipython"),
                        "MPLCONFIGDIR": str(REPORT_DIRECTORY / ".matplotlib"),
                        "JUPYTER_RUNTIME_DIR": str(REPORT_DIRECTORY / ".jupyter_runtime"),
                        "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        client.execute(env=environment)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_connection_file()
    nbformat.validate(notebook)
    temporary = NOTEBOOK.with_suffix(".executed.tmp")
    nbformat.write(notebook, temporary)
    os.replace(temporary, NOTEBOOK)
    html_path = None
    if export_html:
        try:
            from nbconvert import HTMLExporter
        except ImportError:
            print("Notebook executed; HTML export unavailable because nbconvert is missing.", flush=True)
        else:
            exporter = HTMLExporter(template_name="lab")
            exporter.exclude_input = True
            body, _ = exporter.from_notebook_node(notebook)
            html_path = REPORT_DIRECTORY / "15_deduplicated_workflow.html"
            html_path.write_text(body, encoding="utf-8")
    errors = [output for cell in notebook.cells if cell.cell_type == "code"
              for output in cell.get("outputs", []) if output.output_type == "error"]
    if errors:
        raise RuntimeError("Executed report retained error outputs.")
    status = {"status": "complete", "notebook": NOTEBOOK.relative_to(ROOT).as_posix(),
              "html": None if html_path is None else html_path.relative_to(ROOT).as_posix(),
              "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
              "error_outputs": 0, "kernel_interpreter": str(interpreter),
              "training_started": False, "reserved_test_loaded": False}
    (REPORT_DIRECTORY / "execution_receipt.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=600, help="Per-cell execution timeout.")
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--wait-workflow", action="store_true")
    parser.add_argument("--wait-timeout", type=int, default=43200, help="Maximum workflow wait in seconds.")
    args = parser.parse_args()
    if args.wait_workflow:
        if args.wait_timeout < 1:
            raise ValueError("Workflow wait timeout must be positive.")
        wait_for_workflow(args.wait_timeout)
    execute(timeout=args.timeout, export_html=not args.no_html)


if __name__ == "__main__":
    main()
