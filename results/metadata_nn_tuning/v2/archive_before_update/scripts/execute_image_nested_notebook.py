"""Execute only the new read-only image evaluation notebook in the project venv.

The kernel command uses .venv explicitly, its working directory is the notebook
folder, and no global kernel registration is changed. This never launches a
training runner or executes any historical notebook.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from jupyter_client import KernelManager
from nbclient import NotebookClient
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Subset/14_image_nested_evaluation.ipynb"


def execute(*, timeout=600):
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout must be a positive integer number of seconds.")
    interpreter = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not interpreter.is_file():
        raise FileNotFoundError(f"Project venv interpreter is missing: {interpreter}")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [str(interpreter), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=timeout,
                            resources={"metadata": {"path": str(NOTEBOOK.parent)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    temporary = NOTEBOOK.with_suffix(".executed.tmp")
    nbformat.write(notebook, temporary)
    os.replace(temporary, NOTEBOOK)
    print(f"Executed {NOTEBOOK}")
    print(f"Kernel interpreter: {interpreter}")
    print(f"Kernel working directory: {NOTEBOOK.parent}")
    return NOTEBOOK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=600, help="Per-cell timeout in seconds (default: 600).")
    args = parser.parse_args()
    execute(timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
