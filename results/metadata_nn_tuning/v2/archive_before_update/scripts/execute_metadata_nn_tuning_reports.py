"""Execute the eight affected report-only notebooks and preserve their previews."""
from __future__ import annotations

import argparse
import ast
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.metadata_model import file_sha256
from melanoma_detection.metadata_nn_tuning import load_tuning_outputs
from build_metadata_nn_tuning_notebooks import render_html

NOTEBOOKS = [
    "notebooks/Subset/06_metadata_neural_network.ipynb",
    "notebooks/Subset/07_metadata_tuning.ipynb",
    "notebooks/Subset/08_metadata_robustness.ipynb",
    "notebooks/Subset/14_image_nested_evaluation.ipynb",
    "notebooks/Subset_dedup/06_metadata_neural_network.ipynb",
    "notebooks/Subset_dedup/07_metadata_tuning.ipynb",
    "notebooks/Subset_dedup/08_metadata_robustness.ipynb",
    "notebooks/Subset_dedup/12_combined_evaluation.ipynb",
]
REPORT = ROOT / "results/metadata_nn_tuning/v1/report_verification"


def save_receipt(value):
    temporary = REPORT / "execution_receipt.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, REPORT / "execution_receipt.json")


def verify_historical_outputs():
    archive = ROOT / "results/metadata_nn_tuning/v1/archive_before_update"
    receipt = json.loads((archive / "preservation_receipt.json").read_text(encoding="utf-8"))
    for relative, digest in receipt["protected_files"].items():
        if file_sha256(ROOT / relative) != digest:
            raise ValueError(f"A protected historical artifact changed: {relative}")
    for relative, digest in receipt["notebook_snapshots"].items():
        if file_sha256(archive / relative) != digest:
            raise ValueError(f"An archived notebook changed: {relative}")
    return {"historical_files_checked": len(receipt["protected_files"]),
            "archived_notebooks_checked": len(receipt["notebook_snapshots"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="append", choices=NOTEBOOKS)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    paths = args.notebook or NOTEBOOKS
    conditions = {"dedup_v1" if "Subset_dedup/" in path else "original" for path in paths}
    for condition in sorted(conditions):
        if load_tuning_outputs(ROOT, condition)["status"] != "complete":
            raise RuntimeError("Complete five-fold receipts are required before final report execution.")
    REPORT.mkdir(parents=True, exist_ok=True)
    interpreter = ROOT / ".venv/Scripts/python.exe"
    environment = os.environ.copy()
    environment.update({"IPYTHONDIR": str(REPORT / ".ipython"),
        "MPLCONFIGDIR": str(REPORT / ".matplotlib"),
        "JUPYTER_RUNTIME_DIR": str(REPORT / ".jupyter_runtime"),
        "PYTHONDONTWRITEBYTECODE": "1"})
    base = {"training_started": False, "reserved_test_loaded": False,
            "kernel_interpreter": str(interpreter), "historical_preservation": verify_historical_outputs()}
    rows = []
    previous = REPORT / "execution_receipt.json"
    if args.notebook and previous.exists():
        for row in json.loads(previous.read_text(encoding="utf-8")).get("notebooks", []):
            if row["notebook"] not in paths:
                if file_sha256(ROOT / row["notebook"]) != row["sha256"]:
                    raise ValueError("An unselected executed report changed; include it in the refresh.")
                rows.append(row)
    save_receipt({**base, "status": "running", "notebooks": rows})
    for relative in paths:
        path = ROOT / relative
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            # These updated cells are report dispatch and saved-result analysis.
            # Reject accidentally restored fitting calls before kernel execution.
            source = "\n".join(line for line in cell.source.splitlines() if not line.lstrip().startswith("%"))
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call):
                    name = (node.func.attr if isinstance(node.func, ast.Attribute)
                            else node.func.id if isinstance(node.func, ast.Name) else "")
                    if name in {"fit", "fit_transform", "fit_final_bundle", "run_tuning",
                                "evaluate_outer_fold", "evaluate_metadata_holdout"}:
                        raise ValueError(f"Training/test call in report {relative}: {name}")
        manager = KernelManager(kernel_name="python3", connection_file=str(REPORT / "kernel.json"))
        manager.kernel_spec.argv = [str(interpreter), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(notebook, km=manager, timeout=args.timeout,
                                resources={"metadata": {"path": str(path.parent)}})
        print(f"Executing {relative}", flush=True)
        try:
            client.execute(env=environment)
        except Exception as error:
            save_receipt({**base, "status": "failed", "notebooks": rows,
                          "failed_notebook": relative, "error": str(error)})
            raise
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_connection_file()
        nbformat.validate(notebook)
        temporary = path.with_suffix(".executed.tmp")
        nbformat.write(notebook, temporary)
        os.replace(temporary, path)
        directory = REPORT / Path(relative).parent.name
        directory.mkdir(parents=True, exist_ok=True)
        figures = []
        for index, cell in enumerate(notebook.cells):
            for number, output in enumerate(cell.get("outputs", [])):
                if output.get("output_type") == "error":
                    raise RuntimeError(f"Error retained in {relative}")
                png = output.get("data", {}).get("image/png")
                if png:
                    image = directory / f"{path.stem}_cell{index}_{number}.png"
                    image.write_bytes(base64.b64decode(png))
                    figures.append(image.relative_to(ROOT).as_posix())
        preview = directory / path.with_suffix(".html").name
        preview.write_text(render_html(notebook, relative), encoding="utf-8")
        row = {"notebook": relative, "sha256": file_sha256(path), "errors": 0,
               "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
               "figures": figures, "html": preview.relative_to(ROOT).as_posix()}
        rows.append(row)
        save_receipt({**base, "status": "running", "notebooks": rows})
        print(f"Completed {relative}: {row['code_cells']} code cells, {len(figures)} figures", flush=True)
    save_receipt({**base, "status": "complete", "notebooks": rows,
                  "historical_preservation": verify_historical_outputs(),
                  "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "visual_inspection": "pending side-by-side review"})


if __name__ == "__main__":
    main()
