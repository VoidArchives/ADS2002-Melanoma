"""Execute the 21 affected ROC-first report-only notebooks and preserve their previews."""
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
from melanoma_detection.metadata_tuning_v2 import load_tuning_outputs
from build_metadata_nn_tuning_notebooks import render_html

NOTEBOOKS = [p.relative_to(ROOT).as_posix() for folder, lo, hi in [("Subset", 4, 15), ("Subset_dedup", 4, 12)]
             for p in sorted((ROOT / "notebooks" / folder).glob("[0-9][0-9]_*.ipynb"))
             if lo <= int(p.name[:2]) <= hi]
if len(NOTEBOOKS) != 21:
    raise ValueError("The ROC report collection requires exactly 21 affected notebooks.")
REPORT = ROOT / "results/metadata_nn_tuning/v2/report_verification"


def save_receipt(value):
    temporary = REPORT / "execution_receipt.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, REPORT / "execution_receipt.json")


def verify_historical_outputs():
    from melanoma_detection.metadata_roc_archive import verify_preservation
    return verify_preservation(ROOT)


def kernel_guard():
    """Restrict kernel writes and block indirect fitting or test-data access."""
    return '''import os, sys
from pathlib import Path
_report_workspace = Path(%r).resolve()
sys.path.insert(0, str(_report_workspace / "src"))
_report_allowed_writes = [(_report_workspace / p).resolve() for p in %r]
_report_readonly_archive = (_report_workspace / "results/metadata_nn_tuning/v2/archive_before_update").resolve()
def _report_audit(event, args):
    if event in {"os.remove", "os.rmdir", "os.mkdir", "os.rename"}:
        _targets = args[:2] if event == "os.rename" else args[:1]
        for _target in _targets:
            if isinstance(_target, (str, bytes, os.PathLike)):
                _mutation = Path(os.fsdecode(_target)).resolve()
                if _mutation.is_relative_to(_report_readonly_archive):
                    raise PermissionError("Historical presentation archive is read-only: " + str(_mutation))
                if not any(_mutation.is_relative_to(p) for p in _report_allowed_writes):
                    raise PermissionError("Report filesystem mutations must remain in v2: " + str(_mutation))
        return
    if event != "open" or not args or isinstance(args[0], int) or args[0] is None:
        return
    _path = Path(os.fsdecode(args[0])).resolve()
    _mode = args[1] if len(args) > 1 else None
    _flags = args[2] if len(args) > 2 else 0
    _write = (isinstance(_mode, str) and any(c in _mode for c in "wax+")) or (isinstance(_flags, int) and bool(_flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND)))
    if _path.is_relative_to(_report_workspace / "data") and ("test" in _path.name.lower() or any(part.lower() == "test" for part in _path.parts)):
        raise PermissionError("Reserved-test access is prohibited during report execution.")
    if _write and _path.is_relative_to(_report_readonly_archive):
        raise PermissionError("Historical presentation archive is read-only: " + str(_path))
    if _write and not any(_path.is_relative_to(p) for p in _report_allowed_writes):
        raise PermissionError("Report writes must remain in the new v2 namespaces: " + str(_path))
sys.addaudithook(_report_audit)
def _report_no_fit(*args, **kwargs):
    raise RuntimeError("Model fitting and image inference are prohibited in report execution.")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
for _model in [LogisticRegression, RandomForestClassifier, DummyClassifier, OneHotEncoder, StandardScaler, SplineTransformer, SimpleImputer, XGBClassifier]:
    _model.fit = _report_no_fit
from melanoma_detection import dedup_metadata as _nn_fit_module
_nn_fit_module.fit_network = _report_no_fit
from melanoma_detection import image_uncertainty as _image_inference_module
for _entrypoint in ["run_smoke", "run_inference", "load_selected_checkpoint", "prepare_tensor_cache", "predict_cached"]:
    setattr(_image_inference_module, _entrypoint, _report_no_fit)
''' % (str(ROOT), ["results/metadata_nn_tuning/v2", "results/dedup_v1/metadata_nn_tuning/v2"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="append", choices=NOTEBOOKS)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    paths = args.notebook or NOTEBOOKS
    conditions = {"dedup_v1" if "Subset_dedup/" in path else "original" for path in paths}
    for condition in sorted(conditions):
        if load_tuning_outputs(ROOT, condition, experiment_version=2, run_id="roc_auc_grid36_v1", expected_selection_metric="roc_auc")["status"] != "complete":
            raise RuntimeError("Complete five-fold receipts are required before final report execution.")
    REPORT.mkdir(parents=True, exist_ok=True)
    interpreter = ROOT / ".venv/Scripts/python.exe"
    environment = os.environ.copy()
    environment.update({"IPYTHONDIR": str(REPORT / ".ipython"),
        "MPLCONFIGDIR": str(REPORT / ".matplotlib"),
        "JUPYTER_RUNTIME_DIR": str(REPORT / ".jupyter_runtime"),
        "PYTHONDONTWRITEBYTECODE": "1"})
    temporary_directory = REPORT / ".temporary"
    temporary_directory.mkdir(parents=True, exist_ok=True)
    environment.update({"TMP": str(temporary_directory), "TEMP": str(temporary_directory)})
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
                                "evaluate_outer_fold", "evaluate_metadata_holdout", "run_smoke", "run_inference", "run_suite", "run_baselines", "run_training", "load_test", "evaluate_test", "train_network", "fit_network"}:
                        raise ValueError(f"Training/test call in report {relative}: {name}")
        manager = KernelManager(kernel_name="python3", connection_file=str(REPORT / "kernel.json"))
        manager.kernel_spec.argv = [str(interpreter), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(notebook, km=manager, timeout=args.timeout,
                                resources={"metadata": {"path": str(path.parent)}})
        print(f"Executing {relative}", flush=True)
        notebook.cells.insert(0, nbformat.v4.new_code_cell(kernel_guard()))
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
        notebook.cells.pop(0)
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
