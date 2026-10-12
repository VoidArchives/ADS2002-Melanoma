"""Accept saved reports without executing notebooks or opening modelling data.

This checks the eight edited reports, their original templates, the existing
execution receipt and a separately supplied manual visual-inspection receipt.
Only its own acceptance receipt is written; no experiment or notebook is edited.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

import nbformat

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIRECTORY = Path("results/metadata_nn_tuning/v1/report_verification")
ARCHIVE = Path("results/metadata_nn_tuning/v1/archive_before_update/notebooks/Subset")
NOTEBOOKS = {
    "original": [
        "notebooks/Subset/06_metadata_neural_network.ipynb",
        "notebooks/Subset/07_metadata_tuning.ipynb",
        "notebooks/Subset/08_metadata_robustness.ipynb",
        "notebooks/Subset/14_image_nested_evaluation.ipynb",
    ],
    "dedup_v1": [
        "notebooks/Subset_dedup/06_metadata_neural_network.ipynb",
        "notebooks/Subset_dedup/07_metadata_tuning.ipynb",
        "notebooks/Subset_dedup/08_metadata_robustness.ipynb",
        "notebooks/Subset_dedup/12_combined_evaluation.ipynb",
    ],
}
FORBIDDEN_CALLS = {
    "fit", "fit_transform", "train", "run_training", "run_tuning", "fit_nn_pair",
    "fit_network", "fit_final_bundle", "run_suite", "run_comparator",
    "evaluate_outer_fold", "evaluate_metadata_holdout", "evaluate_test", "load_test",
}
HEADING_ADAPTATIONS = {
    "before training": "in the saved experiment",
    "Train the prescribed runs": "Review the prescribed runs",
    "Save predictions and run settings": "Read predictions and run settings",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prose_without_code(source):
    """Remove Markdown code spans/blocks before inspecting links or headings."""
    lines, fence = [], None
    for line in source.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            delimiter = marker.group(1)
            if fence is None:
                fence = delimiter
            elif delimiter[0] == fence[0] and len(delimiter) >= len(fence):
                fence = None
            continue
        if fence is None:
            lines.append(line)
    prose = "\n".join(lines)
    prose = re.sub(r"<(?:pre|code)\b[^>]*>.*?</(?:pre|code)>", "", prose,
                   flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"(`+)(?!`)(.*?)\1(?!`)", "", prose, flags=re.DOTALL)


def markdown_destinations(source):
    prose = prose_without_code(source)
    # Match bracket links, including images, and reference-link definitions.
    inline = re.finditer(r"!?\[(?:\\.|[^\]\\])*\]\(\s*(<[^>]+>|(?:\\.|[^\s)\\])+)"
                         r"(?:\s+[^)]*)?\)", prose)
    destinations = [match.group(1).strip("<>") for match in inline]
    references = {match.group(1).strip().casefold(): match.group(2).strip("<>")
                  for match in re.finditer(r"^\s{0,3}\[([^]]+)\]:\s*(<[^>]+>|\S+)", prose, re.MULTILINE)}
    for match in re.finditer(r"!?\[([^]]+)\]\[([^]]*)\]", prose):
        label = (match.group(2) or match.group(1)).strip().casefold()
        if label in references:
            destinations.append(references[label])
    return destinations


def headings(notebook):
    result = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            for line in prose_without_code(cell.source).splitlines():
                match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
                if match:
                    text = match.group(2).replace("original Subset data", "Subset data")
                    text = text.replace("deduplicated Subset data", "Subset data")
                    result.append((len(match.group(1)), text))
    return result


def parse_code(source):
    # IPython line magics have no Python AST. Preserve all ordinary code lines.
    ordinary = "\n".join(line for line in source.splitlines()
                         if not line.lstrip().startswith(("%", "!")))
    return ast.parse(ordinary)


class NormaliseCondition(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in {"original", "dedup_v1", "deduplicated"}:
            return ast.copy_location(ast.Constant(value="__CONDITION__"), node)
        return node


def normalised_code(notebook):
    return [{"python_ast": ast.dump(NormaliseCondition().visit(parse_code(cell.source)), include_attributes=False),
             "ipython_lines": [line.strip() for line in cell.source.splitlines()
                               if line.lstrip().startswith(("%", "!"))]}
            for cell in notebook.cells if cell.cell_type == "code"]


def notebook_checks(root, relative):
    path = root / relative
    errors = []
    try:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
    except Exception as error:
        return None, {"notebook": relative, "status": "failed", "sha256": sha256(path) if path.is_file() else None,
            "code_cells": None, "executed_code_cells": 0, "saved_errors": [], "local_links": [],
            "headings": [], "markdown_words": 0, "errors": [str(error)]}
    code_cells, executed, saved_errors, links = 0, 0, [], []
    for number, cell in enumerate(notebook.cells, 1):
        if cell.cell_type == "markdown":
            for destination in markdown_destinations(cell.source):
                parsed = urlsplit(destination)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                target = (path.parent / unquote(parsed.path)).resolve()
                valid = target.is_relative_to(root) and target.exists()
                links.append({"cell": number, "destination": destination, "exists": valid})
                if not valid:
                    errors.append(f"Cell {number}: broken local Markdown link {destination}")
        elif cell.cell_type == "code":
            code_cells += 1
            executed += int(cell.execution_count is not None)
            for line in cell.source.splitlines():
                if line.lstrip().startswith(("%", "!")) and re.search(
                    r"(?:fit_final|run_tuning|run_training|train_model|evaluate_metadata_holdout|reserved_test|test_clean)", line):
                    errors.append(f"Cell {number}: fitting/final-test command in IPython syntax")
            saved_errors += [{"cell": number, "ename": output.get("ename"), "evalue": output.get("evalue")}
                             for output in cell.outputs if output.output_type == "error"]
            try:
                tree = parse_code(cell.source)
            except SyntaxError as error:
                errors.append(f"Cell {number}: Python parse failed: {error}")
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if name in FORBIDDEN_CALLS:
                    errors.append(f"Cell {number}: fitting/final-test call {name}")
                if name in {"read_csv", "read_parquet", "read_excel", "read_pickle", "load", "open"}:
                    arguments = [child.value if isinstance(child, ast.Constant) else child.id
                                 for argument in node.args[:1] for child in ast.walk(argument)
                                 if isinstance(child, (ast.Constant, ast.Name)) and
                                 (isinstance(child, ast.Name) or isinstance(child.value, str))]
                    if any(re.search(r"(?:^|[^a-z])(?:reserved_?test|test_?(?:labels|clean)|test)(?:[^a-z]|$)",
                                     value.lower()) for value in arguments):
                        errors.append(f"Cell {number}: possible reserved-test input in {name}")
    if saved_errors:
        errors.append("Saved notebook contains execution errors")
    return notebook, {"notebook": relative, "status": "failed" if errors else "complete",
        "sha256": sha256(path), "code_cells": code_cells, "executed_code_cells": executed,
        "saved_errors": saved_errors, "local_links": links, "headings": headings(notebook),
        "markdown_words": sum(len(cell.source.split()) for cell in notebook.cells if cell.cell_type == "markdown"),
        "errors": errors}


def template_checks(root, notebooks):
    checks = []
    for stage, name in [("06", "06_metadata_neural_network.ipynb"), ("07", "07_metadata_tuning.ipynb")]:
        original_key, clean_key = (f"notebooks/{directory}/{name}" for directory in ("Subset", "Subset_dedup"))
        errors = []
        original, clean = notebooks.get(original_key), notebooks.get(clean_key)
        if original is None or clean is None:
            checks.append({"stage": stage, "status": "failed", "errors": ["A current notebook is unavailable"]})
            continue
        try:
            archive = nbformat.read(root / ARCHIVE / name, as_version=4)
            nbformat.validate(archive)
        except Exception as error:
            checks.append({"stage": stage, "status": "failed", "errors": [f"Original archive unavailable: {error}"]})
            continue
        expected = headings(archive)
        expected = [(level, _adapt_heading(text)) for level, text in expected]
        archive_words = sum(len(cell.source.split()) for cell in archive.cells if cell.cell_type == "markdown")
        ratios = {}
        for key, notebook in [(original_key, original), (clean_key, clean)]:
            current = headings(notebook)
            position = 0
            for heading in expected:
                try:
                    position = current.index(heading, position) + 1
                except ValueError:
                    errors.append(f"{key}: archived heading missing or reordered: {heading}")
                    break
            words = sum(len(cell.source.split()) for cell in notebook.cells if cell.cell_type == "markdown")
            ratios[key] = words / archive_words if archive_words else 1.
            if ratios[key] < .90:
                errors.append(f"{key}: Markdown depth is below 90% of the archived original")
        if headings(original) != headings(clean):
            errors.append("Original and clean heading hierarchies differ")
        try:
            if normalised_code(original) != normalised_code(clean):
                errors.append("Original and clean report code/dispatch differ beyond the condition")
        except SyntaxError:
            errors.append("Report code could not be compared because a cell does not parse")
        checks.append({"stage": stage, "status": "failed" if errors else "complete",
            "archive": (ARCHIVE / name).as_posix(), "archive_sha256": sha256(root / ARCHIVE / name),
            "markdown_word_ratios": ratios, "heading_adaptations": HEADING_ADAPTATIONS, "errors": errors})
    return checks


def _adapt_heading(text):
    for before, after in HEADING_ADAPTATIONS.items():
        text = text.replace(before, after)
    return text


def plot_specification_check(root):
    """Inspect figures made from artificial rows; do not load any saved run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    import numpy as np
    import pandas as pd
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from melanoma_detection.metadata_nn_tuning_reporting import (
        plot_nn_comparison, plot_tuning_bars, plot_tuning_curves,
    )
    frame = pd.DataFrame([{"fold": fold, "target": target} for fold in range(5) for target in [0, 1]])
    for name in ["spline", "spline_tuned", "random_forest", "random_forest_tuned", "xgboost",
                 "xgboost_tuned", "nn_mean", "nn_tuned", "nn_seed_42", "nn_seed_142", "nn_seed_242"]:
        frame[f"p_{name}"] = .1 + .6 * frame.target
    figures, errors = [], []
    try:
        figures = [plot_nn_comparison(frame), plot_tuning_bars(frame), plot_tuning_curves(frame)]
        nn, bars, curves = figures
        if len(nn.axes) != 2 or not np.allclose(nn.get_size_inches(), [11, 4.3]):
            errors.append("NN comparison must preserve the original two-panel 11 x 4.3 composition")
        expected_nn = ["#2563eb", "#d97706", "#d97706", "#d97706", "#303030"]
        for axis in nn.axes:
            if [to_hex(bar.get_facecolor()) for bar in axis.patches] != expected_nn:
                errors.append("NN comparison blue/orange/charcoal colors differ from the original")
        if len(bars.axes) != 1 or [to_hex(bar.get_facecolor()) for bar in bars.axes[0].patches] != \
                ["#2563eb"] * 4 + ["#c58a00"] * 4:
            errors.append("Four-family tuning bars must use fixed blue and tuned gold")
        if len(curves.axes) != 8 or not np.allclose(curves.get_size_inches(), [14, 17.333], atol=.001):
            errors.append("Tuning curves must contain four rows and two columns")
        for axis in curves.axes:
            models = [line for line in axis.lines if line.get_color().lower() in {"#2563eb", "#c58a00"}]
            if [(line.get_color().lower(), line.get_linestyle()) for line in models] != \
                    [("#2563eb", "--"), ("#c58a00", "-")]:
                errors.append("Tuning curves must preserve fixed dashed blue and tuned solid gold")
    except Exception as error:
        errors.append(str(error))
    finally:
        for figure in figures:
            plt.close(figure)
    return {"status": "failed" if errors else "complete", "data": "artificial style-validation rows only",
            "saved_runs_loaded": False, "errors": errors}


def execution_checks(root, rows, path, selected):
    if not path.exists():
        return {"status": "pending", "selected_status": "pending", "reason": "Execution receipt is absent", "notebooks": []}
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        return {"status": "failed", "selected_status": "failed", "notebooks": [],
                "errors": [f"Execution receipt is unreadable: {error}"]}
    errors, results = [], []
    if receipt.get("training_started") is not False or receipt.get("reserved_test_loaded") is not False:
        errors.append("Execution receipt must explicitly confirm no training or reserved-test loading")
    entries = {row["notebook"]: row for row in receipt.get("notebooks", [])}
    if len(entries) != len(receipt.get("notebooks", [])):
        errors.append("Execution receipt repeats a notebook")
    for row in rows:
        key, entry = row["notebook"], entries.get(row["notebook"])
        problems = []
        if entry is None:
            status = "pending"
        else:
            if row["status"] != "complete":
                problems.append("Current notebook did not pass static validation")
            if entry.get("sha256") != row.get("sha256"):
                problems.append("Executed notebook fingerprint differs from the current file")
            if entry.get("errors") != 0 or row["saved_errors"]:
                problems.append("Execution did not finish without errors")
            if entry.get("code_cells") != row["code_cells"] or row["executed_code_cells"] != row["code_cells"]:
                problems.append("Every code cell must have an execution count")
            status = "failed" if problems else "complete"
        results.append({"notebook": key, "status": status, "errors": problems})
    if receipt.get("status") == "failed":
        errors.append(f"Execution receipt failed: {receipt.get('failed_notebook', 'unspecified notebook')}")
    selected_rows = [row for row in results if row["notebook"] in selected]
    selected_status = _status([row["status"] for row in selected_rows], errors)
    status = _status([row["status"] for row in results], errors)
    if status == "complete" and receipt.get("status") != "complete":
        status = "pending"
    return {"status": status, "selected_status": selected_status, "receipt": path.relative_to(root).as_posix(),
            "receipt_sha256": sha256(path), "recorded_status": receipt.get("status"), "notebooks": results, "errors": errors}


def visual_checks(root, rows, path):
    if not path.exists():
        return {"status": "pending", "reason": "Manual visual-inspection receipt is absent"}
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        return {"status": "failed", "errors": [f"Visual-inspection receipt is unreadable: {error}"]}
    if receipt.get("status") == "failed":
        return {"status": "failed", "errors": ["Manual visual inspection recorded a failure"]}
    if receipt.get("status") != "complete":
        return {"status": "pending", "reason": "Manual visual inspection is not complete"}
    fingerprints = receipt.get("notebook_sha256", {})
    errors = [f"Visual review fingerprint is absent or stale: {row['notebook']}" for row in rows
              if fingerprints.get(row["notebook"]) != row.get("sha256")]
    return {"status": "failed" if errors else "complete", "receipt": path.relative_to(root).as_posix(),
            "receipt_sha256": sha256(path), "errors": errors}


def _status(statuses, errors=()):
    if errors or "failed" in statuses:
        return "failed"
    return "complete" if statuses and all(value == "complete" for value in statuses) else "pending"


def verify_reports(project_root=ROOT, *, condition=None, execution_receipt=None, visual_receipt=None):
    root = Path(project_root).resolve()
    if condition not in (None, "original", "dedup_v1"):
        raise ValueError("condition must be original, dedup_v1 or omitted")
    selected = NOTEBOOKS[condition] if condition else sum(NOTEBOOKS.values(), [])
    notebooks, rows = {}, []
    for relative in sum(NOTEBOOKS.values(), []):
        notebook, row = notebook_checks(root, relative)
        notebooks[relative] = notebook
        rows.append(row)
    templates = template_checks(root, notebooks)
    plot_spec = plot_specification_check(root)
    execution_path = (root / (execution_receipt or REPORT_DIRECTORY / "execution_receipt.json")).resolve()
    visual_path = (root / (visual_receipt or REPORT_DIRECTORY / "visual_inspection_receipt.json")).resolve()
    for path in (execution_path, visual_path):
        if not path.is_relative_to(root / REPORT_DIRECTORY):
            raise ValueError("Acceptance inputs must stay within the report-verification directory")
    execution = execution_checks(root, rows, execution_path, selected)
    visual = visual_checks(root, rows, visual_path)
    static = _status([row["status"] for row in rows] + [row["status"] for row in templates] + [plot_spec["status"]])
    return {"schema_version": 1, "status": _status([static, execution["status"], visual["status"]]),
        "verified_utc": datetime.now(timezone.utc).isoformat(), "condition_checked": condition or "both",
        "training_started": False, "reserved_test_loaded": False, "notebooks_modified": False,
        "checks": {"static": static, "all_execution": execution["status"],
                   "selected_execution": execution["selected_status"], "manual_visual_inspection": visual["status"]},
        "notebooks": rows, "templates": templates, "plot_specification": plot_spec,
        "execution": execution, "manual_visual_inspection": visual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--condition", choices=("original", "dedup_v1", "both"), default="both")
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument("--visual-receipt", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    receipt = verify_reports(root, condition=None if args.condition == "both" else args.condition,
                             execution_receipt=args.execution_receipt, visual_receipt=args.visual_receipt)
    directory = root / REPORT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "report_acceptance.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps({"status": receipt["status"], "checks": receipt["checks"],
        "errors": [error for row in receipt["notebooks"] + receipt["templates"] for error in row["errors"]]
                  + receipt["plot_specification"]["errors"] + receipt["execution"].get("errors", [])
                  + [row["notebook"] + ": " + error for row in receipt["execution"]["notebooks"] for error in row["errors"]]
                  + receipt["manual_visual_inspection"].get("errors", []),
        "receipt": destination.relative_to(root).as_posix()}, indent=2))
    return 1 if receipt["status"] == "failed" else (2 if args.require_complete and receipt["status"] != "complete" else 0)


if __name__ == "__main__":
    raise SystemExit(main())
