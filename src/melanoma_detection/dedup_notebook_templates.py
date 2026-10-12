"""Adapt original notebook cells without executing historical experiment code.

Only Markdown and explicitly supplied report-code replacements are transferred.
Saved original outputs, execution counts and fitting code are never copied.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import nbformat

ORIGINAL_DIRECTORY = Path(__file__).resolve().parents[2] / "notebooks" / "Subset"
MARKDOWN_CELL = "__markdown__"
CODE_CELL = "__code__"


def original_template_path(root, relative):
    """Bind the updated downstream report to its preserved template too."""
    from .metadata_source_archive import ARCHIVE, historical_notebook_path
    root = Path(root).resolve()
    if relative == "notebooks/Subset/14_image_nested_evaluation.ipynb":
        receipt = root / ARCHIVE / "preservation_receipt.json"
        if receipt.exists():
            digest = json.loads(receipt.read_text(encoding="utf-8"))["notebook_snapshots"].get(relative)
            snapshot = root / ARCHIVE / relative
            if digest is None or not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
                raise ValueError("Downstream evaluation template archive is missing or changed.")
            return snapshot
    return historical_notebook_path(root, relative)


def original_cells(name: str):
    """Read the authoritative original template, without executing any cells."""
    if Path(name).name != name or not name.endswith(".ipynb"):
        raise ValueError("An original template must be a simple notebook filename.")
    root = ORIGINAL_DIRECTORY.parents[1]
    return nbformat.read(original_template_path(root, "notebooks/Subset/" + name), as_version=4).cells


def original_plot_style(name: str) -> str:
    """Extract literal rcParams declarations only; never execute notebook code."""
    result = []
    for cell in original_cells(name):
        if cell.cell_type != "code":
            continue
        # IPython magics are not Python AST input. Keep line positions stable.
        source = "\n".join("" if line.lstrip().startswith(("%", "!")) else line
                           for line in cell.source.splitlines())
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if ast.unparse(call.func) != "plt.rcParams.update" or len(call.args) != 1:
                continue
            try:
                settings = ast.literal_eval(call.args[0])
            except (ValueError, TypeError):
                continue
            if isinstance(settings, dict):
                result.append(f"plt.rcParams.update({settings!r})")
    return "\n".join(dict.fromkeys(result))


def original_export_dpi(name: str, default: int = 150) -> int:
    """Recover the most common literal PNG export resolution in the template."""
    values = []
    for cell in original_cells(name):
        if cell.cell_type == "code":
            values.extend(int(value) for value in re.findall(r"\bdpi\s*=\s*(\d+)", cell.source))
    return Counter(values).most_common(1)[0][0] if values else default


def template_sections(name: str, code_replacements: dict[int, str],
                      markdown_updates: dict[int, str] | None = None,
                      after_cells: dict[int, list] | None = None,
                      context_note: str = ""):
    """Return original-order cell specifications, using 1-based cell numbers.

    Every code cell requires an explicit replacement, including a deliberately
    empty one. This prevents a historical fitting/test cell entering a report.
    An original Markdown cell is retained verbatim unless its updated source is
    supplied. Extra dedup diagnostics are attached to their relevant sections.
    """
    markdown_updates = markdown_updates or {}
    after_cells = after_cells or {}
    cells = original_cells(name)
    sections = []
    first_markdown = True
    first_code = True
    for index, cell in enumerate(cells, start=1):
        if cell.cell_type == "markdown":
            source = markdown_updates.get(index, cell.source)
            if first_markdown:
                source += ("\n\n**Deduplicated workflow adaptation.** This notebook uses the "
                    "frozen `dedup_v1` development manifest and original patient assignments. "
                    "It reads saved clean artifacts; executing it starts no training and "
                    "does not load the reserved test. The original notebook is preserved as "
                    f"the presentation template: [{name}](../Subset/{name}).")
                if context_note:
                    source += "\n\n" + context_note
                first_markdown = False
            sections.append((MARKDOWN_CELL, source, ""))
        elif cell.cell_type == "code":
            if index not in code_replacements:
                raise ValueError(f"No report-code replacement for {name}, cell {index}.")
            code = code_replacements[index]
            if first_code:
                style = original_plot_style(name)
                if style:
                    code += "\n\n# Preserve the original notebook's plotting defaults.\n" + style
                code += f"\nORIGINAL_EXPORT_DPI = {original_export_dpi(name)}\n"
                first_code = False
            if code.strip():
                sections.append((CODE_CELL, "", code))
        else:
            raise ValueError(f"Unsupported template cell type: {cell.cell_type}")
        sections.extend(after_cells.get(index, []))
    return sections


def markdown_words(sections) -> int:
    return sum(len(re.findall(r"\b[\w]+\b", source))
               for kind, source, _ in sections if kind == MARKDOWN_CELL)
