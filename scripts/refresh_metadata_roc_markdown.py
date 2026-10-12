"""Refresh prose/HTML fingerprints only when executed code and outputs are unchanged."""
import argparse
import json
import os
from pathlib import Path
import nbformat
from build_metadata_nn_tuning_notebooks import render_html
from melanoma_detection.metadata_model import file_sha256

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/metadata_nn_tuning/v2/report_verification"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", default="metadata_markdown_priority_patch.json",
                        help="Relative path to a Markdown patch receipt inside report_verification.")
    patch_name = parser.parse_args().patch
    patch_path = (REPORT / patch_name).resolve()
    if not patch_path.is_relative_to(REPORT.resolve()) or patch_path.suffix != ".json":
        raise ValueError("Patch receipt must be a JSON file inside report_verification")
    patch_name = patch_path.relative_to(REPORT).as_posix()
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    changes = {row["notebook"]: row for row in patch["notebooks"]}
    receipt_path = REPORT / "execution_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    refreshed = []
    for row in receipt["notebooks"]:
        change = changes.get(row["notebook"])
        if change is None or row["sha256"] == change["after_sha256"]:
            continue
        source = ROOT / change["before_snapshot"]
        current = ROOT / row["notebook"]
        if row["sha256"] != change["before_sha256"] or file_sha256(source) != row["sha256"] or file_sha256(current) != change["after_sha256"]:
            raise ValueError("Markdown refresh does not match the executed before/after bytes")
        before, after = (nbformat.read(path, as_version=4) for path in [source, current])
        if before.metadata != after.metadata or len(before.cells) != len(after.cells):
            raise ValueError("Markdown refresh altered notebook metadata or cell count")
        for a, b in zip(before.cells, after.cells):
            if a.cell_type != b.cell_type:
                raise ValueError("Markdown refresh changed cell types")
            if a.cell_type == "markdown":
                a, b = dict(a), dict(b)
                a.pop("source"); b.pop("source")
            if a != b:
                raise ValueError("Markdown refresh changed code, outputs or non-source fields")
        (ROOT / row["html"]).write_text(render_html(after, row["notebook"]), encoding="utf-8")
        row["sha256"] = file_sha256(current)
        refreshed.append({"notebook": row["notebook"], "before_sha256": change["before_sha256"],
                          "after_sha256": row["sha256"], "code_and_outputs_unchanged": True})
    existing = receipt.get("verified_markdown_only_refreshes", [])
    receipt["verified_markdown_only_refreshes"] = existing + [row for row in refreshed if row not in existing]
    receipt.setdefault("markdown_patch_receipts", {})[patch_name] = file_sha256(patch_path)
    receipt["markdown_refresh_source_sha256"] = file_sha256(Path(__file__))
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
    print(json.dumps({"status": "complete", "prose_only_refreshed_reports": refreshed}, indent=2))
