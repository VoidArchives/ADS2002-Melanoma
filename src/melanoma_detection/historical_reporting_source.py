"""Verify the exact historical clean-report source after reporting-only edits.

This binding is restricted to the clean reporting module. Training sources
continue to use their existing strict checks, and no saved receipt is edited.
"""
from pathlib import Path
import hashlib
import json

from .metadata_source_archive import ARCHIVE, verified_source_path

REPORT_SOURCE = "src/melanoma_detection/dedup_workflow.py"
ARCHIVE_RECEIPT = "historical_reporting_source_archive.json"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_reporting_source_path(root, relative, expected_digest):
    root = Path(root).resolve()
    try:
        return verified_source_path(root, relative, expected_digest)
    except ValueError:
        current = (root / relative).resolve()
        if not current.is_relative_to(root) or current.relative_to(root).as_posix() != REPORT_SOURCE:
            raise
    archive = root / ARCHIVE
    binding_path = archive / ARCHIVE_RECEIPT
    if not binding_path.is_file():
        raise ValueError("Historical reporting-source binding is missing.")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("source") != REPORT_SOURCE or binding.get("source_sha256") != expected_digest:
        raise ValueError("Historical reporting-source binding differs from the experiment.")
    snapshot = archive / REPORT_SOURCE
    if binding.get("snapshot") != snapshot.relative_to(root).as_posix() or not snapshot.is_file():
        raise ValueError("Historical reporting snapshot is missing or misbound.")
    preserved = json.loads((archive / "preservation_receipt.json").read_text(encoding="utf-8"))
    historical_relative = binding.get("historical_receipt", "")
    historical = (root / historical_relative).resolve()
    if not historical.is_relative_to(root / "results/dedup_v1/clean_workflow/evaluation"):
        raise ValueError("Historical reporting receipt is outside the saved evaluation namespace.")
    digest = preserved.get("protected_files", {}).get(historical_relative)
    if not digest or binding.get("historical_receipt_sha256") != digest or not historical.is_file() or _sha(historical) != digest:
        raise ValueError("Historical reporting receipt changed or lacks its preserved binding.")
    receipt = json.loads(historical.read_text(encoding="utf-8"))
    if receipt.get("source_sha256", {}).get(REPORT_SOURCE) != expected_digest or _sha(snapshot) != expected_digest:
        raise ValueError("Historical reporting source differs from the saved experiment.")
    return snapshot
