"""Strict bindings to metadata source snapshots preserved before NN tuning.

Report notebooks may evolve while completed experiments keep their immutable
training sources. Only the three explicitly archived metadata notebooks may
use this fallback; arbitrary changed Python sources are still rejected.
"""
from pathlib import Path
import hashlib
import json

ARCHIVE = "results/metadata_nn_tuning/v1/archive_before_update"
NOTEBOOKS = {
    "notebooks/Subset/06_metadata_neural_network.ipynb",
    "notebooks/Subset/07_metadata_tuning.ipynb",
    "notebooks/Subset/08_metadata_robustness.ipynb",
}


def verified_source_path(root, relative, expected_digest):
    root = Path(root).resolve()
    current = (root / relative).resolve()
    if not current.is_relative_to(root):
        raise ValueError("Source path is outside the project.")
    if current.is_file() and hashlib.sha256(current.read_bytes()).hexdigest() == expected_digest:
        return current
    relative = current.relative_to(root).as_posix()
    if relative not in NOTEBOOKS:
        raise ValueError(f"Completed experiment source changed: {relative}")
    receipt_path = root / ARCHIVE / "preservation_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("Metadata source archive receipt is missing.")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["notebook_snapshots"].get(relative) != expected_digest:
        raise ValueError("Archived notebook binding differs from the experiment.")
    snapshot = root / ARCHIVE / relative
    if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected_digest:
        raise ValueError("Archived metadata notebook changed.")
    return snapshot


def historical_notebook_path(root, relative):
    """Resolve an explicitly preserved notebook, checking its original bytes."""
    root = Path(root).resolve()
    receipt_path = root / ARCHIVE / "preservation_receipt.json"
    if relative not in NOTEBOOKS or not receipt_path.is_file():
        return root / relative
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    digest = receipt["notebook_snapshots"].get(relative)
    if digest is None:
        raise ValueError("Metadata notebook lacks its preserved source binding.")
    snapshot = root / ARCHIVE / relative
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
        raise ValueError("Metadata notebook snapshot changed.")
    return snapshot
