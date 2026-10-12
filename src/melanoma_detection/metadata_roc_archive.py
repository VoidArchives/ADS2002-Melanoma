"""Exact, receipt-bound historical sources for ROC-first reports and reuse."""
from pathlib import Path
import hashlib
import json

ARCHIVE = "results/metadata_nn_tuning/v2/archive_before_update"
SUITE_RECEIPT = "results/dedup_v1/metadata_suite/metadata_full_dedup_v1/receipt.json"
BOUND_NOTEBOOKS = {
    "notebooks/Subset/04_dummy_logistic_baseline.ipynb": SUITE_RECEIPT,
    "notebooks/Subset/05_metadata_models.ipynb": SUITE_RECEIPT,
}
EXTERNAL_SNAPSHOTS = {
    ("src/melanoma_detection/image_model.py", "2ca3ae8edbdb14f344b1b834d0ba2bed277879ba45a62904ff2ea6fd4a13e2ab"):
        "results/notebook_organization/20261004_sequence_v1/original_sources/src/melanoma_detection/image_model.py",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Historical source escapes the workspace.")
    return path, path.relative_to(root).as_posix()


def original_snapshot(project_root, relative):
    """Read a precisely hashed presentation snapshot, without executing it."""
    root = Path(project_root).resolve()
    _, relative = _relative(root, relative)
    archive = root / ARCHIVE
    receipt = json.loads((archive / "snapshot_receipt.json").read_text(encoding="utf-8"))
    digest = {**receipt["notebook_snapshots"], **receipt["source_snapshots"]}.get(relative)
    snapshot = archive / relative
    if not digest or not snapshot.is_file() or sha256(snapshot) != digest:
        raise ValueError(f"Historical presentation snapshot is missing or changed: {relative}")
    return snapshot


def verified_source_path(project_root, relative, digest, *, parent_receipt=None):
    """Keep live checks strict; accept archives only through a saved parent."""
    root = Path(project_root).resolve()
    current, relative = _relative(root, relative)
    if current.is_file() and sha256(current) == digest:
        return current
    # Preserve the previously established, narrowly bounded archive bindings.
    from .historical_reporting_source import verified_reporting_source_path
    try:
        return verified_reporting_source_path(root, relative, digest)
    except ValueError:
        pass
    parent_receipt = parent_receipt or BOUND_NOTEBOOKS.get(relative)
    if not parent_receipt:
        raise ValueError(f"Changed source has no explicit historical receipt binding: {relative}")
    parent, parent_relative = _relative(root, parent_receipt)
    if not parent.is_relative_to(root / "results"):
        raise ValueError("Historical receipt must remain under results.")
    preservation = json.loads((root / ARCHIVE / "preservation_receipt.json").read_text(encoding="utf-8"))
    expected_parent = preservation["protected_files"].get(parent_relative)
    if preservation.get("status") != "complete" or not expected_parent or sha256(parent) != expected_parent:
        raise ValueError("Historical parent receipt is changed or not preserved.")
    recorded = json.loads(parent.read_text(encoding="utf-8"))
    recorded_sources = {str(key).replace("\\", "/"): value for key, value in recorded.get("source_sha256", {}).items()}
    if len(recorded_sources) != len(recorded.get("source_sha256", {})):
        raise ValueError("Historical source keys have ambiguous normalized paths.")
    if recorded_sources.get(relative) != digest:
        raise ValueError("Historical receipt does not bind this source digest.")
    external = EXTERNAL_SNAPSHOTS.get((relative, digest))
    if external:
        snapshot = root / external
        if preservation["protected_files"].get(external) != digest:
            raise ValueError("The historical source snapshot lacks its preservation binding.")
    else:
        snapshot = original_snapshot(root, relative)
    if sha256(snapshot) != digest:
        raise ValueError("Historical source bytes do not match the experiment.")
    return snapshot


def verify_preservation(project_root):
    root = Path(project_root).resolve()
    archive = root / ARCHIVE
    receipt = json.loads((archive / "preservation_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        raise ValueError("Pre-update historical preservation is incomplete.")
    for relative, digest in receipt["protected_files"].items():
        if sha256(root / relative) != digest:
            raise ValueError(f"A protected historical artifact changed: {relative}")
    snapshots = {**receipt["notebook_snapshots"], **receipt["source_snapshots"]}
    for relative, digest in snapshots.items():
        if sha256(archive / relative) != digest:
            raise ValueError(f"An archived source changed: {relative}")
    for parent in receipt["ap_parent_runs"].values():
        for relative, digest in parent["source_sha256"].items():
            if sha256(root / relative) != digest:
                raise ValueError(f"Frozen AP fitting dependency changed: {relative}")
    return {"historical_files_checked": len(receipt["protected_files"]),
            "archived_notebooks_checked": len(receipt["notebook_snapshots"]),
            "archived_sources_checked": len(receipt["source_snapshots"]),
            "ap_fitting_dependencies_unchanged": True}
