"""Keep recorded notebook code checks strict across filename-only relocation."""
from pathlib import Path
import hashlib
import json


def code_sha256(document):
    code = "\n\n".join("".join(cell["source"]) for cell in document["cells"]
                       if cell["cell_type"] == "code")
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def relocated_code_source(source, renames):
    """Translate exact former notebook references, leaving computation intact."""
    for entry in sorted(renames, key=lambda item: len(item["old_path"]), reverse=True):
        source = source.replace(entry["old_path"], entry["new_path"])
    for entry in renames:
        source = source.replace(Path(entry["old_path"]).name, Path(entry["new_path"]).name)
    return source


def verify_recorded_notebook_code(root, current_path, recorded_path, expected_hash):
    """Accept exact current code or the narrowly recorded path substitutions.

    Previous prediction manifests remain unchanged. Explicitly archived
    metadata reports bind to their verified pre-tuning source. For an older run, the
    snapshot must reproduce its recorded code hash, and every current code cell
    must match that snapshot with only declared filename substitutions. Any
    computational change still rejects the stale predictions.
    """
    root = Path(root).resolve()
    current_path = (root / current_path).resolve()
    if not current_path.is_relative_to(root / "notebooks"):
        raise ValueError("Notebook path is outside the notebook directory.")
    from .metadata_source_archive import historical_notebook_path
    preserved = historical_notebook_path(root, current_path.relative_to(root).as_posix())
    current = json.loads(preserved.read_text(encoding="utf-8"))
    if code_sha256(current) == expected_hash:
        return "exact_current_code"
    registry = json.loads((root / "notebooks/filename_map.json").read_text(encoding="utf-8"))
    entries = registry["renames"]
    entry = next((entry for entry in entries if entry["old_path"] == recorded_path
                  and entry["new_path"] == current_path.relative_to(root).as_posix()), None)
    if entry is None:
        raise ValueError("Notebook code changed after its saved run.")
    snapshot = (root / entry["snapshot_path"]).resolve()
    if not snapshot.is_relative_to(root / "results/notebook_organization"):
        raise ValueError("Notebook snapshot is outside the rename record.")
    original_bytes = snapshot.read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != entry["original_sha256"]:
        raise ValueError("Original notebook snapshot changed.")
    original = json.loads(original_bytes)
    if code_sha256(original) != expected_hash:
        raise ValueError("Original notebook code does not match the recorded run.")
    old_cells = [relocated_code_source("".join(c["source"]), entries)
                 for c in original["cells"] if c["cell_type"] == "code"]
    new_cells = ["".join(c["source"]) for c in current["cells"] if c["cell_type"] == "code"]
    if old_cells != new_cells:
        raise ValueError("Notebook computation changed after its saved run.")
    return "verified_filename_substitutions_only"
