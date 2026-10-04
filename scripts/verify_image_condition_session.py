"""Verify historical preservation without rewriting any protected artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from melanoma_detection.image_evaluation import (notebook_rename_registry,
    preserved_artifact_path, workspace_relative_path)


AUTHORISED_EDITS = {
    "src/melanoma_detection/image_model.py",
    "configs/image_model.yaml",
    "docs/image_methodology.md",
    "tests/test_image_model.py",
    "scripts/execute_image_notebook.py",
    "src/melanoma_detection/image_evaluation.py",
    "scripts/build_image_notebook.py",
    "scripts/build_image_architecture_notebook.py",
    "scripts/build_image_condition_notebooks.py",
    "scripts/build_image_uncertainty_notebook.py",
    "scripts/run_image_uncertainty.py",
    "scripts/verify_image_condition_session.py",
    "tests/test_metadata_validation.py",
    "README.md",
    "notebooks/README.md",
    "docs/metadata_methodology.md",
}
UNCERTAINTY_ORIGINAL = "notebooks/05a_image_uncertainty_d1_wip.ipynb"
UNCERTAINTY_ARCHIVE = "notebooks/archive/05a_image_uncertainty_d1_historical_three_epoch_fold0.ipynb"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(root: Path, inventory: Path) -> dict:
    root = root.resolve()
    baseline = json.loads(inventory.read_text(encoding="utf-8"))
    unchanged, authorised, relocated, failures = [], [], [], []
    try:
        registry = notebook_rename_registry(root)
    except ValueError as error:
        return {"status": "failed", "protected_file_count": len(baseline["files"]),
                "unchanged_count": 0, "authorised_source_changes": [],
                "authorised_historical_archives": [],
                "historical_artifact_failures": [{"path": "notebooks/filename_map.json", "reason": str(error)}],
                "current_architecture_notebook_preserved": False}
    source_snapshots = registry.get("authorized_source_snapshots", {})
    for relative, record in baseline["files"].items():
        try:
            path = workspace_relative_path(root, relative)
        except ValueError as error:
            failures.append({"path": relative, "reason": str(error)})
            continue
        actual = digest(path) if path.is_file() else None
        if actual == record["sha256"]:
            unchanged.append(relative)
        elif relative.endswith(".ipynb") and relative.startswith("notebooks/"):
            try:
                original = preserved_artifact_path(root, relative, record["sha256"])
            except ValueError as error:
                failures.append({"path": relative, "reason": str(error)})
            else:
                relocated.append({"path": relative, "archive": original.relative_to(root).as_posix(),
                                  "sha256": record["sha256"]})
        elif relative in AUTHORISED_EDITS and path.is_file():
            snapshot = inventory.parent / "baseline_sources" / relative
            valid_original = snapshot.is_file() and digest(snapshot) == record["sha256"]
            registry_source = source_snapshots.get(relative)
            if not valid_original and registry_source is not None and registry_source["original_sha256"] == record["sha256"]:
                snapshot = workspace_relative_path(root, registry_source["snapshot_path"])
                valid_original = snapshot.is_file() and digest(snapshot) == record["sha256"]
            if not valid_original:
                failures.append({"path": relative, "reason": "original source snapshot missing or changed"})
            else:
                authorised.append({"path": relative, "before": record["sha256"], "after": actual,
                                   "original_snapshot": snapshot.resolve().relative_to(root).as_posix()})
        else:
            failures.append({"path": relative, "before": record["sha256"], "after": actual})
    return {"status": "passed" if not failures else "failed", "protected_file_count": len(baseline["files"]),
            "unchanged_count": len(unchanged), "authorised_source_changes": authorised,
            "authorised_historical_archives": relocated,
            "historical_artifact_failures": failures,
            "current_architecture_notebook_preserved": "notebooks/05e_image_architecture_comparison_d1.ipynb" in
                (unchanged + [record["path"] for record in relocated])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inventory", type=Path,
                        default=Path("results/image_d1_runs/b0_augmentation_12e_v1/session/preservation_inventory.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    inventory = args.inventory if args.inventory.is_absolute() else root / args.inventory
    report = verify(root, inventory)
    if args.output:
        target = args.output if args.output.is_absolute() else root / args.output
        protected = [workspace_relative_path(root, relative) for relative in json.loads(inventory.read_text())["files"]]
        protected.extend([inventory.resolve(), root / UNCERTAINTY_ARCHIVE,
                          root / "notebooks/filename_map.json"])
        registry = notebook_rename_registry(root)
        for record in registry["renames"]:
            protected.extend(workspace_relative_path(root, record[key]) for key in ("new_path", "snapshot_path"))
        protected.extend(workspace_relative_path(root, record["snapshot_path"])
                         for record in registry.get("authorized_source_snapshots", {}).values())
        if (target.resolve() in [path.resolve() for path in protected]
                or target.resolve().is_relative_to((inventory.parent / "baseline_sources").resolve())):
            raise ValueError("Verification output cannot overwrite a preserved artifact.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
