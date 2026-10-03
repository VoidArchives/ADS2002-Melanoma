"""Verify historical preservation without rewriting any protected artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


AUTHORISED_EDITS = {
    "src/melanoma_detection/image_model.py",
    "configs/image_model.yaml",
    "docs/image_methodology.md",
    "tests/test_image_model.py",
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(root: Path, inventory: Path) -> dict:
    baseline = json.loads(inventory.read_text(encoding="utf-8"))
    unchanged, authorised, failures = [], [], []
    for relative, record in baseline["files"].items():
        path = root / relative
        actual = digest(path) if path.is_file() else None
        if actual == record["sha256"]:
            unchanged.append(relative)
        elif relative in AUTHORISED_EDITS and path.is_file():
            snapshot = inventory.parent / "baseline_sources" / relative
            if not snapshot.is_file() or digest(snapshot) != record["sha256"]:
                failures.append({"path": relative, "reason": "original source snapshot missing or changed"})
            else:
                authorised.append({"path": relative, "before": record["sha256"], "after": actual})
        else:
            failures.append({"path": relative, "before": record["sha256"], "after": actual})
    return {"status": "passed" if not failures else "failed", "protected_file_count": len(baseline["files"]),
            "unchanged_count": len(unchanged), "authorised_source_changes": authorised,
            "historical_artifact_failures": failures,
            "current_architecture_notebook_preserved": "notebooks/05e_image_architecture_comparison_d1.ipynb" in unchanged}


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
        if target in [root / relative for relative in json.loads(inventory.read_text())["files"]]:
            raise ValueError("Verification output cannot overwrite a preserved artifact.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
