"""Verified development variants; never resolves or opens reserved-test inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from .metadata_validation import validate_metadata_frame

CANONICAL_CSV = "data/subset/processed/train_folds.csv"
VARIANT_DIRECTORY = "data/subset/processed/dedup_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def retained_id_hash(frame: pd.DataFrame) -> str:
    return stable_sha256(sorted(frame.image_name.astype(str).tolist()))


def _restricted_variant_path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    path = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not path.is_relative_to((root / VARIANT_DIRECTORY).resolve()):
        raise ValueError("Development variant artifacts must stay within data/subset/processed/dedup_v1.")
    return path


def _frame_identity(frame: pd.DataFrame, variant: str) -> str:
    # Do not make fitting identity depend on held-out labels.
    rows = frame[["image_name", "patient_id", "fold"]].sort_values("image_name").to_dict("records")
    return stable_sha256({"variant": variant, "memberships": rows})


def load_development_dataset(project_root, dataset_manifest=None) -> tuple[pd.DataFrame, dict]:
    """Return checked metadata and provenance, preserving original default behaviour.

    A variant must verify its original canonical bytes and each exported artifact.
    All masking is by deterministic image ID; labels never choose retained rows.
    """
    root = Path(project_root).resolve()
    canonical_path = root / CANONICAL_CSV
    canonical_digest = sha256_file(canonical_path)
    canonical = validate_metadata_frame(pd.read_csv(canonical_path)).sort_values("image_name").reset_index(drop=True)
    identity = {"variant": "original", "dataset_manifest": None, "dataset_manifest_sha256": None,
                "canonical_source_sha256": canonical_digest, "retained_ids_sha256": retained_id_hash(canonical),
                "dataset_identity": _frame_identity(canonical, "original"), "membership_sha256": {},
                "preserved_inner_memberships": {}}
    if dataset_manifest is None:
        return canonical, identity
    manifest_path = _restricted_variant_path(root, dataset_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("variant") != "dedup_v1":
        raise ValueError("Unsupported development variant manifest.")
    if manifest.get("original_source") != {"path": CANONICAL_CSV, "sha256": canonical_digest}:
        raise ValueError("Canonical development source changed since variant freezing.")
    artifacts = manifest.get("artifacts", {})
    required = {f"{VARIANT_DIRECTORY}/retention_ledger.csv", f"{VARIANT_DIRECTORY}/train_folds.csv"}
    if not required.issubset(artifacts):
        raise ValueError("Manifest is missing required variant exports.")
    for relative, digest in artifacts.items():
        if sha256_file(_restricted_variant_path(root, relative)) != digest:
            raise ValueError(f"Frozen variant artifact changed: {relative}")
    ledger = pd.read_csv(root / f"{VARIANT_DIRECTORY}/retention_ledger.csv", dtype={"image_name": str})
    if not ledger.image_name.is_unique or set(ledger.image_name) != set(canonical.image_name):
        raise ValueError("Retention ledger must partition all original development IDs exactly once.")
    if not ledger.decision.isin(["retained", "excluded"]).all():
        raise ValueError("Unknown retention decision.")
    for _, group in ledger.groupby("pixel_sha256", sort=False):
        chosen = group[group.decision.eq("retained")]
        if len(chosen) != 1 or chosen.image_name.iloc[0] != min(group.image_name):
            raise ValueError("Each pixel group must retain exactly its lexicographically smallest ID.")
        if not group.retained_image_name.eq(chosen.image_name.iloc[0]).all():
            raise ValueError("Ledger representative disagrees with retained decision.")
    retained_ids = set(ledger.loc[ledger.decision.eq("retained"), "image_name"])
    expected = canonical[canonical.image_name.isin(retained_ids)].reset_index(drop=True)
    exported = validate_metadata_frame(pd.read_csv(root / f"{VARIANT_DIRECTORY}/train_folds.csv"))
    exported = exported.sort_values("image_name").reset_index(drop=True)
    try:
        assert_frame_equal(exported, expected, check_dtype=False)
    except AssertionError as error:
        raise ValueError("Variant export differs from the canonical retained records.") from error
    if retained_id_hash(expected) != manifest.get("retained_ids_sha256"):
        raise ValueError("Retained-ID digest differs from the manifest.")
    counts = {"images": len(expected), "malignant_images": int(expected.target.sum()),
              "patients": int(expected.patient_id.nunique()),
              "malignant_patients": int(expected.loc[expected.target.eq(1), "patient_id"].nunique())}
    if counts != manifest.get("counts"):
        raise ValueError("Variant counts differ from the frozen manifest.")
    memberships = manifest.get("preserved_inner_memberships", {})
    if set(memberships) != {str(fold) for fold in range(5)}:
        raise ValueError("Manifest must preserve all five outer partitions' inner memberships.")
    if any(relative not in artifacts for relative in memberships.values()):
        raise ValueError("Inner membership artifact lacks a manifest digest.")
    identity.update({"variant": "dedup_v1", "dataset_manifest": manifest_path.relative_to(root).as_posix(),
                     "dataset_manifest_sha256": sha256_file(manifest_path),
                     "retained_ids_sha256": manifest["retained_ids_sha256"],
                     "dataset_identity": _frame_identity(expected, "dedup_v1"),
                     "membership_sha256": {fold: artifacts[path] for fold, path in memberships.items()},
                     "preserved_inner_memberships": memberships})
    for fold in range(5):
        _validate_inner(expected, pd.read_csv(root / memberships[str(fold)]), fold)
    return expected, identity


def _validate_inner(frame: pd.DataFrame, memberships: pd.DataFrame, outer_fold: int) -> pd.DataFrame:
    expected = frame.loc[frame.fold.ne(outer_fold), ["image_name", "patient_id", "target", "fold"]].sort_values("image_name").reset_index(drop=True)
    required = {"image_name", "patient_id", "target", "fold", "inner_fold"}
    if not required.issubset(memberships) or not memberships.image_name.is_unique:
        raise ValueError("Invalid preserved inner membership schema or duplicate IDs.")
    memberships = memberships.sort_values("image_name").reset_index(drop=True)
    try:
        assert_frame_equal(memberships[list(expected.columns)], expected, check_dtype=False)
    except AssertionError as error:
        raise ValueError("Preserved inner memberships differ from the outer-training pool.") from error
    if set(memberships.inner_fold) != {0, 1, 2} or not memberships.groupby("patient_id").inner_fold.nunique().eq(1).all():
        raise ValueError("Preserved inner assignment must keep every patient in one of three folds.")
    if not memberships.groupby("inner_fold").target.nunique().eq(2).all():
        raise ValueError("Every retained inner fold must contain both classes.")
    return memberships


def load_preserved_inner_memberships(project_root, identity: dict, outer_fold: int) -> pd.DataFrame:
    """Verify and load frozen memberships; never regenerates patient splits."""
    if identity.get("variant") != "dedup_v1" or outer_fold not in range(5):
        raise ValueError("Preserved variant memberships require dedup_v1 and an outer fold 0..4.")
    root = Path(project_root).resolve()
    frame, current_identity = load_development_dataset(root, identity["dataset_manifest"])
    if current_identity != identity:
        raise ValueError("Development variant identity changed.")
    relative = identity["preserved_inner_memberships"][str(outer_fold)]
    return _validate_inner(frame, pd.read_csv(_restricted_variant_path(root, relative)), outer_fold)
