"""Development-only audit and immutable exact-copy dataset freezing.

JPEG pixels choose retention. Metadata and coarse hashes can block or flag a
review, but never select a representative or exclude a distinct photograph.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .development_dataset import (CANONICAL_CSV, VARIANT_DIRECTORY, _validate_inner,
                                  load_development_dataset, retained_id_hash,
                                  sha256_file, stable_sha256)
from .metadata_validation import validate_binary_target, validate_metadata_frame

AUDIT_DIRECTORY = "results/dedup_v1/audit"
HISTORICAL_AUDIT = "results/image_d1_runs/b0_pilot_v1/image_audit.json"
HISTORICAL_RUN = "results/image_d1_runs/b0_nested_5fold_5h_v1"
EXPECTED = {"original_images": 6864, "duplicate_groups": 158, "excluded_benign": 157,
            "excluded_malignant": 1, "images": 6706, "malignant_images": 127,
            "patients": 422, "malignant_patients": 88}
SITES = {"upper extremity", "lower extremity", "torso", "head/neck", "palms/soles", "oral/genital"}


def read_development_source(path: Path, development_ids: set[str]) -> pd.DataFrame:
    """Select IDs before parsing any source labels or other metadata values.

    DictReader necessarily reads CSV fields as strings, but unrelated rows are
    immediately discarded: they are never converted, profiled or retained.
    """
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("image_name") not in development_ids:
                continue
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty or not result.image_name.is_unique or set(result.image_name) != development_ids:
        raise ValueError(f"Development-only source coverage or uniqueness failure: {path.name}")
    return result


def _normalise(value, column):
    if pd.isna(value) or str(value).strip().casefold() in {"", "unknown", "missing"}:
        return None
    if column in {"age_approx", "target", "fold"}:
        return float(value)
    return str(value)


def compare_development_source(canonical: pd.DataFrame, source: pd.DataFrame, name: str) -> None:
    columns = [column for column in canonical if column != "fold"]
    if not set(columns).issubset(source):
        raise ValueError(f"Required source columns missing: {name}")
    # Reject malformed/high-precision source outcomes before float equality can
    # accidentally round them to a seemingly valid canonical 0/1 outcome.
    validate_binary_target(source.target)
    left, right = canonical.set_index("image_name"), source.set_index("image_name")
    for column in columns:
        if column == "image_name":
            continue
        a = left[column].map(lambda value: _normalise(value, column))
        b = right.loc[left.index, column].map(lambda value: _normalise(value, column))
        matching = a.eq(b) | (a.isna() & b.isna())
        if not matching.all():
            examples = left.index[~matching].tolist()[:5]
            raise ValueError(f"Source metadata disagrees with development records: {name}/{column}/{examples}")


def validate_audit_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    frame = validate_metadata_frame(frame).sort_values("image_name").reset_index(drop=True)
    if not frame.image_name.str.fullmatch(r"ISIC_\d{7}").all():
        raise ValueError("Unsafe or malformed development image ID.")
    if not frame.patient_id.str.fullmatch(r"IP_\d{7}").all():
        raise ValueError("Unsafe or malformed patient ID.")
    for column, valid in [("sex", {"male", "female"}), ("anatom_site_general_challenge", SITES)]:
        known = frame[column].map(lambda value: _normalise(value, column)).dropna()
        if not known.isin(valid).all():
            raise ValueError(f"Unknown category in {column}.")
    if not frame.groupby("patient_id").sex.nunique(dropna=True).le(1).all():
        raise ValueError("Sex is inconsistent within a patient.")
    if not frame.benign_malignant.eq(frame.target.map({0: "benign", 1: "malignant"})).all():
        raise ValueError("Binary target disagrees with benign_malignant.")
    # Benign 'unknown' diagnosis is explicitly valid; do not treat it as an absent outcome.
    malignant = frame.target.eq(1)
    if not frame.loc[malignant, "diagnosis"].eq("melanoma").all() or frame.loc[~malignant, "diagnosis"].eq("melanoma").any():
        raise ValueError("Diagnosis conflicts with binary malignancy.")
    return frame


def _decode(path: Path) -> Image.Image:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Image.open(path) as image:
            if image.format != "JPEG":
                raise ValueError(f"Required image is not JPEG: {path.name}")
            pixels = image.convert("RGB")
            pixels.load()
        if caught:
            raise ValueError(f"Image decoding warned for {path.name}: {caught[0].message}")
    return pixels


def image_fingerprint(pixels: Image.Image) -> str:
    # Matches historical _audit_images, including dimensions before RGB bytes.
    return hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()


def audit_images(root: Path, frame: pd.DataFrame, saved_audit: dict, progress=None) -> tuple[pd.DataFrame, dict]:
    saved = {row["image_id"]: row for row in saved_audit["rows"]}
    if len(saved) != len(frame) or set(saved) != set(frame.image_name):
        raise ValueError("Historical image audit does not cover current development IDs exactly.")
    records, groups = [], defaultdict(list)
    for index, row in enumerate(frame.itertuples(index=False)):
        path = root / "data/subset/train" / f"{row.image_name}.jpg"
        pixels = _decode(path)
        content_hash, byte_hash = image_fingerprint(pixels), sha256_file(path)
        old = saved[row.image_name]
        if (content_hash != old["pixel_sha256"] or byte_hash != old["source_sha256"]
                or list(pixels.size) != old["size"] or str(row.patient_id) != old["patient_id"] or int(row.fold) != old["fold"]):
            raise ValueError(f"Current JPEG or membership differs from saved audit: {row.image_name}")
        rgb = np.asarray(pixels)
        gray = np.asarray(pixels.convert("L"), dtype=float)
        thumbnail = np.asarray(pixels.convert("L").resize((8, 8), Image.Resampling.BILINEAR))
        coarse = np.packbits((thumbnail >= thumbnail.mean()).reshape(-1)).tobytes().hex()
        border = np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3),
                                 rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3)])
        records.append({"image_name": row.image_name, "patient_id": row.patient_id, "target": int(row.target),
                        "fold": int(row.fold), "width": pixels.width, "height": pixels.height,
                        "source_sha256": byte_hash, "pixel_sha256": content_hash, "coarse_average_hash_8": coarse,
                        "gray_mean": float(gray.mean()), "gray_std": float(gray.std()),
                        "gradient_proxy": float((np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2),
                        "near_white_fraction": float(np.all(rgb >= 245, axis=2).mean()),
                        "near_black_border_fraction": float(np.all(border <= 10, axis=1).mean())})
        groups[content_hash].append(row.image_name)
        if progress and (index + 1) % 1000 == 0:
            progress(f"Verified {index + 1}/{len(frame)} development JPEGs")
    lookup = frame.set_index("image_name")
    duplicates = [sorted(names) for names in groups.values() if len(names) > 1]
    for names in duplicates:
        metadata = lookup.loc[names]
        conflicts = [column for column in metadata if metadata[column].map(lambda value: _normalise(value, column)).nunique(dropna=False) > 1]
        if conflicts:
            raise ValueError(f"Identical image copies conflict in {conflicts}: {names}")
        representative = _decode(root / "data/subset/train" / f"{names[0]}.jpg")
        for name in names[1:]:
            candidate = _decode(root / "data/subset/train" / f"{name}.jpg")
            if representative.size != candidate.size or representative.tobytes() != candidate.tobytes():
                raise ValueError("Pixel hash collision: exact pixel confirmation failed.")
    fingerprint_frame = pd.DataFrame(records)
    fingerprint_frame["retained_image_name"] = fingerprint_frame.pixel_sha256.map(lambda value: min(groups[value]))
    fingerprint_frame["decision"] = np.where(fingerprint_frame.image_name.eq(fingerprint_frame.retained_image_name), "retained", "excluded")
    fingerprint_frame["duplicate_group_size"] = fingerprint_frame.pixel_sha256.map(lambda value: len(groups[value]))
    fingerprint_frame["exclusion_reason"] = np.where(fingerprint_frame.decision.eq("excluded"), "confirmed_identical_rgb_copy", "")
    coarse_groups = [group.image_name.tolist() for _, group in fingerprint_frame.groupby("coarse_average_hash_8")
                     if len(group) > 1 and group.pixel_sha256.nunique() > 1]
    return fingerprint_frame, {"duplicate_groups": duplicates, "duplicate_group_sizes": [len(names) for names in duplicates],
                               "coarse_hash_groups_with_distinct_pixels": coarse_groups,
                               "coarse_hash_rule": "8x8 grayscale average hash; review-only, never an exclusion rule",
                               "all_images_match_historical_source_and_pixel_hashes": True,
                               "pixel_equality_confirmed_for_every_duplicate": True}


def fold_counts(frame: pd.DataFrame) -> list[dict]:
    return [{"fold": int(fold), "images": len(group), "benign_images": int(group.target.eq(0).sum()),
             "malignant_images": int(group.target.sum()), "patients": int(group.patient_id.nunique()),
             "malignant_patients": int(group.loc[group.target.eq(1), "patient_id"].nunique())}
            for fold, group in frame.groupby("fold")]


def metadata_profile(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    ages = pd.to_numeric(frame.age_approx, errors="coerce")
    work = frame.assign(numeric_age=ages, site_missing=frame.anatom_site_general_challenge.map(lambda value: _normalise(value, "site") is None))
    patients = work.groupby("patient_id").agg(images=("image_name", "size"), malignant_images=("target", "sum"),
                    fold=("fold", "first"), age_min=("numeric_age", "min"), age_max=("numeric_age", "max"),
                    distinct_ages=("numeric_age", "nunique"), missing_site=("site_missing", "sum")).reset_index()
    patients["age_span"] = patients.age_max - patients.age_min
    missing = work.groupby(["fold", "target", "site_missing"]).agg(images=("image_name", "size"), patients=("patient_id", "nunique")).reset_index()
    summary = {"images": len(frame), "patients": len(patients), "malignant_images": int(frame.target.sum()),
               "malignant_patients": int(frame.loc[frame.target.eq(1), "patient_id"].nunique()),
               "patients_with_multiple_recorded_ages": int(patients.distinct_ages.gt(1).sum()),
               "patients_with_age_span_at_least_15": int(patients.age_span.ge(15).sum()),
               "missing_age_images": int(ages.isna().sum()), "missing_site_images": int(work.site_missing.sum()),
               "patient_image_count_min": int(patients.images.min()), "patient_image_count_max": int(patients.images.max()),
               "top_ten_patients_image_fraction": float(patients.images.nlargest(10).sum() / len(frame))}
    return summary, patients, missing


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _review_sheet(root: Path, ids: list[str], title: str, rotation=False) -> bytes:
    columns = 4
    cell_width, cell_height = (360, 220) if rotation else (240, 275)
    sheet = Image.new("RGB", (columns * cell_width, 50 + ((len(ids) + columns - 1) // columns) * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill="black")
    for index, name in enumerate(ids):
        pixels = _decode(root / "data/subset/train" / f"{name}.jpg")
        x, y = (index % columns) * cell_width, 50 + (index // columns) * cell_height
        draw.text((x + 5, y + 5), name, fill="black")
        if rotation:
            original = pixels.resize((224, 224), Image.Resampling.BILINEAR)
            rotated = original.rotate(20, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0)
            sheet.paste(original.resize((170, 170)), (x + 5, y + 28))
            sheet.paste(rotated.resize((170, 170)), (x + 180, y + 28))
        else:
            sheet.paste(pixels.resize((224, 224), Image.Resampling.BILINEAR), (x + 5, y + 28))
    output = BytesIO()
    sheet.save(output, format="PNG")
    return output.getvalue()


def freeze_development_variant(project_root, *, expected_counts=EXPECTED, progress=print) -> dict:
    """Audit and atomically freeze file contents without changing original inputs.

    Every audit finishes before outputs are written. Existing artifacts are
    immutable: an identical rerun is allowed, any byte change stops writing.
    ``expected_counts`` is injectable only for small synthetic regression tests.
    """
    root = Path(project_root).resolve()
    canonical_path = root / CANONICAL_CSV
    original = validate_audit_metadata(pd.read_csv(canonical_path))
    if len(original) != expected_counts["original_images"]:
        raise ValueError("Original development count differs from the declared population.")
    receipt_path = root / "results/metrics/metadata_nn_run_config_d1.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("input_path") != CANONICAL_CSV or receipt.get("input_sha256") != sha256_file(canonical_path):
            raise ValueError("Canonical metadata differs from the historical prediction input receipt.")
    elif expected_counts == EXPECTED:
        raise FileNotFoundError("Historical metadata input receipt is required to freeze this development dataset.")
    ids = set(original.image_name)
    source_paths = ["data/raw/train.csv", "data/subset/train_labels.csv", "data/subset/processed/train_clean.csv"]
    source_hashes = {CANONICAL_CSV: sha256_file(canonical_path)}
    for relative in source_paths:
        source = read_development_source(root / relative, ids)
        compare_development_source(original, source, relative)
        source_hashes[relative] = sha256_file(root / relative)
    historical_path = root / HISTORICAL_AUDIT
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    source_hashes[HISTORICAL_AUDIT] = sha256_file(historical_path)
    ledger, image_audit = audit_images(root, original, historical, progress=progress)
    kept = set(ledger.loc[ledger.decision.eq("retained"), "image_name"])
    retained = original[original.image_name.isin(kept)].reset_index(drop=True)
    excluded = ledger[ledger.decision.eq("excluded")]
    counts = {"images": len(retained), "malignant_images": int(retained.target.sum()),
              "patients": int(retained.patient_id.nunique()),
              "malignant_patients": int(retained.loc[retained.target.eq(1), "patient_id"].nunique())}
    observed = {"original_images": len(original), "duplicate_groups": len(image_audit["duplicate_groups"]),
                "excluded_benign": int(excluded.target.eq(0).sum()), "excluded_malignant": int(excluded.target.eq(1).sum()), **counts}
    if observed != expected_counts:
        raise ValueError(f"Deduplication counts differ from the declared contract: {observed}")
    if any(size != 2 for size in image_audit["duplicate_group_sizes"]):
        raise ValueError("Expected duplicate pairs; a larger group requires investigation.")
    before_profile, before_patients, before_missing = metadata_profile(original)
    after_profile, after_patients, after_missing = metadata_profile(retained)
    original_fold_counts, retained_fold_counts = fold_counts(original), fold_counts(retained)
    dataset_outputs = {f"{VARIANT_DIRECTORY}/retention_ledger.csv": _csv_bytes(ledger),
                       f"{VARIANT_DIRECTORY}/train_folds.csv": _csv_bytes(retained)}
    membership_paths, membership_sources = {}, {}
    for fold in range(5):
        relative = f"{HISTORICAL_RUN}/folds/{fold}/inner_memberships.csv"
        membership = _validate_inner(original, pd.read_csv(root / relative), fold)
        filtered = membership[membership.image_name.isin(kept)].reset_index(drop=True)
        _validate_inner(retained, filtered, fold)
        # Confirm every retained patient's assignment is unchanged, not merely a valid split.
        if not filtered.set_index("image_name").inner_fold.equals(membership[membership.image_name.isin(kept)].set_index("image_name").inner_fold):
            raise ValueError("Inner patient assignment changed during masking.")
        destination = f"{VARIANT_DIRECTORY}/folds/{fold}/inner_memberships.csv"
        dataset_outputs[destination] = _csv_bytes(filtered)
        membership_paths[str(fold)] = destination
        membership_sources[str(fold)] = {"path": relative, "sha256": sha256_file(root / relative)}
        source_hashes[relative] = membership_sources[str(fold)]["sha256"]
    artifacts = {relative: hashlib.sha256(content).hexdigest() for relative, content in dataset_outputs.items()}
    manifest = {"schema_version": 1, "variant": "dedup_v1", "original_source": {"path": CANONICAL_CSV, "sha256": source_hashes[CANONICAL_CSV]},
                "retention_rule": "lexicographically_smallest_image_name_per_confirmed_identical_full_resolution_rgb_group",
                "fingerprint_version": "sha256_str_dimensions_plus_decoded_rgb_v1", "source_fingerprint": "sha256_file_bytes",
                "retained_ids_sha256": retained_id_hash(retained), "counts": counts, "original_counts": before_profile,
                "artifacts": artifacts, "preserved_inner_memberships": membership_paths,
                "original_inner_memberships": membership_sources, "audited_input_sha256": source_hashes,
                "reserved_test_loaded": False, "labels_used_to_choose_representative": False}
    original_images_digest = stable_sha256(ledger[["image_name", "source_sha256", "pixel_sha256"]].to_dict("records"))
    audit_summary = {"status": "complete", "development_ids_only": True, "source_comparisons_match": True,
                     "reserved_test_loaded": False, "training_started": False, "original_images_sha256": original_images_digest,
                     "original_profile": before_profile, "retained_profile": after_profile,
                     "original_fold_counts": original_fold_counts, "retained_fold_counts": retained_fold_counts,
                     "retention_counts": observed, **image_audit,
                     "quality_rules": {"near_white_fraction": "fraction with all RGB channels >=245",
                                       "near_black_border_fraction": "fraction of 8px borders with all channels <=10",
                                       "gradient_proxy": "mean absolute adjacent grayscale difference; review only"},
                     "weak_rotation_review": {"angle_degrees": 20, "resize": 224, "rotation_interpolation": "nearest",
                                              "fill": "black", "expand": False, "changes_to_historical_transform": False,
                                              "interpretation": "black corners and potential edge clipping need review; images not excluded"}}
    audit_outputs = {f"{AUDIT_DIRECTORY}/audit_summary.json": _json_bytes(audit_summary),
                     f"{AUDIT_DIRECTORY}/patient_profile_original.csv": _csv_bytes(before_patients),
                     f"{AUDIT_DIRECTORY}/patient_profile_retained.csv": _csv_bytes(after_patients),
                     f"{AUDIT_DIRECTORY}/missing_site_by_fold_label_original.csv": _csv_bytes(before_missing),
                     f"{AUDIT_DIRECTORY}/missing_site_by_fold_label_retained.csv": _csv_bytes(after_missing)}
    quality_selections = {}
    for title, column, ascending in [("Washed-out review", "near_white_fraction", False),
                                      ("Low-detail review", "gradient_proxy", True),
                                      ("Border-heavy review", "near_black_border_fraction", False)]:
        selected = ledger.sort_values([column, "image_name"], ascending=[ascending, True]).head(8).image_name.tolist()
        quality_selections[column] = selected
        audit_outputs[f"{AUDIT_DIRECTORY}/figures/{column}.png"] = _review_sheet(root, selected, title)
    rotation_ids = quality_selections["near_black_border_fraction"][:4]
    audit_outputs[f"{AUDIT_DIRECTORY}/figures/weak_rotation_review.png"] = _review_sheet(root, rotation_ids, "Original / +20 degree weak rotation: black fill and nearest interpolation", rotation=True)
    audit_outputs[f"{AUDIT_DIRECTORY}/quality_review_selection.json"] = _json_bytes(quality_selections)
    table_rows = "\n".join(f"| {a['fold']} | {a['images']} -> {b['images']} | {a['malignant_images']} -> {b['malignant_images']} | {b['patients']} |" for a, b in zip(original_fold_counts, retained_fold_counts))
    report = f"""# Development dataset audit and exact-copy variant

The audit selects only the original development image IDs before converting source metadata. No reserved-test files were opened and no training occurred.

## Verified retention

{len(image_audit['duplicate_groups'])} exact pixel-identical pairs were confirmed against current JPEGs and historical byte/pixel fingerprints. The smallest image ID in each group is retained. {observed['excluded_benign']} benign and {observed['excluded_malignant']} malignant copies are excluded through a manifest; original files remain intact. Retained: {counts['images']} images, {counts['malignant_images']} malignant images, {counts['patients']} patients and {counts['malignant_patients']} malignant patients.

| Fold | Images original -> retained | Malignant original -> retained | Retained patients |
|---|---:|---:|---:|
{table_rows}

Original outer folds and every saved inner patient assignment are preserved. Source metadata matches after the historical missing-value cleaning and numeric age formatting are accounted for. Duplicate groups agree on all labels, patient IDs, folds and metadata fields. Pixel equality is explicitly checked after fingerprint matching.

## Data-quality findings

- {before_profile['patients_with_multiple_recorded_ages']} patients have multiple recorded ages; {before_profile['patients_with_age_span_at_least_15']} span at least 15 years. Values remain unchanged: capture dates are unavailable, so longitudinal differences cannot be resolved here.
- {before_profile['missing_site_images']} images have missing site; fold/label and patient concentration are exported separately. Sites are not filled from other images belonging to the patient.
- Patients contribute {before_profile['patient_image_count_min']} to {before_profile['patient_image_count_max']} images; the ten largest patients supply {before_profile['top_ten_patients_image_fraction']:.2%} of original rows. Patient bootstrap handles dependence; equal-patient image weighting is a separate sensitivity diagnostic.
- {len(image_audit['coarse_hash_groups_with_distinct_pixels'])} coarse average-hash groups contain distinct pixels. They are review candidates and remain in the dataset.
- Deterministic quality contact sheets show extremes of near-white content, low grayscale gradients and dark borders. These proxies do not automatically establish unusable images or exclude them.
- The historical weak rotation has nearest interpolation, black fill and no expansion. The fixed +20-degree preview illustrates corner fill and possible edge clipping. Replication retains the original transform; revisions belong to a separate experiment.

## Limits and next stage

Diagnosis `unknown` for benign images is allowed. Multiple patient ages/sites and coarse similarity are not duplicate rules. This audit cannot verify annotation truth, imaging dates or clinical diagnoses. Frozen-prediction filtering measures evaluation weighting only; clean retraining is needed to remove training influence. Existing uncertainty outputs describe historical checkpoints until regenerated.

The variant loader verifies canonical-source bytes, exported ledgers/metadata/memberships and retained-ID identity before any model or report can use this dataset.
"""
    audit_outputs[f"{AUDIT_DIRECTORY}/audit_report.md"] = report.encode()
    all_outputs = {**dataset_outputs, **audit_outputs, f"{VARIANT_DIRECTORY}/dataset_manifest.json": _json_bytes(manifest)}
    # Check every existing destination before any writes; never overwrite divergent artifacts.
    for relative, content in all_outputs.items():
        destination = root / relative
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"Immutable dedup_v1 artifact already exists with different contents: {relative}")
    for relative, content in all_outputs.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with destination.open("xb") as handle:
                handle.write(content)
    loaded, identity = load_development_dataset(root, f"{VARIANT_DIRECTORY}/dataset_manifest.json")
    # Recheck source hashes and all image bytes after exports. Original records/checkpoints are never written.
    for relative, digest in source_hashes.items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"Original source changed during audit: {relative}")
    current_images = [{"image_name": row.image_name, "source_sha256": sha256_file(root / "data/subset/train" / f"{row.image_name}.jpg"),
                       "pixel_sha256": row.pixel_sha256} for row in ledger.itertuples(index=False)]
    if stable_sha256(current_images) != original_images_digest:
        raise ValueError("Original images changed during audit.")
    return {"status": "complete", "variant": identity["variant"], "dataset_manifest": identity["dataset_manifest"],
            "dataset_manifest_sha256": identity["dataset_manifest_sha256"], "retained_counts": counts,
            "audit_report": f"{AUDIT_DIRECTORY}/audit_report.md", "reserved_test_loaded": False, "training_started": False}
