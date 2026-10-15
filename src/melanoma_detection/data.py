"""Getting data in: label loading/cleaning, the image-folder audit, duplicate
removal, and the PyTorch ``Dataset``/``DataLoader`` builders.

Images are expected as individual JPEG files in one folder (e.g.
``data/train/ISIC_0000000.jpg``), named ``<image_name>.jpg``.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from .utils import RANDOM_STATE

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg")

REQUIRED_COLUMNS = (
    "image_name", "patient_id", "sex", "age_approx",
    "anatom_site_general_challenge", "diagnosis", "benign_malignant", "target",
)
CATEGORICAL_COLUMNS = ("sex", "anatom_site_general_challenge", "diagnosis", "benign_malignant")
AGE_BAND_BINS = [-np.inf, 19, 29, 39, 49, 59, 69, 79, np.inf]
AGE_BAND_LABELS = ["0-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
SPLITS = ("train", "validation", "test")


# --------------------------------------------------------------------------- #
# Labels / metadata
# --------------------------------------------------------------------------- #
def load_labels(path: str | Path) -> pd.DataFrame:
    """Read ``train_labels.csv`` and fail early if a required column is missing."""
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def check_record_integrity(df: pd.DataFrame) -> pd.Series:
    """Count record-level problems. Every value should be 0."""
    return pd.Series({
        "duplicate rows": int(df.duplicated().sum()),
        "duplicate image IDs": int(df["image_name"].duplicated().sum()),
        "target values outside {0, 1}": int((~df["target"].isin([0, 1])).sum()),
        "ages outside 0-120": int(((df["age_approx"] < 0) | (df["age_approx"] > 120)).sum()),
    })


def summarise_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count, missing percent and unique-value count per column."""
    return pd.DataFrame({
        "missing_records": df.isna().sum(),
        "missing_record_percent": df.isna().mean().mul(100),
        "unique_values": df.nunique(dropna=True),
    })


def clean_train_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise text columns, flag and fill missing sites, fix dtypes.

    Adds ``site_missing`` (0/1); missing anatomical sites become ``"unknown"``.
    """
    clean = df.copy()
    for column in CATEGORICAL_COLUMNS:
        clean[column] = clean[column].astype("string").str.strip().str.lower()

    clean["site_missing"] = clean["anatom_site_general_challenge"].isna().astype("int8")
    clean["anatom_site_general_challenge"] = clean["anatom_site_general_challenge"].fillna("unknown")
    clean["age_approx"] = pd.to_numeric(clean["age_approx"], errors="raise").round().astype("Int64")
    clean["target"] = pd.to_numeric(clean["target"], errors="raise").astype("int8")
    return clean


def build_patient_table(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: median age, sex, ``has_malignant`` (any image with target 1),
    image count, whether any site was missing, and an age band."""
    patient_level = (
        clean.groupby("patient_id", as_index=False)
        .agg(
            age_approx=("age_approx", "median"),
            sex=("sex", "first"),
            has_malignant=("target", "max"),
            image_count=("image_name", "size"),
            any_site_missing=("site_missing", "max"),
        )
    )
    patient_level["age_band"] = pd.cut(
        patient_level["age_approx"].astype(float),
        bins=AGE_BAND_BINS,
        labels=AGE_BAND_LABELS,
    )
    return patient_level


# --------------------------------------------------------------------------- #
# Image-folder audit
# --------------------------------------------------------------------------- #
def inspect_image(file_path: Path) -> dict:
    """Decode one image file and record its properties.

    Records format, size, quality measures (brightness, contrast, sharpness) and
    three fingerprints: file hash, decoded-pixel hash, and a coarse difference hash.
    Never raises on a bad image; the error is stored in the row instead.
    """
    row = {
        "file_path": str(file_path), "image_name": file_path.stem,
        "extension": file_path.suffix.lower(),
        "file_kib": file_path.stat().st_size / 1024,
        "readable": False, "error": "", "warnings": "",
        "format": None, "mode": None, "width": np.nan, "height": np.nan,
        "channels": np.nan, "aspect_ratio": np.nan,
        "brightness": np.nan, "contrast": np.nan, "sharpness": np.nan,
        "file_hash": None, "pixel_hash": None, "dhash": None,
    }
    try:
        raw = file_path.read_bytes()
        row["file_hash"] = hashlib.sha256(raw).hexdigest()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with Image.open(BytesIO(raw)) as original:
                original.load()  # decode fully, not just the header
                row.update(
                    format=original.format, mode=original.mode,
                    width=original.width, height=original.height,
                    channels=len(original.getbands()),
                    aspect_ratio=original.width / original.height,
                )
                rgb = original.convert("RGB")
            row["warnings"] = "; ".join(str(w.message) for w in caught)

        pixel_hash = hashlib.sha256(str(rgb.size).encode())
        pixel_hash.update(rgb.tobytes())
        row["pixel_hash"] = pixel_hash.hexdigest()

        thumb = rgb.copy()
        thumb.thumbnail((256, 256))
        grey = np.asarray(thumb.convert("L"), dtype=np.float32)
        row["brightness"], row["contrast"] = float(grey.mean()), float(grey.std())
        if min(grey.shape) >= 3:
            laplacian = (
                -4 * grey[1:-1, 1:-1] + grey[:-2, 1:-1] + grey[2:, 1:-1]
                + grey[1:-1, :-2] + grey[1:-1, 2:]
            )
            row["sharpness"] = float(laplacian.var())

        tiny = np.asarray(rgb.convert("L").resize((9, 8)))
        row["dhash"] = np.packbits(tiny[:, 1:] > tiny[:, :-1]).tobytes().hex()
        row["readable"] = True
    except Exception as error:  # noqa: BLE001 - audit must survive any bad file
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def audit_images(
    images_dir: str | Path,
    progress_every: int = 250,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Inspect every JPEG in ``images_dir`` and return one row per file.

    The audit depends only on the folder's contents, so it can be cached: if
    ``cache_path`` exists it is loaded instead of re-hashing every image; otherwise
    the result is written there. Delete the cache if the folder's images change.
    """
    if cache_path is not None and Path(cache_path).is_file():
        logger.info("Loading cached audit from %s", cache_path)
        return pd.read_pickle(cache_path)

    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {images_dir}")

    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False  # truncated files should be flagged
    records = []
    try:
        files = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
        )
        if not files:
            raise ValueError(f"No JPEG files found in {images_dir}.")
        for count, file_path in enumerate(files, start=1):
            records.append(inspect_image(file_path))
            if count % progress_every == 0 or count == len(files):
                logger.info("Audited %s/%s files", f"{count:,}", f"{len(files):,}")
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous

    audit = pd.DataFrame(records)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        audit.to_pickle(cache_path)
    return audit


def attach_metadata(zip_audit: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Join audit rows to ``image_name, patient_id, target, split`` from the labels."""
    meta = clean[["image_name", "patient_id", "target", "split"]]
    return zip_audit.merge(meta, on="image_name", how="left", validate="many_to_one")


def duplicate_summary(image_audit: pd.DataFrame, hash_column: str) -> pd.DataFrame:
    """Groups of files sharing ``hash_column`` (file_hash, pixel_hash or dhash)."""
    groups = (
        image_audit.dropna(subset=[hash_column])
        .groupby(hash_column)
        .agg(
            files=("file_path", "size"),
            images=("image_name", lambda s: ", ".join(s.unique()[:6])),
            patients=("patient_id", "nunique"),
            split_count=("split", "nunique"),
            label_count=("target", "nunique"),
            pixel_versions=("pixel_hash", "nunique"),
        )
    )
    return groups.loc[groups["files"].gt(1)].reset_index()


def summarise_audit(image_audit: pd.DataFrame, clean: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Headline audit counts plus the detail tables behind them.

    ``image_audit`` must already have metadata attached (:func:`attach_metadata`).
    Returns ``(counts, tables)``; ``tables`` maps a title to a DataFrame you can inspect.
    """
    meta = clean[["image_name", "patient_id", "target", "split"]]
    match_counts = meta["image_name"].map(image_audit["image_name"].value_counts()).fillna(0)
    missing_images = meta.loc[match_counts.eq(0)]
    ambiguous_images = meta.loc[match_counts.gt(1)]
    unmatched_files = image_audit.loc[image_audit["patient_id"].isna()]
    unreadable_files = image_audit.loc[~image_audit["readable"]]
    readable = image_audit.loc[image_audit["readable"]]

    file_duplicates = duplicate_summary(image_audit, "file_hash")
    pixel_duplicates = duplicate_summary(image_audit, "pixel_hash")
    similar = duplicate_summary(image_audit, "dhash").query("pixel_versions > 1")
    cross_split = pixel_duplicates.query("split_count > 1")

    counts = pd.Series({
        "Metadata images": len(meta),
        "Files checked": len(image_audit),
        "Readable images": len(readable),
        "Missing metadata images": len(missing_images),
        "Metadata IDs matching multiple entries": len(ambiguous_images),
        "Files without metadata": len(unmatched_files),
        "Unreadable or non-image entries": len(unreadable_files),
        "Readable images that are not JPEG": int(readable["format"].ne("JPEG").sum()),
        "Readable images with mode other than RGB": int(readable["mode"].ne("RGB").sum()),
        "Images with decoder warnings": int(image_audit["warnings"].ne("").sum()),
        "Identical-file groups": len(file_duplicates),
        "Identical-RGB-pixel groups": len(pixel_duplicates),
        "Identical-pixel groups crossing splits": len(cross_split),
        "Identical-pixel groups with conflicting labels": int(pixel_duplicates["label_count"].gt(1).sum()),
        "Coarse similarity groups to review": len(similar),
    }, name="Count")

    tables = {
        "Missing images": missing_images,
        "Ambiguous image IDs": ambiguous_images,
        "Unmatched files": unmatched_files[["file_path"]],
        "Unreadable/non-image entries": unreadable_files[["file_path", "error"]],
        "Decoder warnings": image_audit.loc[image_audit["warnings"].ne(""), ["file_path", "warnings"]],
        "Exact pixel duplicates": pixel_duplicates,
        "Identical file duplicates": file_duplicates,
        "Similarity candidates (NOT confirmed duplicates)": similar,
    }
    return counts, tables


def remove_duplicates(
    clean: pd.DataFrame,
    image_audit: pd.DataFrame,
    confirmed_pairs: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop exact-pixel duplicates and any visually confirmed duplicate pairs.

    ``confirmed_pairs`` is a list of ``(kept_id, dropped_id)`` image-name pairs that
    you checked by eye (their pixel hashes differ, so they are merged by hand).
    Within each duplicate group the lowest image ID is retained. Raises if duplicate
    records disagree on patient, split, label, sex, age or site.

    Returns ``(clean_images, duplicate_removal_log)``.
    """
    confirmed_pairs = list(confirmed_pairs or [])
    audit_columns = ["image_name", "file_path", "pixel_hash", "readable"]

    working = (
        clean.merge(image_audit[audit_columns], on="image_name", how="left", validate="one_to_one")
        .sort_values("image_name")
        .reset_index(drop=True)
    )
    required = audit_columns + ["patient_id", "split", "target"]
    assert working[required].notna().all().all(), "Missing metadata or audit results."
    assert working["readable"].eq(True).all(), "An image is unreadable."
    assert working["target"].isin([0, 1]).all()
    assert working["split"].isin(SPLITS).all()
    working["target"] = working["target"].astype("int64")

    # Map the hash of each hand-confirmed duplicate onto the hash of the image kept.
    by_name = working.set_index("image_name")["pixel_hash"]
    hash_map = {by_name[dropped]: by_name[kept] for kept, dropped in confirmed_pairs}
    working["duplicate_group"] = working["pixel_hash"].replace(hash_map)

    metadata_columns = ["patient_id", "split", "target", "sex", "age_approx", "anatom_site_general_challenge"]
    conflicts = (
        working.groupby("duplicate_group")[metadata_columns]
        .nunique(dropna=False).gt(1).any(axis=1)
    )
    conflicting = working.loc[
        working["duplicate_group"].isin(conflicts[conflicts].index),
        ["image_name", "duplicate_group"] + metadata_columns,
    ]
    if not conflicting.empty:
        raise ValueError(f"Duplicate metadata disagrees; review before excluding:\n{conflicting}")

    working["retained_image"] = working.groupby("duplicate_group")["image_name"].transform("first")
    working["retained_hash"] = working.groupby("duplicate_group")["pixel_hash"].transform("first")
    remove = working["image_name"].ne(working["retained_image"])

    log = (
        working.loc[remove, ["image_name", "retained_image", "patient_id", "split", "target", "file_path"]]
        .rename(columns={"image_name": "excluded_image"})
        .copy()
    )
    log["reason"] = np.where(
        working.loc[remove, "pixel_hash"].eq(working.loc[remove, "retained_hash"]),
        "Exact pixel duplicate", "Visually confirmed duplicate",
    )
    log = log.reset_index(drop=True)

    clean_images = (
        working.loc[~remove]
        .drop(columns=["readable", "duplicate_group", "retained_image", "retained_hash"])
        .reset_index(drop=True)
    )

    assert clean_images["image_name"].is_unique
    assert clean_images["pixel_hash"].is_unique
    for kept, dropped in confirmed_pairs:
        assert clean_images["image_name"].isin([kept, dropped]).sum() == 1
    assert clean_images.groupby("patient_id")["split"].nunique().eq(1).all()
    assert set(clean_images["patient_id"]) == set(clean["patient_id"])
    assert len(clean_images) + len(log) == len(clean)
    return clean_images, log


# --------------------------------------------------------------------------- #
# PyTorch dataset / loaders
# --------------------------------------------------------------------------- #
class MelanomaImageDataset(Dataset):
    """Reads images straight from a folder of JPEGs.

    Each item is ``(image_tensor, float_label, image_name)``. ``rows`` needs the
    columns ``image_name, file_path, target`` (``file_path`` is whatever
    :func:`remove_duplicates` / the audit recorded). ``num_workers`` can be > 0,
    since each worker just opens plain files rather than sharing one ZIP handle.
    """

    def __init__(self, rows: pd.DataFrame, transform):
        self.rows = rows[["image_name", "file_path", "target"]].reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        with Image.open(row["file_path"]) as source:
            image = source.convert("RGB")

        image = self.transform(image)
        label = torch.tensor(float(row["target"]), dtype=torch.float32)
        return image, label, row["image_name"]


def _initialise_image_worker(worker_id: int) -> None:
    """Avoid nested OpenCV thread pools and seed each worker's augmentation."""
    import random

    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import cv2
    except ImportError:
        pass  # Basic torchvision transforms do not need OpenCV.
    else:
        cv2.setNumThreads(1)
    transform = torch.utils.data.get_worker_info().dataset.transform
    pipeline = getattr(transform, "pipeline", None)
    if hasattr(pipeline, "set_random_seed"):
        pipeline.set_random_seed(seed)


def make_loaders(
    clean_images: pd.DataFrame,
    train_transform,
    eval_transform,
    batch_size: int = 32,
    seed: int = RANDOM_STATE,
    num_workers: int = 0,
    worker_timeout: float = 60,
    device=None,
) -> tuple[dict, dict, torch.Generator]:
    """Build ``(datasets, loaders, generator)`` keyed by ``train/validation/test``.

    Only the training loader shuffles (using the returned seeded generator) and
    only the training dataset gets the augmenting transform. ``clean_images`` must
    have a ``file_path`` column (present after :func:`remove_duplicates`).
    With workers, queue one batch per worker and raise if a batch takes longer
    than worker_timeout seconds. Set worker_timeout=0 to disable the timeout.
    Worker startup on Windows may take several seconds at each new iterator.
    """
    if num_workers < 0 or not np.isfinite(worker_timeout) or worker_timeout < 0:
        raise ValueError("num_workers and worker_timeout must be nonnegative.")
    datasets = {
        split: MelanomaImageDataset(
            rows=clean_images.loc[clean_images["split"].eq(split)],
            transform=train_transform if split == "train" else eval_transform,
        )
        for split in SPLITS
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=(torch.device(device).type == "cuda" if device is not None
                        else torch.cuda.is_available()),
            drop_last=False,
            generator=generator if split == "train" else None,
            worker_init_fn=_initialise_image_worker if num_workers else None,
            prefetch_factor=1 if num_workers else None,
            timeout=worker_timeout if num_workers else 0,
        )
        for split, dataset in datasets.items()
    }
    return datasets, loaders, generator
