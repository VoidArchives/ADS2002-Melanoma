"""Reusable checks for CSV-to-image-file inventory consistency."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ImageInventoryReport:
    """Results of comparing CSV image identifiers with image directories."""

    csv_row_count: int
    unique_csv_names: int
    jpeg_file_count: int
    dicom_file_count: int
    duplicate_csv_names: tuple[str, ...]
    missing_jpeg: tuple[str, ...]
    missing_dicom: tuple[str, ...]
    orphan_jpeg: tuple[str, ...]
    orphan_dicom: tuple[str, ...]
    duplicate_jpeg_stems: tuple[str, ...]
    duplicate_dicom_stems: tuple[str, ...]

    @property
    def all_csv_rows_have_both_formats(self) -> bool:
        """Whether every CSV identifier has both a JPEG and a DICOM file."""

        return not self.missing_jpeg and not self.missing_dicom

    @property
    def directories_exactly_match_csv(self) -> bool:
        """Whether identifiers and both directories match one-to-one."""

        return (
            self.all_csv_rows_have_both_formats
            and not self.duplicate_csv_names
            and not self.orphan_jpeg
            and not self.orphan_dicom
            and not self.duplicate_jpeg_stems
            and not self.duplicate_dicom_stems
        )

    def summary_rows(self) -> list[tuple[str, int]]:
        """Return compact label/count rows suitable for notebooks or logs."""

        return [
            ("CSV rows", self.csv_row_count),
            ("unique CSV image names", self.unique_csv_names),
            ("JPEG files", self.jpeg_file_count),
            ("DICOM files", self.dicom_file_count),
            ("duplicated CSV image names", len(self.duplicate_csv_names)),
            ("CSV names missing JPEG", len(self.missing_jpeg)),
            ("CSV names missing DICOM", len(self.missing_dicom)),
            ("JPEG files absent from CSV", len(self.orphan_jpeg)),
            ("DICOM files absent from CSV", len(self.orphan_dicom)),
            ("duplicate JPEG stems", len(self.duplicate_jpeg_stems)),
            ("duplicate DICOM stems", len(self.duplicate_dicom_stems)),
        ]

    def mismatch_examples(self, limit: int = 20) -> dict[str, tuple[str, ...]]:
        """Return bounded mismatch examples for readable notebook output."""

        if limit < 0:
            raise ValueError("limit must be non-negative")

        return {
            "duplicate_csv_names": self.duplicate_csv_names[:limit],
            "missing_jpeg": self.missing_jpeg[:limit],
            "missing_dicom": self.missing_dicom[:limit],
            "orphan_jpeg": self.orphan_jpeg[:limit],
            "orphan_dicom": self.orphan_dicom[:limit],
            "duplicate_jpeg_stems": self.duplicate_jpeg_stems[:limit],
            "duplicate_dicom_stems": self.duplicate_dicom_stems[:limit],
        }

    def assert_complete(self, *, strict: bool = False) -> None:
        """Raise when required images are missing, or on any mismatch if strict."""

        if not self.all_csv_rows_have_both_formats:
            raise AssertionError(
                "Training images are incomplete: "
                f"{len(self.missing_jpeg)} missing JPEG and "
                f"{len(self.missing_dicom)} missing DICOM files."
            )

        if strict and not self.directories_exactly_match_csv:
            raise AssertionError(
                "Image directories do not match the CSV one-to-one. "
                "Inspect the report for duplicates or orphan files."
            )


def _index_files_by_stem(
    directory: Path,
    allowed_suffixes: Iterable[str],
) -> tuple[dict[str, Path], tuple[str, ...]]:
    allowed = {suffix.lower() for suffix in allowed_suffixes}
    indexed: dict[str, Path] = {}
    duplicate_stems: set[str] = set()

    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue

        if path.stem in indexed:
            duplicate_stems.add(path.stem)
        else:
            indexed[path.stem] = path

    return indexed, tuple(sorted(duplicate_stems))


def check_image_inventory(
    train_csv: str | Path,
    jpeg_directory: str | Path,
    dicom_directory: str | Path,
    *,
    image_column: str = "image_name",
) -> ImageInventoryReport:
    """Compare CSV image identifiers with JPEG and DICOM directory contents.

    The check is read-only. File extensions are matched case-insensitively,
    and both ``.jpg`` and ``.jpeg`` are accepted for JPEG files.
    """

    train_csv = Path(train_csv)
    jpeg_directory = Path(jpeg_directory)
    dicom_directory = Path(dicom_directory)

    required_paths = (train_csv, jpeg_directory, dicom_directory)
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required paths: {missing_paths}")
    if not train_csv.is_file():
        raise FileNotFoundError(f"CSV path is not a file: {train_csv}")
    if not jpeg_directory.is_dir() or not dicom_directory.is_dir():
        raise NotADirectoryError("JPEG and DICOM paths must both be directories.")

    with train_csv.open(newline="", encoding="utf-8-sig") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        fieldnames = csv_reader.fieldnames or []
        if image_column not in fieldnames:
            raise KeyError(f"CSV does not contain an {image_column!r} column.")

        image_names = []
        for row_number, row in enumerate(csv_reader, start=2):
            image_name = (row.get(image_column) or "").strip()
            if not image_name:
                raise ValueError(f"Missing {image_column!r} on CSV row {row_number}.")
            image_names.append(image_name)

    image_name_counts = Counter(image_names)
    duplicate_csv_names = tuple(
        sorted(name for name, count in image_name_counts.items() if count > 1)
    )

    jpeg_files, duplicate_jpeg_stems = _index_files_by_stem(
        jpeg_directory,
        {".jpg", ".jpeg"},
    )
    dicom_files, duplicate_dicom_stems = _index_files_by_stem(
        dicom_directory,
        {".dcm"},
    )

    csv_names = set(image_name_counts)
    jpeg_names = set(jpeg_files)
    dicom_names = set(dicom_files)

    return ImageInventoryReport(
        csv_row_count=len(image_names),
        unique_csv_names=len(csv_names),
        jpeg_file_count=len(jpeg_files),
        dicom_file_count=len(dicom_files),
        duplicate_csv_names=duplicate_csv_names,
        missing_jpeg=tuple(sorted(csv_names - jpeg_names)),
        missing_dicom=tuple(sorted(csv_names - dicom_names)),
        orphan_jpeg=tuple(sorted(jpeg_names - csv_names)),
        orphan_dicom=tuple(sorted(dicom_names - csv_names)),
        duplicate_jpeg_stems=duplicate_jpeg_stems,
        duplicate_dicom_stems=duplicate_dicom_stems,
    )


__all__ = ["ImageInventoryReport", "check_image_inventory"]
