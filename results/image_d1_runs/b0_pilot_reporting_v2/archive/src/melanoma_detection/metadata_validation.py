"""Checks that stop malformed labels or unsafe patient folds before modelling.

Validation happens before integer conversion. Otherwise a value such as 1.5
can silently become 1 and look like a valid malignant label.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Iterable

import numpy as np
import pandas as pd


METADATA_FEATURES = ("age_approx", "sex", "anatom_site_general_challenge")


def validate_binary_target(values: pd.Series, name: str = "target") -> pd.Series:
    """Return 0/1 labels as ``int8``, rejecting bad original values first.

    Numeric strings (for example ``"0"`` or ``"1.0"``) are accepted. Missing,
    non-numeric, infinite, and fractional labels are rejected, rather than
    filled, rounded, or truncated. The input, index, and Series name are kept.
    """
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if values.empty:
        raise ValueError(f"{name} must contain at least one label.")
    if values.isna().any():
        raise ValueError(f"{name} contains missing labels.")
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only numeric 0/1 labels.") from error
    if pd.api.types.is_complex_dtype(numeric.dtype):
        raise ValueError(f"{name} must contain real numeric 0/1 labels.")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains missing or non-finite labels.")
    if not numeric.isin([0, 1]).all():
        raise ValueError(f"{name} must contain exactly 0 or 1 before integer conversion.")

    # Numeric conversion may round a high-precision string to 0 or 1. Check
    # strings and Decimal values at their original precision as well.
    for value in values:
        if isinstance(value, (str, Decimal)):
            try:
                exact = Decimal(value.strip() if isinstance(value, str) else value)
            except InvalidOperation as error:
                raise ValueError(f"{name} must contain only numeric 0/1 labels.") from error
            if not exact.is_finite() or exact not in (Decimal(0), Decimal(1)):
                raise ValueError(f"{name} must contain exactly 0 or 1 before integer conversion.")
    return numeric.astype("int8")


def _validate_identifiers(values: pd.Series, name: str) -> None:
    """Require cleaned identifiers so grouping and joining use the same keys."""
    text = values.astype("string")
    if values.isna().any() or text.str.strip().eq("").any():
        raise ValueError(f"{name} contains missing or blank identifiers.")
    if text.ne(text.str.strip()).any():
        raise ValueError(f"{name} contains identifiers with surrounding whitespace.")


def validate_metadata_frame(
    frame: pd.DataFrame,
    *,
    required_features: Iterable[str] = METADATA_FEATURES,
    expected_folds: Iterable[int] = (0, 1, 2, 3, 4),
    require_folds: bool = True,
) -> pd.DataFrame:
    """Validate development metadata and return a copy with checked labels.

    Each row describes one image. ``image_name`` must be unique; ``patient_id``
    may repeat because a patient can have several images. When folds are
    supplied, every patient must belong to one fold and every expected fold
    must contain both outcomes. This prevents an image from one patient being
    used to train a model that evaluates another image from the same patient.

    The default contract is the existing five-fold d1 development dataset.
    Set ``require_folds=False`` for a cleaned training table before splitting.
    Age may be numeric, missing, or the cleaned labels ``Unknown``/``Missing``;
    known ages must be finite and within 0 to 120 years. Feature imputation is
    left to the training pipeline, so it can learn from training rows only.

    This function deliberately validates development labels. It is not a
    reason to load the reserved test labels during model development.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("frame must contain at least one row.")
    if not frame.columns.is_unique:
        raise ValueError("frame contains duplicate column names.")
    features = tuple(required_features)
    required = {"image_name", "patient_id", "target", *features}
    if require_folds:
        required.add("fold")
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Required metadata columns are missing: {sorted(missing_columns)}")

    validated = frame.copy()
    for identifier in ("image_name", "patient_id"):
        _validate_identifiers(validated[identifier], identifier)
    if not validated["image_name"].astype("string").is_unique:
        raise ValueError("image_name contains duplicate image identifiers.")
    validated["target"] = validate_binary_target(validated["target"])

    if "age_approx" in features:
        ages = validated["age_approx"]
        age_text = ages.astype("string").str.strip().str.casefold()
        missing_age = ages.isna() | age_text.isin(["unknown", "missing"])
        try:
            known_ages = pd.to_numeric(ages.loc[~missing_age], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError("age_approx contains a malformed age value.") from error
        if pd.api.types.is_complex_dtype(known_ages.dtype):
            raise ValueError("age_approx must contain real ages.")
        if not np.isfinite(known_ages.to_numpy(dtype=float)).all():
            raise ValueError("age_approx contains a non-finite age value.")
        if not known_ages.between(0, 120).all():
            raise ValueError("age_approx contains values outside 0 to 120 years.")

    if "fold" in validated.columns:
        expected = tuple(expected_folds)
        if (
            not expected
            or len(set(expected)) != len(expected)
            or any(isinstance(fold, bool) or not isinstance(fold, (int, np.integer)) for fold in expected)
            or any(fold < 0 for fold in expected)
        ):
            raise ValueError("expected_folds must contain distinct non-negative integers.")
        try:
            folds = pd.to_numeric(validated["fold"], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError("fold must contain integer fold labels.") from error
        if pd.api.types.is_complex_dtype(folds.dtype):
            raise ValueError("fold must contain real integer fold labels.")
        if folds.isna().any() or not np.isfinite(folds.to_numpy(dtype=float)).all():
            raise ValueError("fold contains missing or non-finite labels.")
        if not folds.mod(1).eq(0).all() or set(folds) != set(expected):
            raise ValueError(f"fold must contain exactly the expected integer labels: {list(expected)}.")
        validated["fold"] = folds.astype("int64")
        if not validated.groupby("patient_id", dropna=False)["fold"].nunique().eq(1).all():
            raise ValueError("A patient occurs in more than one fold.")
        if not validated.groupby("fold")["target"].nunique().eq(2).all():
            raise ValueError("Every fold must contain both benign and malignant images.")
    return validated
