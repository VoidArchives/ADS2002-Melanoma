"""Patient-safe cross-validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assign_patient_folds(
    frame: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    target_col: str = "target",
    group_col: str = "patient_id",
) -> pd.DataFrame:
    """Return a copy of ``frame`` with patient-safe stratified fold labels.

    Stratification is performed on the image-level target while all rows for a
    patient are kept in the same fold. The input frame is never modified.

    Parameters
    ----------
    frame:
        Image-level training data.
    n_splits:
        Number of cross-validation folds. Must be at least two.
    random_state:
        Seed used when shuffling groups before fold assignment.
    target_col:
        Name of the binary image-level target column.
    group_col:
        Name of the patient identifier column.

    Returns
    -------
    pandas.DataFrame
        A copy of ``frame`` with a non-null ``int8`` column named ``fold``.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2.")
    if "fold" in frame.columns:
        raise ValueError("frame already contains a 'fold' column.")

    missing_columns = {target_col, group_col} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Required columns are missing: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("frame must contain at least one row.")

    targets = frame[target_col]
    if targets.isna().any():
        raise ValueError(f"{target_col!r} contains missing values.")
    if not targets.isin([0, 1]).all():
        invalid_targets = sorted(
            (value for value in targets.unique() if value not in {0, 1}),
            key=str,
        )
        raise ValueError(
            f"{target_col!r} must be binary with values 0 and 1; "
            f"found invalid values: {invalid_targets}"
        )

    groups = frame[group_col]
    if groups.isna().any():
        raise ValueError(f"{group_col!r} contains missing values.")
    blank_group = groups.astype("string").str.strip().eq("").fillna(False)
    if blank_group.any():
        raise ValueError(f"{group_col!r} contains blank values.")

    patient_targets = frame.groupby(group_col, sort=False)[target_col].max()
    patient_count = len(patient_targets)
    if patient_count < n_splits:
        raise ValueError(
            f"At least {n_splits} unique patients are required; found {patient_count}."
        )

    patient_class_counts = patient_targets.value_counts().reindex([0, 1], fill_value=0)
    if (patient_class_counts < n_splits).any():
        raise ValueError(
            f"At least {n_splits} positive and {n_splits} negative patient "
            "groups are required for stratified splitting."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_values = np.full(len(frame), fill_value=-1, dtype=np.int8)
    placeholder_features = np.zeros((len(frame), 1), dtype=np.int8)

    for fold, (_, validation_positions) in enumerate(
        splitter.split(placeholder_features, targets, groups=groups)
    ):
        fold_values[validation_positions] = fold

    if (fold_values < 0).any():
        raise RuntimeError("At least one row was not assigned to a fold.")

    result = frame.copy(deep=True)
    result["fold"] = fold_values

    patient_fold_counts = result.groupby(group_col)["fold"].nunique()
    if patient_fold_counts.gt(1).any():
        raise RuntimeError("At least one patient was assigned to multiple folds.")

    return result
