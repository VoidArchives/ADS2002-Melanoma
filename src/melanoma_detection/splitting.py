"""Stratified patient-level cross-validation (prevents patient leakage)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .utils import RANDOM_STATE

SPLIT_ORDER = ["train", "validation"]


def patient_level_split(
    patient_level: pd.DataFrame,
    seed: int = RANDOM_STATE,
    n_splits: int = 3,
    patient_col: str = "patient_id",
    label_col: str = "has_malignant",
) -> list[dict]:
    """Return one patient-to-split mapping per stratified fold.

    Every patient validates once and trains in the other ``n_splits - 1`` folds.
    Three folds give approximately 67/33 training/validation in each run.
    Initialise a fresh model and optimiser for each mapping.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2.")

    target = patient_level.set_index(patient_col)[label_col]
    if not target.index.is_unique:
        raise ValueError(f"{patient_col} must be unique in the patient table.")
    if target.index.isna().any() or target.isna().any():
        raise ValueError("Patient IDs and patient labels must not be missing.")
    if target.empty or target.nunique() < 2 or target.value_counts().min() < n_splits:
        raise ValueError("At least two classes and n_splits patients per class are required.")

    # Plain numpy arrays: pandas' Arrow-backed string indexes break sklearn's indexing.
    ids = target.index.to_numpy()
    labels = target.to_numpy()
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        {
            **{p: "train" for p in ids[train_indices]},
            **{p: "validation" for p in ids[validation_indices]},
        }
        for train_indices, validation_indices in folds.split(ids, labels)
    ]


def iter_patient_folds(df: pd.DataFrame, split_maps: list[dict], patient_col: str = "patient_id"):
    """Yield ``(fold_number, image_table)`` with one-based fold numbers."""
    for fold, split_map in enumerate(split_maps, start=1):
        assigned = assign_splits(df, split_map, patient_col=patient_col)
        assert_no_patient_overlap(assigned, patient_col=patient_col)
        yield fold, assigned


def assign_splits(df: pd.DataFrame, split_map: dict, patient_col: str = "patient_id") -> pd.DataFrame:
    """Return a copy of ``df`` with a ``split`` column mapped from the patient IDs."""
    out = df.copy()
    out["split"] = out[patient_col].map(split_map)
    if out["split"].isna().any():
        raise ValueError("Some patients have no split assignment.")
    return out



def assert_no_patient_overlap(
    df: pd.DataFrame, patient_col: str = "patient_id", split_col: str = "split"
) -> None:
    """Raise if any patient appears in more than one split."""
    per_patient = df.groupby(patient_col)[split_col].nunique()
    leaked = per_patient[per_patient > 1]
    if not leaked.empty:
        raise AssertionError(f"{len(leaked)} patient(s) appear in more than one split: {list(leaked.index[:5])}")


def summarise_splits(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    image_col: str = "image_name",
    target_col: str = "target",
) -> pd.DataFrame:
    """Patients, images and melanoma counts/percent per split (image-level table)."""
    summary = (
        df.groupby("split")
        .agg(patients=(patient_col, "nunique"), images=(image_col, "size"), melanoma=(target_col, "sum"))
        .reindex(SPLIT_ORDER)
    )
    summary["non_melanoma"] = summary["images"] - summary["melanoma"]
    # Cast to float first so int8 targets cannot overflow.
    summary["melanoma_percent"] = summary["melanoma"].astype(float).div(summary["images"]).mul(100)
    return summary


def save_splits(split_map: dict, path: str | Path) -> Path:
    """Write the patient-to-split assignment to CSV so every notebook shares it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(split_map, name="split").rename_axis("patient_id").reset_index().to_csv(path, index=False)
    return path


def load_splits(path: str | Path) -> dict:
    """Read a split file written by :func:`save_splits` back into a dict."""
    table = pd.read_csv(path)
    return dict(zip(table["patient_id"], table["split"]))


def load_or_create_patient_folds(df: pd.DataFrame, directory: str | Path, seed: int = RANDOM_STATE) -> list[dict]:
    """Reuse notebook 02's three folds, or create them for an older cleaned table."""
    paths = [Path(directory) / f"splits_fold_{fold}.csv" for fold in range(1, 4)]
    if all(path.exists() for path in paths):
        return [load_splits(path) for path in paths]
    if any(path.exists() for path in paths):
        raise ValueError("Incomplete fold files: rerun the patient splitting notebook.")
    if df["patient_id"].isna().any():
        raise ValueError("Patient IDs must not be missing.")
    if "split" in df and df["split"].eq("test").any():
        raise ValueError("Remove the held-out test set before creating folds.")
    patients = df.groupby("patient_id", as_index=False)["target"].max()
    folds = patient_level_split(patients, seed=seed, label_col="target")
    for mapping, path in zip(folds, paths):
        save_splits(mapping, path)
    return folds
