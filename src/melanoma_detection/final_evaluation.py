"""Held-out cohort validation and patient-cluster uncertainty for frozen models."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .evaluation import compute_metrics, confusion_counts


def validate_test_cohort(test: pd.DataFrame, development: pd.DataFrame) -> None:
    """Reject invalid labels, duplicate images and development/test patient overlap."""
    for name, rows in (("Test", test), ("Development", development)):
        required = {"image_name", "patient_id", "target"}
        if not required.issubset(rows.columns):
            raise ValueError(f"{name}: missing columns {sorted(required - set(rows.columns))}.")
        if rows.empty or rows[["image_name", "patient_id", "target"]].isna().any().any():
            raise ValueError(f"{name}: IDs/labels must be non-empty and non-missing.")
        if not rows.image_name.is_unique or not rows.target.isin([0, 1]).all():
            raise ValueError(f"{name}: require unique images and binary labels.")
    if test.target.nunique() != 2:
        raise ValueError("Test metrics require both target classes.")
    if set(test.patient_id) & set(development.patient_id):
        raise ValueError("Patient leakage: test patients overlap development patients.")
    if set(test.image_name) & set(development.image_name):
        raise ValueError("Test image IDs overlap development image IDs.")


def align_test_predictions(
    predictions: pd.DataFrame, reference: pd.DataFrame, score_columns: list[str],
) -> pd.DataFrame:
    """Require complete test coverage, matching IDs/labels and valid probabilities."""
    needed = {"image_name", *score_columns}
    if not needed.issubset(predictions.columns):
        raise ValueError(f"Predictions missing columns: {sorted(needed - set(predictions.columns))}.")
    if not predictions.image_name.is_unique or set(predictions.image_name) != set(reference.image_name):
        raise ValueError("Predictions must cover every test image exactly once; no inner join is allowed.")
    aligned = predictions.set_index("image_name").loc[reference.image_name].reset_index()
    for column in ("patient_id", "target"):
        if column in aligned and not np.array_equal(aligned[column], reference[column]):
            raise ValueError(f"Test predictions disagree on {column}.")
        aligned[column] = reference[column].to_numpy()
    values = aligned[score_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Test probabilities must be finite and in [0, 1].")
    return aligned


def patient_bootstrap_report(
    rows: pd.DataFrame, thresholds: Mapping[str, float], *,
    n_boot: int = 1000, seed: int = 42, confidence: float = 0.95,
) -> pd.DataFrame:
    """Image-level metrics with percentile CIs from resampling whole patients.

    Thresholds stay fixed in every resample. One-class resamples are skipped;
    undefined precision/F1 samples are excluded per metric and counted explicitly.
    The same seed pairs patient resamples across models on the same cohort.
    """
    if n_boot < 2 or not 0 < confidence < 1 or not thresholds:
        raise ValueError("Require n_boot >= 2, confidence in (0, 1), and thresholds.")
    rows = align_test_predictions(rows, rows, ["score"])
    if rows.patient_id.isna().any() or not rows.target.isin([0, 1]).all() or rows.target.nunique() != 2:
        raise ValueError("Require patient IDs and both binary target classes.")
    y, scores = rows.target.to_numpy(), rows.score.to_numpy()
    groups = rows.groupby("patient_id", sort=True).indices
    members = list(groups.values())
    rng = np.random.default_rng(seed)
    resamples = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[i] for i in chosen])
        if np.unique(y[indices]).size == 2:
            resamples.append(indices)
    if not resamples:
        raise ValueError("No bootstrap sample contains both target classes.")
    names = {"ROC-AUC": "ROC_AUC", "Average precision": "AP",
             "Sensitivity / recall": "Sensitivity", "Specificity": "Specificity",
             "average_recall": "Balanced accuracy", "Precision": "Precision", "F1": "F1"}
    quantiles = [(1 - confidence) * 50, (1 + confidence) * 50]
    reports = []
    for policy, threshold in thresholds.items():
        point = compute_metrics(y, scores, threshold)
        samples = pd.DataFrame([compute_metrics(y[idx], scores[idx], threshold) for idx in resamples])
        report = {"policy": policy, "threshold": float(threshold), "images": len(rows),
                  "patients": len(members), "bootstrap_requested": n_boot,
                  "bootstrap_two_class": len(resamples), "confidence": confidence,
                  **{k.upper(): v for k, v in confusion_counts(y, scores >= threshold).items()}}
        for source, label in names.items():
            valid = samples[source].to_numpy()
            valid = valid[np.isfinite(valid)]
            lower, upper = np.percentile(valid, quantiles) if len(valid) else (np.nan, np.nan)
            report.update({label: point[source], f"{label}_lower": lower,
                           f"{label}_upper": upper, f"{label}_n_valid": len(valid)})
        reports.append(report)
    return pd.DataFrame(reports)
