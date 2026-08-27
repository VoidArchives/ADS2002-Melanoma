"""Metadata-only baseline models and leakage-safe evaluation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_METADATA_FEATURES = (
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
)


def build_logistic_metadata_pipeline(
    feature_columns: Sequence[str] = DEFAULT_METADATA_FEATURES,
    *,
    class_weight: str | dict[int, float] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build the deterministic metadata-only logistic-regression baseline.

    All supplied features are treated as categorical and one-hot encoded inside
    the pipeline. Keeping preprocessing inside the estimator ensures categories
    are learned from each training fold rather than from validation data. The
    default unweighted fit preserves the observed class prevalence; callers can
    pass ``class_weight="balanced"`` for an explicit imbalance-aware variant.
    """

    features = tuple(feature_columns)
    if not features:
        raise ValueError("feature_columns must contain at least one feature.")
    if len(features) != len(set(features)):
        raise ValueError("feature_columns must not contain duplicates.")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(features),
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = LogisticRegression(
        C=1.0,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=1_000,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def generate_oof_probabilities(
    frame: pd.DataFrame,
    estimator: BaseEstimator,
    feature_columns: Sequence[str] = DEFAULT_METADATA_FEATURES,
    *,
    target_col: str = "target",
    fold_col: str = "fold",
    group_col: str = "patient_id",
    prediction_name: str = "probability",
) -> pd.Series:
    """Generate out-of-fold positive-class probabilities aligned to ``frame``.

    The estimator is cloned and refitted once per fold. A patient-safety check
    rejects input where a group appears in multiple validation folds.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict_proba"):
        raise TypeError("estimator must implement fit and predict_proba.")

    features = tuple(feature_columns)
    required_columns = {*features, target_col, fold_col, group_col}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Required columns are missing: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("frame must contain at least one row.")
    if not features:
        raise ValueError("feature_columns must contain at least one feature.")
    if frame[list(features)].isna().any().any():
        raise ValueError("Metadata features contain missing values.")

    targets = frame[target_col]
    if targets.isna().any() or not targets.isin([0, 1]).all():
        raise ValueError(f"{target_col!r} must contain only binary values 0 and 1.")
    if frame[fold_col].isna().any():
        raise ValueError(f"{fold_col!r} contains missing values.")
    if frame[group_col].isna().any():
        raise ValueError(f"{group_col!r} contains missing values.")

    fold_values = sorted(frame[fold_col].unique().tolist())
    if len(fold_values) < 2:
        raise ValueError("At least two folds are required for OOF prediction.")
    if frame.groupby(group_col)[fold_col].nunique().gt(1).any():
        raise ValueError("At least one patient appears in multiple folds.")

    probabilities = np.full(len(frame), np.nan, dtype=float)
    for fold in fold_values:
        validation_mask = frame[fold_col].eq(fold).to_numpy()
        training_mask = ~validation_mask
        training_targets = targets.iloc[np.flatnonzero(training_mask)]
        if training_targets.nunique() != 2:
            raise ValueError(f"Training data for fold {fold!r} does not contain both classes.")

        fold_estimator = clone(estimator)
        fold_estimator.fit(
            frame.iloc[np.flatnonzero(training_mask)][list(features)],
            training_targets,
        )
        validation_features = frame.iloc[np.flatnonzero(validation_mask)][list(features)]
        class_labels = list(fold_estimator.classes_)
        if 1 not in class_labels:
            raise RuntimeError(f"Estimator for fold {fold!r} has no positive class.")
        positive_class_position = class_labels.index(1)
        probabilities[validation_mask] = fold_estimator.predict_proba(
            validation_features
        )[:, positive_class_position]

    if np.isnan(probabilities).any():
        raise RuntimeError("At least one row did not receive an OOF prediction.")

    return pd.Series(probabilities, index=frame.index, name=prediction_name)


def binary_probability_metrics(
    targets: Sequence[int] | pd.Series,
    probabilities: Sequence[float] | pd.Series,
) -> dict[str, float]:
    """Calculate discrimination and probability-quality metrics."""

    target_array = np.asarray(targets)
    probability_array = np.asarray(probabilities, dtype=float)
    if target_array.shape != probability_array.shape:
        raise ValueError("targets and probabilities must have the same shape.")
    if target_array.size == 0:
        raise ValueError("targets and probabilities must not be empty.")
    if not np.isin(target_array, [0, 1]).all():
        raise ValueError("targets must contain only binary values 0 and 1.")
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must be finite.")
    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("probabilities must lie between 0 and 1.")
    if np.unique(target_array).size != 2:
        raise ValueError("targets must contain both classes.")

    return {
        "roc_auc": float(roc_auc_score(target_array, probability_array)),
        "average_precision": float(
            average_precision_score(target_array, probability_array)
        ),
        "log_loss": float(log_loss(target_array, probability_array, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target_array, probability_array)),
    }
