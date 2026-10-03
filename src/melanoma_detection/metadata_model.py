"""Fixed metadata baseline and training-only threshold/calibration procedures.

The outcome is lesion malignancy. Patient IDs define validation groups and are
never predictors. No function in this module opens a reserved test dataset.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

from melanoma_detection.metadata_evaluation import select_recall_threshold, validate_probabilities
from melanoma_detection.metadata_validation import validate_metadata_frame


FEATURES = ["age_approx", "sex", "anatom_site_general_challenge"]
SPECIFICATION = {
    "model": "fixed_spline_logistic", "features": FEATURES,
    "age_knots": 4, "age_degree": 3, "age_knot_placement": "quantile",
    "logistic_C": 1.0, "class_weight": None, "seed": 42,
    "primary_probability": "raw", "threshold_rule": "highest cutoff attaining training OOF recall",
    "target_recall": 0.8, "threshold_purpose": "academic sensitivity scenario; not a clinical target",
    "calibration_comparator": "logistic recalibration of training-only OOF logits, C=1.0",
    "metric_unit": "image/lesion", "validation_and_bootstrap_unit": "patient",
}


def file_sha256(path: str | Path) -> str:
    """Fingerprint the exact input bytes used by an experiment."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_age(values):
    """Convert the validated numeric/Missing age field for training imputation."""
    return pd.DataFrame(values).apply(pd.to_numeric, errors="coerce")


def categorical_values(values):
    """Make missing sex/site an explicit category; learn levels during fit only."""
    return pd.DataFrame(values).fillna("Missing").astype(str)


def build_spline_logistic() -> Pipeline:
    """Build the existing four-knot cubic baseline without parameter searching."""
    age = Pipeline([
        ("numeric", FunctionTransformer(numeric_age, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=4, degree=3, knots="quantile", include_bias=False)),
        ("scaler", StandardScaler()),
    ])
    categories = Pipeline([
        ("missing", FunctionTransformer(categorical_values, feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("age", age, ["age_approx"]),
        ("categories", categories, ["sex", "anatom_site_general_challenge"]),
    ])
    return Pipeline([
        ("preprocessor", preprocessing),
        ("classifier", LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)),
    ])


def probability_logits(probabilities) -> np.ndarray:
    """Convert probabilities to a finite score, clipping only at numerical ends."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("Probabilities must be finite, one-dimensional and between zero and one.")
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_recalibrator(targets, probabilities) -> LogisticRegression:
    """Fit a two-coefficient sigmoid on OOF scores, never fitted training scores.

    C=1 provides a fixed modest penalty because positive cases are scarce.
    This is logistic recalibration on logits, not sklearn's exact Platt smoother.
    """
    targets, probabilities = validate_probabilities(targets, probabilities)
    if np.unique(targets).size != 2:
        raise ValueError("Recalibration needs both classes.")
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(probability_logits(probabilities), targets)
    if calibrator.coef_[0, 0] <= 0:
        raise ValueError("Recalibration would reverse or erase ranking; investigate the training scores.")
    return calibrator


def predict_recalibrated(calibrator, probabilities) -> np.ndarray:
    return calibrator.predict_proba(probability_logits(probabilities))[:, 1]


def grouped_oof(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Hold out each existing patient fold once within the supplied data.

    For an outer training set there are four remaining folds: fit on three,
    predict the fourth. Each patient's OOF score comes from a model that has
    never seen that patient. Preprocessing is fitted inside that same model.
    """
    probabilities = np.full(len(frame), np.nan)
    records = []
    for fold in sorted(frame["fold"].unique()):
        heldout = frame["fold"].eq(fold).to_numpy()
        training = ~heldout
        training_patients = set(frame.loc[training, "patient_id"])
        validation_patients = set(frame.loc[heldout, "patient_id"])
        if not training_patients.isdisjoint(validation_patients):
            raise ValueError("A patient crossed an OOF boundary.")
        if frame.loc[training, "target"].nunique() != 2 or frame.loc[heldout, "target"].nunique() != 2:
            raise ValueError("Each OOF training and validation partition needs both classes.")
        estimator = build_spline_logistic()
        estimator.fit(frame.loc[training, FEATURES], frame.loc[training, "target"])
        probabilities[heldout] = estimator.predict_proba(frame.loc[heldout, FEATURES])[:, 1]
        records.append({
            "inner_heldout_fold": int(fold), "fit_patients": len(training_patients),
            "validation_patients": len(validation_patients), "patient_overlap": 0,
            "fit_positive_images": int(frame.loc[training, "target"].sum()),
            "validation_positive_images": int(frame.loc[heldout, "target"].sum()),
        })
    validate_probabilities(frame["target"], probabilities)
    return probabilities, pd.DataFrame(records)


def evaluate_outer_fold(frame: pd.DataFrame, fold: int, target_recall: float = 0.8):
    """Evaluate complete threshold and calibration procedures on untouched patients.

    Outer labels are carried to the result table but never passed to any fit,
    calibration fit or threshold selection. Both raw and recalibrated procedures
    are specified in advance and reported; this function chooses no winner.
    """
    frame = validate_metadata_frame(frame)
    validation = frame["fold"].eq(fold)
    if not validation.any():
        raise ValueError("Requested outer fold is absent.")
    fitting = frame.loc[~validation].reset_index(drop=True)
    heldout = frame.loc[validation].reset_index(drop=True)
    if not set(fitting.patient_id).isdisjoint(heldout.patient_id):
        raise ValueError("A patient crossed an outer boundary.")
    inner_probabilities, audit = grouped_oof(fitting)
    calibrator = fit_recalibrator(fitting.target, inner_probabilities)
    raw_threshold = select_recall_threshold(fitting.target, inner_probabilities, target_recall)
    calibrated_inner = predict_recalibrated(calibrator, inner_probabilities)
    calibrated_threshold = select_recall_threshold(fitting.target, calibrated_inner, target_recall)
    estimator = build_spline_logistic().fit(fitting[FEATURES], fitting.target)
    raw_probabilities = estimator.predict_proba(heldout[FEATURES])[:, 1]
    calibrated_probabilities = predict_recalibrated(calibrator, raw_probabilities)
    result = heldout[["image_name", "patient_id", "target", "fold"]].copy()
    result["p_spline_raw"] = raw_probabilities
    result["p_spline_calibrated"] = calibrated_probabilities
    result["raw_threshold"] = raw_threshold
    result["calibrated_threshold"] = calibrated_threshold
    result["decision_raw"] = (raw_probabilities >= raw_threshold).astype(int)
    result["decision_calibrated"] = (calibrated_probabilities >= calibrated_threshold).astype(int)
    audit.insert(0, "outer_fold", fold)
    audit["outer_validation_patients"] = heldout.patient_id.nunique()
    audit["calibrator_slope"] = float(calibrator.coef_[0, 0])
    audit["calibrator_intercept"] = float(calibrator.intercept_[0])
    return result, audit


def fit_final_bundle(frame: pd.DataFrame, *, input_sha256: str, target_recall: float = 0.8) -> dict:
    """Fit the frozen raw baseline on all development patients, without test data.

    The threshold is chosen on five-fold development OOF scores. Recalibration
    remains an explicitly secondary comparator; the primary model stays raw.
    This training step is not a final test evaluation.
    """
    frame = validate_metadata_frame(frame)
    oof_probabilities, _ = grouped_oof(frame)
    estimator = build_spline_logistic().fit(frame[FEATURES], frame.target)
    threshold = select_recall_threshold(frame.target, oof_probabilities, target_recall)
    return {
        "schema_version": 1, "model": estimator,
        "specification": {**SPECIFICATION, "target_recall": target_recall},
        "threshold": threshold, "input_sha256": input_sha256,
        "training_patient_ids": sorted(frame.patient_id.astype(str).unique()),
        "training_image_ids": sorted(frame.image_name.astype(str).unique()),
    }
