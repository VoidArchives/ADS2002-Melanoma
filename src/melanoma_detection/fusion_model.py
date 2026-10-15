"""Two-score logistic stacking of existing patient-level OOF predictions.

Second-stage CV is exploratory: base predictions are not regenerated inside
each outer fold. Use an untouched test set for a final independent evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression

from .evaluation import compare_models
from .splitting import iter_patient_folds
from .utils import RANDOM_STATE


def _require_columns(table: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}.")


def _validate_rows(table: pd.DataFrame, name: str, *, both_classes: bool = False) -> None:
    _require_columns(table, ["image_name", "target"], name)
    if table.empty or table.image_name.isna().any() or not table.image_name.is_unique:
        raise ValueError(f"{name}: image_name must be non-missing, unique and non-empty.")
    if not table.target.isin([0, 1]).all():
        raise ValueError(f"{name}: target must contain only binary labels 0 and 1.")
    if both_classes and table.target.nunique() != 2:
        raise ValueError(f"{name}: both target classes are required.")
    if "patient_id" in table and table.patient_id.isna().any():
        raise ValueError(f"{name}: patient_id must not be missing.")


def _probabilities(values: ArrayLike, name: str) -> NDArray[np.float64]:
    scores = np.asarray(values, dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"{name}: probabilities must be finite and in [0, 1].")
    return scores


def safe_logit(p: ArrayLike, eps: float = 1e-6) -> NDArray[np.float64]:
    """Validate probabilities, then clip endpoints and return finite log-odds."""
    if not 0 < eps < 0.5:
        raise ValueError("eps must be strictly between 0 and 0.5.")
    clipped = np.clip(_probabilities(p, "safe_logit"), eps, 1 - eps)
    return np.log(clipped) - np.log1p(-clipped)


def prepare_fusion_data(
    image_oof: pd.DataFrame,
    metadata_oof: pd.DataFrame,
    clean_images: pd.DataFrame | None = None,
    *,
    metadata_score_col: str = "p_nn_mean",
    include_spline: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Validate and inner-join OOF scores; return matched rows and a full ID audit.

    Missing image patient IDs are recovered only from the supplied clean_images
    table. Unmatched IDs are reported and returned; conflicting labels/IDs raise.
    """
    _require_columns(image_oof, ["melanoma_score"], "Image OOF")
    _require_columns(metadata_oof, ["patient_id", metadata_score_col], "Metadata OOF")
    for name, table in (("Image OOF", image_oof), ("Metadata OOF", metadata_oof)):
        _validate_rows(table, name)
        if "fold" in table:
            if table.fold.isna().any():
                raise ValueError(f"{name}: fold must not be missing.")
            if "patient_id" in table and table.groupby("patient_id").fold.nunique().gt(1).any():
                raise ValueError(f"{name}: a patient appears in multiple base-model folds.")

    image_cols = ["image_name", "target", "melanoma_score"]
    image_cols += [c for c in ("patient_id", "fold") if c in image_oof]
    images = image_oof[image_cols].rename(
        columns={"melanoma_score": "image_score", "fold": "image_fold"})
    metadata_cols = ["image_name", "patient_id", "target", metadata_score_col]
    metadata_cols += ["fold"] if "fold" in metadata_oof else []
    if include_spline:
        _require_columns(metadata_oof, ["p_spline"], "Metadata OOF")
        if "p_spline" not in metadata_cols:
            metadata_cols.append("p_spline")
    metadata = metadata_oof[metadata_cols].rename(
        columns={metadata_score_col: "metadata_score", "fold": "metadata_fold"})
    images["image_score"] = _probabilities(images.image_score, "image_score")
    metadata["metadata_score"] = _probabilities(metadata.metadata_score, "metadata_score")
    if "p_spline" in metadata:
        metadata["p_spline"] = _probabilities(metadata.p_spline, "p_spline")

    if "patient_id" not in images:
        if clean_images is None:
            raise ValueError("Image OOF has no patient_id; supply the clean_images lookup table.")
        _require_columns(clean_images, ["image_name", "patient_id"], "Clean images")
        lookup = clean_images[["image_name", "patient_id"]]
        if lookup.isna().any().any() or not lookup.image_name.is_unique:
            raise ValueError("Clean images: IDs must be non-missing and image_name unique.")
        images = images.merge(lookup, on="image_name", how="left", validate="one_to_one", sort=False)
        if images.patient_id.isna().any():
            missing = images.loc[images.patient_id.isna(), "image_name"].tolist()
            raise ValueError(f"No clean_images patient ID for image IDs: {missing[:10]}.")
    if "image_fold" in images and images.groupby("patient_id").image_fold.nunique().gt(1).any():
        raise ValueError("Image OOF: a patient appears in multiple base-model folds.")

    only_image = images.loc[~images.image_name.isin(metadata.image_name), "image_name"].tolist()
    only_metadata = metadata.loc[~metadata.image_name.isin(images.image_name), "image_name"].tolist()
    matched = images.merge(metadata, on="image_name", how="inner", validate="one_to_one",
                           suffixes=("", "_metadata"), sort=False)
    print(f"Image OOF rows: {len(images)}\nMetadata OOF rows: {len(metadata)}\n"
          f"Matched fusion rows: {len(matched)}\nOnly in image model: {len(only_image)}\n"
          f"Only in metadata model: {len(only_metadata)}")
    for name, ids in (("image", only_image), ("metadata", only_metadata)):
        if ids:
            print(f"Only in {name} (first 10 IDs): {ids[:10]}; full list in audit.")
    for column in ("target", "patient_id"):
        mismatch = matched[column].ne(matched[f"{column}_metadata"])
        if mismatch.any():
            ids = matched.loc[mismatch, "image_name"].tolist()
            raise ValueError(f"{column} mismatch between OOF sources for image IDs: {ids[:10]}.")
    matched = matched.drop(columns=["target_metadata", "patient_id_metadata"])
    _validate_rows(matched, "Matched fusion data", both_classes=True)
    audit = {
        "Image OOF rows": len(images), "Metadata OOF rows": len(metadata),
        "Matched rows": len(matched), "Matched patients": matched.patient_id.nunique(),
        "Matched melanoma positives": int(matched.target.sum()),
        "Unmatched image rows": len(only_image), "Unmatched metadata rows": len(only_metadata),
        "only_image_ids": only_image, "only_metadata_ids": only_metadata,
    }
    return matched, audit


def build_fusion_features(
    rows: pd.DataFrame, *, image_score_col: str = "image_score",
    metadata_score_col: str = "metadata_score", eps: float = 1e-6,
) -> pd.DataFrame:
    """Build exactly two logit features, preserving the input row order/index."""
    _require_columns(rows, [image_score_col, metadata_score_col], "Fusion features")
    return pd.DataFrame({
        "image_logit": safe_logit(rows[image_score_col], eps),
        "metadata_logit": safe_logit(rows[metadata_score_col], eps),
    }, index=rows.index)


@dataclass
class FusionModel:
    """Serializable estimator plus the complete score-to-feature recipe."""

    estimator: LogisticRegression
    eps: float
    image_score_col: str
    metadata_score_col: str
    random_state: int
    feature_names: tuple[str, str] = ("image_logit", "metadata_logit")
    source_paths: dict[str, str] = field(default_factory=dict)


def fit_fusion_model(
    rows: pd.DataFrame, *, image_score_col: str = "image_score",
    metadata_score_col: str = "metadata_score", eps: float = 1e-6,
    random_state: int = RANDOM_STATE,
) -> FusionModel:
    """Fit a fixed C=1 logistic stacker on two model scores."""
    _validate_rows(rows, "Fusion training data", both_classes=True)
    features = build_fusion_features(rows, image_score_col=image_score_col,
                                     metadata_score_col=metadata_score_col, eps=eps)
    estimator = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000,
                                   random_state=random_state).fit(features, rows.target)
    return FusionModel(estimator, eps, image_score_col, metadata_score_col, random_state)


def predict_fusion_model(model: FusionModel, rows: pd.DataFrame) -> NDArray[np.float64]:
    """Predict using the saved transform; inference needs only the two scores."""
    features = build_fusion_features(rows, image_score_col=model.image_score_col,
                                     metadata_score_col=model.metadata_score_col, eps=model.eps)
    scores = model.estimator.predict_proba(features.loc[:, list(model.feature_names)])[:, 1]
    return _probabilities(scores, "fusion_score")


def evaluate_fusion_models(
    rows: pd.DataFrame, score_columns: Mapping[str, str], threshold: float = 0.5,
) -> pd.DataFrame:
    """Use shared evaluation helpers on exactly the same images for every model."""
    _validate_rows(rows, "Evaluation data", both_classes=True)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")
    _require_columns(rows, list(score_columns.values()), "Evaluation data")
    predictions = {}
    for name, column in score_columns.items():
        _probabilities(rows[column], column)
        predictions[name] = rows[["image_name", "target", column]].rename(columns={column: "score"})
    return compare_models(predictions, rows.set_index("image_name").target, threshold).rename(
        columns={"recall": "Sensitivity", "specificity": "Specificity",
                 "average_recall": "Balanced accuracy"})


def validate_fusion_predictions(
    predictions: pd.DataFrame, reference: pd.DataFrame, split_maps: list[dict],
) -> None:
    """Require complete aligned OOF output and exactly one validation per patient."""
    _validate_rows(reference, "Fusion reference", both_classes=True)
    _validate_rows(predictions, "Fusion OOF", both_classes=True)
    _require_columns(reference, ["patient_id"], "Fusion reference")
    _require_columns(predictions, ["patient_id", "fusion_score", "fusion_fold"], "Fusion OOF")
    if len(split_maps) != 3:
        raise ValueError("Exactly three saved patient-fold mappings are required.")
    if len(predictions) != len(reference) or set(predictions.image_name) != set(reference.image_name):
        raise ValueError("Fusion OOF must predict every matched image exactly once.")
    expected = reference.set_index("image_name").loc[predictions.image_name]
    for column in ("target", "patient_id"):
        if not np.array_equal(predictions[column].to_numpy(), expected[column].to_numpy()):
            raise ValueError(f"Fusion OOF {column} values differ from the matched reference.")
    for column in ("image_score", "metadata_score", "p_spline", "fusion_score"):
        if column in predictions:
            _probabilities(predictions[column], column)
    if predictions.fusion_fold.isna().any() or predictions.groupby("patient_id").fusion_fold.nunique().gt(1).any():
        raise ValueError("Each patient must have exactly one fusion-validation fold.")
    counts = np.zeros(len(predictions), dtype=int)
    for fold, rows in iter_patient_folds(predictions, split_maps):
        if not rows.split.isin(["train", "validation"]).all():
            raise ValueError(f"Invalid split labels in saved patient fold {fold}.")
        validation = rows.split.eq("validation").to_numpy()
        if not np.array_equal(validation, predictions.fusion_fold.eq(fold).to_numpy()):
            raise ValueError(f"Fusion OOF assignments disagree with saved patient fold {fold}.")
        counts += validation
        assert set(rows.loc[~validation, "patient_id"]).isdisjoint(rows.loc[validation, "patient_id"])
    if not np.all(counts == 1):
        raise ValueError("Every matched image must receive exactly one fusion OOF prediction.")


def cross_validate_fusion_model(
    rows: pd.DataFrame, split_maps: list[dict], *, metadata_score_col: str = "metadata_score",
    eps: float = 1e-6, random_state: int = RANDOM_STATE,
    return_models: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, dict[int, FusionModel]]:
    """Predict held-out patients; optionally return the exact fitted fold models."""
    _validate_rows(rows, "Fusion CV data", both_classes=True)
    _require_columns(rows, ["patient_id"], "Fusion CV data")
    if len(split_maps) != 3:
        raise ValueError("Exactly three saved patient-fold mappings are required.")
    assignments = pd.DataFrame([rows.patient_id.map(mapping).to_numpy() for mapping in split_maps])
    if (not assignments.isin(["train", "validation"]).all().all()
            or not assignments.eq("validation").sum(axis=0).eq(1).all()):
        raise ValueError("Every matched patient must validate once and train twice in saved folds.")
    # Validate every fold before fitting; iter_patient_folds also asserts patient isolation.
    folds = list(iter_patient_folds(rows.reset_index(drop=True), split_maps))
    for fold, assigned in folds:
        for split in ("train", "validation"):
            _validate_rows(assigned.loc[assigned.split.eq(split)], f"Fold {fold} {split}", both_classes=True)
    output = rows.reset_index(drop=True).copy()
    output["fusion_score"] = np.nan
    output["fusion_fold"] = 0
    counts = np.zeros(len(output), dtype=int)
    metrics = []
    fold_models = {}
    for fold, assigned in folds:
        training = assigned.loc[assigned.split.eq("train")]
        validation = assigned.loc[assigned.split.eq("validation")]
        assert set(training.patient_id).isdisjoint(validation.patient_id), "Patient leakage"
        model = fit_fusion_model(training, metadata_score_col=metadata_score_col,
                                eps=eps, random_state=random_state)
        fold_models[fold] = model
        output.loc[validation.index, "fusion_score"] = predict_fusion_model(model, validation)
        output.loc[validation.index, "fusion_fold"] = fold
        counts[validation.index] += 1
        fold_metrics = evaluate_fusion_models(output.loc[validation.index], {"Fusion": "fusion_score"}).iloc[0]
        metrics.append({"fusion_fold": fold, "train_patients": training.patient_id.nunique(),
                        "validation_patients": validation.patient_id.nunique(),
                        "validation_images": len(validation), **fold_metrics.to_dict()})
    assert np.all(counts == 1), "Each matched image must be predicted exactly once"
    validate_fusion_predictions(output, rows, split_maps)
    if return_models:
        return output, pd.DataFrame(metrics).set_index("fusion_fold"), fold_models
    return output, pd.DataFrame(metrics).set_index("fusion_fold")


def fit_final_fusion_model(
    rows: pd.DataFrame, *, metadata_score_col: str = "metadata_score", eps: float = 1e-6,
    random_state: int = RANDOM_STATE, source_paths: Mapping[str, str] | None = None,
) -> FusionModel:
    """Fit all development OOF rows for future test inference, never OOF evaluation."""
    model = fit_fusion_model(rows, metadata_score_col=metadata_score_col,
                            eps=eps, random_state=random_state)
    model.source_paths = dict(source_paths or {})
    return model
