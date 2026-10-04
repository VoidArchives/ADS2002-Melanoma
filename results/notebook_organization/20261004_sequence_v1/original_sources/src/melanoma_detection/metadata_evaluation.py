"""Checks and transparent evaluation for image-level metadata predictions.

Each prediction concerns one lesion image. A patient can have several images,
so confidence intervals resample patients, keeping all their images together.
These helpers do not train models, select features, or read the reserved test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


PROBABILITY_METRICS = ("average_precision", "roc_auc", "log_loss", "brier")


def _numeric_vector(values, name: str) -> np.ndarray:
    """Reject missing, infinite, empty, or multidimensional input explicitly."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real numeric values.")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite, non-missing values.")
    return array


def _binary_vector(values, name: str) -> np.ndarray:
    array = _numeric_vector(values, name)
    # Validate before casting: converting 1.5 to an integer would hide bad data.
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must be binary with values 0 and 1.")
    return array.astype(np.int8)


def _group_codes(groups, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(groups, dtype=object)
    if array.ndim != 1 or len(array) != n_rows:
        raise ValueError("groups must be one-dimensional and match the number of predictions.")
    series = pd.Series(array)
    if series.isna().any():
        raise ValueError("groups must not contain missing patient identifiers.")
    if series.astype("string").str.strip().eq("").any():
        raise ValueError("groups must not contain blank patient identifiers.")
    try:
        codes, patients = pd.factorize(series, sort=False)
    except TypeError as exc:
        raise ValueError("groups must contain hashable patient identifiers.") from exc
    return codes, np.asarray(patients)


def validate_probabilities(y, p) -> tuple[np.ndarray, np.ndarray]:
    """Return validated target/probability vectors without silently correcting them.

    Targets must be exactly 0 or 1 before integer conversion. Probabilities must
    be finite numbers in [0, 1]. Exact 0 and 1 predictions are allowed; log loss
    uses scikit-learn's numerical clipping when computing their loss.
    """
    target = _binary_vector(y, "y")
    probability = _numeric_vector(p, "p")
    if len(target) != len(probability):
        raise ValueError("y and p must have the same length.")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("p must contain probabilities between 0 and 1.")
    return target, probability


def probability_metrics(y, p) -> dict[str, float]:
    """Measure ranking and probability quality at the image level.

    Average precision is missing (NaN) when there are no malignant images.
    ROC-AUC is missing when either class is absent. Log loss and Brier score
    remain defined for a single-class subgroup. Missing values mean the measure
    cannot be estimated, rather than that the model achieved zero performance.
    """
    target, probability = validate_probabilities(y, p)
    return {
        "average_precision": (
            float(average_precision_score(target, probability)) if target.sum() else np.nan
        ),
        "roc_auc": (
            float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else np.nan
        ),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(target, probability)),
    }


def select_recall_threshold(y, p, target_recall: float = 0.8) -> float:
    """Choose the highest score threshold reaching the requested training recall.

    Classify an image as flagged when ``p >= threshold``. Sort malignant-image
    scores from highest to lowest and keep enough malignant images to reach the
    requested recall. The last required score is the threshold. Every image
    tied at that score is flagged together, which can make recall exceed the
    requested value. Fit this threshold only on internal training predictions;
    evaluation labels must never be used to choose it.
    """
    target, probability = validate_probabilities(y, p)
    if (
        isinstance(target_recall, (bool, np.bool_))
        or not isinstance(target_recall, Real)
        or not np.isfinite(target_recall)
        or not 0 < target_recall <= 1
    ):
        raise ValueError("target_recall must be a finite number greater than 0 and at most 1.")
    positive_scores = probability[target == 1]
    if len(positive_scores) == 0:
        raise ValueError("A recall threshold requires at least one malignant image.")
    # Compare achievable recalls directly. ceil(recall * n) can incorrectly
    # require another positive because, for example, .14 * 50 exceeds 7 by a
    # floating-point rounding error even though 7/50 reaches .14 exactly.
    achieved_recalls = np.arange(1, len(positive_scores) + 1) / len(positive_scores)
    required = int(np.searchsorted(achieved_recalls, float(target_recall), side="left")) + 1
    return float(np.sort(positive_scores)[::-1][required - 1])


def decision_metrics(y, decisions) -> dict[str, float | int]:
    """Count flagged/missed images and report rates with explicit denominators.

    Sensitivity = TP/(TP+FN), specificity = TN/(TN+FP), precision = TP/(TP+FP).
    A zero denominator produces NaN, avoiding an invented rate for a subgroup
    with no malignant images, no benign images, or no flagged images.
    """
    target = _binary_vector(y, "y")
    flagged = _binary_vector(decisions, "decisions")
    if len(target) != len(flagged):
        raise ValueError("y and decisions must have the same length.")
    tp = int(((target == 1) & (flagged == 1)).sum())
    fp = int(((target == 0) & (flagged == 1)).sum())
    tn = int(((target == 0) & (flagged == 0)).sum())
    fn = int(((target == 1) & (flagged == 0)).sum())

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "sensitivity": rate(tp, tp + fn),
        "specificity": rate(tn, tn + fp),
        "precision": rate(tp, tp + fp),
        "flagged_fraction": (tp + fp) / len(target),
    }


def _ranking_cache(target: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    # A bootstrap changes image weights, not score order. Sort just once/model.
    order = np.argsort(-probability, kind="stable")
    sorted_probability = probability[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_probability)), len(probability) - 1]
    clipped = np.clip(probability, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
    return {
        "order": order,
        "target": target[order],
        "ends": ends,
        "log_losses": -(target * np.log(clipped) + (1 - target) * np.log1p(-clipped)),
        "brier_losses": (target - probability) ** 2,
    }


def _weighted_probability_metrics(cache: dict, weights: np.ndarray) -> np.ndarray:
    """Exactly match repeated-row metrics using patient multiplicity weights."""
    sorted_weights = weights[cache["order"]]
    positive = np.cumsum(sorted_weights * cache["target"])[cache["ends"]]
    total = np.cumsum(sorted_weights)[cache["ends"]]
    negative = total - positive
    recall = positive / positive[-1]
    precision = np.divide(positive, total, out=np.zeros_like(positive), where=total > 0)
    ap = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    fpr = np.r_[0.0, negative / negative[-1]]
    tpr = np.r_[0.0, recall]
    auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
    denominator = weights.sum()
    return np.array([
        ap,
        auc,
        np.dot(weights, cache["log_losses"]) / denominator,
        np.dot(weights, cache["brier_losses"]) / denominator,
    ])


def patient_bootstrap(
    y,
    groups,
    probabilities: Mapping[str, Sequence[float]],
    n_bootstrap: int = 2000,
    seed: int = 42,
    comparisons: Sequence[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Return image-level estimates and 95% patient-bootstrap percentile intervals.

    Draw the original number of patients with replacement for each replicate.
    A selected patient contributes all their images; repeated selection repeats
    all those images. The same draw evaluates every model, so candidate-minus-
    reference intervals are paired. Multiplicity weights avoid building large
    copied tables and sorting scores again for each replicate.

    Single-class replicates cannot estimate ROC-AUC and are excluded from every
    metric to keep a shared, transparent set of draws. The requested, valid, and
    rejected counts are returned. If no draw is valid, intervals are NaN. This
    can happen in a subgroup with no malignant images. Intervals describe the
    supplied predictions conditional on the fitted/selected procedure; they do
    not correct for repeatedly choosing a winner from the same development data
    and do not include retraining or random-seed variation.

    Each row has ``estimate_kind`` (absolute or difference), ``model_name``,
    ``reference_model``, ``metric``, ``estimate``, ``ci_lower``, ``ci_upper``,
    ``n_requested``, ``n_valid``, and ``n_rejected_single_class``. Difference rows
    always mean candidate minus reference; lower log loss/Brier is better.
    """
    target = _binary_vector(y, "y")
    codes, patients = _group_codes(groups, len(target))
    if len(patients) < 2:
        raise ValueError("Patient bootstrap requires at least two unique patients.")
    if not isinstance(n_bootstrap, (int, np.integer)) or isinstance(n_bootstrap, (bool, np.bool_)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not isinstance(probabilities, Mapping) or not probabilities:
        raise ValueError("probabilities must be a non-empty mapping of model names to predictions.")
    if any(not isinstance(name, str) or not name.strip() for name in probabilities):
        raise ValueError("Every model name must be a non-empty string.")
    validated = {name: validate_probabilities(target, p)[1] for name, p in probabilities.items()}
    model_names = list(validated)
    pairs = list(comparisons) if comparisons is not None else []
    for pair in pairs:
        if (
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or any(not isinstance(name, str) or name not in validated for name in pair)
        ):
            raise ValueError("Each comparison must contain existing candidate and reference model names.")

    caches = [_ranking_cache(target, validated[name]) for name in model_names]
    estimates = {name: probability_metrics(target, validated[name]) for name in model_names}
    draws = np.empty((n_bootstrap, len(model_names), len(PROBABILITY_METRICS)))
    rng = np.random.default_rng(seed)
    valid = 0
    for _ in range(n_bootstrap):
        counts = np.bincount(rng.integers(len(patients), size=len(patients)), minlength=len(patients))
        weights = counts[codes].astype(np.float64)
        positive_weight = np.dot(weights, target)
        if positive_weight == 0 or positive_weight == weights.sum():
            continue
        for model_index, cache in enumerate(caches):
            draws[valid, model_index] = _weighted_probability_metrics(cache, weights)
        valid += 1
    draws = draws[:valid]

    rows = []

    def add_row(model: str, reference: str | None, metric_index: int, samples: np.ndarray):
        metric = PROBABILITY_METRICS[metric_index]
        estimate = estimates[model][metric]
        if reference is not None:
            estimate -= estimates[reference][metric]
        lower, upper = np.quantile(samples, [0.025, 0.975]) if len(samples) else (np.nan, np.nan)
        rows.append({
            "estimate_kind": "absolute" if reference is None else "difference",
            "model_name": model,
            "reference_model": reference,
            "metric": metric,
            "estimate": estimate,
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "n_requested": n_bootstrap,
            "n_valid": valid,
            "n_rejected_single_class": n_bootstrap - valid,
        })

    for index, name in enumerate(model_names):
        for metric_index in range(len(PROBABILITY_METRICS)):
            add_row(name, None, metric_index, draws[:, index, metric_index])
    for candidate, reference in pairs:
        difference = draws[:, model_names.index(candidate)] - draws[:, model_names.index(reference)]
        for metric_index in range(len(PROBABILITY_METRICS)):
            add_row(candidate, reference, metric_index, difference[:, metric_index])
    return pd.DataFrame(rows)


def calibration_bins(y, p, n_bins: int = 5, groups=None) -> pd.DataFrame:
    """Summarize predicted and observed risk in approximate equal-count bins.

    Bins use empirical probability quantiles. Boundaries are shared by all equal
    scores, so tied predictions are never split arbitrarily and fewer than the
    requested bins can result. The observed rate is malignant images / images;
    it is descriptive and must not be interpreted as a patient malignancy rate.
    Optional patient counts show how much independent support each bin has. A
    patient's different images may occur in several bins.
    """
    target, probability = validate_probabilities(y, p)
    if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, (bool, np.bool_)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    if groups is not None:
        _group_codes(groups, len(target))
    boundaries = np.unique(np.quantile(probability, np.linspace(0, 1, n_bins + 1)[1:-1]))
    labels = np.searchsorted(boundaries, probability, side="left")
    frame = pd.DataFrame({"bin": labels, "target": target, "probability": probability})
    if groups is not None:
        frame["patient_id"] = np.asarray(groups)
    result = frame.groupby("bin", sort=True).agg(
        n_images=("target", "size"),
        n_malignant_images=("target", "sum"),
        mean_probability=("probability", "mean"),
        observed_rate=("target", "mean"),
        min_probability=("probability", "min"),
        max_probability=("probability", "max"),
    )
    if groups is not None:
        result["n_patients"] = frame.groupby("bin")["patient_id"].nunique()
    result = result.reset_index(drop=True)
    result.insert(0, "bin", np.arange(1, len(result) + 1))
    return result


def subgroup_metrics(data: pd.DataFrame, probabilities, decisions, site_col: str | None = None) -> pd.DataFrame:
    """Describe age, sex, and anatomical-site performance with support counts.

    Required columns: target, patient_id, age_approx, sex, and anatomical site.
    The default site column is anatom_site_general_challenge (used by d1), or
    anatom_site_general when the challenge column is absent. Set site_col to
    choose an explicit source column.
    Age groups are under 50, 50 or over, and missing. Sex/site retain their source
    values, with missing values labelled Missing. Each image is evaluated in its
    own subgroup. A patient can contribute to multiple subgroups, so subgroup
    counts must not be summed to estimate the total number of patients. This
    descriptive table does not tune the model or threshold on subgroup outcomes.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if site_col is None:
        site_col = "anatom_site_general_challenge" if "anatom_site_general_challenge" in data.columns else "anatom_site_general"
    required = {"target", "patient_id", "age_approx", "sex", site_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Required subgroup columns are missing: {sorted(missing)}")
    target, probability = validate_probabilities(data["target"], probabilities)
    flagged = _binary_vector(decisions, "decisions")
    if len(flagged) != len(target):
        raise ValueError("decisions must match the number of data rows.")
    _group_codes(data["patient_id"], len(target))
    try:
        age = pd.to_numeric(data["age_approx"], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    except (ValueError, TypeError) as exc:
        raise ValueError("age_approx must contain numeric ages or missing values.") from exc
    if np.isinf(age).any() or (age[np.isfinite(age)] < 0).any():
        raise ValueError("age_approx must not contain infinite or negative ages.")
    category_groups = {
        "age": np.where(np.isnan(age), "Missing", np.where(age < 50, "Under 50", "50 or over")),
        "sex": data["sex"].astype("string").fillna("Missing").to_numpy(),
        "site": data[site_col].astype("string").fillna("Missing").to_numpy(),
    }
    patients = data["patient_id"].to_numpy()
    rows = []
    for attribute, categories in category_groups.items():
        for subgroup in pd.unique(categories):
            mask = categories == subgroup
            rows.append({
                "attribute": attribute,
                "subgroup": subgroup,
                "n_images": int(mask.sum()),
                "n_patients": int(pd.Series(patients[mask]).nunique()),
                "n_malignant_images": int(target[mask].sum()),
                "n_positive_patients": int(pd.Series(patients[mask & (target == 1)]).nunique()),
                **probability_metrics(target[mask], probability[mask]),
                **decision_metrics(target[mask], flagged[mask]),
            })
    return pd.DataFrame(rows)
