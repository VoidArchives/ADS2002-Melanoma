"""Inspectable inner-validation uncertainty diagnostics; no fitting or selection.

Prediction scores are sigmoid outputs, not established calibrated probabilities.
The primary referral comparison holds deterministic scores fixed across MC and
TTA rankings. An explicit per-mode-mean sensitivity analysis is also available.
Descending score is a malignant-risk referral baseline; entropy prioritises
scores near 0.5. Neither is an estimate of epistemic uncertainty.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .metadata_evaluation import validate_probabilities
from .metadata_validation import validate_binary_target


DEFAULT_FRACTIONS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
RANKINGS = ("uncertainty_sd", "prediction_score", "predictive_entropy")


def _identifier(values, name):
    text = values.astype("string")
    if values.isna().any() or text.str.strip().eq("").any():
        raise ValueError(f"{name} contains missing or blank identifiers.")
    if text.ne(text.str.strip()).any():
        raise ValueError(f"{name} contains surrounding whitespace.")
    return text.astype(str)


def _frame(frame, required, name):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame.")
    if frame.empty or not frame.columns.is_unique or not set(required).issubset(frame.columns):
        raise ValueError(f"{name} must be nonempty with unique columns including {list(required)}.")
    result = frame.copy()
    for column in ("image_id", "patient_id"):
        result[column] = _identifier(result[column], column)
    result["target"] = validate_binary_target(result["target"])
    return result


def _pass_ids(values):
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, int, np.integer)):
            raise ValueError("pass_ids must be nonblank strings or integers.")
        value = str(value)
        if not value.strip() or value != value.strip():
            raise ValueError("pass_ids must be nonblank strings or integers without surrounding whitespace.")
        result.append(value)
    return result


def validate_raw(raw: pd.DataFrame, expected: pd.DataFrame, pass_ids: Sequence) -> pd.DataFrame:
    """Validate every prediction and the exact image/pass Cartesian product.

    Raw rows require image_id, patient_id, target, pass_id and score. Expected
    metadata require image_id, patient_id and target, with one row per image.
    Integer and string pass identifiers share their string representation, so
    0 and '0' cannot accidentally count as distinct passes. No inner join can
    silently remove missing images, metadata conflicts or unexpected passes.
    Returns a copy, ordered by expected image order and declared pass order.
    """
    metadata = _frame(expected, ("image_id", "patient_id", "target"), "expected")
    predictions = _frame(raw, ("image_id", "patient_id", "target", "pass_id", "score"), "raw")
    passes = _pass_ids(pass_ids)
    if not passes or len(set(passes)) != len(passes):
        raise ValueError("Expected pass_ids must be nonempty and unique after normalisation.")
    if metadata.image_id.duplicated().any():
        raise ValueError("Expected image_id values must be unique.")
    predictions["pass_id"] = _pass_ids(predictions.pass_id)
    _, predictions["score"] = validate_probabilities(predictions.target, predictions.score)
    if predictions.duplicated(["image_id", "pass_id"]).any():
        raise ValueError("Duplicate image/pass prediction keys.")
    if set(predictions.image_id) != set(metadata.image_id):
        raise ValueError("Raw predictions do not have complete exact expected image coverage.")
    if set(predictions.pass_id) != set(passes):
        raise ValueError("Raw predictions have missing or unexpected passes.")
    if len(predictions) != len(metadata) * len(passes):
        raise ValueError("Raw predictions do not have complete expected passes for every image.")
    # Uniqueness + exact image/pass sets + Cartesian-product row count proves
    # complete per-image coverage; additionally reconcile metadata before use.
    lookup = metadata.set_index("image_id")
    for column in ("patient_id", "target"):
        if not np.array_equal(predictions[column].to_numpy(), lookup.loc[predictions.image_id, column].to_numpy()):
            raise ValueError(f"Raw prediction {column} disagrees with expected metadata.")
    image_order = {value: index for index, value in enumerate(metadata.image_id)}
    pass_order = {value: index for index, value in enumerate(passes)}
    predictions["_image_order"] = predictions.image_id.map(image_order)
    predictions["_pass_order"] = predictions.pass_id.map(pass_order)
    return predictions.sort_values(["_image_order", "_pass_order"]).drop(
        columns=["_image_order", "_pass_order"]).reset_index(drop=True)


def aggregate_raw(raw: pd.DataFrame, expected: pd.DataFrame, pass_ids: Sequence) -> pd.DataFrame:
    """Validate first, then calculate image means and sample SD (ddof=1).

    A single deterministic pass has SD=0 by convention; its value is not an
    uncertainty estimate. No inference mode is combined with another mode.
    """
    checked = validate_raw(raw, expected, pass_ids)
    return checked.groupby("image_id", sort=False).agg(
        patient_id=("patient_id", "first"), target=("target", "first"),
        mean_score=("score", "mean"), score_sd=("score", lambda x: x.std(ddof=1) if len(x) > 1 else 0.),
        n_passes=("score", "size"),
    ).reset_index()


def predictive_entropy(scores) -> np.ndarray:
    """Binary entropy in natural-log units; 0/1 have exactly zero entropy."""
    _, p = validate_probabilities(np.zeros(len(scores)), scores)
    result = np.zeros_like(p)
    interior = (p > 0) & (p < 1)
    q = p[interior]
    result[interior] = -(q * np.log(q) + (1 - q) * np.log1p(-q))
    return result


def convergence_summary(raw, expected, pass_ids, smaller_pass_counts=(5, 10, 20)) -> pd.DataFrame:
    """Compare nested pass prefixes with all passes, never generate new draws.

    Spearman correlation is undefined for a constant SD vector and stays NaN.
    Top-20% overlap uses ceil(0.2*n_images), resolving ties by image identifier.
    These checks diagnose Monte Carlo stability, not predictive correctness.
    """
    passes = _pass_ids(pass_ids)
    checked = validate_raw(raw, expected, passes)
    complete = aggregate_raw(checked, expected, passes).set_index("image_id")
    rows = []
    k = max(1, int(np.ceil(.2 * len(complete))))
    full_top = set(complete.reset_index().sort_values(["score_sd", "image_id"], ascending=[False, True]).image_id.iloc[:k])
    for count in smaller_pass_counts:
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not 1 <= count < len(passes):
            raise ValueError("smaller_pass_counts must be positive integers below the full pass count.")
        subset = checked.loc[checked.pass_id.isin(passes[:count])]
        prefix = aggregate_raw(subset, expected, passes[:count]).set_index("image_id").loc[complete.index]
        prefix_top = set(prefix.reset_index().sort_values(["score_sd", "image_id"], ascending=[False, True]).image_id.iloc[:k])
        correlation = (prefix.score_sd.corr(complete.score_sd, method="spearman")
                       if prefix.score_sd.nunique() > 1 and complete.score_sd.nunique() > 1 else np.nan)
        rows.append({"n_passes": count, "reference_passes": len(passes),
                     "mean_score_mae": float((prefix.mean_score - complete.mean_score).abs().mean()),
                     "score_sd_mae": float((prefix.score_sd - complete.score_sd).abs().mean()),
                     "sd_spearman": correlation, "top20_overlap_fraction": len(prefix_top & full_top) / k})
    return pd.DataFrame(rows)


def _validate_summary(summary):
    required = ("image_id", "patient_id", "target", "base_score", "mc_mean_score", "mc_score_sd", "tta_mean_score", "tta_score_sd")
    result = _frame(summary, required, "summary")
    if result.image_id.duplicated().any():
        raise ValueError("Summary image_id values must be unique.")
    for column in ("base_score", "mc_mean_score", "tta_mean_score"):
        _, result[column] = validate_probabilities(result.target, result[column])
    for column in ("mc_score_sd", "tta_score_sd"):
        try:
            values = result[column].to_numpy(dtype=float)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{column} must contain finite, nonnegative SD values.") from error
        if np.iscomplexobj(result[column]) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} must contain finite, nonnegative SD values.")
        result[column] = values
    return result.sort_values("image_id").reset_index(drop=True)


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _configuration(fractions, n_random, seed):
    fractions = np.asarray(fractions, dtype=float)
    if (fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all()
            or (fractions < 0).any() or (fractions >= 1).any() or len(np.unique(fractions)) != len(fractions)):
        raise ValueError("fractions must be distinct finite numbers in [0,1), leaving retained images.")
    _positive_integer(n_random, "n_random")
    _positive_integer(seed, "seed", minimum=0)
    return fractions


def _score_column(method, score_source):
    if score_source not in ("base", "mean"):
        raise ValueError("score_source must be 'base' (common deterministic scores) or 'mean' (per-mode sensitivity).")
    return "base_score" if score_source == "base" else f"{method}_mean_score"


def _rankings(frame, method, score_source):
    score = frame[_score_column(method, score_source)].to_numpy()
    # Stable order uses lexicographically sorted image IDs; repeated bootstrap
    # occurrences retain their occurrence order. The ranking never reads labels.
    return {"uncertainty_sd": np.argsort(-frame[f"{method}_score_sd"].to_numpy(), kind="stable"),
            "prediction_score": np.argsort(-score, kind="stable"),
            "predictive_entropy": np.argsort(-predictive_entropy(score), kind="stable")}


def _referral_values(frame, score, order, n_referred):
    referred = order[:n_referred]
    retained = order[n_referred:]
    target = frame.target.to_numpy()
    patients = frame.patient_id.to_numpy()
    malignant = target == 1
    n_positive = int(target.sum())
    positive_patients = set(patients[malignant])
    referred_positive_patients = set(patients[referred][malignant[referred]])
    return {"n_images": len(frame), "n_patients": frame.patient_id.nunique(),
            "n_malignant_images": n_positive, "n_malignant_patients": len(positive_patients),
            "n_referred": n_referred, "achieved_referral_fraction": n_referred / len(frame),
            "n_retained": len(retained), "n_patients_referred": len(set(patients[referred])),
            "n_patients_retained": len(set(patients[retained])),
            "retained_error": float(((score[retained] >= .5) != target[retained]).mean()),
            "retained_brier": float(((score[retained] - target[retained]) ** 2).mean()),
            "retained_score_mae": float(np.abs(score[retained] - target[retained]).mean()),
            "n_malignant_images_referred": int(target[referred].sum()),
            "malignant_referred_fraction": float(target[referred].sum() / n_positive) if n_positive else np.nan,
            "n_malignant_patients_referred": len(referred_positive_patients),
            "malignant_patients_referred_fraction": len(referred_positive_patients) / len(positive_patients) if positive_patients else np.nan}


def evaluate_referral(summary, fractions=DEFAULT_FRACTIONS, n_random=100, seed=42,
                      score_source="base") -> pd.DataFrame:
    """Compare separate MC/TTA rankings using common underlying base scores.

    Image referral uses floor(fraction*n_images). Label-independent ties use
    image_id. The default base_score provides the ordinary score/entropy
    rankings and retained-image errors for BOTH uncertainty modes. This holds
    predictions fixed, so differences reflect referral ranking alone. Explicit
    score_source='mean' instead uses each mode's own mean_score for a separate
    sensitivity analysis; do not compare those errors across modes as ranking
    differences alone. All methods classify using the fixed score >=0.5;
    this cutoff is descriptive and never
    selected using validation labels. Random rows contain every seeded repeat,
    enabling mean and between-seed intervals rather than one lucky baseline.
    Patient counts describe images referred, not whole-patient referral; a
    patient can appear in both retained and referred sets. A malignant patient
    is one with >=1 malignant image. Zero malignant denominators stay NaN.
    """
    frame = _validate_summary(summary)
    fractions = _configuration(fractions, n_random, seed)
    random_orders = [(seed + index, np.random.default_rng(seed + index).permutation(len(frame))) for index in range(n_random)]
    rows = []
    for method in ("mc", "tta"):
        score_column = _score_column(method, score_source)
        score = frame[score_column].to_numpy()
        orders = [(name, None, order) for name, order in _rankings(frame, method, score_source).items()]
        orders += [("random", random_seed, order) for random_seed, order in random_orders]
        for ranking, random_seed, order in orders:
            for fraction in fractions:
                rows.append({"method": method, "ranking": ranking, "random_seed": random_seed,
                             "referral_fraction": fraction, "score_column": score_column,
                             "fixed_score_cutoff": .5,
                             **_referral_values(frame, score, order, int(np.floor(fraction * len(frame))))})
    return pd.DataFrame(rows)


def _bootstrap_metrics(frame, fractions, random_orders, score_source):
    """Vectorised cumulative errors/positives; no fixed original referral set."""
    n = len(frame)
    cutoffs = np.floor(fractions * n).astype(int)
    positive = frame.target.to_numpy()
    total_positive = positive.sum()
    result = np.empty((2, 4, len(fractions), 2))
    for index, method in enumerate(("mc", "tta")):
        errors = ((frame[_score_column(method, score_source)].to_numpy() >= .5) != positive).astype(int)
        orders = list(_rankings(frame, method, score_source).values())
        for ranking, order in enumerate(orders):
            error_cumulative = np.r_[0, np.cumsum(errors[order])]
            positive_cumulative = np.r_[0, np.cumsum(positive[order])]
            result[index, ranking, :, 0] = (errors.sum() - error_cumulative[cutoffs]) / (n - cutoffs)
            result[index, ranking, :, 1] = positive_cumulative[cutoffs] / total_positive if total_positive else np.nan
        error_cumulative = np.concatenate([np.zeros((len(random_orders), 1)), np.cumsum(errors[random_orders], axis=1)], axis=1)
        positive_cumulative = np.concatenate([np.zeros((len(random_orders), 1)), np.cumsum(positive[random_orders], axis=1)], axis=1)
        result[index, 3, :, 0] = ((errors.sum() - error_cumulative[:, cutoffs]) / (n - cutoffs)).mean(axis=0)
        result[index, 3, :, 1] = (positive_cumulative[:, cutoffs] / total_positive).mean(axis=0) if total_positive else np.nan
    return result


def referral_patient_bootstrap(summary, fractions=DEFAULT_FRACTIONS, n_bootstrap=2000,
                               n_random=100, seed=42, score_source="base", referral_seed=None) -> pd.DataFrame:
    """Paired 95% patient-bootstrap intervals for SD minus each comparator.

    Resample the original number of patients with replacement and repeat ALL
    images of each draw. Recompute rankings, floor referral counts and random
    permutations for each replicate; the same patient draw and random orders
    serve both inference modes. The default uses common deterministic base_score
    for score/entropy ranking and error, matching evaluate_referral. Explicit
    score_source='mean' is a separate per-mode sensitivity analysis. The random
    baseline is the mean of n_random
    seeded rankings, not one draw. In each replicate random seeds derive from
    [seed, replicate, 104729], separate from the patient-sampling RNG. The
    point estimate uses referral_seed + [0..n_random-1], exactly matching
    evaluate_referral(seed=referral_seed). If omitted, referral_seed=seed.

    Negative retained-error differences favour SD; positive malignant-referred
    differences favour SD. If a bootstrap draw has no malignant images its
    malignant fraction is undefined (NaN), while retained error stays valid.
    Counts report valid/undefined draws per metric. These intervals condition
    on the checkpoint, selected epoch/recipe and existing inference draws.
    They omit retraining and selection uncertainty and do not validate a
    clinical referral policy or correct repeated pilot comparisons.
    """
    frame = _validate_summary(summary)
    fractions = _configuration(fractions, n_random, seed)
    _positive_integer(n_bootstrap, "n_bootstrap")
    referral_seed = seed if referral_seed is None else referral_seed
    _positive_integer(referral_seed, "referral_seed", minimum=0)
    patients = frame.patient_id.unique()
    if len(patients) < 2:
        raise ValueError("Patient bootstrap requires at least two unique patients.")
    groups = [np.flatnonzero(frame.patient_id.to_numpy() == patient) for patient in patients]
    original_random = np.stack([np.random.default_rng(referral_seed + index).permutation(len(frame)) for index in range(n_random)])
    absolute = _bootstrap_metrics(frame, fractions, original_random, score_source)
    estimate = absolute[:, :1] - absolute[:, 1:]
    draws = np.empty((n_bootstrap, 2, 3, len(fractions), 2))
    patient_rng = np.random.default_rng(seed)
    for replicate in range(n_bootstrap):
        selected = patient_rng.integers(len(patients), size=len(patients))
        indices = np.concatenate([groups[index] for index in selected])
        sample = frame.iloc[indices].sort_values("image_id", kind="stable").reset_index(drop=True)
        random_rng = np.random.default_rng(np.random.SeedSequence([seed, replicate, 104729]))
        orders = random_rng.permuted(np.tile(np.arange(len(sample)), (n_random, 1)), axis=1)
        metrics = _bootstrap_metrics(sample, fractions, orders, score_source)
        draws[replicate] = metrics[:, :1] - metrics[:, 1:]
    rows = []
    for method_index, method in enumerate(("mc", "tta")):
        for comparator_index, comparator in enumerate((*RANKINGS[1:], "random")):
            for fraction_index, fraction in enumerate(fractions):
                for metric_index, metric in enumerate(("retained_error", "malignant_referred_fraction")):
                    values = draws[:, method_index, comparator_index, fraction_index, metric_index]
                    valid = values[np.isfinite(values)]
                    lower, upper = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
                    rows.append({"method": method, "ranking": "uncertainty_sd", "comparator": comparator,
                                 "score_column": _score_column(method, score_source),
                                 "referral_fraction": fraction, "metric": metric,
                                 "estimate": estimate[method_index, comparator_index, fraction_index, metric_index],
                                 "ci_lower": lower, "ci_upper": upper, "n_requested": n_bootstrap,
                                 "n_valid": len(valid), "n_undefined": n_bootstrap - len(valid),
                                 "n_random_per_replicate": n_random, "bootstrap_seed": seed,
                                 "referral_seed": referral_seed})
    return pd.DataFrame(rows)
