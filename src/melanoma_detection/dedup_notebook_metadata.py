"""Detailed, inference-only clean metadata notebook sections and plot helpers.

All estimators come from verified saved folds. No helper fits a model, changes a
dataset, chooses a training recipe or opens the reserved test.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

from .dedup_metadata import verify_saved_receipt
from .dedup_metadata_suite import verify_suite_receipt
from .historical_metadata_artifacts import verify_historical_suite
from .metadata_evaluation import (calibration_bins, decision_metrics, probability_metrics,
                                 _ranking_cache, _weighted_probability_metrics)

LABELS = {
    "dummy": "Training-prevalence dummy", "categorical_logistic": "Categorical-age logistic",
    "spline_raw": "Fixed spline", "random_forest": "Random forest", "xgboost": "XGBoost",
    "spline_tuned": "Nested-tuned spline", "random_forest_tuned": "Nested-tuned forest",
    "xgboost_tuned": "Nested-tuned XGBoost", "metadata_nn_mean": "Fixed three-seed NN mean",
    "nn_seed_42": "NN seed 42", "nn_seed_142": "NN seed 142", "nn_seed_242": "NN seed 242",
    "spline_calibrated": "Training-only recalibrated spline",
    "ablation_sex": "Sex only", "ablation_age": "Age only", "ablation_site": "Site only",
    "ablation_sex_age": "Sex + age", "ablation_sex_site": "Sex + site",
    "ablation_age_site": "Age + site", "ablation_all_features": "All three features",
}
COLORS = ["#2563eb", "#d97706", "#0f766e", "#9333ea", "#dc2626", "#64748b"]


def saved_metadata(root, bundle):
    root = Path(root).resolve()
    base = root / "results/dedup_v1/metadata"
    suite = root / "results/dedup_v1/metadata_suite/metadata_full_dedup_v1"
    core = json.loads((base / "comparator_receipt.json").read_text())
    verify_saved_receipt(root, core)
    receipt = json.loads((suite / "receipt.json").read_text())
    verify_historical_suite(root, receipt)
    if core["dataset_identity"] != bundle["identity"] or receipt["dataset_identity"] != bundle["identity"]:
        raise ValueError("Saved metadata diagnostics belong to another dataset.")
    expected = bundle["expected"].rename(columns={"image_id": "image_name"})
    calibration = pd.read_csv(base / "metadata_calibrated_oof.csv", float_precision="round_trip")
    calibration = calibration.set_index("image_name").loc[expected.image_name].reset_index()
    for column in ("patient_id", "fold", "target"):
        if not np.array_equal(calibration[column], expected[column]):
            raise ValueError("Saved calibration records do not match clean scoring records.")
    if not np.allclose(calibration.p_spline_raw, bundle["scores"]["spline_raw"], rtol=0, atol=1e-14):
        raise ValueError("Saved calibration raw scores differ from the fixed comparator.")
    for scale in ("raw", "calibrated"):
        score = "p_spline_raw" if scale == "raw" else "p_spline_calibrated"
        if not np.array_equal(calibration[score].ge(calibration[f"{scale}_threshold"]).astype(int), calibration[f"decision_{scale}"]):
            raise ValueError("Saved training-only decisions disagree with their cutoffs.")
    return {"root": root, "base": base, "suite": suite, "core": core,
            "receipt": receipt, "calibration": calibration}


def metric_table(frame, scores, names):
    rows = []
    for name in names:
        if name not in scores:
            continue
        values = np.asarray(scores[name])
        fold_metrics = []
        for fold in sorted(frame.fold.unique()):
            mask = frame.fold.eq(fold).to_numpy()
            metrics = probability_metrics(frame.loc[mask, "target"], values[mask])
            fold_metrics.append(metrics)
            rows.append({"model": name, "scope": "fold", "fold": fold, **metrics})
        rows.append({"model": name, "scope": "mean_fold", "fold": None,
                     **{key: np.mean([record[key] for record in fold_metrics]) for key in fold_metrics[0]}})
        rows.append({"model": name, "scope": "pooled", "fold": None,
                     **probability_metrics(frame.target, values)})
    return pd.DataFrame(rows)


def curve_figure(frame, scores, names, title, *, pr_limit=None):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), layout="constrained")
    for index, name in enumerate(names):
        values = scores[name]
        fpr, tpr, _ = roc_curve(frame.target, values)
        precision, recall, _ = precision_recall_curve(frame.target, values)
        metrics = probability_metrics(frame.target, values)
        color = COLORS[index % len(COLORS)]
        axes[0].plot(fpr, tpr, color=color, label=f"{LABELS.get(name, name)} (AUC {metrics['roc_auc']:.3f})")
        axes[1].plot(recall, precision, color=color, label=f"{LABELS.get(name, name)} (AP {metrics['average_precision']:.4f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#64748b", label="Chance ranking")
    axes[1].axhline(frame.target.mean(), linestyle="--", color="#64748b", label=f"Prevalence {frame.target.mean():.2%}")
    axes[0].set(title="Pooled held-out ROC", xlabel="Benign images falsely flagged (fraction)", ylabel="Malignant images found (recall)", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(title="Pooled held-out precision–recall", xlabel="Malignant images found (recall)", ylabel="Malignant fraction among flagged images", xlim=(0, 1), ylim=(0, 1 if pr_limit is None else pr_limit))
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(title + "; pooled curves are secondary to mean-fold scores", fontsize=11)
    return fig


def exploratory_cutoff(target, scores):
    """Replicate the original OOF-balanced-accuracy illustration efficiently."""
    target, scores = np.asarray(target), np.asarray(scores, dtype=float)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("Exploratory balanced accuracy requires both classes.")
    candidates = np.unique(np.r_[0., scores, 1.])
    positive, negative = np.sort(scores[target == 1]), np.sort(scores[target == 0])
    recall = (len(positive) - np.searchsorted(positive, candidates, side="left")) / len(positive)
    specificity = np.searchsorted(negative, candidates, side="left") / len(negative)
    balanced = (recall + specificity) / 2
    best = np.flatnonzero(np.isclose(balanced, balanced.max()))
    selected = best[np.argmin(np.abs(candidates[best] - .5))]
    return float(candidates[selected])


def cutoff_table(frame, scores, names, *, exploratory=False):
    rows = []
    for name in names:
        cutoff = exploratory_cutoff(frame.target, scores[name]) if exploratory and name != "dummy" else .5
        metrics = decision_metrics(frame.target, np.asarray(scores[name]) >= cutoff)
        rows.append({"model": name, "cutoff": cutoff,
                     "role": "OOF-selected illustration; selection and assessment labels overlap" if exploratory and name != "dummy" else "Fixed descriptive cutoff",
                     "balanced_accuracy": (metrics["sensitivity"] + metrics["specificity"]) / 2, **metrics})
    return pd.DataFrame(rows)


def confusion_figure(target, scenarios, title):
    """Each scenario is (label, fixed saved decisions); never tunes a cutoff."""
    columns = min(3, len(scenarios))
    rows = int(np.ceil(len(scenarios) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 4.2 * rows), squeeze=False, layout="constrained")
    matrices = [confusion_matrix(target, decision, labels=[0, 1]) for _, decision in scenarios]
    maximum = max(matrix.max() for matrix in matrices)
    for axis, (label, _), matrix in zip(axes.flat, scenarios, matrices):
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        percentages = matrix / matrix.sum(axis=1, keepdims=True) * 100
        for row in range(2):
            for column in range(2):
                code = [["TN", "FP"], ["FN", "TP"]][row][column]
                axis.text(column, row, f"{code}\n{matrix[row, column]:,}\n{percentages[row, column]:.1f}% of class",
                          ha="center", va="center", color="white" if matrix[row, column] > maximum / 2 else "#0f172a", fontsize=11)
        axis.set(title=label, xticks=[0, 1], xticklabels=["Left unflagged", "Flagged"],
                 yticks=[0, 1], yticklabels=["Benign", "Malignant"], xlabel="Decision", ylabel="Observed class")
    for axis in list(axes.flat)[len(scenarios):]:
        axis.axis("off")
    fig.suptitle(title + "; percentages are within the observed class", fontsize=12)
    return fig


def metric_bar_figure(table, names, title):
    selected = table.loc[table.scope.eq("mean_fold")].set_index("model").loc[names]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    positions = np.arange(len(names))
    for axis, metric, label in zip(axes, ("average_precision", "roc_auc"), ("Average precision", "ROC-AUC")):
        values = selected[metric].to_numpy()
        axis.bar(positions, values, color=[COLORS[index % len(COLORS)] for index in positions])
        axis.set(xticks=positions, xticklabels=[LABELS.get(name, name) for name in names], ylabel=label,
                 title="Mean outer-fold " + label, ylim=(0, max(values) * 1.28))
        axis.tick_params(axis="x", labelrotation=18)
        axis.grid(axis="y", alpha=.2)
        for position, value in zip(positions, values):
            axis.text(position, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(title)
    return fig


def estimator_diagnostics(meta):
    coefficients, importances, profiles = [], [], []
    ages = np.arange(15, 91)
    for fold in range(5):
        models = joblib.load(meta["suite"] / f"fold_{fold}/models.joblib")["estimators"]
        logistic = models["categorical_logistic"]
        for feature, value in zip(logistic.named_steps["preprocessor"].get_feature_names_out(), logistic.named_steps["classifier"].coef_[0]):
            coefficients.append({"fold": fold, "encoded_feature": feature, "logit_coefficient": value})
        for name in ("random_forest", "xgboost"):
            estimator = models[name]
            names = estimator.named_steps["preprocessor"].get_feature_names_out()
            values = estimator.named_steps["classifier"].feature_importances_
            for feature in ("sex", "age_approx", "anatom_site_general_challenge"):
                mask = np.array([value.startswith(f"categorical__{feature}_") for value in names])
                importances.append({"fold": fold, "model": name, "feature": feature,
                                    "importance": float(np.sum(values[mask]))})
        fixed = joblib.load(meta["base"] / f"fold_{fold}/models.joblib")["spline"]
        for sex in ("female", "male"):
            profile = pd.DataFrame({"age_approx": ages, "sex": sex, "anatom_site_general_challenge": "torso"})
            values = fixed.predict_proba(profile[["age_approx", "sex", "anatom_site_general_challenge"]])[:, 1]
            profiles.extend({"fold": fold, "sex": sex, "age": age, "probability": probability}
                            for age, probability in zip(ages, values))
    return pd.DataFrame(coefficients), pd.DataFrame(importances), pd.DataFrame(profiles)


def patient_draws(frame, repetitions=2000, seed=42):
    """Whole-patient multiplicities independently sampled within outer folds."""
    patients = frame[["patient_id", "fold"]].drop_duplicates().sort_values("patient_id")
    if not patients.patient_id.is_unique:
        raise ValueError("A patient crosses outer folds in diagnostic draws.")
    lookup = {patient: index for index, patient in enumerate(patients.patient_id)}
    codes = frame.patient_id.map(lookup).to_numpy()
    draws = np.zeros((repetitions, len(patients)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _, cohort in patients.groupby("fold", sort=True):
        indices = cohort.patient_id.map(lookup).to_numpy()
        draws[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=repetitions)
    return draws, codes


def reliability_table(frame, scores, *, repetitions=2000):
    draws, codes = patient_draws(frame, repetitions)
    result = []
    for name, values in scores.items():
        values = np.asarray(values)
        bins = calibration_bins(frame.target, values, n_bins=5, groups=frame.patient_id)
        for row in bins.to_dict("records"):
            mask = (values >= row["min_probability"]) & (values <= row["max_probability"])
            denominator = draws @ np.bincount(codes, weights=mask, minlength=draws.shape[1])
            numerator = draws @ np.bincount(codes, weights=mask * frame.target.to_numpy(), minlength=draws.shape[1])
            rates = np.divide(numerator, denominator, out=np.full(len(draws), np.nan), where=denominator > 0)
            finite = rates[np.isfinite(rates)]
            lower, upper = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
            result.append({"model": name, **row, "observed_lower": lower,
                           "observed_upper": upper, "valid_draws": len(finite)})
    return pd.DataFrame(result)


def decision_intervals(frame, decisions, *, repetitions=2000):
    draws, codes = patient_draws(frame, repetitions)
    target, flagged = frame.target.to_numpy(), np.asarray(decisions, dtype=bool)
    quantities = np.array([np.bincount(codes, weights=mask, minlength=draws.shape[1]) for mask in
                          ((target == 1) & flagged, (target == 1) & ~flagged,
                           (target == 0) & ~flagged, (target == 0) & flagged)])
    tp, fn, tn, fp = quantities @ draws.T
    ratios = {"sensitivity": (tp, tp + fn), "specificity": (tn, tn + fp),
              "precision": (tp, tp + fp), "flagged_fraction": (tp + fp, tp + fn + tn + fp)}
    point = decision_metrics(target, flagged)
    rows = []
    for metric, (numerator, denominator) in ratios.items():
        values = np.divide(numerator, denominator, out=np.full(repetitions, np.nan), where=denominator > 0)
        finite = values[np.isfinite(values)]
        lower, upper = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
        rows.append({"metric": metric, "estimate": point[metric], "ci_lower": lower,
                     "ci_upper": upper, "valid_draws": len(finite), "requested_draws": repetitions})
    return pd.DataFrame(rows)


def subgroup_diagnostics(frame, values, decisions, *, repetitions=2000):
    ages = pd.to_numeric(frame.age_approx, errors="coerce")
    sites = frame.anatom_site_general_challenge.astype("string").fillna("Missing")
    unknown = sites.str.strip().str.casefold().isin(["", "unknown", "missing", "nan", "none", "null", "n/a", "na"])
    groupings = {"Age band": pd.Series(np.where(ages.isna(), "Missing", np.where(ages < 50, "<50", ">=50")), index=frame.index),
                 "Sex": frame.sex.astype("string").fillna("Missing"), "Site": sites,
                 "Site availability": pd.Series(np.where(unknown, "Unknown", "Known"), index=frame.index)}
    result = []
    for field, grouping in groupings.items():
        for group in sorted(grouping.unique()):
            mask = grouping.eq(group).to_numpy(dtype=bool)
            subset = frame.loc[mask].reset_index(drop=True)
            probability, flagged = np.asarray(values)[mask], np.asarray(decisions)[mask]
            positives = int(subset.loc[subset.target.eq(1), "patient_id"].nunique())
            row = {"field": field, "group": group, "images": len(subset),
                   "patients": subset.patient_id.nunique(), "malignant_images": int(subset.target.sum()),
                   "malignant_patients": positives, "prevalence": subset.target.mean(),
                   **probability_metrics(subset.target, probability), **decision_metrics(subset.target, flagged),
                   "average_precision_lower": np.nan, "average_precision_upper": np.nan,
                   "roc_auc_lower": np.nan, "roc_auc_upper": np.nan,
                   "evidence_note": "Sparse: fewer than 10 malignant patients; intervals suppressed" if positives < 10 else "Exploratory; shared frozen cutoff; no subgroup tuning"}
            if positives >= 10 and subset.target.nunique() == 2:
                draws, codes = patient_draws(subset, repetitions)
                cache = _ranking_cache(subset.target.to_numpy(), probability)
                sampled = []
                for draw in draws:
                    weight = draw[codes].astype(float)
                    positive = float(np.dot(weight, subset.target))
                    if positive > 0 and weight.sum() > positive:
                        sampled.append(_weighted_probability_metrics(cache, weight)[:2])
                if sampled:
                    limits = np.quantile(sampled, [.025, .975], axis=0)
                    for index, metric in enumerate(("average_precision", "roc_auc")):
                        row[f"{metric}_lower"], row[f"{metric}_upper"] = limits[:, index]
                row["valid_draws"] = len(sampled)
            else:
                row["valid_draws"] = 0
            result.append(row)
    return pd.DataFrame(result)


def save_metadata_table(root, stage, name, table):
    """Persist new explanatory evidence outside all frozen training artifacts."""
    for token in (stage, name):
        if not token or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in token):
            raise ValueError("Metadata report stage and table names require simple identifiers.")
    root = Path(root).resolve()
    if stage.startswith(("06", "07", "08", "12")):
        from .metadata_nn_tuning import condition_directory, DEFAULT_RUN_ID
        prefix = condition_directory(root, "dedup_v1", DEFAULT_RUN_ID) / "report"
    else:
        prefix = root / "results/dedup_v1/clean_workflow/metadata_depth"
    folder = (prefix / stage).resolve()
    if not folder.is_relative_to(prefix.resolve()):
        raise ValueError("Supporting metadata tables must stay under the separate report root.")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    table.to_csv(path, index=False)
    return path


def _section(heading, description, code):
    return heading, description, dedent(code).strip()


INITIALISE = '''
import json
from melanoma_detection.metadata_evaluation import decision_metrics
from melanoma_detection.dedup_notebook_metadata import (
    LABELS, COLORS, saved_metadata, metric_table, curve_figure, metric_bar_figure,
    cutoff_table, confusion_figure, estimator_diagnostics, reliability_table,
    decision_intervals, subgroup_diagnostics, save_metadata_table,
)
bundle = score_bundle()
frame, scores = bundle["expected"], bundle["scores"]
meta = saved_metadata(ROOT, bundle)
display(pd.DataFrame([{"Dataset identity": bundle["identity"]["dataset_identity"],
    "New CPU fits": meta["receipt"]["executed_fits"], "Reused clean fits": meta["receipt"]["reused_fits"],
    "Images with matching predictions": len(frame), "Reserved test loaded": False}]))
'''


def metadata_specs():
    """Use the originals as literal Markdown, cell-order and figure templates."""
    from .dedup_notebook_metadata_templates import metadata_template_specs
    return metadata_template_specs()


def metadata_depth_specs():
    """Return full methodological/visual counterparts of original stages 04–08."""
    specs = []
    specs.append(("04_metadata_baselines.ipynb", "04 · Compare clean dummy and logistic baselines",
        "Does learning recorded age, sex and site improve on assigning everyone the training prevalence? Reproduce the original baseline comparison on retained images, with transparent exploratory cutoff illustrations.", [
        _section("1. Load matching saved predictions and explain the protocol", "The dummy learns only training prevalence. Categorical-age logistic regression uses one-hot age, sex and site, C=1 and liblinear. Every prediction comes from a fit excluding that patient's entire outer fold. ID, diagnosis, target and fold are never predictors; category levels are learned within fitting patients. These cells load the approved fits rather than train again.", INITIALISE + '''
names = ["dummy", "categorical_logistic"]
metrics = metric_table(frame, scores, names)
display(pd.DataFrame([{"model": LABELS[name], "inputs": "sex, age category, site" if name != "dummy" else "training prevalence only", "outer fits": 5} for name in names]))
'''),
        _section("2. Check the prior baseline against preserved folds", "The dummy must give one constant score within each outer fold. Its held-out fold AUC is 0.5 and its AP equals that fold's malignant prevalence. The training prevalence need not match validation prevalence, so pooled dummy AUC need not be exactly 0.5.", '''
check = frame.assign(dummy_score=scores["dummy"]).groupby("fold").agg(images=("target", "size"), malignant_images=("target", "sum"), validation_prevalence=("target", "mean"), training_prior=("dummy_score", "first"), distinct_scores=("dummy_score", "nunique"))
dummy_fold = metrics.loc[metrics.model.eq("dummy") & metrics.scope.eq("fold")].set_index("fold")
check["heldout_AP"] = dummy_fold.average_precision
check["heldout_AUC"] = dummy_fold.roc_auc
assert check.distinct_scores.eq(1).all()
assert np.allclose(check.heldout_AUC, .5)
assert np.allclose(check.heldout_AP, check.validation_prevalence)
display(check.round(5))
'''),
        _section("3. Compare primary and secondary probability metrics", "Read mean-fold AP first: it gives each preserved patient fold equal weight. Larger AP/AUC means better ranking; smaller log loss/Brier means smaller probability error. Pooled results are secondary and weight larger folds more heavily. Low Brier alone can disguise poor detection when malignancy is rare.", '''
display(metrics.loc[metrics.scope.ne("fold")].round(5))
display(metrics.loc[metrics.scope.eq("fold")].pivot(index="fold", columns="model", values=["average_precision", "roc_auc"]).round(4))
finish_figure(metric_bar_figure(metrics, names, "Clean dummy versus categorical-age logistic"), "baseline_ranking")
'''),
        _section("4. Read the held-out ROC and precision–recall curves", "ROC plots false-positive fraction against recall; nearer the upper left is preferable. PR plots recall against the malignant fraction among flagged images; compare precision at a common recall. The horizontal baseline is observed prevalence. The recall-zero endpoint does not establish perfect detection.", '''
finish_figure(curve_figure(frame, scores, names, "Clean baseline curves"), "baseline_roc_pr")
'''),
        _section("5. Inspect decisions at the fixed 0.50 cutoff", "A fixed cutoff is a descriptive scenario, not a clinical rule. Each matrix prints TN/FP/FN/TP and percentages within benign or malignant rows. Compare missed malignant cases with false alarms and flagged fraction; an excellent overall error rate can coexist with poor malignant recall.", '''
fixed = cutoff_table(frame, scores, names)
display(fixed.round(5))
scenarios = [(LABELS[name] + " · cutoff 0.50", np.asarray(scores[name]) >= .5) for name in names]
finish_figure(confusion_figure(frame.target, scenarios, "Fixed-cutoff baseline decisions"), "baseline_confusions_05")
'''),
        _section("6. Reproduce the original exploratory cutoff illustration", "This secondary illustration chooses the logistic cutoff maximising OOF balanced accuracy, then uses those same OOF labels to report it. That overlap can overstate future performance. It is excluded from the primary model-choice claims and never exported as the final cutoff. The dummy remains at 0.50; notebook 08 uses training-only cutoff rules instead.", '''
exploratory = cutoff_table(frame, scores, names, exploratory=True)
display(exploratory.round(5))
cutoff = exploratory.set_index("model").loc["categorical_logistic", "cutoff"]
scenarios = [("Logistic · fixed 0.50", scores["categorical_logistic"] >= .5), (f"Logistic · OOF-selected {cutoff:.4f}", scores["categorical_logistic"] >= cutoff)]
finish_figure(confusion_figure(frame.target, scenarios, "Secondary illustration: OOF-selected cutoff"), "baseline_exploratory_confusions")
'''),
        _section("7. Inspect saved logistic associations and state the limits", "Coefficients describe regularised associations among encoded categories; they are neither causes nor stable patient risk factors. A category absent from one training fold has no coefficient in that fit. Error bars below show variability across available saved fits, not sampling confidence intervals. This remains a development comparison before image information.", '''
coefficients, _, _ = estimator_diagnostics(meta)
summary = coefficients.groupby("encoded_feature").logit_coefficient.agg(["mean", "std", "count"])
selected = summary.reindex(summary["mean"].abs().nlargest(15).index).sort_values("mean")
display(selected.round(4))
fig, ax = plt.subplots(figsize=(10.5, 6), layout="constrained")
ax.barh(selected.index.str.replace("categorical__", "", regex=False), selected["mean"], xerr=selected["std"].fillna(0), color="#2563eb")
ax.axvline(0, color="#64748b", linewidth=1)
ax.set(title="Largest categorical-logistic associations across five saved fits", xlabel="Mean logit coefficient; bars ± one fold standard deviation")
finish_figure(fig, "baseline_logistic_coefficients")
gain = metrics.loc[metrics.scope.eq("mean_fold")].set_index("model").average_precision
display(Markdown(f"Mean-fold AP: dummy **{gain['dummy']:.4f}**, logistic **{gain['categorical_logistic']:.4f}**. The fixed-cutoff and OOF-selected matrices describe different decision scenarios; only the latter reuses scoring labels to choose a cutoff. No reserved-test conclusion follows."))
''')]))

    specs.append(("05_metadata_models.ipynb", "05 · Compare clean metadata models and feature sets",
        "Reproduce the original five-model and seven-feature-set comparisons. Read ranking, probability error, fold variation, nonlinear age and saved model diagnostics together.", [
        _section("1. Check matched records and the fixed model specifications", "The fixed spline uses four quantile knots, cubic degree, training-median age imputation and scaling; the classifier has C=1. Trees retain categorical-age encoding. Forest settings are 300 trees/depth 8/minimum leaf 5; XGBoost uses 200 trees/depth 3/learning rate 0.05. These are historical starting settings, with no outer-label tuning in this notebook.", INITIALISE + '''
names = ["dummy", "categorical_logistic", "spline_raw", "random_forest", "xgboost"]
metrics = metric_table(frame, scores, names)
display(pd.DataFrame([{"Model": LABELS[name], "Features": "Age, sex, site", "Preprocessing fit": "Outer training patients only", "OOF images": len(frame)} for name in names]))
display(pd.DataFrame([meta["core"]["specification"]["spline"]]).T)
'''),
        _section("2. Compare ranking and probability error", "Mean-fold AP is primary; pooled AP/AUC/log loss/Brier are secondary. Metadata has no lesion appearance, and a larger AP does not make its score a calibrated risk. The wider model-family comparison remains secondary development evidence.", '''
display(metrics.loc[metrics.scope.ne("fold")].round(5))
finish_figure(metric_bar_figure(metrics, names, "Five fixed metadata model families"), "metadata_five_model_scores")
'''),
        _section("3. Read variation across the preserved patient folds", "A score can change because different held-out patients carry different malignant cases, ages and sites. These lines connect the same model across the five patient folds; they are not a learning curve or five independent datasets.", '''
fold_metrics = metrics.loc[metrics.scope.eq("fold")]
display(fold_metrics.pivot(index="fold", columns="model", values=["average_precision", "roc_auc"]).round(4))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), layout="constrained")
for index, name in enumerate(names):
    rows = fold_metrics.loc[fold_metrics.model.eq(name)]
    for ax, metric in zip(axes, ["average_precision", "roc_auc"]):
        ax.plot(rows.fold, rows[metric], "o-", color=COLORS[index], label=LABELS[name])
for ax, label in zip(axes, ["Average precision", "ROC-AUC"]):
    ax.set(xlabel="Preserved outer patient fold", ylabel=label, title="Held-out fold " + label, xticks=range(5))
    ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
finish_figure(fig, "metadata_fold_variation")
'''),
        _section("4. Read the model ROC and PR curves", "Each pooled curve traces a changing cutoff on held-out predictions. Curves can cross, meaning one method is better in one recall range and worse in another. PR is focused to 0–30% precision to reveal the low-prevalence trade-offs; the complete AP still uses the entire curve.", '''
finish_figure(curve_figure(frame, scores, names, "Fixed metadata models", pr_limit=.30), "metadata_five_model_curves")
'''),
        _section("5. Inspect the learned nonlinear age relationship", "Use the five saved outer-training spline models, without fitting all development patients. Hold site at torso and sex at the two recorded values while varying age. The line averages model predictions and shading spans the five fits. These conditional profiles are descriptive, omit lesion appearance and do not show a causal age effect or a risk for an actual patient.", '''
coefficients, importances, age_profiles = estimator_diagnostics(meta)
summary = age_profiles.groupby(["sex", "age"]).probability.agg(["mean", "min", "max"]).reset_index()
fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
for index, (sex, rows) in enumerate(summary.groupby("sex")):
    ax.plot(rows.age, rows["mean"], color=COLORS[index], label=f"{sex}; torso")
    ax.fill_between(rows.age, rows["min"], rows["max"], alpha=.15, color=COLORS[index])
ax.set(xlabel="Recorded age (years)", ylabel="Saved spline probability estimate", title="Age profiles: mean and range across five training-patient fits")
ax.legend(frameon=False); ax.grid(alpha=.2)
finish_figure(fig, "metadata_nonlinear_age_profiles")
'''),
        _section("6. Compare the seven original feature ablations", "Each partial feature set uses the same spline/logistic procedure and patient folds. The all-feature row reuses the exact fixed-spline predictions. A lower score after dropping a field describes its value in this procedure; it does not prove a causal effect. Correlated fields and changing prevalence complicate interpretation.", '''
ablation_names = ["ablation_sex", "ablation_age", "ablation_site", "ablation_sex_age", "ablation_sex_site", "ablation_age_site", "ablation_all_features"]
ablation_metrics = metric_table(frame, scores, ablation_names)
display(ablation_metrics.loc[ablation_metrics.scope.ne("fold")].round(5))
assert np.allclose(scores["ablation_all_features"], scores["spline_raw"], rtol=0, atol=1e-14)
selected = ablation_metrics.loc[ablation_metrics.scope.eq("mean_fold")].sort_values("average_precision")
dummy_ap = metrics.loc[metrics.model.eq("dummy") & metrics.scope.eq("mean_fold"), "average_precision"].iloc[0]
fig, ax = plt.subplots(figsize=(10, 5.3), layout="constrained")
ax.barh([LABELS[name] for name in selected.model], selected.average_precision, color="#2563eb")
ax.axvline(dummy_ap, color="#64748b", linestyle="--", label=f"Dummy mean-fold AP {dummy_ap:.4f}")
for index, value in enumerate(selected.average_precision): ax.text(value, index, f"  {value:.4f}", va="center")
ax.set(xlabel="Mean outer-fold average precision", title="Feature ablation on identical retained scoring records", xlim=(0, float(selected.average_precision.max()) * 1.25))
ax.legend(frameon=False); ax.grid(axis="x", alpha=.2)
finish_figure(fig, "metadata_seven_feature_ablations")
'''),
        _section("7. Inspect tree importances with the proper cautions", "Sum encoded-column importances back to age, sex and site. Forest values use impurity-based importance; XGBoost uses its saved normalised tree importance. Many age categories can receive more opportunities for splits. These measures are associations within fitted trees and cannot establish out-of-sample field importance or causation; compare fields within a model, not magnitudes across algorithms.", '''
summary = importances.groupby(["model", "feature"]).importance.agg(["mean", "std"]).reset_index()
display(summary.round(4))
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
for ax, model in zip(axes, ["random_forest", "xgboost"]):
    rows = summary.loc[summary.model.eq(model)]
    ax.barh(rows.feature, rows["mean"], xerr=rows["std"], color="#0f766e")
    ax.set(xlabel="Mean normalised importance; ± fold standard deviation", title=LABELS[model])
finish_figure(fig, "metadata_saved_tree_importances")
'''),
        _section("8. Reproduce the exploratory confusion-matrix comparison", "Original notebook 05 selected each learned model's balanced-accuracy cutoff from the same OOF labels it reported. Retain that educational illustration here as explicitly secondary. These cutoffs never replace training-only decision rules, and matrices do not establish clinical usefulness. The dummy remains fixed at 0.50.", '''
fixed = cutoff_table(frame, scores, names)
illustrative = cutoff_table(frame, scores, names, exploratory=True)
display(fixed.round(5)); display(illustrative.round(5))
scenarios = [(LABELS[row.model] + f" · {row.cutoff:.4f}", scores[row.model] >= row.cutoff) for row in illustrative.itertuples()]
finish_figure(confusion_figure(frame.target, scenarios, "Secondary OOF-selected decision illustrations"), "metadata_exploratory_confusions")
'''),
        _section("9. Collect the supported findings", "Keep the original fixed spline as the predeclared metadata comparator even if a secondary model leads one score. Read AP, AUC, fold variation and review workload together; do not promote a model from whichever table looks favourable.", '''
primary = metrics.loc[metrics.scope.eq("mean_fold")].sort_values("average_precision", ascending=False)
display(primary[["model", "average_precision", "roc_auc", "log_loss", "brier"]].round(5))
display(Markdown("The clean five-model results and feature ablations are complete development comparisons. Calibration and patient intervals follow in notebook 08; the reserved test remains untouched."))
''')]))

    specs.append(("06_metadata_neural_network.ipynb", "06 · Examine the clean three-seed metadata NN",
        "Reproduce the network protocol and inspect its saved fitting histories, stopping decisions, seed stability and equal-mean ensemble on the same retained OOF records.", [
        _section("1. Explain architecture, preprocessing and optimisation", "The model is a 32-unit dense layer, ReLU, 20% dropout and one output logit. Training median/scaling prepare age; training one-hot categories prepare sex/site. AdamW uses learning rate 0.001, weight decay 0.001 and batches of 128, with unweighted BCE loss. Seeds 42, 142 and 242 are all retained with equal averaging; there is no favourable-seed selection.", INITIALISE + '''
spec = meta["core"]["specification"]["neural_network"]
display(pd.DataFrame([spec]).T)
epochs = pd.read_csv(meta["base"] / "epochs.csv")
history = pd.read_csv(meta["base"] / "history.csv")
names = ["spline_raw", "nn_seed_42", "nn_seed_142", "nn_seed_242", "metadata_nn_mean"]
metrics = metric_table(frame, scores, names)
'''),
        _section("2. Verify patient-safe stopping and fresh outer refits", "For each seed/fold, a different preserved training fold supplies stopping labels: (outer fold + 1 + seed index) modulo 5. Best stopping BCE chooses duration; at least ten epochs are attempted with patience 12. That duration is then used by a fresh preprocessor and network fitted on all four outer-training folds. Outer labels never select epochs. These runs vary both seed and stopping membership, not independent datasets.", '''
display(epochs[["base_seed", "outer_fold", "training_seed", "stop_fold", "selected_epochs", "attempted_stopping_epochs", "stop_log_loss", "epoch_fit_patients", "epoch_stop_patients", "score_patients"]].round(5))
checks = []
for fold in range(5):
    members = pd.read_csv(meta["base"] / f"fold_{fold}/selection_memberships.csv")
    for seed, rows in members.groupby("base_seed"):
        fit = set(rows.loc[rows.role.eq("epoch_fitting"), "patient_id"])
        stop = set(rows.loc[rows.role.eq("epoch_stopping"), "patient_id"])
        outer = set(rows.loc[rows.role.eq("outer_validation"), "patient_id"])
        assert fit.isdisjoint(stop) and (fit | stop).isdisjoint(outer)
        checks.append({"outer_fold": fold, "base_seed": seed, "fit_patients": len(fit), "stop_patients": len(stop), "outer_patients": len(outer), "overlap": 0})
display(pd.DataFrame(checks))
'''),
        _section("3. Read all fifteen stopping trajectories", "Each panel is one outer-training pool. Solid lines show fitting BCE and dashed lines show patient-disjoint stopping BCE for each seed. Vertical dotted lines mark selected durations. Falling fitting loss with flat or rising stopping loss can indicate overfitting; these are training-only selection diagnostics, not outer-fold performance.", '''
fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
for fold, ax in enumerate(axes.flat[:5]):
    for index, seed in enumerate([42, 142, 242]):
        rows = history.loc[history.outer_fold.eq(fold) & history.base_seed.eq(seed) & history.phase.eq("stopping")]
        selected = int(epochs.loc[epochs.outer_fold.eq(fold) & epochs.base_seed.eq(seed), "selected_epochs"].iloc[0])
        ax.plot(rows.epoch, rows.fit_bce, color=COLORS[index], label=f"{seed}: fitting")
        ax.plot(rows.epoch, rows.stop_bce, "--", color=COLORS[index], label=f"{seed}: stopping")
        ax.axvline(selected, linestyle=":", color=COLORS[index], alpha=.5)
    ax.set(title=f"Outer fold {fold}: training-only stopping", xlabel="Attempted epoch", ylabel="Binary cross-entropy")
    ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=7)
axes.flat[5].axis("off")
finish_figure(fig, "nn_fifteen_stopping_trajectories")
'''),
        _section("4. Read the fresh refit histories and selected durations", "The outer refit runs for its selected duration using all four training folds. Only fitting BCE is available here; it must not be interpreted as validation performance. The duration table makes the workload and stopping-fold sensitivity visible without selecting a new duration from these plots.", '''
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
for index, seed in enumerate([42, 142, 242]):
    rows = epochs.loc[epochs.base_seed.eq(seed)]
    axes[0].plot(rows.outer_fold, rows.selected_epochs, "o-", color=COLORS[index], label=f"Seed {seed}")
    for fold in range(5):
        line = history.loc[history.base_seed.eq(seed) & history.outer_fold.eq(fold) & history.phase.eq("outer_refit")]
        axes[1].plot(line.epoch, line.fit_bce, color=COLORS[index], alpha=.5, label=f"Seed {seed}" if fold == 0 else None)
axes[0].set(title="Selected duration across training pools", xlabel="Outer fold", ylabel="Fresh outer-refit epochs", xticks=range(5))
axes[1].set(title="Fifteen fresh refit training histories", xlabel="Refit epoch", ylabel="Fitting BCE")
for ax in axes: ax.grid(alpha=.2); ax.legend(frameon=False)
finish_figure(fig, "nn_selected_epochs_and_refits")
'''),
        _section("5. Compare every seed and the prescribed ensemble", "Report all seeds and their equal-mean ensemble rather than selecting the strongest seed. The primary bars use mean-fold AP and AUC; pooled results remain secondary. Sample variability over three runs reflects seed plus stopping-fold choices and is not a confidence interval for future patients.", '''
display(metrics.loc[metrics.scope.ne("fold")].round(5))
stability = []
for scope in ["mean_fold", "pooled"]:
    seed_rows = metrics.loc[metrics.scope.eq(scope) & metrics.model.str.startswith("nn_seed")]
    for metric in ["average_precision", "roc_auc", "log_loss", "brier"]:
        values = seed_rows[metric]
        stability.append({"scope": scope, "metric": metric, "runs": len(values), "mean": values.mean(), "sample_std": values.std(ddof=1), "minimum": values.min(), "maximum": values.max(), "range": values.max()-values.min()})
display(pd.DataFrame(stability).round(5))
finish_figure(metric_bar_figure(metrics, names, "Network seed and stopping-fold sensitivity"), "nn_seed_stability")
'''),
        _section("6. Verify the ensemble and inspect seed disagreement", "The ensemble uses the equal mean of all three seed probabilities. Verify it within the float32 arithmetic precision of the saved NN inference and report the small numerical difference after CSV loading. Between-seed score spread is an exploratory stability diagnostic; it is not the later image MC-dropout uncertainty measure. Listed images are development review examples, not automatic exclusions or new training rules.", '''
seed_values = np.column_stack([scores[f"nn_seed_{seed}"] for seed in [42, 142, 242]])
mean_difference = float(np.max(np.abs(seed_values.mean(axis=1) - scores["metadata_nn_mean"])))
assert np.allclose(seed_values.mean(axis=1), scores["metadata_nn_mean"], rtol=2*np.finfo(np.float32).eps, atol=1e-8)
display(pd.DataFrame([{"Maximum float64 reconstruction difference": mean_difference, "Saved inference arithmetic": "float32 equal mean; original retained scores preserved"}]))
review = frame[["image_id", "patient_id", "fold", "target"]].copy()
review["ensemble_score"] = scores["metadata_nn_mean"]
review["seed_score_sd"] = seed_values.std(axis=1, ddof=1)
display(review.sort_values(["seed_score_sd", "image_id"], ascending=[False, True]).head(12).round(5))
fig, ax = plt.subplots(figsize=(9.5, 4.5), layout="constrained")
ax.hist(review.seed_score_sd, bins=30, color="#d97706")
ax.set(title="Between-seed spread on retained held-out images", xlabel="Sample standard deviation of three probabilities", ylabel="Images")
finish_figure(fig, "nn_between_seed_disagreement")
'''),
        _section("7. Read held-out curves and the supported conclusion", "Read the NN and spline at a common recall; a higher AP is not a calibrated risk or a clinical decision rule. Notebook 08 supplies the paired whole-patient interval for NN mean minus fixed spline. These curves and seed diagnostics do not justify replacing the primary model by whichever seed looks best.", '''
finish_figure(curve_figure(frame, scores, ["spline_raw", "metadata_nn_mean"], "Fixed spline versus prescribed metadata NN"), "nn_spline_roc_pr")
display(Markdown("All three prescribed runs are included. Stopping and refit histories are complete; outer predictions remain held out by patient. Probability calibration, decision support and paired intervals follow in notebook 08."))
''')]))

    specs.append(("07_metadata_tuning.ipynb", "07 · Inspect clean nested metadata tuning",
        "Reproduce the historical bounded searches and their patient-safe evaluation. Show every candidate, reconstructed inner membership, selected setting, fixed-versus-tuned curve and fold-level change.", [
        _section("1. Load scores and declare the exact bounded search", "Inside each outer-training pool, mean AP across three inner patient folds selects a setting. Grid order resolves ties deterministically, and a fresh fit on all outer-training patients supplies the outer prediction. Candidate counts are 18 spline, 12 forest and 12 XGBoost; 630 inner fits plus 15 selected outer refits. Fixed comparator fits are reused.", INITIALISE + '''
from melanoma_detection.dedup_metadata_suite import GRIDS
from sklearn.model_selection import ParameterGrid
grid_rows = [{"model": name, "parameter": parameter.rsplit("__", 1)[-1], "values": str(values), "candidate_recipes": len(list(ParameterGrid(grid)))} for name, grid in GRIDS.items() for parameter, values in grid.items()]
display(pd.DataFrame(grid_rows))
selected = pd.read_csv(meta["suite"] / "selected_parameters.csv")
candidates = pd.read_csv(meta["suite"] / "inner_candidates.csv")
families = [("spline_raw", "spline_tuned"), ("random_forest", "random_forest_tuned"), ("xgboost", "xgboost_tuned")]
names = [name for pair in families for name in pair]
metrics = metric_table(frame, scores, names)
'''),
        _section("2. Verify reconstruction and inner patient support", "No historical tuning membership receipt was saved. The separate reconstruction applies the original seed 42 + outer fold to original 6,864-row CSV order, freezes patient memberships, then retains the clean IDs. It is recorded with library versions and is not represented as a recovered execution receipt. Never regenerate the split on the 6,706 retained rows.", '''
receipt = read_json(meta["suite"] / "memberships_receipt.json")
display(pd.DataFrame([{key: receipt[key] for key in ["status", "historical_membership_receipt_available", "original_row_order_used", "split_on_retained_rows", "seed_rule"]}]))
display(pd.DataFrame([receipt["versions"]]).T)
members = pd.read_csv(meta["suite"] / "inner_memberships_retained.csv")
assert members.groupby(["outer_fold", "patient_id"]).inner_fold.nunique().eq(1).all()
inner_audit = pd.read_csv(meta["suite"] / "inner_audit.csv")
assert inner_audit.patient_overlap.eq(0).all() and inner_audit.outer_patient_overlap.eq(0).all()
display(inner_audit)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
for ax, column, label in zip(axes, ["validation_patients", "validation_positive_images"], ["Inner-validation patients", "Inner-validation malignant images"]):
    table = inner_audit.pivot(index="outer_fold", columns="inner_fold", values=column)
    table.plot.bar(ax=ax, rot=0)
    ax.set(title=label, xlabel="Outer-training pool", ylabel=label)
    ax.legend(title="Preserved inner fold", frameon=False)
finish_figure(fig, "nested_inner_patient_and_case_support")
'''),
        _section("3. Inspect selected settings and every candidate", "The complete 210-row table contains 42 recipes per outer pool, their three inner AP values, mean, spread and selected flag. These are training-only selection scores, not final OOF performance. The error bars below are standard deviation across three inner folds and are not confidence intervals.", '''
display(selected)
display(candidates.loc[candidates.outer_fold.eq(0)].sort_values(["model", "candidate_index"]).round(5))
assert len(candidates) == 210 and int(candidates.selected.sum()) == 15
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
for ax, model in zip(axes, ["spline", "random_forest", "xgboost"]):
    rows = candidates.loc[candidates.outer_fold.eq(0) & candidates.model.eq(model)]
    ax.errorbar(rows.candidate_index, rows.mean_inner_average_precision, yerr=rows.std_inner_average_precision, fmt="o", color="#2563eb", capsize=2)
    winner = rows.loc[rows.selected]
    ax.scatter(winner.candidate_index, winner.mean_inner_average_precision, marker="*", s=180, color="#d97706", label="Selected within training")
    ax.set(title=model.replace("_", " ") + " · outer pool 0", xlabel="Deterministic grid candidate index", ylabel="Mean inner AP ± inner-fold SD")
    ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
finish_figure(fig, "nested_all_candidate_scores_outer0")
'''),
        _section("4. Check how often each recipe was selected", "Different training pools can choose different settings because malignant support is limited. Frequency describes selection stability; it does not identify a single recipe to refit on all development data or retrospectively change this experiment.", '''
frequency = selected.groupby(["model", "best_parameters"]).size().reset_index(name="outer_pools_selected")
display(frequency)
fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
for ax, model in zip(axes, ["spline", "random_forest", "xgboost"]):
    rows = frequency.loc[frequency.model.eq(model)]
    labels = [", ".join(key.rsplit("__", 1)[-1] + "=" + str(value) for key, value in json.loads(parameters).items()) for parameters in rows.best_parameters]
    ax.barh(labels, rows.outer_pools_selected, color="#d97706")
    ax.set(title=model.replace("_", " "), xlabel="Outer-training pools selecting recipe", xlim=(0, 5))
    ax.tick_params(axis="y", labelsize=8)
finish_figure(fig, "nested_recipe_selection_frequency")
'''),
        _section("5. Compare fixed and tuned ranking scores", "Use mean-fold AP first and inspect AUC beside it. A tuned procedure can score worse on held-out patients even when its grid contains the fixed setting: it selects by noisy inner AP. Wider tuned-versus-fixed comparisons remain secondary; higher observed AP is not automatic promotion.", '''
display(metrics.loc[metrics.scope.ne("fold")].round(5))
mean = metrics.loc[metrics.scope.eq("mean_fold")].set_index("model")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
x = np.arange(3)
for ax, metric, label in zip(axes, ["average_precision", "roc_auc"], ["Average precision", "ROC-AUC"]):
    for offset, version, color in [(-.17, 0, "#2563eb"), (.17, 1, "#d97706")]:
        values = [mean.loc[pair[version], metric] for pair in families]
        ax.bar(x + offset, values, width=.34, color=color, label="Fixed" if version == 0 else "Nested-tuned")
        for position, value in zip(x + offset, values): ax.text(position, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set(xticks=x, xticklabels=["Spline", "Forest", "XGBoost"], ylabel=label, title="Mean outer-fold " + label, ylim=(0, float(mean[metric].max()) * 1.25))
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
finish_figure(fig, "nested_fixed_versus_tuned_scores")
'''),
        _section("6. Read the six pooled ROC and PR comparisons", "Each row compares one model family. Dashed blue is fixed and solid gold is tuned. PR precision is focused to 0–30%; AP calculations still use the full ranking. Crossing curves indicate cutoff-dependent trade-offs. Pooled curves are secondary to the mean-fold ranking table.", '''
from sklearn.metrics import roc_curve, precision_recall_curve
fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout="constrained")
for row, pair in enumerate(families):
    for version, name in enumerate(pair):
        color, style = ("#2563eb", "--") if version == 0 else ("#d97706", "-")
        fpr, tpr, _ = roc_curve(frame.target, scores[name]); precision, recall, _ = precision_recall_curve(frame.target, scores[name])
        axes[row, 0].plot(fpr, tpr, style, color=color, label=LABELS[name])
        axes[row, 1].plot(recall, precision, style, color=color, label=LABELS[name])
    axes[row, 0].plot([0, 1], [0, 1], ":", color="#64748b")
    axes[row, 1].axhline(frame.target.mean(), linestyle=":", color="#64748b")
    axes[row, 0].set(xlabel="False-positive fraction", ylabel="Recall", title=LABELS[pair[0]] + " · ROC", xlim=(0, 1), ylim=(0, 1))
    axes[row, 1].set(xlabel="Recall", ylabel="Precision", title=LABELS[pair[0]] + " · PR", xlim=(0, 1), ylim=(0, .30))
    for ax in axes[row]: ax.legend(frameon=False); ax.grid(alpha=.2)
finish_figure(fig, "nested_three_family_roc_pr")
'''),
        _section("7. Collect fold-level improvements, ties and losses", "A mean change can hide different directions across patient folds. Count improved/tied/worse folds and report AP/AUC changes without claiming that five reused folds establish repeatable superiority. Recipe selection is training-only, but wider development model exploration still limits final conclusions.", '''
folds = metrics.loc[metrics.scope.eq("fold")].pivot(index="fold", columns="model", values="average_precision")
rows = []
for fixed, tuned in families:
    differences = folds[tuned] - folds[fixed]
    ties = np.isclose(differences, 0)
    rows.append({"family": LABELS[fixed], "mean_fold_AP_change": mean.loc[tuned, "average_precision"] - mean.loc[fixed, "average_precision"], "mean_fold_AUC_change": mean.loc[tuned, "roc_auc"] - mean.loc[fixed, "roc_auc"], "folds_improved": int(((differences > 0) & ~ties).sum()), "folds_tied": int(ties.sum()), "folds_worse": int(((differences < 0) & ~ties).sum())})
display(pd.DataFrame(rows).round(5))
display(folds.round(5))
display(Markdown("The fixed raw spline remains the prescribed metadata comparator. Any change in the final procedure requires a separate declared experiment; these searches never inspect reserved-test labels."))
''')]))

    specs.append(("08_metadata_robustness.ipynb", "08 · Check clean metadata uncertainty, calibration and decisions",
        "Reproduce the original robustness analysis using saved training-only maps and cutoffs, paired patient intervals, reliability diagrams, class support and review workload. Defer the final full-development fit.", [
        _section("1. Match scores and audit the training-only procedure", "For each outer pool, four preserved training folds supply inner OOF spline scores. Those scores alone fit the sigmoid recalibrator and select the highest cutoff meeting the academic 80% training recall target. A fresh outer-training spline then scores untouched outer patients. Outer recall can differ from 80%; this is a research scenario, not a clinical target.", INITIALISE + '''
calibration = meta["calibration"]
audit = pd.read_csv(meta["base"] / "spline_inner_audit.csv")
assert audit.patient_overlap.eq(0).all()
display(audit.round(5))
display(calibration.groupby("fold").agg(raw_cutoff=("raw_threshold", "first"), recalibrated_cutoff=("calibrated_threshold", "first"), images=("target", "size"), malignant_images=("target", "sum")))
'''),
        _section("2. Read paired whole-patient ranking intervals", "The saved comparison uses 2,000 seed-42 draws within outer folds, repeating every image of a sampled patient together. NN and spline receive identical multiplicities. Mean-fold AP is primary; pooled differences are secondary. Intervals hold fitted predictions fixed and omit retraining, architecture search and distribution shift.", '''
evaluation = latest_evaluation(ROOT)
if "output_directory" in evaluation:
    folder = ROOT / evaluation["output_directory"]
    absolute = pd.read_csv(folder / "bootstrap_intervals.csv")
    paired = pd.read_csv(folder / "paired_comparisons.csv")
    left = absolute.loc[absolute.model.isin(["spline_raw", "metadata_nn_mean"]) & absolute.aggregation.eq("mean_fold") & absolute.metric.eq("average_precision")]
    right = paired.loc[paired.comparison.eq("metadata_nn_mean minus spline_raw") & paired.aggregation.eq("mean_fold") & paired.metric.isin(["average_precision", "roc_auc"])]
    display(left.round(5)); display(right.round(5))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for index, row in enumerate(left.itertuples()):
        axes[0].plot([row.ci_lower, row.ci_upper], [index, index], color="#2563eb", linewidth=2)
        axes[0].plot(row.estimate, index, "o", color="#2563eb")
    axes[0].set(yticks=range(len(left)), yticklabels=[LABELS[name] for name in left.model], xlabel="Mean-fold AP", title="Conditional 95% patient intervals", xlim=(0, max(.12, float(left.ci_upper.max()) * 1.1)))
    ap = right.loc[right.metric.eq("average_precision")]
    for row in ap.itertuples():
        axes[1].plot([row.ci_lower, row.ci_upper], [0, 0], color="#d97706", linewidth=3)
        axes[1].plot(row.difference, 0, "s", color="#d97706")
    axes[1].axvline(0, linestyle="--", color="#64748b")
    axes[1].set(yticks=[0], yticklabels=["NN mean − fixed spline"], xlabel="Paired mean-fold AP change", title="Read the paired interval against zero")
    for ax in axes: ax.grid(axis="x", alpha=.2)
    finish_figure(fig, "metadata_paired_patient_intervals")
else:
    display(Markdown("**Pending:** verified clean paired intervals have not been generated."))
'''),
        _section("3. Compare probability error and equal-patient weighting", "Raw versus recalibrated score error must be assessed on outer patients. Equal-patient image weights 1/(retained images for that patient) provide a secondary diagnostic for unequal patient contribution; they do not convert the task into patient cancer diagnosis or change the primary metric.", '''
probabilities = {"spline_raw": calibration.p_spline_raw.to_numpy(), "spline_calibrated": calibration.p_spline_calibrated.to_numpy(), "metadata_nn_mean": scores["metadata_nn_mean"]}
display(metric_table(frame, probabilities, list(probabilities)).query("scope != 'fold'").round(5))
from sklearn.metrics import average_precision_score, roc_auc_score
weights = 1 / frame.groupby("patient_id").patient_id.transform("size").to_numpy()
display(pd.DataFrame([{"model": LABELS[name], "pooled_image_AP": average_precision_score(frame.target, values), "equal_patient_AP": average_precision_score(frame.target, values, sample_weight=weights), "pooled_image_AUC": roc_auc_score(frame.target, values), "equal_patient_AUC": roc_auc_score(frame.target, values, sample_weight=weights)} for name, values in probabilities.items()]).round(5))
'''),
        _section("4. Read reliability diagrams and score distributions", "Five approximate equal-count bins keep equal scores together. The horizontal axis is mean estimated risk and the vertical axis is observed malignant-image fraction; the diagonal marks agreement. Vertical intervals use 2,000 whole-patient draws within folds with fixed bins. A patient can appear in several bins, and sparse malignant support makes the curve noisy. Histogram movements do not prove better calibration.", '''
reliability = reliability_table(frame, {name: values for name, values in probabilities.items() if name != "metadata_nn_mean"})
display(reliability.round(5))
save_metadata_table(ROOT, "08_metadata_robustness", "reliability_bins_with_patient_intervals", reliability)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout="constrained")
maximum = max(float(values.max()) for name, values in probabilities.items() if name != "metadata_nn_mean")
for index, name in enumerate(["spline_raw", "spline_calibrated"]):
    rows = reliability.loc[reliability.model.eq(name)]
    axes[0].plot(rows.mean_probability, rows.observed_rate, "o-", color=COLORS[index], label=LABELS[name])
    axes[0].vlines(rows.mean_probability, rows.observed_lower, rows.observed_upper, color=COLORS[index])
    axes[1].hist(probabilities[name], bins=np.linspace(0, maximum + .001, 25), histtype="step", linewidth=2, color=COLORS[index], label=LABELS[name])
limit = max(float(reliability.mean_probability.max()), float(reliability.observed_upper.max())) * 1.15
axes[0].plot([0, limit], [0, limit], "--", color="#64748b", label="Agreement")
axes[0].set(xlabel="Mean estimated malignant probability", ylabel="Observed malignant-image fraction", title="Five-bin calibration with patient intervals", xlim=(0, limit), ylim=(0, limit))
axes[1].set(xlabel="Estimated malignant probability", ylabel="Images", title="Held-out raw and adjusted score distributions")
for ax in axes: ax.legend(frameon=False, fontsize=9); ax.grid(alpha=.2)
finish_figure(fig, "metadata_reliability_and_score_distributions")
'''),
        _section("5. Assess frozen cutoffs on outer patients", "Each outer fold uses its own training-selected cutoff. Combine its saved decisions without choosing another cutoff from outer labels. Sensitivity counts malignant images found; specificity counts benign images left unflagged; precision is malignant share among flags; flagged fraction estimates review workload. These are image-level decisions with patient-cluster uncertainty.", '''
decision_rows, interval_rows = [], []
for scale, name in [("raw", "spline_raw"), ("calibrated", "spline_calibrated")]:
    decisions = calibration[f"decision_{scale}"].to_numpy()
    decision_rows.append({"procedure": LABELS[name], **decision_metrics(frame.target, decisions)})
    ci = decision_intervals(frame, decisions)
    ci.insert(0, "procedure", LABELS[name]); interval_rows.append(ci)
decision_summary = pd.DataFrame(decision_rows)
decision_ci = pd.concat(interval_rows, ignore_index=True)
display(decision_summary.round(5)); display(decision_ci.round(5))
save_metadata_table(ROOT, "08_metadata_robustness", "training_only_decision_summary", decision_summary)
save_metadata_table(ROOT, "08_metadata_robustness", "training_only_decision_intervals", decision_ci)
scenarios = [(LABELS[name] + " · training-only cutoff", calibration[column].to_numpy()) for name, column in [("spline_raw", "decision_raw"), ("spline_calibrated", "decision_calibrated")]]
finish_figure(confusion_figure(frame.target, scenarios, "Training-selected metadata decision support"), "metadata_training_only_confusions")
'''),
        _section("6. Read detection and review-workload intervals", "The same 2,000 patient draws assess raw and recalibrated frozen decisions. Intervals do not reselect cutoffs in each draw or guarantee a clinical workload. Undefined rates retain missing values, and valid-draw counts show whether scarce classes limit estimation.", '''
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
for ax, metric, label in zip(axes, ["sensitivity", "flagged_fraction"], ["Malignant images found (recall)", "All images flagged (review fraction)"]):
    rows = decision_ci.loc[decision_ci.metric.eq(metric)].reset_index(drop=True)
    for index, row in rows.iterrows():
        ax.plot([row.ci_lower, row.ci_upper], [index, index], color=COLORS[index], linewidth=3)
        ax.plot(row.estimate, index, "o", color=COLORS[index])
    ax.set(yticks=range(len(rows)), yticklabels=rows.procedure, xlabel=label, title=label + " with patient intervals", xlim=(0, 1))
    ax.grid(axis="x", alpha=.2)
axes[0].axvline(.8, color="#64748b", linestyle="--", label="Training target")
axes[0].legend(frameon=False)
finish_figure(fig, "metadata_detection_and_review_intervals")
'''),
        _section("7. Check age, sex, site and site availability with support", "Use raw spline scores and the same frozen decision procedure for every subgroup; no subgroup-specific cutoff is fitted. AP depends partly on malignant prevalence. One patient's images may occupy multiple age/site groups, so counts cannot be summed as independent patients. Secondary intervals use 2,000 seed-42 whole-patient draws within outer folds. Fewer than ten malignant patients triggers interval suppression as a reporting caution, not a zero metric; undefined values remain missing.", '''
diagnostic_frame = bundle["frame"].set_index("image_name").loc[frame.image_id].reset_index()
subgroups = subgroup_diagnostics(diagnostic_frame, calibration.p_spline_raw.to_numpy(), calibration.decision_raw.to_numpy())
save_metadata_table(ROOT, "08_metadata_robustness", "subgroup_diagnostics_with_patient_support", subgroups)
display(subgroups[["field", "group", "images", "patients", "malignant_images", "malignant_patients", "prevalence", "average_precision", "roc_auc", "sensitivity", "precision", "flagged_fraction", "valid_draws", "evidence_note"]].round(5))
fig, axes = plt.subplots(1, 4, figsize=(18, 6), layout="constrained")
limit = max(float(subgroups.average_precision.max()), float(subgroups.average_precision_upper.max())) * 1.15
for ax, field in zip(axes, ["Age band", "Sex", "Site", "Site availability"]):
    rows = subgroups.loc[subgroups.field.eq(field)].reset_index(drop=True)
    for index, row in rows.iterrows():
        if pd.notna(row.average_precision_lower): ax.plot([row.average_precision_lower, row.average_precision_upper], [index, index], color="#2563eb", linewidth=2)
        if pd.notna(row.average_precision): ax.plot(row.average_precision, index, "o" if row.malignant_patients >= 10 else "x", color="#2563eb")
        else: ax.text(limit * .04, index, "AP undefined", va="center", fontsize=9)
    ax.set(yticks=range(len(rows)), yticklabels=[f"{row.group}\\n({row.malignant_patients} malignant patients)" for row in rows.itertuples()], xlabel="Pooled subgroup AP", title=field, xlim=(0, limit))
    ax.tick_params(axis="y", labelsize=9); ax.grid(axis="x", alpha=.2)
finish_figure(fig, "metadata_subgroup_support_and_intervals")
'''),
        _section("8. State the current evidence and final-fit boundary", "The original robustness notebook fitted a final all-development bundle. This stage deliberately postpones that fit until the model procedure is chosen. The saved clean OOF models, maps and cutoffs support development comparisons only; no final reserved-test result or clinical safety claim follows.", '''
raw = decision_summary.iloc[0]
display(Markdown(f"The frozen training-selected raw spline procedure finds **{int(raw.TP)} of {int(frame.target.sum())}** malignant images, flags **{int(raw.TP + raw.FP)} of {len(frame):,}** images and has precision **{raw.precision:.2%}**. Read those outcomes with the patient intervals and subgroup support above."))
display(Markdown("**Final all-development fitting is deferred.** The existing historical final model remains preserved; these notebooks neither replace it nor open the reserved test."))
''')]))
    return specs
