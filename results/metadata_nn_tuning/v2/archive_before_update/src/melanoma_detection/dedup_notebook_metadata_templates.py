"""Original metadata notebook templates adapted to verified clean evidence.

Every original code cell is deliberately replaced. Safe metric and plotting
cells are copied explicitly so their figure composition and styles remain
identical; fitting, splitting and final-test operations are never transferred.
"""
from __future__ import annotations

import re
from textwrap import dedent

import joblib
import numpy as np
import pandas as pd

from .dedup_notebook_templates import original_cells, template_sections


def marginal_age_profile(train, meta):
    """Average saved outer-fit scores over the clean sex/site distribution."""
    features = ["sex", "age_approx", "anatom_site_general_challenge"]
    models = [joblib.load(meta["base"] / f"fold_{fold}/models.joblib")["spline"]
              for fold in range(5)]
    records = []
    for age in sorted(pd.to_numeric(train.age_approx, errors="coerce").dropna().unique()):
        scenario = train[features].copy()
        scenario["age_approx"] = str(int(age))
        estimates = [float(model.predict_proba(scenario)[:, 1].mean()) for model in models]
        records.append({"age": int(age), "model_implied_average_risk": np.mean(estimates)})
    return pd.DataFrame(records)


def metadata_patient_intervals(frame, probability_sets, repetitions=2000):
    """Original table shape with current paired draws within outer folds."""
    from .image_nested_reporting import patient_fold_bootstrap
    probabilities = {("image" if name == "Raw spline" else name): values
                     for name, values in probability_sets.items()}
    absolute, paired = patient_fold_bootstrap(frame, probabilities,
        n_bootstrap=repetitions, seed=42)
    absolute = absolute.rename(columns={"model": "model_name", "valid_replicates": "n_valid"})
    absolute["model_name"] = absolute.model_name.replace({"image": "Raw spline"})
    absolute["reference_model"] = None
    comparisons = []
    for name in probability_sets:
        if name == "Raw spline":
            continue
        rows = paired.loc[paired.comparison.eq("image minus " + name)].copy()
        lower = -rows.ci_upper.to_numpy()
        rows["ci_upper"] = -rows.ci_lower.to_numpy()
        rows["ci_lower"] = lower
        rows["estimate"] = -rows.difference
        rows["model_name"], rows["reference_model"] = name, "Raw spline"
        rows = rows.rename(columns={"valid_replicates": "n_valid"})
        comparisons.append(rows)
    return pd.concat([absolute, *comparisons], ignore_index=True)


def _code(name, index, *, plot=None, replacements=None):
    """Transfer a specifically reviewed non-fitting original code cell."""
    source = original_cells(name)[index - 1].source
    for old, new in (replacements or {}).items():
        if old not in source:
            raise ValueError(f"Missing replacement in {name} cell {index}: {old!r}")
        source = source.replace(old, new)
    if plot:
        source = source.replace("plt.show()", f"finish_figure(fig, {plot!r})")
    if re.search(r"\.(?:fit|fit_transform)\s*\(|(?:search|evaluate_outer_fold|fit_final_bundle)\s*\(", source):
        raise ValueError(f"Historical fitting code cannot enter {name} cell {index}.")
    return source


def _md(name, updates=None):
    """Keep full original prose, changing clean identities and explicit findings."""
    result = {}
    for index, cell in enumerate(original_cells(name), start=1):
        if cell.cell_type != "markdown":
            continue
        text = cell.source
        for old, new in {
            "Subset data": "deduplicated Subset data",
            "03_patient_splitting.ipynb": "03_patient_fold_verification.ipynb",
            "04_dummy_logistic_baseline.ipynb": "04_metadata_baselines.ipynb",
            "09_image_b0_pilot.ipynb": "09_image_modelling.ipynb",
            "6,864": "6,706", "6,736": "6,579", "1.86%": "1.89%",
        }.items():
            text = text.replace(old, new)
        text = re.sub(r"\b128(?= (?:malignant images|images are malignant))", "127", text)
        replacement = (updates or {}).get(index)
        if isinstance(replacement, str):
            text = dedent(replacement).strip()
        elif replacement:
            for old, new in replacement.items():
                if old not in text:
                    raise ValueError(f"Missing Markdown replacement in {name} cell {index}: {old!r}")
                text = text.replace(old, new)
        result[index] = text
    return result


COMMON = '''
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from sklearn.metrics import (
    PrecisionRecallDisplay, RocCurveDisplay, average_precision_score,
    balanced_accuracy_score, brier_score_loss, confusion_matrix, log_loss,
    roc_auc_score, roc_curve, precision_recall_curve,
)
from melanoma_detection.metadata_evaluation import probability_metrics, decision_metrics, calibration_bins
from melanoma_detection.dedup_notebook_metadata import (
    saved_metadata, metric_table, exploratory_cutoff, reliability_table,
    decision_intervals, subgroup_diagnostics, save_metadata_table,
)
from melanoma_detection.dedup_notebook_metadata_templates import marginal_age_profile, metadata_patient_intervals
bundle = score_bundle()
frame, scores = bundle["expected"], bundle["scores"]
meta = saved_metadata(ROOT, bundle)
PROJECT_ROOT = ROOT
train = bundle["frame"].set_index("image_name").loc[frame.image_id].reset_index()
assert np.array_equal(train.patient_id, frame.patient_id)
assert np.array_equal(train.target, frame.target)
assert np.array_equal(train.fold, frame.fold)
assert train.groupby("patient_id").fold.nunique().eq(1).all()
assert train.image_name.is_unique and len(train) == 6706
assert int(train.target.sum()) == 127
FEATURE_COLUMNS = ["sex", "age_approx", "anatom_site_general_challenge"]
FEATURES = FEATURE_COLUMNS
DATA_PATH = ROOT / "data/subset/processed/dedup_v1/dataset_manifest.json"
TRAIN_FOLDS_PATH = DATA_PATH
REPORT_DIR = ROOT / "results/dedup_v1/clean_workflow/metadata_depth" / NOTEBOOK_STAGE
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES = FIGURE_DIR
TABLES = METRICS = PREDICTIONS = REPORT_DIR
plt.rcdefaults()
'''

CHECK_DATA = '''
print(f"Manifest: {DATA_PATH.relative_to(ROOT)}")
print(f"Dataset identity: {bundle['identity']['dataset_identity']}")
print(f"Images: {len(train):,}; patients: {train.patient_id.nunique():,}")
print(f"Malignant images: {int(train.target.sum()):,} ({train.target.mean():.2%})")
print(f"Malignant patients: {train.loc[train.target.eq(1), 'patient_id'].nunique():,}")
print("Original patient folds retained; reserved test not loaded.")
'''

BINARY_METRICS = '''
def binary_probability_metrics(targets, probabilities):
    values = probability_metrics(targets, probabilities)
    return {"roc_auc": values["roc_auc"], "average_precision": values["average_precision"],
            "log_loss": values["log_loss"], "brier_score": values["brier"]}
'''

THRESHOLD_HELPER = '''
def select_balanced_accuracy_threshold(y_true, probabilities):
    cutoff = exploratory_cutoff(y_true, probabilities)
    return cutoff, float(balanced_accuracy_score(y_true, np.asarray(probabilities) >= cutoff))
'''

PRIMARY_NOTE = "The clean workflow's primary ranking measure is **unweighted mean outer-fold AP**. The original pooled tables and figures are retained for presentation consistency and are explicitly secondary; their values are not interchangeable with mean-fold scores."
ROBUSTNESS_NOTE = "The clean workflow's primary ranking measure is **unweighted mean outer-fold AP**. Original figure composition and colour meanings are retained. The paired interval plot explicitly displays mean-fold AP, while pooled values remain secondary tables; their values are not interchangeable."


def _baseline():
    name = "04_dummy_logistic_baseline.ipynb"
    codes = {
        4: COMMON + BINARY_METRICS + '''
MODEL_LABELS = {"dummy": "Prevalence dummy", "logistic": "Logistic regression"}
COLORS = {"dummy": "#94A3B8", "logistic": "#2563EB"}
''' + CHECK_DATA,
        6: '''
estimators = {"dummy": "saved dummy", "logistic": "saved categorical logistic"}
oof_predictions = train[["image_name", "patient_id", "fold", "target"]].copy()
oof_predictions["dummy_probability"] = scores["dummy"]
oof_predictions["logistic_probability"] = scores["categorical_logistic"]
assert oof_predictions.filter(like="probability").notna().all().all()
print(f"Loaded verified OOF probabilities for {len(oof_predictions):,} images.")
print("Distinct dummy probabilities:")
print(np.sort(oof_predictions.dummy_probability.unique()))
''',
        8: _code(name, 8) + '''
display(metrics.round(4))
primary = metric_table(frame, scores, ["dummy", "categorical_logistic"]).query("scope == 'mean_fold'")
display(primary.set_index("model")[["average_precision", "roc_auc", "log_loss", "brier"]].round(5))
''',
        9: _code(name, 9) + '''
assert np.allclose(dummy_fold_check["ROC-AUC"], .5)
assert np.allclose(dummy_fold_check["Average precision"], dummy_fold_check["Validation prevalence"])
display(dummy_fold_check.round(6))
''',
        11: _code(name, 11, plot="baseline_roc_pr"),
        13: THRESHOLD_HELPER + _code(name, 13, replacements={
            original_cells(name)[12].source.split("y_true =", 1)[0]: ""}),
        15: _code(name, 15, plot="baseline_exploratory_confusions") + "\ndisplay(confusion_results.round(4))\n",
    }
    md = _md(name, {
        1: {"0.0418": "0.0430", "0.0184": "0.0184 (pooled); primary mean-fold AP is 0.0567 versus 0.0189"},
        3: {"We check that each patient belongs to one fold before training.": "We check that each patient belongs to one preserved fold and match saved clean predictions. Exact pixel-copy removal retains all 422 patients and 88 malignant patients; 158 copies were removed, leaving 6,706 images and 127 malignant images."},
        5: {"### 2. Generate held-out probabilities": "### 2. Load and verify held-out probabilities", "Both methods train and predict": "The saved clean methods trained and predicted"},
        7: {"They need not be identical.": "They need not be identical. The added mean-fold table is primary for this clean workflow; mean-fold AP is 0.0189 for the dummy and 0.0567 for logistic regression."},
        14: {"51 of 127 malignant images (39.8%)": "51 of 127 malignant images (40.2%)", "1,170 of 6,579 benign images (17.4%)": "1,170 of 6,579 benign images (17.8%)"},
        17: {"0.0418": "0.0430", "0.4952": "0.4877", "0.5945": "0.5943", "Keep the dummy": "On the primary mean-fold measure, AP is 0.0189 for the dummy and 0.0567 for logistic regression. Keep the dummy"},
    })
    return ("04_metadata_baselines.ipynb", "04 · Compare clean dummy and logistic baselines", "", template_sections(name, codes, md, context_note=PRIMARY_NOTE))


def _models():
    name = "05_metadata_models.ipynb"
    labels = '''
MODEL_LABELS = {"dummy": "Prevalence dummy", "logistic": "Standard logistic regression",
    "spline_logistic": "Spline logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost"}
ALIASES = {"dummy": "dummy", "logistic": "categorical_logistic", "spline_logistic": "spline_raw",
    "random_forest": "random_forest", "xgboost": "xgboost"}
CURVES_PATH = FIGURE_DIR / "metadata_baseline_curves_dedup_v1.png"
set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 120
'''
    ablation = original_cells(name)[22].source
    ablation_prefix, ablation_plot = ablation.split("plot_data =", 1)
    age_plot = original_cells(name)[25].source.split("observed_age_risk =", 1)[1]
    codes = {
        5: COMMON + BINARY_METRICS + labels,
        7: '''
display(pd.DataFrame([{"Model": "Categorical models", "Age": "one-hot age categories", "Other fields": "training-learned one-hot sex/site"},
    {"Model": "Spline logistic", "Age": "training median; four quantile knots; cubic spline; training scale", "Other fields": "training-learned one-hot sex/site"}]))
display(pd.DataFrame([meta["core"]["specification"]["spline"]]).T)
''',
        9: '''
assert train.groupby("patient_id").fold.nunique().eq(1).all()
display(train.groupby("fold").agg(images=("target", "size"), malignant_images=("target", "sum"), patients=("patient_id", "nunique")))
print("Saved receipt verified; no fresh model fitting is performed by this notebook.")
''',
        11: CHECK_DATA,
        13: '''
estimators = dict(ALIASES)
oof_predictions = train[["image_name", "patient_id", "fold", "target"]].copy()
for model_name, score_name in ALIASES.items():
    oof_predictions[f"{model_name}_probability"] = scores[score_name]
assert oof_predictions.filter(like="probability").notna().all().all()
print(f"Loaded {len(oof_predictions):,} matched clean OOF predictions for five models.")
''',
        15: _code(name, 15) + '''
display(overall_metrics.round(4))
primary_metrics = metric_table(frame, scores, list(ALIASES.values())).query("scope == 'mean_fold'")
display(primary_metrics.set_index("model")[["average_precision", "roc_auc", "log_loss", "brier"]].round(5))
''',
        18: _code(name, 18) + "\ndisplay(fold_model_metrics.round(4))\n",
        21: _code(name, 21, plot="metadata_five_model_curves"),
        23: '''
ablation_feature_sets = {"Sex only": ["sex"], "Age only": ["age_approx"], "Site only": ["anatom_site_general_challenge"],
    "Sex + age": ["sex", "age_approx"], "Sex + site": ["sex", "anatom_site_general_challenge"],
    "Age + site": ["age_approx", "anatom_site_general_challenge"], "All features": FEATURE_COLUMNS}
ablation_keys = ["ablation_sex", "ablation_age", "ablation_site", "ablation_sex_age", "ablation_sex_site", "ablation_age_site", "ablation_all_features"]
ablation_rows = [{"Feature set": label, "Features": ", ".join(ablation_feature_sets[label]),
    **binary_probability_metrics(train.target, scores[key])}
    for label, key in zip(ablation_feature_sets, ablation_keys)]
assert np.allclose(scores["ablation_all_features"], scores["spline_raw"], rtol=0, atol=1e-14)
ablation_results = pd.DataFrame(ablation_rows).set_index("Feature set")
dummy_average_precision = overall_metrics.loc["Prevalence dummy", "average_precision"]
ablation_results["Average precision gain vs dummy"] = ablation_results.average_precision - dummy_average_precision
display(ablation_results.round(4))
display(metric_table(frame, scores, ablation_keys).query("scope == 'mean_fold'").set_index("model").round(5))
plot_data = ''' + ablation_plot.replace("plt.show()", "finish_figure(fig, 'metadata_seven_feature_ablations')"),
        26: "marginal_age_risk = marginal_age_profile(train, meta)\nobserved_age_risk = " + age_plot.replace("plt.show()", "finish_figure(fig, 'metadata_observed_and_marginal_age')"),
        29: _code(name, 29) + "\ndisplay(category_prevalence.round(2))\n",
        32: THRESHOLD_HELPER + _code(name, 32, replacements={original_cells(name)[31].source.split("y_true =", 1)[0]: ""}),
        34: _code(name, 34, plot="metadata_exploratory_confusions") + "\ndisplay(confusion_results.round(4))\n",
    }
    md = _md(name, {
        1: {"0.0484": "0.0492 pooled (0.0648 mean-fold)", "0.0184": "0.0184 pooled (0.0189 mean-fold)"},
        8: {"We create a fresh copy": "The separate clean runner created a fresh copy", "The helper keeps those predictions in the original image order.": "This report reads verified saved predictions and aligns them with retained image identities; it does not refit models."},
        10: {"Before fitting": "Before reading the saved fits"},
        12: {"### 1. Generate patient-safe out-of-fold predictions": "### 1. Load patient-safe out-of-fold predictions", "For each fold, the model trains": "In the completed clean run, each model trained", "and predicts": "and predicted"},
        14: {"Use AP first": "Use the added mean-fold AP table first; pooled metrics retain the original descriptive view. Use AP first"},
        16: {"0.0484": "0.0492", "The spline leads on pooled AP and ROC-AUC.": "The spline leads on pooled and mean-fold AP and on pooled ROC-AUC among these fixed models. Its primary mean-fold AP is 0.0648."},
        19: {"0.0386 to 0.1103": "0.0373 to 0.1122"},
        22: {"We refit": "The separate clean runner refitted"},
        24: {"0.0410": "0.0412", "0.0264": "0.0271", "0.0479": "0.0487", "0.0196": "0.0197", "0.0484": "0.0492", "0.655 to 0.650": "0.653 to 0.643", "Age and site account": "These are pooled AP values, matching the original chart. Primary mean-fold AP is 0.0458 for age, 0.0636 for age plus site and 0.0648 for all three fields. Age and site account"},
        25: {"This descriptive view fits the spline on all development images.": "**Experiment-scope adaptation:** the original fitted a new all-development spline for this descriptive view. The clean report instead averages predictions from the five saved outer-training spline fits, so no additional model is fitted.", "average the model scores over their recorded sex and site.": "average each saved model's scores over the clean records' sex and site, then take the equal mean across saved models. The observed-rate points use the retained development images.", "The gold line is the average model score": "The gold line is this mean marginal model score"},
        27: {"age 90 has only 16 images": "age 90 has only 16 retained images"},
        37: {"0.0484": "0.0492", "0.6499": "0.6428", "0.0225": "0.0227", "66 of 127 malignant images (51.6%)": "65 of 127 malignant images (51.2%)", "1,498 of 6,579 benign images (22.2%)": "1,484 of 6,579 benign images (22.6%)", "Retain the fixed spline": "Its primary mean-fold AP is 0.0648, ahead of XGBoost (0.0591), categorical logistic (0.0567) and forest (0.0548). Retain the fixed spline"},
    })
    return (name, "05 · Compare clean metadata models and feature sets", "", template_sections(name, codes, md, context_note=PRIMARY_NOTE))


def _network():
    name = "06_metadata_neural_network.ipynb"
    codes = {
        3: COMMON + '''
BASE_SEEDS = [42, 142, 242]
spec = meta["core"]["specification"]["neural_network"]
epochs = pd.read_csv(meta["base"] / "epochs.csv")
history = pd.read_csv(meta["base"] / "history.csv")
print("Verified clean NN receipt:", (meta["base"] / "comparator_receipt.json").relative_to(ROOT))
''',
        5: CHECK_DATA + '''
for fold, rows in train.groupby("fold"):
    assert rows.target.nunique() == 2
display(train.groupby("fold").agg(images=("target", "size"), patients=("patient_id", "nunique"), malignant_images=("target", "sum")))
''',
        7: '''
display(pd.DataFrame([{"Input": "Age", "Saved transformation": "Training-median imputation then standard scaling"},
    {"Input": "Sex/site", "Saved transformation": "Fixed Missing label; training one-hot categories; unknown ignored"}]))
display(pd.DataFrame([spec]).T)
''',
        8: '''
display(epochs[["base_seed", "outer_fold", "training_seed", "stop_fold", "selected_epochs", "attempted_stopping_epochs", "stop_log_loss", "epoch_fit_patients", "epoch_stop_patients", "score_patients"]].round(5))
print("Saved stopping BCE chose duration; fresh saved outer refits used that duration.")
''',
        10: '''
nn_oof = {seed: scores[f"nn_seed_{seed}"] for seed in BASE_SEEDS}
nn_mean, spline_oof = scores["metadata_nn_mean"], scores["spline_raw"]
assert np.allclose(np.column_stack(list(nn_oof.values())).mean(axis=1), nn_mean,
    rtol=2*np.finfo(np.float32).eps, atol=1e-8)
training_log = epochs.copy()
checks = []
for fold in range(5):
    members = pd.read_csv(meta["base"] / f"fold_{fold}/selection_memberships.csv")
    for seed, rows in members.groupby("base_seed"):
        fitting = set(rows.loc[rows.role.eq("epoch_fitting"), "patient_id"])
        stopping = set(rows.loc[rows.role.eq("epoch_stopping"), "patient_id"])
        scoring = set(rows.loc[rows.role.eq("outer_validation"), "patient_id"])
        assert fitting.isdisjoint(stopping) and (fitting | stopping).isdisjoint(scoring)
        checks.append({"outer_fold": fold, "base_seed": seed, "fitting_patients": len(fitting),
            "stopping_patients": len(stopping), "scoring_patients": len(scoring), "overlap": 0})
display(pd.DataFrame(checks))
print("All fifteen saved patient-separated NN runs verified; no retraining.")
''',
        12: _code(name, 12, plot="nn_seed_stability", replacements={
            "Same d1 patients and outer folds": "Same dedup_v1 patients and outer folds",
            "PROJECT_ROOT / 'results/figures/metadata_nn_stability_d1.png'": "FIGURE_DIR / 'metadata_nn_stability_dedup_v1.png'"}) + '''
display(metric_table(frame, scores, ["spline_raw", "nn_seed_42", "nn_seed_142", "nn_seed_242", "metadata_nn_mean"]).query("scope == 'mean_fold'").set_index("model").round(5))
''',
        14: '''
output = train[["image_name", "patient_id", "target", "fold"]].copy()
for seed in BASE_SEEDS:
    output[f"p_nn_seed_{seed}"] = nn_oof[seed]
output["p_nn_mean"], output["p_spline"] = nn_mean, spline_oof
paths = []
for report_name, report in [("matched_predictions", output), ("pooled_metrics", results),
    ("training_variability", stability), ("selected_epochs", epochs)]:
    paths.append(save_metadata_table(ROOT, NOTEBOOK_STAGE, report_name, report))
print("New explanatory report copies:")
for path in paths:
    print(path.relative_to(ROOT))
display(pd.DataFrame([{"dataset_identity": bundle["identity"]["dataset_identity"],
    "source_receipt": str((meta["base"] / "comparator_receipt.json").relative_to(ROOT)),
    "test_set_loaded": False, "training_on_notebook_execute": False}]))
''',
    }
    md = _md(name, {
        1: {"0.0465": "0.0477 (pooled; 0.0596 mean-fold)", "0.0484": "0.0492 (pooled; 0.0648 mean-fold)", "0.0458 to 0.0486": "0.0464 to 0.0486 pooled (0.0539 to 0.0588 mean-fold)"},
        2: {"Use `data/subset/processed/train_folds.csv`": "Use the frozen `data/subset/processed/dedup_v1/dataset_manifest.json`, which retains 6,706 exact-copy-cleaned development records from the canonical CSV,", "Before these runs": "Before the completed clean runs"},
        9: {"### 2. Train the prescribed runs on five patient-held-out folds": "### 2. Verify the prescribed runs on five patient-held-out folds"},
        11: {"Each score pools held-out predictions from all five outer folds.": "The original chart pools held-out predictions from all five outer folds. Its format, colour meaning and annotations are retained here; the added mean-fold table is primary for the clean workflow.", "the primary measure": "the primary ranking measure (mean-fold aggregation in this clean workflow)"},
        13: '''
### 4. Save predictions and run settings

Save one report row per retained image so later checks compare identical patients. SHA-256 fingerprints identify the frozen dataset manifest, retained IDs, original patient memberships, prediction bytes and training source code. The verified clean comparator receipt records preprocessing, network design, stopping and refit settings, package versions and every saved artifact. Fingerprints support traceability rather than proving correctness.

**Execution adaptation:** the original notebook trained and replaced its `metadata_nn_*` outputs. This clean notebook reads those separately completed clean fits and writes explanatory report copies under `results/dedup_v1/clean_workflow/metadata_depth/06_metadata_neural_network/`. Training receipts and predictions remain the authority. The `dedup_v1` filenames identify retained development data; no historical `d1` result is relabelled or overwritten. The reserved test remains outside these reports.
''',
        16: {"0.0465": "0.0477 pooled (0.0596 mean-fold)", "0.6723": "0.6695 pooled (0.6767 mean-fold)", "0.0484": "0.0492 pooled (0.0648 mean-fold)", "0.6499": "0.6428 pooled (0.6556 mean-fold)", "0.0458 to 0.0486": "0.0464 to 0.0486 pooled; mean-fold AP ranges from 0.0539 to 0.0588"},
    })
    return (name, "06 · Examine the clean three-seed metadata NN", "", template_sections(name, codes, md, context_note=PRIMARY_NOTE))


def _tuning():
    name = "07_metadata_tuning.ipynb"
    codes = {
        4: COMMON + '''
from melanoma_detection.dedup_metadata_suite import GRIDS
from sklearn.model_selection import ParameterGrid
selected = pd.read_csv(meta["suite"] / "selected_parameters.csv")
candidates = pd.read_csv(meta["suite"] / "inner_candidates.csv")
PARAMETERS_PATH = REPORT_DIR / "metadata_tuning_best_parameters_dedup_v1.csv"
METRICS_PATH = REPORT_DIR / "metadata_tuning_metrics_dedup_v1.csv"
ROC_AUC_BARS_PATH = FIGURE_DIR / "metadata_tuning_roc_auc_dedup_v1.png"
AVERAGE_PRECISION_BARS_PATH = FIGURE_DIR / "metadata_tuning_ap_dedup_v1.png"
CURVES_PATH = FIGURE_DIR / "metadata_tuning_curves_dedup_v1.png"
''',
        6: CHECK_DATA,
        8: '''
grid_rows = [{"model": model, "parameter": parameter.rsplit("__", 1)[-1],
    "values": str(values), "candidates": len(list(ParameterGrid(grid)))}
    for model, grid in GRIDS.items() for parameter, values in grid.items()]
display(pd.DataFrame(grid_rows))
display(pd.DataFrame([meta["core"]["specification"]["spline"]]).T)
''',
        10: '''
model_setups = {"Spline logistic": ("spline_raw", "spline_tuned"),
    "Random forest": ("random_forest", "random_forest_tuned"), "XGBoost": ("xgboost", "xgboost_tuned")}
untuned_oof = {name: scores[keys[0]] for name, keys in model_setups.items()}
tuned_oof = {name: scores[keys[1]] for name, keys in model_setups.items()}
family_labels = {"spline": "Spline logistic", "random_forest": "Random forest", "xgboost": "XGBoost"}
parameter_records = []
for row in selected.itertuples():
    text = ", ".join(f"{key.rsplit('__', 1)[-1]}={value}" for key, value in json.loads(row.best_parameters).items())
    parameter_records.append({"Outer fold": row.outer_fold, "Model": family_labels[row.model],
        "Inner best average precision": row.inner_best_average_precision, "Best parameters": text})
assert len(parameter_records) == 15 and len(candidates) == 210
print("Loaded completed patient-nested searches: 630 inner fits and 15 tuned outer refits.")
''',
        12: _code(name, 12),
        14: _code(name, 14) + '''
primary_rows = []
for model_name, keys in model_setups.items():
    for version, key in zip(["Untuned", "Nested-tuned"], keys):
        row = metric_table(frame, scores, [key]).query("scope == 'mean_fold'").iloc[0]
        primary_rows.append({"Model": model_name, "Version": version, "Mean-fold AP": row.average_precision,
            "Mean-fold ROC-AUC": row.roc_auc, "Mean-fold log loss": row.log_loss, "Mean-fold Brier": row.brier})
display(pd.DataFrame(primary_rows).set_index(["Model", "Version"]).round(5))
''',
        16: _code(name, 16, replacements={"plt.show()": "finish_figure(fig, save_path.stem)"}),
        18: _code(name, 18, plot="nested_three_family_roc_pr"),
        20: _code(name, 20),
    }
    md = _md(name, {
        1: {"0.0484 to 0.0521": "0.0492 to 0.0502 pooled (0.0648 to 0.0688 mean-fold)", "0.6499 to 0.6440": "0.6428 to 0.6332 pooled"},
        2: {"Split the other patients into three new inner folds, keeping each patient together.": "Use three inner patient folds within the other patients. **Membership qualification:** no original execution membership receipt was saved, so the separate clean runner reconstructed the original seed-42-plus-outer split using the original 6,864-row CSV order, froze those patient assignments and then filtered retained images. This is a documented reconstruction, not a recovered execution receipt; no splitter is rerun on the 6,706 retained rows."},
        5: {"development data": "retained development data"},
        7: {"### 3. Define the model pipelines": "### 3. Inspect the saved model pipelines and search grids", "The next cell": "The saved clean specification"} if "The next cell" in original_cells(name)[6].source else {"### 3. Define the model pipelines": "### 3. Inspect the saved model pipelines and search grids"},
        9: {"### 4. Train the fixed models and run the setting searches": "### 4. Load the fixed models and completed setting searches", "both versions train": "the completed versions trained", "and predict": "and predicted", "also uses": "also used"},
        13: {"The first table pools": "The original first table pools", "Read the pooled score": "The additional mean-fold table is the primary clean ranking view. Read the pooled score"},
        15: {"both tree models fall on both": "both tree models lose AP, while forest ROC-AUC rises and XGBoost ROC-AUC falls", "There are no uncertainty bars": "These preserve the original separate AP and ROC-AUC figures and pooled aggregation. Read them alongside primary mean-fold scores. There are no uncertainty bars"},
        22: {"0.0484 to 0.0521": "0.0492 to 0.0502", "0.6499 to 0.6440": "0.6428 to 0.6332", "0.0443 to 0.0373": "0.0433 to 0.0406", "0.0439 to 0.0383": "0.0434 to 0.0379", "Retain the fixed spline": "These are the original-style pooled comparisons. Mean-fold AP changes from 0.0648 to 0.0688 for spline, 0.0548 to 0.0521 for forest and 0.0591 to 0.0526 for XGBoost. Retain the fixed spline"},
    })
    return (name, "07 · Inspect clean nested metadata tuning", "", template_sections(name, codes, md, context_note=PRIMARY_NOTE))


def _robustness():
    name = "08_metadata_robustness.ipynb"
    bootstrap_plot = original_cells(name)[9].source.split("display(intervals.loc", 1)[1]
    calibration_plot = original_cells(name)[11].source.split("calibration_tables =", 1)[1]
    decision_report = original_cells(name)[13].source.split("decision_rows, decision_ci_tables =", 1)[1]
    subgroup_plot = original_cells(name)[15].source.split("fig, axes =", 1)[1]
    codes = {
        4: COMMON + '''
TARGET_RECALL = .80
BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED = 2000, 42
''',
        6: CHECK_DATA + '''
nn_predictions = pd.read_csv(meta["base"] / "metadata_nn_oof.csv")
nn_predictions = nn_predictions.set_index("image_name").loc[train.image_name].reset_index()
for column in ["patient_id", "target", "fold"]:
    assert np.array_equal(train[column], nn_predictions[column])
display(pd.DataFrame([{"Images": len(train), "Patients": train.patient_id.nunique(),
    "Malignant images": int(train.target.sum()), "Patients with malignancy": train.loc[train.target.eq(1), "patient_id"].nunique(),
    "Malignant image fraction": train.target.mean()}]).round(4))
''',
        8: '''
outer_predictions = meta["calibration"].copy()
split_audit = pd.read_csv(meta["base"] / "spline_inner_audit.csv")
assert split_audit.patient_overlap.eq(0).all()
np.testing.assert_allclose(outer_predictions.p_spline_raw, scores["spline_raw"], rtol=0, atol=1e-14)
fold_rows = []
for fold, rows in outer_predictions.groupby("fold"):
    for version, p_column, decision_column, threshold_column in [
        ("Raw spline", "p_spline_raw", "decision_raw", "raw_threshold"),
        ("Recalibrated spline", "p_spline_calibrated", "decision_calibrated", "calibrated_threshold")]:
        fold_rows.append({"fold": fold, "procedure": version, "training_selected_threshold": rows[threshold_column].iloc[0],
            **probability_metrics(rows.target, rows[p_column]), **decision_metrics(rows.target, rows[decision_column])})
fold_metrics = pd.DataFrame(fold_rows)
display(split_audit.round(5))
display(fold_metrics[["fold", "procedure", "training_selected_threshold", "sensitivity", "specificity", "precision", "flagged_fraction"]].round(4))
save_metadata_table(ROOT, NOTEBOOK_STAGE, "training_only_fold_metrics", fold_metrics)
''',
        10: '''
probability_sets = {"Raw spline": scores["spline_raw"], "Recalibrated spline": outer_predictions.p_spline_calibrated.to_numpy(),
    "NN seed 42": scores["nn_seed_42"], "NN seed 142": scores["nn_seed_142"], "NN seed 242": scores["nn_seed_242"],
    "NN mean of three": scores["metadata_nn_mean"]}
intervals_all = metadata_patient_intervals(frame, probability_sets, BOOTSTRAP_REPETITIONS)
save_metadata_table(ROOT, NOTEBOOK_STAGE, "paired_patient_intervals_all_aggregations", intervals_all)
intervals = intervals_all.query("aggregation == 'mean_fold'").copy()
display(intervals.loc''' + bootstrap_plot.replace('"Average precision")', '"Mean outer-fold average precision")')
            .replace('"AP with patient bootstrap intervals"', '"Mean-fold AP with patient bootstrap intervals"')
            .replace("plt.show()", "finish_figure(fig, 'metadata_paired_patient_intervals')"),
        12: '''
def bin_rate_intervals(analysis_frame, probabilities, bins, repetitions=2000):
    table = reliability_table(analysis_frame, {"current": probabilities}, repetitions=repetitions)
    return table[["bin", "observed_lower", "observed_upper"]]
calibration_tables = ''' + calibration_plot.replace("plt.show()", "finish_figure(fig, 'metadata_reliability_and_score_distributions')"),
        14: "decision_rows, decision_ci_tables = " + decision_report,
        16: '''
subgroup_table = subgroup_diagnostics(train, outer_predictions.p_spline_raw.to_numpy(),
    outer_predictions.decision_raw.to_numpy(), repetitions=1000)
subgroup_table = subgroup_table.loc[subgroup_table.field.ne("Site availability")].copy()
subgroup_table = subgroup_table.rename(columns={"malignant_images": "positive_images", "malignant_patients": "positive_patients"})
subgroup_table["sensitivity_lower"], subgroup_table["sensitivity_upper"] = np.nan, np.nan
ages = pd.to_numeric(train.age_approx, errors="coerce")
report_groups = {"Age band": pd.Series(np.where(ages.isna(), "Missing", np.where(ages < 50, "<50", ">=50")), index=train.index),
    "Sex": train.sex.fillna("Missing").astype(str), "Site": train.anatom_site_general_challenge.fillna("Missing").astype(str)}
for index, row in subgroup_table.iterrows():
    if row.positive_patients >= 10:
        mask = report_groups[row.field].eq(row.group).to_numpy()
        subset = train.loc[mask].reset_index(drop=True)
        ci = decision_intervals(subset, outer_predictions.loc[mask, "decision_raw"].to_numpy(), repetitions=1000)
        interval = ci.query("metric == 'sensitivity'").iloc[0]
        subgroup_table.loc[index, ["sensitivity_lower", "sensitivity_upper"]] = [interval.ci_lower, interval.ci_upper]
save_metadata_table(ROOT, NOTEBOOK_STAGE, "subgroup_checks", subgroup_table)
display(subgroup_table[["field", "group", "images", "patients", "positive_images", "positive_patients", "prevalence",
    "average_precision", "average_precision_lower", "average_precision_upper", "sensitivity", "evidence_note"]].round(4))
fig, axes = ''' + subgroup_plot.replace("plt.show()", "finish_figure(fig, 'metadata_subgroup_support_and_intervals')"),
        18: '''
display(pd.DataFrame([{"chosen_development_comparator": "Raw four-knot spline; C=1",
    "fitted_clean_final_model": False, "reserved_test_loaded": False,
    "historical_final_bundle": "models/metadata_d1/frozen_spline.joblib",
    "next_required_step": "Settle the complete recipe, fit a separate clean all-development bundle, then test once"}]))
print("No final fit or final-test evaluation is performed by this report.")
''',
        20: '''
recap = metric_table(frame, {"spline_raw": probability_sets["Raw spline"], "metadata_nn_mean": probability_sets["NN mean of three"]},
    ["spline_raw", "metadata_nn_mean"]).query("scope == 'mean_fold'").set_index("model")
raw_ap, nn_ap = recap.loc["spline_raw", "average_precision"], recap.loc["metadata_nn_mean", "average_precision"]
delta = intervals[(intervals.model_name == "NN mean of three") & (intervals.reference_model == "Raw spline") & intervals.metric.eq("average_precision")].iloc[0]
raw_decisions = decision_summary.query("procedure == 'Raw spline'").iloc[0]
print(f"Primary mean-fold AP: raw spline {raw_ap:.4f}; mean-of-three NN {nn_ap:.4f}.")
print(f"NN minus spline mean-fold AP: {delta.estimate:+.4f}, patient 95% interval [{delta.ci_lower:+.4f}, {delta.ci_upper:+.4f}].")
print(f"Training-target recall: {TARGET_RECALL:.0%}; held-out raw-spline sensitivity: {raw_decisions.sensitivity:.1%}.")
print(f"At that policy, {raw_decisions.flagged_fraction:.1%} of images are flagged; precision is {raw_decisions.precision:.1%}.")
print(f"Sparse reporting groups: {int(subgroup_table.positive_patients.lt(10).sum())} of {len(subgroup_table)}.")
print("Primary development comparator remains the raw spline. Final clean fit is deferred; reserved test remains closed.")
''',
    }
    md = _md(name, {
        1: {"79.7%": "80.3%", "67.0%": "68.6%"},
        2: {"repeatedly sample patients with replacement": "repeatedly sample patients with replacement separately within each original outer fold", "Save the chosen model, cutoff and source fingerprints before any later final-test evaluation.": "The completed clean fits and training-selected cutoffs are already fingerprinted. **Scope qualification:** the original notebook also fitted a final all-development bundle. This report defers that separate clean fit until modelling choices are settled; the historical bundle remains preserved."},
        3: {"Use the project's Python environment.": "Use the project's Python environment."} if "Use the project's Python environment." in original_cells(name)[2].source else {},
        5: {"recorded filename relocations are accepted only when the executable method is otherwise preserved": "the clean manifest and retained-record identity must match both metadata receipts"},
        7: {"### 3. Select calibration and cutoffs using training patients": "### 3. Verify calibration and cutoffs selected using training patients"},
        9: {"Draw 422 patients with replacement 2,000 times": "Draw the original number of patients within each outer fold with replacement, 2,000 times (422 patients in total per draw)", "AP is horizontal": "mean-fold AP is horizontal", "network AP minus spline AP": "network mean-fold AP minus spline mean-fold AP", "[-0.0190, +0.0059]": "[-0.0255, +0.0076]", "The network mean's interval": "**Aggregation adaptation:** the original figure displayed pooled AP. The clean workflow retains its blue-dot/gold-square layout but displays the primary mean-fold AP. Pooled intervals are also saved as a secondary table. The network mean's interval"},
        13: {"102 of 127 malignant images": "102 of 127 malignant images", "4,599 of 6,706": "4,597 of 6,706"},
        15: {"1,000 patient draws within the group": "1,000 whole-patient draws within the group and its original outer-fold strata"},
        17: '''
### 8. Prepare the retained model for later final fitting and testing

Keep the previously chosen raw four-knot spline as the development comparator. The original stage fitted it on all development patients and selected its academic cutoff from five-fold development OOF scores, then saved the model, cutoff, training identities, source fingerprints and dependency versions. That combination gives a later test script a complete fixed recipe.

**Experiment-scope change:** this clean report deliberately defers all-development fitting until the final procedure is chosen. It reads the five clean outer models and their training-selected calibration/cutoff artifacts; these support development evaluation, rather than providing a final deployable model. The historical `models/metadata_d1` bundle belongs to the original 6,864-image data and cannot be relabelled as deduplicated.

After settling features, model settings, calibration and cutoff rules, fit a separately named clean bundle on all 6,706 development records, preserving patient identities and exact deduplication fingerprints. Verify reserved-test patient/image separation and the consistent duplicate policy before scoring. Evaluate once with the saved procedure, without adjusting it to improve the test result. This notebook performs neither that fit nor the test.
''',
        19: {"Read these as development findings": "The recap uses primary mean-fold AP and its matching paired interval, alongside pooled image-level decision rates. Read these as development findings"},
        21: {"three of eleven": "three of eleven", "Flagging two-thirds": "Flagging roughly 69%"},
        22: '''
## Summary

The raw spline has primary mean-fold AP 0.0648; the three-network mean has 0.0596. Their difference is -0.0053 with a 95% paired patient interval of [-0.0255, +0.0076], giving no clear network improvement. The original-style pooled AP values are 0.0492 and 0.0477. Recalibration slightly lowers probability error; within-fold ranking is unchanged by its monotonic mapping, although pooled ranking shifts because folds use different mappings.

Keep the raw spline as the predeclared development comparator. The training-selected 80% recall scenario finds 102 of 127 malignant images (80.3%), flags 4,597 of 6,706 images (68.6%) and has 2.2% precision. Raw and recalibrated procedures make the same decisions here. That workload limits metadata alone, and three of eleven age/sex/site reporting groups have fewer than ten malignant patients.

**Final clean fitting remains deferred.** Existing outer models and cutoffs are saved for development assessment; the historical original-data final bundle remains preserved. The reserved test has not been evaluated. Next, [09_image_modelling.ipynb](09_image_modelling.ipynb) assesses clean image information. Complete and freeze the modelling procedure before fitting a separate clean final bundle and evaluating the reserved test once.
''',
    })
    return (name, "08 · Read clean metadata uncertainty, calibration and decisions", "", template_sections(name, codes, md, context_note=ROBUSTNESS_NOTE))


def _append_diagnostics(spec, selected_sections):
    """Keep clean diagnostic additions after the full original narrative."""
    from .dedup_notebook_metadata import INITIALISE, metadata_depth_specs
    filename, title, purpose, sections = spec
    legacy = next(item for item in metadata_depth_specs() if item[0] == filename)
    added = [("__markdown__", "## Additional deduplicated-workflow diagnostics\n\n"
        "The complete original section sequence above is retained as the template. "
        "The following clearly labelled additions document clean artifact verification, "
        "training support and explanatory diagnostics that were not figures in the original. "
        "They remain secondary development evidence and do not select another model or cutoff.", ""),
        ("__code__", "", INITIALISE)]
    if filename.startswith("05"):
        added.append(("__code__", "", "coefficients, importances, age_profiles = estimator_diagnostics(meta)"))
    if filename.startswith("04"):
        added.append(("__code__", "", "metrics = metric_table(frame, scores, ['dummy', 'categorical_logistic'])"))
    if filename.startswith("08"):
        added.append(("__code__", "", 'calibration = meta["calibration"]'))
        added.append(("__code__", "", "COLORS = ['#2563EB', '#C58A00', '#6B7F2A', '#EA580C', '#303030', '#94A3B8']"))
    for number in selected_sections:
        heading, description, code = legacy[3][number - 1]
        if filename.startswith(("07", "08")):
            code = code.replace("#d97706", "#C58A00")
        if filename.startswith("05"):
            code = code.replace('color="#0f766e"', 'color={"random_forest": "#6B7F2A", "xgboost": "#EA580C"}[model]')
        added.append(("__markdown__", f"### Added diagnostic: {heading}\n\n{description}", ""))
        added.append(("__code__", "", code))
    return filename, title, purpose, sections + added


def metadata_template_specs():
    """Five original metadata narratives with their actual figure templates."""
    from .dedup_notebooks import SETUP
    from .metadata_nn_tuning_reporting import metadata_nn_notebook_sections
    from .metadata_nn_tuning_reporting import metadata_nn_assessment_sections as assessment_sections
    specifications = [
        _append_diagnostics(_baseline(), [7]),
        _append_diagnostics(_models(), [7]),
        ("06_metadata_neural_network.ipynb", "06 · Examine the clean fixed metadata NN", "",
         metadata_nn_notebook_sections("dedup_v1", "06")),
        ("07_metadata_tuning.ipynb", "07 · Inspect clean nested tuning including NN", "",
         metadata_nn_notebook_sections("dedup_v1", "07")),
        _append_diagnostics(_robustness(), [3, 5, 6, 7]),
    ]
    standalone = []
    for filename, title, purpose, sections in specifications:
        if filename.startswith(("06", "07")):
            standalone.append((filename, title, purpose, sections))
            continue
        if filename.startswith("08"):
            sections = sections + assessment_sections("dedup_v1")
            sections = [(kind, markdown.replace("NN mean of three", "Fixed NN mean of three"), code.replace("NN mean of three", "Fixed NN mean of three").replace(
                'REPORT_DIR = ROOT / "results/dedup_v1/clean_workflow/metadata_depth" / NOTEBOOK_STAGE',
                'REPORT_DIR = NN_REPORT / "tables"')) for kind, markdown, code in sections]
        for index, (kind, markdown, code) in enumerate(sections):
            if kind == "__code__":
                sections[index] = (kind, markdown, f"NOTEBOOK_STAGE = {filename[:-6]!r}\n" + SETUP + "\n" + code)
                break
        standalone.append((filename, title, purpose, sections))
    return standalone
