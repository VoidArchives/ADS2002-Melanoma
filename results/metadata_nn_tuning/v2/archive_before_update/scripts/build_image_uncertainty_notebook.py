"""Build the current educational uncertainty notebook; never execute training."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    cells = []
    def md(source):
        cells.append(nbf.v4.new_markdown_cell(source.strip() + "\n"))
    def code(source):
        cells.append(nbf.v4.new_code_cell(source.strip() + "\n"))

    md("""
# 13 — Does image uncertainty help prioritise review?

An image model produces a score between 0 and 1: higher means more strongly
ranked toward melanoma. **Uncertainty** here means how much that score varies
when we repeat prediction under specified changes. A high score and a high
uncertainty estimate answer different questions.

We test whether variation improves the order in which images are sent for
review, beyond simply sending the highest ordinary scores. We load an existing
**checkpoint** (saved model weights) and perform inference only. The default is
the weak-augmentation B0 control, using its recorded best epoch rather than the
final epoch 12. An epoch is one pass through the fitting data.

These are **inner-validation diagnostics**: patients excluded from fitting also
helped choose the checkpoint and pilot recipes. Direct training leakage is
checked, but that selection still makes results exploratory. Full out-of-fold
(OOF) evaluation would test the complete selection process on independent outer
patients; the reserved test comes last. Neither is performed here, and a pilot
AP leader does not become the final architecture.

The [historical three-epoch experiment](../archive/image_uncertainty_three_epoch_fold0.ipynb)
and its result files are preserved. Every current execution writes a new directory
under `results/image_uncertainty_runs/`. See the [concept guide](../concepts.md)
when a term is unfamiliar.
""")
    md("""
## 1. Choose a saved checkpoint explicitly

The shared loader rebuilds the trained network and exact deterministic image
preparation. Checks verify the run, recipe (training settings), best epoch,
checkpoint hash (file fingerprint), patient memberships and saved predictions.

The defaults below select the experimental B0 control. To inspect another
supported checkpoint, set `SOURCE_RUN`, `ARCHITECTURE` and `RECIPE_ID` together:
for example `architecture_pilot_3h_v1`, `densenet121`, `low_none`, or
`architecture_pilot_3h_v1`, `convnext_tiny`, `high_sqrt`. This is an explicit
diagnostic choice, not automatic architecture selection. `CONDITION_ID` applies
only to a controlled B0 condition. Missing or incompatible inputs stop execution.
""")
    code("""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from IPython.display import display, Markdown

PROJECT_ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
    if (path / "src/melanoma_detection").is_dir() and (path / "configs/image_model.yaml").is_file())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from melanoma_detection.image_uncertainty import (
    load_selected_checkpoint, run_smoke, run_inference, tta_transform,
    snapshot_preserved_artifacts, assert_preserved_artifacts,
)
from melanoma_detection.image_uncertainty_evaluation import (
    validate_raw, aggregate_raw, convergence_summary,
    evaluate_referral, referral_patient_bootstrap,
)
from melanoma_detection.image_model import file_sha256, transformation_profile, make_transform
from melanoma_detection.metadata_evaluation import probability_metrics, calibration_bins

SOURCE_RUN = "b0_augmentation_12e_v1"
ARCHITECTURE = "efficientnet_b0"
RECIPE_ID = "lr3e5_sqrt"
CONDITION_ID = "A"
MC_PASSES = 30
INFERENCE_SEED = 20261004
REFERRAL_SEED = 17042
BOOTSTRAP_SEED = 20242
REFERRAL_FRACTIONS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
RANDOM_REPEATS = 100
PATIENT_BOOTSTRAPS = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUTPUT_DIR = PROJECT_ROOT / "results/image_uncertainty_runs" / f"{ARCHITECTURE}_{RECIPE_ID}_mc30_tta4_{stamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
(OUTPUT_DIR / "figures").mkdir()
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})
COLORS = {"uncertainty_sd": "#245D8C", "prediction_score": "#B46A20",
    "predictive_entropy": "#747133", "random": "#666666"}
LINESTYLES = {"uncertainty_sd": "-", "prediction_score": "--",
    "predictive_entropy": "-.", "random": ":"}
def save_json(name, value):
    (OUTPUT_DIR / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
def save_figure(name, figure):
    figure.savefig(OUTPUT_DIR / "figures" / name, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figure)
print("Source:", SOURCE_RUN, "| architecture:", ARCHITECTURE, "| recipe:", RECIPE_ID)
print("Device:", DEVICE, "| outputs:", OUTPUT_DIR)
""")
    md("""
## 2. Recover the exact evaluation patients

We use the saved first inner-validation fold inside outer-training folds 1–4.
The outer fold 0 patients remain excluded. Patient IDs define separation; they
are not model inputs. The adapter verifies the ordered frame identities against
the checkpoint and validates source-image hashes before prediction. It never
loads a reserved-test table or predicts the excluded outer fold.

The weak B0 control has 3,660 fitting images / 225 patients and 1,834 validation
images / 113 patients, including 33 malignant images. Membership is reconstructed
from recorded artifacts, not recreated by a newly sampled split.
""")
    code("""
archive_path = PROJECT_ROOT / "notebooks/archive/image_uncertainty_three_epoch_fold0.ipynb"
protected = snapshot_preserved_artifacts(PROJECT_ROOT, archive_path=archive_path)
context = load_selected_checkpoint(PROJECT_ROOT, SOURCE_RUN, RECIPE_ID,
    architecture=ARCHITECTURE, condition_id=CONDITION_ID, device=DEVICE)
frame = context["frame"].assign(image_id=context["frame"].image_name)
save_json("checkpoint_and_split_provenance.json", context["provenance"])
display(pd.DataFrame([
    {"partition": name, "images": len(rows), "patients": rows.patient_id.nunique(),
     "malignant_images": int(rows.target.sum()) if name != "outer_excluded" else None}
    for name, rows in [("fitting", context["fitting_frame"]),
                       ("inner_validation", frame), ("outer_excluded", context["excluded_frame"])]
]))
print("Selected epoch:", context["checkpoint"]["epoch"], "| trained dropout:", context["spec"]["dropout"])
print("Deterministic preparation:", make_transform(context["spec"], training=False))
""")
    md("""
## 3. Check a small sample before full inference

The smoke check uses 16 images to catch a mismatch before a costly full run.
Deterministic predictions must repeat and agree with the saved export. Two
seeded dropout passes must vary under different seeds and repeat under the same
seed. Only explicit dropout may activate; BatchNorm buffers and all model state
must stay unchanged. Identity TTA (no image change) must reproduce the ordinary
score. Any failure stops execution before the 30-pass run.
""")
    code("""
smoke = run_smoke(context, n_images=16, seed=INFERENCE_SEED)
save_json("smoke_receipt.json", smoke)
display(pd.DataFrame(list(smoke.items()), columns=["Smoke check", "Result"]))
print("Smoke check passed; full inner-validation inference is now permitted.")
""")
    md("""
### Inspect the four conservative views

**Read the preview:** each row is one prepared image; columns are identity,
horizontal flip, vertical flip and both flips. `target=0` is benign, `target=1`
malignant. Check that the whole lesion, texture and colour remain visible; only
orientation should change. These operations rearrange pixels without deleting
them or creating reflected border fragments. One example from each class shows
framing, not representative prevalence. Labels do not choose transformations.
""")
    code("""
examples = pd.concat([frame[frame.target == 0].head(1), frame[frame.target == 1].head(1)])
prepare = make_transform(context["spec"], training=False)
mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
fig, axes = plt.subplots(len(examples), 4, figsize=(10, 5.5), constrained_layout=True, squeeze=False)
for row_index, row in enumerate(examples.itertuples()):
    with Image.open(row.image_path) as source:
        prepared = prepare(source.convert("RGB"))
    for col, view in enumerate(["identity", "hflip", "vflip", "hvflip"]):
        pixels = (tta_transform(prepared, view) * std + mean).permute(1, 2, 0).numpy()
        axes[row_index, col].imshow(np.clip(pixels, 0, 1))
        axes[row_index, col].set_title(f"{view} | target={row.target}")
        axes[row_index, col].axis("off")
fig.suptitle("Lossless full-image TTA after exact checkpoint preparation")
save_figure("conservative_tta_examples.png", fig)
""")
    md("""
## 4. Keep dropout variation and image-view variation separate

**MC dropout** (Monte Carlo dropout) predicts the same prepared image 30 times.
Dropout randomly disables some activations, creating a different mask each pass.
Only explicit dropout layers activate at their trained rate (0.2 here).
BatchNorm, which keeps learned normalisation statistics, and stochastic depth,
which can randomly bypass network blocks, stay in evaluation mode. Every pass
has a recorded seed. The mean is an ensemble **prediction score**; sample
standard deviation (**SD**, denominator 29) measures spread under those masks.
It does not capture all sources of uncertainty.

**TTA** (test-time augmentation) instead keeps the whole model in evaluation
mode and averages four views: identity, horizontal, vertical and both flips.
They act on the exact prepared tensor after whole-image resize and any recorded
preprocessing. No crop, padding, arbitrary rotation or colour change is added.
Flips preserve pixels and can be applied after normalisation without changing
their values. Sample SD (denominator 3) measures orientation sensitivity for
these four views, not uncertainty across every plausible image change. Its
numerical size cannot be compared with dropout SD as total uncertainty.

Sigmoid converts the network output to a score in [0,1]. These scores and their
means are **not calibrated probabilities**: class weighting can change their
scale. No score-to-probability mapping or decision threshold is fitted here.
""")
    code("""
inference = run_inference(context, mc_passes=MC_PASSES, seed=INFERENCE_SEED,
    batch_size=context["training"]["microbatch"], progress=lambda message: print(message, flush=True))
mc_raw, tta_raw = inference["mc_raw"], inference["tta_raw"]
mc_pass_ids = list(range(1, MC_PASSES + 1))
tta_pass_ids = ["identity", "hflip", "vflip", "hvflip"]
validate_raw(mc_raw, frame, mc_pass_ids)
validate_raw(tta_raw, frame, tta_pass_ids)
save_json("raw_prediction_checks.json", {
    "all_raw_scores_finite_in_unit_interval": True,
    "unique_image_pass_keys": True, "exact_image_and_pass_coverage": True,
    "patient_and_target_metadata_agree": True,
    "evaluation_images": len(frame), "mc_rows": len(mc_raw), "tta_rows": len(tta_raw),
    "mc_expected_pass_ids": mc_pass_ids, "tta_expected_pass_ids": tta_pass_ids,
})
mc_raw.to_csv(OUTPUT_DIR / "mc_raw_predictions.csv", index=False)
tta_raw.to_csv(OUTPUT_DIR / "tta_raw_predictions.csv", index=False)
inference["deterministic"].to_csv(OUTPUT_DIR / "deterministic_predictions.csv", index=False)
save_json("inference_checks.json", inference["checks"])
save_json("inference_provenance.json", inference["provenance"])
print("Validated raw rows:", len(mc_raw), "MC and", len(tta_raw), "TTA")
""")
    md("""
## 5. Validate before aggregation

Each raw row is one image and one pass/view. The validator rejects NaN, infinity,
scores outside [0,1], repeated image/pass keys, extra images or passes, changed
patient IDs/labels, and incomplete passes or image coverage. Only complete,
validated raw predictions enter a summary. The saved raw tables allow every
mean and SD to be independently recomputed.
""")
    code("""
mc_summary = aggregate_raw(mc_raw, frame, mc_pass_ids).rename(columns={
    "mean_score": "mc_mean_score", "score_sd": "mc_score_sd", "n_passes": "mc_n_passes"})
tta_summary = aggregate_raw(tta_raw, frame, tta_pass_ids).rename(columns={
    "mean_score": "tta_mean_score", "score_sd": "tta_score_sd", "n_passes": "tta_n_passes"})
summary = inference["deterministic"].merge(mc_summary,
    on=["image_id", "patient_id", "target"], validate="one_to_one").merge(tta_summary,
    on=["image_id", "patient_id", "target"], validate="one_to_one")
assert len(summary) == len(frame)
summary["split"] = "inner_validation"
summary["evaluation_status"] = "exploratory_inner_validation_not_outer_oof"
summary["source_run"] = SOURCE_RUN
summary["recipe_id"] = RECIPE_ID
summary["selected_epoch"] = context["checkpoint"]["epoch"]
summary.to_csv(OUTPUT_DIR / "prediction_summary.csv", index=False)
display(summary[["image_id", "target", "base_score", "mc_mean_score", "mc_score_sd",
                 "tta_mean_score", "tta_score_sd"]].head(8).round(5))
""")
    md("""
## 6. Check stability as the number of passes increases

Compare the first 5, 10 and 20 seeded passes with the same 30-pass reference;
labels do not select the pass count. **Read the two panels:** pass count is
horizontal. Left, mean absolute difference from the reference is vertical,
with separate lines for mean score and SD; closer to zero means more stable
values. Right, SD rank correlation is vertical; closer to 1 means a more similar
ordering of images. The table also gives overlap among the highest-SD 20%.

Stable scores need not imply stable rankings. The finite 30-pass reference is
not an exact expectation, so agreement is evidence of stability rather than
proof that all uncertainty has been captured.
""")
    code("""
convergence = convergence_summary(mc_raw, frame, mc_pass_ids)
convergence.to_csv(OUTPUT_DIR / "mc_convergence.csv", index=False)
display(convergence.round(5))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for column, label, style in [("mean_score_mae", "Mean score", "-"), ("score_sd_mae", "Score SD", "--")]:
    axes[0].plot(convergence["n_passes"], convergence[column], marker="o", linestyle=style, label=label)
axes[0].set(xlabel="Number of MC passes", ylabel="Mean absolute difference vs 30 passes",
            title="MC numerical stability — inner validation")
axes[0].legend()
axes[0].set_xticks(convergence.n_passes)
axes[1].plot(convergence["n_passes"], convergence["sd_spearman"], marker="o", color="#245D8C")
axes[1].set(xlabel="Number of MC passes", ylabel="SD rank correlation vs 30 passes",
            ylim=(0, 1.02), title="Uncertainty ranking stability")
axes[1].set_xticks(convergence.n_passes)
save_figure("mc_convergence.png", fig)
""")
    md("""
## 7. Compare ranking and descriptive reliability

**AP** (average precision) summarises precision (malignancy among flags) as
recall (malignant images found) changes with the cutoff; higher is better.
It is useful for rare malignancies. **ROC-AUC** summarises overall ranking;
higher is better and 0.5 is chance. **Log loss** penalises confident wrong scores
and **Brier** averages squared score errors; lower is better, but both can hide
poor recall when benign images dominate. The fitting-prevalence constant is a
simple baseline computed without validation labels, not a fitted calibrator.

**Read the reliability diagram:** each line is deterministic, MC mean or TTA
mean. Up to five equal-count bins group similar scores. Horizontal position is
the bin's mean score, vertical position its observed malignant-image fraction.
The diagonal means agreement; below it scores exceed the observed frequency,
above it scores fall below it. Left shows [0,1]; right enlarges the same observed
range and bins. A point near the diagonal is descriptive agreement, not proof
of calibrated clinical risk. Bin support is saved; sparse malignancies and
patients contributing to several bins make the estimates noisy. The checkpoint
was chosen on this split, so these checks may be optimistic.
""")
    code("""
score_columns = {"Deterministic": "base_score", "MC mean": "mc_mean_score", "TTA mean": "tta_mean_score"}
quality_rows, reliability_parts = [], []
for name, column in score_columns.items():
    quality_rows.append({"prediction": name, **probability_metrics(summary.target, summary[column])})
    bins = calibration_bins(summary.target, summary[column], n_bins=5, groups=summary.patient_id)
    bins["prediction"] = name
    reliability_parts.append(bins)
fit_prevalence = float(context["fitting_frame"].target.mean())
quality_rows.append({"prediction": "Constant fitting prevalence",
    **probability_metrics(summary.target, np.full(len(summary), fit_prevalence))})
quality = pd.DataFrame(quality_rows)
reliability = pd.concat(reliability_parts, ignore_index=True)
quality.to_csv(OUTPUT_DIR / "prediction_quality.csv", index=False)
reliability.to_csv(OUTPUT_DIR / "descriptive_reliability.csv", index=False)
display(quality.round(5))
print("Validation malignant-image fraction:", round(float(summary.target.mean()), 5),
      "| fitting prevalence:", round(fit_prevalence, 5))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
zoom_max = max(0.05, float(max(reliability.mean_probability.max(), reliability.observed_rate.max())) * 1.12)
for ax, limit, title in zip(axes, [1.0, zoom_max], ["Full score range", "Enlarged observed range"]):
    for (name, bins), color, style in zip(reliability.groupby("prediction", sort=False),
        ["#666666", "#245D8C", "#B46A20"], ["-", "--", "-."]):
        ax.plot(bins.mean_probability, bins.observed_rate, marker="o", color=color, linestyle=style, label=name)
    ax.plot([0, limit], [0, limit], linestyle=":", color="#333333", label="Score–frequency agreement")
    ax.set(xlabel="Mean prediction score (uncalibrated)", ylabel="Observed malignant-image fraction",
           xlim=(0, limit), ylim=(0, limit), title=title)
axes[0].legend(fontsize=9)
fig.suptitle("Descriptive reliability — same five bins on one inner-validation split")
save_figure("descriptive_reliability.png", fig)
""")
    md("""
## 8. Examine what the two SD measures capture

**Read the histograms:** the left panel bins MC score SD and the middle bins
TTA score SD horizontally; vertical height counts images. Mass near zero means
little change under that procedure. Each panel has its own SD scale because
dropout masks and image orientations perturb different things.

**Read the scatterplot:** each dot is an image, with MC SD horizontally and
TTA SD vertically. A rising pattern suggests the methods prioritise similar
images; scatter suggests different sensitivities. There is no equal-SD target
because the measures have different meanings. Variation does not prove malignancy
or error, and a model can be consistently, confidently wrong.
""")
    code("""
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
axes[0].hist(summary.mc_score_sd, bins=35, color="#245D8C")
axes[0].set(xlabel="MC score SD (30 passes)", ylabel="Images", title="Dropout-mask sensitivity")
axes[1].hist(summary.tta_score_sd, bins=35, color="#B46A20")
axes[1].set(xlabel="TTA score SD (4 views)", ylabel="Images", title="Orientation sensitivity")
axes[2].scatter(summary.mc_score_sd, summary.tta_score_sd, s=9, alpha=0.4, color="#666666")
axes[2].set(xlabel="MC score SD", ylabel="TTA score SD", title="Agreement between diagnostics")
save_figure("uncertainty_distributions.png", fig)
""")
    md("""
## 9. Does SD improve the referral order?

**Referral** means setting some images aside for review. The **retained set**
contains those still evaluated automatically. All rankings use the **same
deterministic predictions**, labels and predeclared referral fractions, so the
comparison isolates ranking rather than MC/TTA score averaging.

Rank from highest to lowest by SD, ordinary score, or predictive entropy
`−s log(s) − (1−s) log(1−s)`. Score ranking prioritises stronger melanoma scores;
entropy prioritises scores near 0.5, where the binary score is least decisive.
Entropy inherits the uncalibrated score scale. Random referral is repeated with
100 recorded seeds to give a simple baseline. Ties use image ID; no operational
fraction is selected from these labels.

Retained classification error uses the fixed descriptive cutoff 0.5; MAE
(mean absolute score error) and Brier also expose score errors. Malignant images
referred and distinct patients with a malignant image referred have different
denominators. Referral acts on images, not every image from a patient, and it
does not simulate whether a reviewer diagnoses correctly.

**Read the four panels:** the top row uses MC SD, the bottom TTA SD. Horizontal
position is percentage of all images referred. Left is retained classification
error (lower is better); right is percentage of malignant images referred
(higher is better). Lines compare SD, ordinary score, entropy and mean random
referral at the same budget. Compare at the same horizontal position; a lower
left curve alone may hide which malignant cases remain. These curves describe
trade-offs, not a selected clinical policy.
""")
    code("""
referral = evaluate_referral(summary, fractions=REFERRAL_FRACTIONS,
    n_random=RANDOM_REPEATS, seed=REFERRAL_SEED)
referral.to_csv(OUTPUT_DIR / "referral_rankings.csv", index=False)
print("Retained error: fixed descriptive score cutoff 0.5; no threshold fitting")
display(referral[(referral.referral_fraction == 0.20) & (referral.ranking != "random")].round(5))
""")
    code("""
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for row, method in enumerate(["mc", "tta"]):
    data = referral[referral.method == method]
    for ranking in ["uncertainty_sd", "prediction_score", "predictive_entropy", "random"]:
        values = data[data.ranking == ranking].groupby("referral_fraction", as_index=False).mean(numeric_only=True)
        for col, metric in enumerate(["retained_error", "malignant_referred_fraction"]):
            axes[row, col].plot(values.referral_fraction * 100, values[metric] * 100,
                marker="o", markersize=4, color=COLORS[ranking], linestyle=LINESTYLES[ranking], label=ranking.replace("_", " "))
    axes[row, 0].set(title=f"{method.upper()} SD: common deterministic-score error", ylabel="Classification error at score 0.5 (%)")
    axes[row, 1].set(title=f"{method.upper()} SD: malignant images referred", ylabel="Fraction of malignant images referred (%)", ylim=(0, 100))
    for ax in axes[row]:
        ax.set_xlabel("Images referred (%)")
        ax.legend(fontsize=8)
save_figure("referral_comparisons.png", fig)
""")
    md("""
## 10. Estimate paired differences with patient bootstrap intervals

The **patient bootstrap** repeatedly samples whole patient groups with
replacement, keeping each patient's images together. We use 2,000 resamples of
the 113 validation patients and recompute referral rankings and budgets each
time. This respects dependence between images of the same patient. Differences
are SD ranking minus score, entropy or mean random ranking on the same resample.
Undefined malignant fractions in resamples without malignancy remain missing;
valid replicate counts are reported.

**Read the interval plot at the predeclared 20% referral:** each row names the
method and comparator. Dots/squares are paired estimates, horizontal bars are
95% percentile intervals; blue circles are MC and orange squares TTA. Horizontal
units are percentage-point differences. The dotted zero line means no difference.
On the left, negative retained-error differences favour SD; on the right,
positive malignant-referral differences favour SD. An interval crossing zero
includes no difference and should not be read as clear evidence of benefit.

Intervals describe uncertainty conditional on these fitted predictions and
finite masks/views. They omit training-seed variation and do not remove optimism
from checkpoint selection on this split. Testing several fractions also creates
multiple comparisons; a favourable point does not establish a referral policy.
""")
    code("""
differences = referral_patient_bootstrap(summary, fractions=REFERRAL_FRACTIONS,
    n_bootstrap=PATIENT_BOOTSTRAPS, n_random=RANDOM_REPEATS, seed=BOOTSTRAP_SEED,
    referral_seed=REFERRAL_SEED)
differences.to_csv(OUTPUT_DIR / "paired_patient_bootstrap_differences.csv", index=False)
display(differences[differences.referral_fraction == 0.20].round(5))
""")
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for ax, metric, title in zip(axes, ["retained_error", "malignant_referred_fraction"],
    ["Retained error: negative favours SD", "Malignant referral: positive favours SD"]):
    values = differences[(differences.referral_fraction == 0.20) & (differences.metric == metric)].reset_index(drop=True)
    for index, row in enumerate(values.itertuples()):
        color = "#245D8C" if row.method == "mc" else "#B46A20"
        ax.hlines(index, row.ci_lower * 100, row.ci_upper * 100, color=color, linewidth=2)
        ax.plot(row.estimate * 100, index, marker="o" if row.method == "mc" else "s", color=color)
    ax.axvline(0, color="#333333", linestyle=":")
    ax.set_yticks(range(len(values)), [f"{r.method.upper()} − {r.comparator.replace('_', ' ')}" for r in values.itertuples()])
    ax.set(xlabel="SD minus comparator (percentage points)", title=title)
    ax.invert_yaxis()
fig.suptitle("Paired patient-bootstrap 95% intervals at predeclared 20% referral")
save_figure("paired_patient_bootstrap_differences.png", fig)
""")
    md("""
## 11. Preserve the training artifacts and record provenance

File hashes compare checkpoints, histories, old prototype outputs and the
historical notebook before and after inference. We also retain the checkpoint
and split identities, deterministic preprocessing profile, all pass seeds,
batch size, device and package versions, referral seeds/fractions, bootstrap
settings and the current inference source hashes. Raw, summary and figure
outputs belong only to this new run directory.
""")
    code("""
preservation = assert_preserved_artifacts(protected)
save_json("preservation_receipt.json", preservation)
source_paths = [PROJECT_ROOT / "src/melanoma_detection" / name for name in
    ["image_uncertainty.py", "image_uncertainty_evaluation.py", "image_model.py", "image_transforms.py"]]
save_json("analysis_provenance.json", {
    "source_run": SOURCE_RUN, "architecture": ARCHITECTURE, "recipe_id": RECIPE_ID,
    "condition_id": CONDITION_ID, "evaluation_split": "inner_validation",
    "evaluation_status": "exploratory_inner_validation_not_outer_oof",
    "mc_passes": MC_PASSES, "inference_seed": INFERENCE_SEED,
    "tta_views": tta_pass_ids, "referral_fractions": list(REFERRAL_FRACTIONS),
    "random_referral_seed": REFERRAL_SEED, "random_repeats": RANDOM_REPEATS,
    "bootstrap_seed": BOOTSTRAP_SEED, "patient_bootstrap_replicates": PATIENT_BOOTSTRAPS,
    "bootstrap_interval": "paired patient-cluster percentile 95%; conditional on fitted predictions",
    "classification_cutoff": 0.5, "calibration_fitted": False, "threshold_selected": False,
    "reserved_test_loaded": False, "outer_fold_predictions_generated": False,
    "source_sha256": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in source_paths},
    "output_sha256": {path.name: file_sha256(path) for path in OUTPUT_DIR.glob("*.csv")},
})
display(pd.DataFrame([preservation]))
print("Saved all diagnostic outputs to", OUTPUT_DIR)
""")
    md("""
## Limitations

This is one small, imbalanced inner-validation split reused for epoch and pilot
recipe selection. Patient separation prevents direct training leakage but does
not make these diagnostics an unbiased final evaluation. Neither a favourable
architecture AP nor a favourable referral fraction selects the final model.

Dropout SD covers explicit dropout-mask variation only. It excludes backbone
stochastic depth, training-seed variation and distribution shift. TTA covers four
conservative orientations only; it does not estimate total uncertainty. Thirty
passes are a finite reference, and rank convergence can be slower than score
convergence. Low uncertainty can accompany consistent errors.

The class-weighted scores are uncalibrated. Reliability bins have sparse
malignant support, and image-level referral does not establish patient-level
clinical benefit. The fixed cutoff 0.5 is descriptive. Bootstrap intervals are
conditional and do not correct selection optimism or multiple comparisons.
No calibrator, referral threshold, fusion model, outer OOF evaluation or reserved
test evaluation is implemented here.

## Summary

The current workflow attaches uncertainty to an explicitly selected recorded
best checkpoint, reproduces its exact preparation and patient membership, and
checks inference without changing training artifacts. Raw scores, MC means/SDs
and TTA means/SDs are kept separate and saved with provenance. The executed
summary below reports the observed prediction metrics; the referral tables and
paired intervals determine whether SD adds useful information on this split.
All findings remain exploratory until confirmed within a frozen patient-safe
evaluation procedure on independent outer patients.
""")
    code("""
observed = quality.set_index("prediction")
display(Markdown(f"**Executed checkpoint:** {ARCHITECTURE}, run `{SOURCE_RUN}`, recipe `{RECIPE_ID}`, "
    f"best epoch {context['checkpoint']['epoch']}. **Population:** {len(summary):,} inner-validation images "
    f"from {summary.patient_id.nunique()} patients, including {int(summary.target.sum())} malignant images. "
    f"**AP:** deterministic {observed.loc['Deterministic', 'average_precision']:.4f}, "
    f"MC mean {observed.loc['MC mean', 'average_precision']:.4f}, "
    f"TTA mean {observed.loc['TTA mean', 'average_precision']:.4f}. "
    "Averaging performance does not establish that uncertainty improves referral; inspect the matched rankings and paired intervals."))
at_twenty = referral[(referral.referral_fraction == 0.20) & (referral.ranking != "random")]
for method in ["mc", "tta"]:
    rows = at_twenty[at_twenty.method == method].set_index("ranking")
    pair = differences[(differences.method == method) & (differences.referral_fraction == 0.20)
        & (differences.comparator == "prediction_score")].set_index("metric")
    malignant_pair = pair.loc["malignant_referred_fraction"]
    error_pair = pair.loc["retained_error"]
    display(Markdown(f"**{method.upper()} at the predeclared 20% referral:** SD referred "
        f"{int(rows.loc['uncertainty_sd', 'n_malignant_images_referred'])} malignant images versus "
        f"{int(rows.loc['prediction_score', 'n_malignant_images_referred'])} by ordinary score ranking. "
        f"The paired malignant-referral difference is {100 * malignant_pair['estimate']:.2f} percentage points "
        f"(95% patient-bootstrap interval {100 * malignant_pair['ci_lower']:.2f} to {100 * malignant_pair['ci_upper']:.2f}). "
        f"The retained-error difference is {100 * error_pair['estimate']:.3f} percentage points "
        f"(interval {100 * error_pair['ci_lower']:.3f} to {100 * error_pair['ci_upper']:.3f})."))
ordinary_pairs = differences[(differences.referral_fraction == 0.20) & (differences.comparator == "prediction_score")]
zero_count = int(((ordinary_pairs.ci_lower <= 0) & (ordinary_pairs.ci_upper >= 0)).sum())
display(Markdown(f"**Decision:** keep SD as an exploratory diagnostic. {zero_count} of {len(ordinary_pairs)} "
    "displayed intervals against ordinary score ranking include zero. The checkpoint was selected using these "
    "validation patients; this run does not establish independent evidence of improvement or justify a referral "
    "threshold. No architecture is declared final."))
""")
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (project environment)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    for index, cell in enumerate(notebook.cells):
        cell.id = f"uncertainty-{index:02d}"
    nbf.validate(notebook)
    return notebook


if __name__ == "__main__":
    path = ROOT / "notebooks/Subset/13_image_uncertainty.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), path)
    print(f"Built {path}")
