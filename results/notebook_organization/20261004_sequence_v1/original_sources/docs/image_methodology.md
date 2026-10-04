# Image modelling for d1

## Implemented workflow

The main `d1/09_image_b0_pilot.ipynb` now reports the completed EfficientNet-B0
pilot from verified historical predictions and checkpoints. The shared Python
trainer and `configs/image_model.yaml` preserve its declared preparation.
It uses image pixels as predictors. Patient IDs identify validation groups;
they are not model inputs. The WIP model remains a historical experiment. Its
original uncertainty notebook is preserved as
`notebooks/archive/image_uncertainty_three_epoch_fold0.ipynb`;
the current `d1/13_image_uncertainty.ipynb` adapts uncertainty diagnostics
to the shared checkpoint pipeline with separate run-specific outputs.

The original pilot registers B0, the existing weak augmentation and unchanged
preprocessing. The separate three-hour exploration adds DenseNet121 and
ConvNeXt-Tiny through the same trainer. Stronger augmentation, hair removal,
CLAHE and Swin were deferred by that historical exploration. The separate
two-hour controlled experiment below introduces stronger augmentation and
optional CLAHE without changing those completed pilots. Hair removal and Swin
remain later additions. Selecting an unavailable adapter must
raise an error rather than silently substitute another procedure.

The pilot holds out outer fold 0 completely. Inside the other four folds,
three stratified patient-grouped inner folds are constructed. The first inner
fold selects pilot epochs and settings. These are **inner validation scores**,
not final OOF performance. No reserved test data or d2 data are used.

## Current three-hour architecture exploration

The session limit includes implementation, tests, training and reporting.
Reserve the final 15 minutes for inspected results. New training has a separate
50-minute allowance and a watchdog; a paused trial remains incomplete.

The main 05 notebook reports the archived B0 pilot using saved predictions,
histories and provenance. It includes actual ROC/precision-recall curves and
F1, precision, recall/sensitivity, specificity and confusion matrices at the
fixed raw threshold 0.5 (`score >= threshold`). These are exploratory reports.
No threshold is fitted on validation labels. Undefined rates remain missing.
Differently weighted training losses are plotted as separate objectives.

`d1/10_image_architecture_comparison.ipynb` inspects the separate
`architecture_pilot_3h_v1` run. It reuses the exact archived fitting/validation
patients, weak transformations, whole-image 224-pixel bilinear preparation and
ImageNet normalisation. Outer fold 0 remains excluded. DenseNet and ConvNeXt
have their own declared learning rates; B0's provisional choice is not copied.

| Family | Backbone rates | Starting microbatch |
|---|---|---|
| Saved B0 reference | 3e-5, 1e-4 | Historical 64 |
| DenseNet121 | 3e-5, 1e-4 | 32 |
| ConvNeXt-Tiny | 1e-5, 3e-5 | 16 |

Cross rates with unweighted BCE and fitting-only square-root positive weighting.
Each new fit has six total epochs, one classifier-only warmup and the unchanged
30-epoch scheduler horizon. Preserve effective batch 64 through accumulation,
explicit classifier dropout 0.2, frozen DenseNet BatchNorm, and ConvNeXt's
pretrained LayerNorm. Only the new output linear layer trains during warmup;
LayerNorm belongs to the backbone group during full fine-tuning.

Run matching DenseNet/ConvNeXt pairs: lower-rate unweighted, lower-rate weighted,
higher-rate unweighted, then higher-rate weighted. Both members must finish
before the next pair begins. The maximum target is eight new fits and 48 epochs,
not a guarantee. Compare only recipe slots completed for all three families.
Unmatched and paused records remain separate; unfinished candidates cannot lose
a comparison. A short-run AP leader is not an established best architecture,
and slower families remain eligible for later confirmation.

Save histories, actual highest-AP checkpoints, deterministic validation scores,
raw reliability checks, epoch times and peak memory. Exact epoch ties keep the
earlier epoch. PNG figures are embedded in executed notebooks and inspected.
Calibration, tuned recall thresholds, augmentation/preprocessing experiments,
longer three-inner-fold confirmation and final nested evaluation are deferred.

### Measured outcome of this session

All eight new six-epoch fits completed on 3,660 fitting images from 225 patients.
They used the same 1,834 validation images from 113 patients, including 33
melanoma images. New training took 38.5 wall-clock minutes including preparation
and checkpoint overhead. Both starting microbatches fitted without reduction.

| Family | Best observed recipe | Highest-AP epoch | AP | Mean measured epoch time | Peak allocated GPU memory |
|---|---|---:|---:|---:|---:|
| Historical B0 | 3e-5, square-root weighting | 6 | 0.1383 | 33.7 seconds | 2.81 GiB |
| DenseNet121 | 3e-5, unweighted | 2 | 0.1452 | 43.1 seconds | 2.19 GiB |
| ConvNeXt-Tiny | 3e-5, square-root weighting | 6 | 0.1495 | 45.3 seconds | 1.48 GiB |

These maxima reuse one validation split and do not establish an optimal recipe
or architecture. B0 and ConvNeXt peaked at the six-epoch cap; additional training
is justified. DenseNet's early peak warrants monitoring under the later stopping
rules. Independently validate every family with its own settings across three
inner folds under a 30-epoch cap, with minimum 12 epochs and patience seven.

At this image count, simply multiplying measured mean epoch times by 30 gives
about 17, 22 and 23 minutes of epoch work per B0, DenseNet and ConvNeXt fit.
These are rough starting estimates; setup/checkpoints add time, longer fits have
a different warmup proportion, and larger fitting partitions cost more. Twelve
30-epoch fits per family would already total roughly 12 GPU-hours across the
three families before overhead or transformation experiments. Final budgets
must use new measurements and pause when exhausted.

Prediction-only execution reproduced all eight new validation score sets exactly
and preserved all 59 checked training artifacts. The 54 protected historical
files remained unchanged. Reporting embeds and saves 20 inspected architecture
figures, alongside six figures in the completed B0 reporting notebook.

## Two-hour controlled B0 augmentation experiment

The new `d1/11_image_augmentation.ipynb` uses a separate
`b0_augmentation_12e_v1` run. Optional `d1/12_image_lighting.ipynb` uses
`b0_lighting_12e_v1` and references the exact fresh A control. Both notebooks
read executed results by default; an explicit shared runner invocation is
required for training. Neither contains a training loop.

| Condition | Training augmentation | Deterministic preprocessing | Contrast |
|---|---|---|---|
| A | Exact historical weak transformation | Unchanged | Fresh reference |
| B | Predefined stronger transformation | Unchanged | B minus A |
| C, optional | Exact historical weak transformation | CLAHE lightness correction | C minus A |

All conditions reuse the saved first inner split, excluding outer fold 0:
3,660 fitting images/225 patients and 1,834 validation images/113 patients.
Use fresh matched initialisations at seed 42, pretrained B0 V1, dropout 0.2,
backbone LR 3e-5, head LR 1e-3 and fitting-only square-root positive weighting.
This recipe is an exploratory experimental control, not an established optimum.
Keep effective batch 64, AdamW weight decay 1e-4, clipping 1.0, AMP, frozen
BatchNorm, zero workers and the original 30-epoch cosine scheduler horizon.
Train exactly 12 total epochs including one classifier-only warmup; no early
stopping. Save the actual highest-AP checkpoint (exact ties keep the earlier
epoch) and separately report epoch 12. The cap is not a selected refit duration.

The weak chain remains whole-image bilinear resize, horizontal/vertical flips
and nearest-neighbour rotation up to 20 degrees with black fill. Stronger uses
flips, bilinear affine rotation up to 45 degrees, translation up to 5% of the
original image dimensions, scale 0.9--1.05 and reflection padding. Restore the
original output canvas after applying the padded affine. Brightness and
contrast factors are 0.8--1.2, saturation 0.9--1.1 and hue is unchanged. No
random resized crop, cutout or lesion masking is admitted.

RGB decoding, whole-image resize, selected deterministic preprocessing and
normalisation apply identically during fitting and prediction. Stochastic
augmentation applies only during fitting. CLAHE operates on the CIELAB L
channel at clip limit 2 and an 8x8 tile grid, preserving a/b before converting
back to RGB. It is distinct from random colour augmentation. The procedure
uses fixed per-image rules; later learned preparation must fit only fitting
patients and record its state. Source images are never overwritten.

Before training, freeze 24 distinct fitting-patient examples, balanced by image
label and covering lesion framing, hair and lighting. Inspect four seeded weak
and stronger variants per image with matched originals; save and embed every
PNG page. Record clipping, boundary distortion, texture/pigmentation loss,
implausible colour and reflected lesion duplication. Any visible loss of useful
lesion information or implausible lesion appearance blocks that condition.
Minor background-only artifacts may pass with an explicit record. Quality
review is a manual prerequisite, not an automatic consequence of valid tensors.
CLAHE requires an equivalent original/corrected review on the same examples.

The 120-minute session includes implementation, tests, quality inspection,
training and reporting. Target allocations are 40/20/30/20 minutes with ten
minutes contingency. All training stops by minute 100. Preserve the last
committed checkpoint if a watchdog interrupts an unexpectedly slow epoch.
Completed A/B fits and verified predictions must exist by minute 70 before
optional C setup. Allow at most ten minutes for its binary-only dependency,
tests and review; admit C only when its projected work plus five minutes fits
before minute 100. OpenCV is optional and must not upgrade the numerical
environment. Resource/quality gates, not observed AP, decide admission.

Budget exhaustion pauses a compatible run and labels unfinished contrasts
incomplete. It supplies no evidence that a candidate is inferior. Resume keeps
the full optimiser, scheduler, scaler, RNG and history; never bypass historical
checkpoint incompatibility. The new preparer audits global image identity
without excluded labels and validates label consistency only within the
outer-training pool. Outer labels cannot affect preparation or predictions.

Report AP, ROC-AUC, unweighted log loss, Brier, ROC/PR curves and raw reliability.
Classification metrics and confusion matrices use raw probability >=0.5 and
are labelled exploratory; undefined precision remains missing. No threshold or
calibrator is fitted here. The academic 80%-recall scenario awaits training-only
OOF scores. Learning curves mark selected epochs, and all executed figures are
saved as PNGs and inspected for readability.

A quality-passing AP gain >=0.005 marks a promising B0 hypothesis, not statistical
proof or a global winner. Smaller gains retain weak/unchanged as the development
reference. Later stronger+CLAHE testing requires both individual gates and must
beat the better single by >=0.005. B0 findings cannot prune safe candidates for
DenseNet or ConvNeXt. Longer three-inner-fold confirmation and all final
selection remain independently contained within each outer-training partition.
Hair removal, combinations, Swin, nested evaluation, calibration/threshold
fitting, uncertainty, fusion and the reserved test remain outside this session.

### Executed controlled-stage outcome

The fresh weak/unchanged condition A completed all twelve epochs. Its highest
AP was 0.138318 at epoch 6, with ROC-AUC 0.866337, unweighted log loss 0.087642
and Brier score 0.021026. Epoch-12 AP was 0.100529. This split does not support
increasing the epoch budget alone; later three-inner-fold confirmation must
select duration independently. Training took 423.25 seconds (7.05 minutes),
averaged 34.82 seconds per epoch and used 2.81 GiB peak allocated GPU memory.

All twelve matched preview pages were inspected before training. The predefined
stronger condition B failed the quality gate: seven variants contained reflected
lesion fragments or useful boundary clipping. B was recorded as `quality_blocked`
and was not trained or revised after seeing performance. A/B therefore remains
incomplete, with no AP difference or advancement decision. A later experiment
must declare and inspect a safer profile before attempting a matched comparison.

CLAHE condition C was deferred because admission required a complete safe A/B
pair. OpenCV was not installed; no corrected previews or C training were performed.
Its separate notebook displays the exact A reference and the deferred status,
without fabricated C results. Both new notebooks executed without cell errors;
05b embeds eighteen PNGs and 05c embeds six A-reference PNGs.

The 132-test suite passed with one optional real-OpenCV test skipped. A further
checkpoint prediction check changed all 1,370 excluded labels, including an
invalid label, in memory: training identities and both A prediction files stayed
exactly unchanged, with maximum score difference zero. Prediction-only execution
preserved thirteen training artifacts. Preservation verification checked 258
baseline files: 254 remained identical and four authorised source/configuration
changes retained their original snapshots. Historical notebooks, predictions,
checkpoints, figures, histories and completion receipts were unchanged, including
the current architecture notebook rather than its older receipt's notebook hash.

## Historical preservation and current checkpoints

The original pilot source/configuration snapshot is archived under the separate
reporting run. Its preservation receipt hashes the pilot, WIP notebooks and
historical checkpoints. New B0 reporting lives in `b0_pilot_reporting_v2` and new
architecture results in `architecture_pilot_3h_v1`. Original files are not
rewritten. The original uncertainty notebook was copied byte-for-byte to the
clearly named historical archive before its active path was updated. Its saved
prediction, metric and figure files remain historical results. Earlier receipts
refer to the original notebook content, now retained in that archive. The new
runner uses a workspace-local pretrained-weight cache under
`models/image_pretrained` and validates pinned weight identities.

Changing core source intentionally makes direct historical training resume
incompatible. Historical reporting validates saved artifacts and recorded
source identity without asserting that current code produced them. Do not edit
old manifests to bypass a mismatch. A longer continuation requires a separate,
audited state import in a later session.

## Current checkpoint uncertainty diagnostics

`d1/13_image_uncertainty.ipynb` uses existing checkpoints only. It selects a
run, architecture and recipe explicitly, then loads the recorded highest-AP
checkpoint through `image_model.load_model_checkpoint`. The default is the
fresh weak/unchanged B0 control, condition A of `b0_augmentation_12e_v1`, recipe
`lr3e5_sqrt`, best epoch 6. Supported DenseNet121 and ConvNeXt-Tiny recipes in
`architecture_pilot_3h_v1` are alternatives requiring explicit selection. A
pilot AP leader is not automatically the final model.

Reconstruct ordered fitting and validation membership from the checkpoint's
recorded run manifest and validate it against current data. Check image and
patient identities, fitting/validation patient separation and exclusion of
outer fold 0. The present inner-validation split contains 1,834 images from
113 patients, including 33 malignant images; fitting used 3,660 images from
225 patients. These diagnostics reuse labels that selected pilot epochs and
settings. They are neither full OOF nor reserved-test evaluation.

Ordinary inference restores the checkpoint's exact deterministic RGB decoding,
whole-image bilinear resize, selected preprocessing and ImageNet normalisation.
It checks repeatability and agreement with the saved deterministic scores before
continuing. MC dropout repeats the same prepared images 30 times with recorded
seeds. Only explicit dropout modules enter training mode; their trained rates
are preserved. BatchNorm and stochastic depth remain in evaluation mode.
Record per-image mean score and sample standard deviation, and compare 5, 10
and 20 passes with the 30-pass reference. This checks within-run stability; it
does not establish convergence beyond 30 passes or capture all uncertainty.

TTA is a separate orientation-sensitivity experiment with dropout disabled.
Apply identity, horizontal flip, vertical flip and both flips after exact
deterministic preparation. These transformations preserve the entire prepared
canvas without affine rotation, crops or reflection padding, avoiding the
clipping and reflected lesion fragments seen in the stronger training profile.
The identity view must reproduce ordinary deterministic scores. TTA SD and MC
SD describe different perturbations and are not calibrated risk probabilities.

Validate every raw score as finite and within [0, 1] before aggregation. Reject
duplicate image/pass keys, missing or unexpected passes and incomplete image
coverage. Save the raw deterministic, MC and TTA predictions and their summaries
in a new run-specific directory under `results/image_uncertainty_runs`. Provenance
identifies the checkpoint hash and best epoch, selected run/recipe, split and
membership, preparation, seeds, inference settings and integrity checks. Check
that checkpoints and training histories retain their original hashes.

Evaluate ranking value using the same deterministic base scores and referral
fractions for MC SD, TTA SD, ordinary high-score referral, predictive entropy
and repeated seeded random referral. Report retained-set classification error
at the fixed raw cutoff 0.5, retained Brier error and malignant images referred,
with paired patient-bootstrap intervals for key differences. Resample whole
patients so their images stay together. Intervals
are conditional on the existing checkpoint and validation predictions; they do
not account for training variation or pilot selection bias. Referral means
setting images aside for review, not a correct diagnosis or observed reviewer
benefit. Retained metrics also reflect the changing case mix.

Reliability bins, log loss and Brier score describe the raw deterministic and
repeated-mean score scales. Do not fit a calibrator or select a referral
threshold using these pilot validation labels. Use the terms prediction score,
MC-dropout variation and TTA sensitivity separately; no calibrated probability
column is produced. Any eventual rule requires training-only selection and
independent evaluation.

The smoke check precedes full default notebook execution and exercises
checkpoint compatibility, patient separation, deterministic repeatability,
dropout activation, unchanged BatchNorm buffers and identity-TTA agreement.
Dedicated tests also inject invalid scores, duplicate keys, missing passes and
incomplete image coverage to verify rejection before aggregation.
This workflow does not retrain models, implement
fusion, evaluate outer fold 0 or open the reserved test.

## Training and checkpoint rules

Compare backbone learning rates `3e-5` and `1e-4`, each with ordinary BCE and
moderate positive weighting `sqrt(negative_count / positive_count)`. Compute
weights from fitting images only. Each recipe gets six epochs, including one
classifier-only warmup epoch. The classifier learning rate is `1e-3`; AdamW
weight decay is `1e-4`. BatchNorm statistics and affine parameters stay frozen.

Use automatic mixed precision on CUDA, clipping gradients at 1.0. Begin with
microbatch 64 and effective batch 64. If a CUDA memory error occurs, roll back
the epoch, halve the microbatch down to four and use gradient accumulation to
preserve the effective batch. Account for the actual sample count in the last
accumulation window.

The schedule is declared before training: one warmup epoch, then 29 cosine
fine-tuning epochs, ending at 10% of the original learning rates. A six-epoch
pilot does not compress this into a six-epoch decay. A compatible continuation
therefore follows the same learning-rate trajectory.

Save both the highest-AP checkpoint and the latest epoch-boundary checkpoint.
The latter contains optimiser, scheduler, scaler and random-generator states.
An exact AP tie keeps the earlier checkpoint. A changed input, recipe, code or
execution environment requires a separate run or an explicit incompatibility
error. Prediction-only execution must preserve training history.

The 60-minute compute allocation stops between epochs. If any of the four
trials is unfinished, the comparison remains incomplete and has no declared
winner. Completed trials are not compared with shorter trials as equal runs.

The CLI deliberately enforces this six-epoch pilot. A longer experiment must be
introduced as a separate stage with its own budget and output identity. The
shared recipe trainer supports the fixed 30-epoch horizon, but extending a run
must not overwrite the six-epoch pilot's interpretation or comparison summary.

## What the pilot can establish

The saved results show training speed, memory use, learning curves and whether
the pipeline runs correctly. They provide preliminary evidence about B0
settings. They cannot establish the optimal architecture or an unbiased
performance estimate after selecting epochs on those same validation cases.

AP measures how well melanoma images are ranked when they are rare. ROC-AUC
provides another ranking view. Log loss, Brier score and reliability bins
describe the raw scores' probability quality. Include the validation
prevalence and a constant fitting-prevalence baseline.

Reliability bins are **calibration checks**, not a fitted calibration model.
Do not fit a calibrator or select a threshold on the pilot validation labels:
those labels already choose epochs and settings. In the later nested workflow,
three fixed-duration inner refits provide training-only OOF predictions for
sigmoid recalibration (`C=1`) and an academic 80%-recall threshold scenario.
Failed or non-positive-slope recalibration retains raw results and records the
failure; no calibrated result is silently fabricated.

If the provisional winner's best epoch is five or six, recommend extending
compatible training to a 30-epoch cap. An earlier peak suggests that more epochs
may offer little improvement on this split. Either observation remains
exploratory; repeat validation on the other inner folds in the later stage.

## Later controlled experiments

Three thin experiment notebooks will reuse the same loaders, training code,
folds, metrics and manifests. They must not copy the training loop.

| Stage | Candidate set | Initial allocation | Progression |
|---|---|---|---|
| 1. B0 pilot | Four learning-rate/loss recipes; six epochs | 30–60 minutes | Pipeline and integrity checks pass |
| 2. Augmentation | Weak versus stronger; 12 epochs each | 2 GPU-hours | Quality checks pass; stronger improves screening AP by at least 0.005 |
| 3. Preprocessing | Unchanged, hair-only, CLAHE-only; conditional combination | 4 GPU-hours | Test individual changes first; combine only if both pass quality and AP gates |
| 4. Architectures | B0, DenseNet121, ConvNeXt-Tiny; four recipes each | 12 GPU-hours | Each family receives a longer horizon and three-inner-fold confirmation |
| 4b. Swin-Tiny | Optional additional family | 6 GPU-hours | Admit before final evaluation if memory fits and added projected nested cost is at most 12 GPU-hours |
| 5. Nested evaluation | Five outer folds and three inner folds | Initial 48 GPU-hours, resumable | Complete all outer folds before aggregate final conclusions |

These are allocations, not promised completion times. An exhausted budget
pauses the stage and marks it incomplete; it does not eliminate a slow model.

### Augmentation

The weak baseline is whole-image resize, random horizontal/vertical flips and
rotation by at most 20 degrees, without colour jitter. The stronger candidate
uses flips, affine rotation by at most 45 degrees, translation by at most 5%,
scale 0.9–1.05, reflection padding, brightness/contrast variation of 20%,
saturation variation of 10% and unchanged hue. Do not introduce random resized
crops or cutout.

Inspect 24 fitting-partition examples, each with four seeded variants. Cover
lesion framing, hair and low/high lighting, and record clipping, texture loss,
boundary distortion and implausible colour. Unsafe variants cannot advance.
Ordinary validation preparation remains deterministic.

### Preprocessing

Hair removal is optional: at 224 pixels use grayscale black-hat detection with
a 17×17 elliptical kernel, threshold above 20, one 3×3 dilation and Telea
inpainting radius three. Empty masks leave the image unchanged. Masks exceeding
10% of pixels return the original image with an audit flag. Inspect originals,
masks, reconstructions and all flagged fitting examples for lost lesion detail.

The lighting/contrast candidate applies CLAHE only to CIELAB lightness
(`clipLimit=2`, `tileGridSize=(8, 8)`), retaining the colour channels. This is a
contrast-normalisation experiment, not proof that lighting bias is removed.
Inspect colour changes and clipping. Both procedures require a later optional
OpenCV dependency; the first B0 pilot does not require or install it.

Apply deterministic preprocessing identically to training and evaluation
images, before stochastic training augmentation. Keep random brightness and
contrast augmentation separate. Any learned preprocessing must fit solely on
the current fitting patients and save that fitted state and its membership.

Compare unchanged, hair-only and CLAHE-only before testing hair→CLAHE. The
combination is eligible only when both individual procedures pass quality
checks and improve screening AP by at least 0.005. This is a progression rule,
not a statistical significance claim.

### Architecture adapters

Pin weights to `IMAGENET1K_V1`. Preserve pretrained pooling and normalisation,
use a one-logit classifier with explicit dropout 0.2, and keep whole-image
224-pixel framing. Record expected pretrained preparation and the actual
full-image preparation; do not silently introduce centre crops that remove
lesion boundaries. B0 retains its explicit bilinear resize override. Use the
appropriate normalisation/interpolation for each additional family.

| Family | Backbone learning rates | Initial microbatch |
|---|---|---|
| B0 | 3e-5, 1e-4 | 64 |
| DenseNet121 | 3e-5, 1e-4 | 32 |
| ConvNeXt-Tiny | 1e-5, 3e-5 | 16 |
| Swin-Tiny, if admitted | 1e-5, 3e-5 | 16 |

Cross each learning rate with the two loss settings. Accumulate to effective
batch 64. Freeze BatchNorm for B0/DenseNet; LayerNorm stays trainable during
full fine-tuning for ConvNeXt/Swin. Architecture trials have a 30-epoch cap,
minimum 12 epochs before early stopping, patience seven and AP improvement
threshold 0.001. Save the actual highest-AP epoch; the threshold controls
patience only. A six-epoch B0 pilot does not rank these other families.

## Later final nested selection procedure (not implemented)

Freeze candidate sets, quality gates, budgets and rules before looking at outer
results. Within **each** outer-training partition independently:

1. Build three inner patient-grouped folds; use the first for bounded screening.
2. Independently for every family, screen its four recipes with weak/unchanged
   preparation under a 30-epoch cap, minimum 12 epochs and patience seven.
3. With that family's recipe fixed, compare weak/strong augmentation, then
   unchanged, hair-only and CLAHE-only preparation using quality and AP gates.
   B0 findings cannot prune another family's safe candidates.
4. Test hair followed by CLAHE only when both singles qualify; the combination
   must beat the better single by at least 0.005 AP.
5. Retune all four recipes under that family's selected preparation profile,
   reusing its already matching fit; do not repeat the transformation search.
6. Confirm one independently screened complete pipeline for every family across
   all three inner folds, reusing a compatible completed first-inner-fold fit.
7. Select by mean inner AP. Ties within 0.0001 favour unweighted loss, lower
   learning rate, then B0, DenseNet, ConvNeXt and Swin.
8. Refit the selected recipe on all outer-training patients for the median of
   its three best epochs. Exact epoch ties choose the earlier epoch.
9. Generate three fixed-duration inner refits for training-only OOF calibration
   and threshold selection, then evaluate the outer patients once.
10. Also save raw predictions from that partition's independently selected B0
    reference; reuse the selected model when it is B0.

Exploratory winners do not globally prune the candidate library. Outer results
do not choose a global winner or alter selection rules. This procedure avoids
a full Cartesian search but may miss architecture-specific transformation
interactions; report that limitation.

Report fold AP and mean fold AP prominently, alongside pooled AP, ROC-AUC,
log loss, Brier and reliability plots. Raw score scales can differ across
independently selected folds. For 2,000 patient-bootstrap replicates, resample
whole patients within each outer fold when estimating mean-fold AP intervals.
Compare B0 and metadata on matching images and statistics. Bootstrap intervals
are conditional on fitted predictions; separate seeds describe training
variation. Seeds 17/202 must not replace seed 42 because their results look better.

The conservative revised bound is 235 logical fits and at most 7,050 epochs across
unequal architectures; reserve up to 70 additional fits and 2,100 epochs for Swin.
This replaces the earlier
B0-profile-transfer estimate of 180 fits/4,380 epochs. Matching checkpoints can
reduce actual work. Estimate runtime from observed epoch/preprocessing/inference
times for each architecture rather than equating all epochs. This may be a
multi-day laptop workload. The 48-hour allocation is not a completion promise.

## Handoff and verification

Preserve `image_id`, `patient_id`, `fold`, `target`, `model_name` and
`base_probability`. Add split/run/recipe identity to pilot predictions so they
cannot be mistaken for outer OOF scores. Calibrated probabilities are separate
columns, never replacements for the raw deterministic scores.

Manifests must carry candidate library and procedure version, input/image/code
hashes, splits, architecture/head/weights, actual transforms, preprocessing
parameters/order and fitted patient membership, seeds, effective batch, stopping
reason, package versions and resume states. Later preprocessing caches must be
keyed by source image and procedure hashes.

The shared checkpoint loader reconstructs supported models and deterministic
preparation. The current 05a uncertainty workflow uses this loader and recorded
inner membership; the archived three-epoch fold-0 notebook retains its historical
loader for reference. The current adapter activates only explicit dropout for MC
while BatchNorm and stochastic depth remain in evaluation mode. Regenerate
uncertainty outputs after model changes, with identity-TTA agreement checks.
Fusion remains a separate later stage.

Test patient isolation, outer-label invariance, invalid/missing/corrupt images,
exact duplicate content, deterministic preparation, frozen BatchNorm, correct
partial-batch accumulation, finite mixed-precision computation, checkpoint
round-trip/resume and manifest mismatch rejection. Check prediction-only history
preservation and exact output coverage. Inspect the executed notebook and its
actual charts. Later-stage adapters add their own mask/cache/LayerNorm tests.

Verify committed architecture outputs without training with
`scripts/validate_image_architecture_outputs.py`. Add
`--prediction-roundtrip --device cuda` to reconstruct one completed checkpoint
per new family and compare all deterministic validation scores. It also checks
the historical preservation receipt, current numerical source hashes, resumable
states, selected checkpoint identities, complete image coverage and finite
learning histories.
Use `--prediction-only --device cuda` to exercise the real prediction-only CLI
across all available checkpoints and verify training artifacts stay unchanged.
The CLI applies the same deterministic CUDA policy as training before loading
models. Its training-time source snapshot is preserved separately from this
later inference setup.

The final deployment model repeats the frozen selection procedure on all
development patients before the reserved test is evaluated once. The current
uncertainty diagnostics extend the pilot workflow; validated final uncertainty,
fusion and reserved-test evaluation remain later work.
