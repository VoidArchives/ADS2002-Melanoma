from pathlib import Path
import json

ROOT = Path.cwd()
updates = {}

updates['09_image_b0_pilot.ipynb'] = {
0: '''# 09 — Image model: the six-epoch B0 pilot

Can an image model rank melanoma images above benign images? EfficientNet-B0
(B0) is a convolutional neural network: it learns image patterns such as edges,
texture and colour. Its starting weights were learned on ImageNet, a large
general-image dataset, before being adapted to this task.

Four training **recipes** (learning rate and class weighting) were compared.
An **epoch** is one pass through the fitting data; a **checkpoint** stores the
model's learned weights at an epoch. We report each recipe's saved checkpoint
with the highest validation average precision (AP).

This is an **inner-validation pilot**: patients held out from fitting guide early
choices. Because those same labels helped choose recipes and epochs, these are
exploratory results. Full out-of-fold (OOF) evaluation would test the complete
selection process on independent outer folds; the reserved test comes last.
Outer fold 0 and the reserved test are excluded here.

This notebook reads saved results without training or fitting calibration.
Original outputs are preserved; new reports use `b0_pilot_reporting_v2`.
Continue with [architecture comparison](10_image_architecture_comparison.ipynb).
''',
1: '''## Setup and verification

Use the project's `.venv` environment. The checks below verify saved source,
settings, files and patient memberships, then independently recompute the
reported metrics. File hashes act as fingerprints: a changed file fails the
check. This protects the meaning of the historical results after code changes.
No reserved-test files are loaded.
''',
3: '''## Patients and image preparation

Keep all images from one patient in the same partition, so a model cannot learn
from a patient it later evaluates. The table distinguishes **fitting** patients,
**inner-validation** patients, and **outer-excluded** patients whose labels are
not used here.

Only RGB images enter the model. Each whole image is resized to 224 × 224 pixels
using bilinear interpolation. ImageNet normalisation rescales colour channels
to the range expected by the pretrained network. During fitting, weak random
flips and rotations add **augmentation**: varied views intended to discourage
memorising an image's orientation. Validation uses the same preparation every
time, without random augmentation.
''',
5: '''**Read the preview:** each row is one fitting image; the left column is the
original and the next three columns are seeded weak variants. `target=1` means
melanoma and `target=0` benign. Compare lesion edges, texture and colour across
the row. Rotation can create black corners or remove peripheral pixels; an
intact centre alone does not prove that the whole lesion is preserved. This
deliberate label mix is a quality check, not a representative prevalence sample.
''',
7: '''## Prediction quality and computing cost

**AP** summarises precision (how many flagged images are malignant) as recall
(how many malignant images are found) changes across score cutoffs. Higher is
better; the malignant-image fraction is a useful chance-ranking reference.
**ROC-AUC** measures ranking across both classes; higher is better and 0.5 is
chance ranking. AP deserves attention because malignant images are rare.
**Log loss** penalises confident wrong scores; **Brier score** averages squared
score errors. Lower is better for both, but a low loss alone can hide poor
melanoma detection in an imbalanced dataset.

These outputs are prediction **scores**, not verified clinical probabilities.
Class weighting gives malignant fitting examples more influence and can change
the score scale. The constant fitting-prevalence row is a simple baseline.

All four recipes ran six epochs, including one classifier-only warmup before
adapting the whole network. The fixed learning-rate schedule spans 30 epochs;
that schedule is distinct from the executed six-epoch budget and selected epoch.
Exact AP ties retain the earlier checkpoint. The resource table reports measured
epoch time and allocated GPU memory, excluding setup/download overhead.
''',
9: '''## Learning over epochs

**Read the first figure:** the horizontal axis is completed epoch; each line is
one recipe. The upper panels show AP and ROC-AUC (higher is better); the lower
panels show validation log loss and Brier (lower is better). Horizontal guides
mark prevalence for AP and 0.5 for ROC-AUC. Hollow rings mark the epoch chosen
by AP in every panel, rather than each panel's separate optimum. The dotted
vertical line separates classifier warmup from whole-network fine-tuning.

**Read the second figure:** epoch is horizontal and fitting binary cross-entropy
(BCE), the training objective, is vertical. Falling loss shows a closer fit to
training labels; improvement should also appear on validation. Panels separate
weighted and unweighted objectives because their magnitudes are not comparable.
A late AP peak motivates further checking, not proof that longer training helps.
''',
11: '''## ROC and precision–recall curves

**Read the curves:** each line is a recipe's selected checkpoint on the same
1,834 images from 113 patients. Moving along a line changes the score cutoff.
The left panel plots false-positive rate (benign images flagged) horizontally
against recall/sensitivity (malignant images found) vertically; curves closer
to the upper left rank better than the chance diagonal. The right panel plots
recall horizontally against precision vertically. Higher precision at the same
recall is better; the horizontal guide shows malignant-image prevalence.

AP summarises precision–recall ranking; it is not the trapezoidal area under
the plotted steps. Neither curve selects a practical cutoff. These patients
also helped select the checkpoints, so the comparison remains exploratory.
''',
13: '''## Classification at the fixed score cutoff 0.5

An image is flagged when `base_probability >= 0.5`; this stored column contains
an uncalibrated score despite its name. The cutoff is declared, not tuned here.
Recall/sensitivity measures malignant images found; specificity measures benign
images correctly left unflagged. Precision measures malignancy among flags;
F1 balances precision and recall. Undefined ratios remain missing.

**Read each confusion matrix:** rows are true class and columns predicted class.
The diagonal contains correct decisions. Bottom left contains missed melanoma
images (false negatives); top right contains benign images flagged (false
positives). Cell text gives counts and percentages within each true-class row;
darker blue means a larger row fraction. Compare both errors rather than the
many correct benign images. A strong ranking score does not guarantee useful
recall at this cutoff. The later 80%-recall scenario requires training-only OOF
scores; outer evaluation labels must not choose a threshold.
''',
15: '''## Descriptive reliability of the scores

**Read the reliability panels:** each panel is a recipe. Similar scores are
grouped into equal-count bins; horizontal position is the bin's mean score and
vertical position is its observed melanoma fraction. The diagonal means score
and frequency agree. Points below it indicate scores exceeding the observed
frequency; points above it indicate the reverse. The shared axes cover the
observed bin means, so check their range before comparing other plots.

The table gives each bin's image, patient and malignant-image counts. Few
malignant images and repeated images from a patient make the estimates noisy.
This describes reliability; it does not fit a calibrator (a score-to-probability
mapping) or establish accurate clinical probabilities.
''',
17: '''## Interpretation and next steps

The preferred B0 recipe has AP 0.1383 at epoch 6. Its late peak supports checking
duration on further patient splits; it does not establish an optimal duration
or make B0 the final architecture. [Notebook 10](10_image_architecture_comparison.ipynb)
compares other families. [Notebook 11](11_image_augmentation.ipynb) reports the
later augmentation quality gate; [notebook 12](12_image_lighting.ipynb) explains
why lighting correction was deferred. [Notebook 13](13_image_uncertainty.ipynb)
contains current inference diagnostics.

Longer confirmation across three inner folds, calibration, full outer evaluation
and any fusion remain later work. The reserved test stays outside development.
''',
19: '''## Limitations

- One inner patient split and six epochs show early behaviour, not final performance.
- Recipe and epoch choices reuse these validation labels, which can make results optimistic.
- Class-weighted scores are uncalibrated; 0.5 is a reporting cutoff with potentially low recall.
- Weighted/unweighted fitting losses have different scales and cannot rank recipes against each other.
- A finite image preview cannot certify every lesion; full outer evaluation and the reserved test remain pending.
''',
20: '''## Summary

The saved six-epoch experiment provides a verified B0 reference. Read AP alongside
the fixed-cutoff errors, score reliability and computing cost. The observations
below guide further experiments; they do not select the final architecture.
'''
}

updates['10_image_architecture_comparison.ipynb'] = {
0: '''# 10 — Compare image-model architectures

Does changing the image network improve early ranking at an affordable cost?
This notebook compares pretrained convolutional neural networks on matching
patients: **EfficientNet-B0** is a compact reference; **DenseNet121** passes
earlier features to later layers; **ConvNeXt-Tiny** uses a modernised convolutional
design. Different designs may learn useful patterns at different speeds.

An architecture is the network design; a **recipe** specifies how it is trained.
A **checkpoint** stores weights after an epoch (one pass through fitting data).
For each recipe, the highest validation AP checkpoint is reported.

These are **inner-validation diagnostics**: held-out fitting-pool patients guide
recipe and epoch choices. Full out-of-fold (OOF) evaluation would test the entire
selection process on independent outer patients. Outer fold 0 and the reserved
test are excluded here. Eight new fits were budgeted; the saved completion
ledger establishes what actually ran. Short pilots cannot eliminate a slower
family or select the final model.

Training defaults to disabled. The notebook reads executed results using shared
training/reporting utilities. See the [concept guide](../concepts.md) for terms.
''',
2: '''## Shared preparation and the completion ledger

Every family uses the same patient memberships and whole RGB images resized
to 224 × 224 pixels, with the same weak fitting flips/rotation and ImageNet
colour normalisation. Ordinary validation has no random augmentation. These
controls make the network comparison easier to interpret.

Each family tries its own low/high backbone learning rates (the size of weight
updates), with either no class weighting or fitting-only square-root positive
weighting (more influence for rare malignant examples). Thus `low_sqrt` is a
matched recipe **slot**, not an identical numerical learning rate across networks.
Effective batch size is 64 images. Each admitted fit runs six epochs, including
one classifier-only warmup, under a fixed 30-epoch learning-rate schedule. The
selected epoch is the highest AP so far; exact ties retain the earlier epoch.

B0/DenseNet keep BatchNorm statistics and parameters fixed, limiting changes
to their pretrained normalisation with small image batches. ConvNeXt retains
its native LayerNorm, trained during full fine-tuning. All new classifiers use
dropout 0.2 and one logit, converted to a score between 0 and 1. The
[concept guide](../concepts.md) explains these terms.

Read `status` and `completed_epochs` before comparing results. Only completed
matching slots rank families; a paused trial is unfinished, not inferior. The
original session allowed at most 50 training minutes within its wider deadline.
''',
4: '''**Read the preparation preview:** each row contains one fitting image; the
original is left and three seeded weak variants follow. `target=1` means melanoma,
`target=0` benign. Compare the full lesion boundary, texture and colour. Rotation
can create black corners or clip peripheral pixels. This small deliberate label
mix checks image preparation; it does not certify every lesion or represent
dataset prevalence. Later augmentation and lighting stages are separate.
''',
6: '''## Learning over epochs

For each family, the first figure uses completed epoch horizontally and four
validation metrics vertically. Each line is a recipe; paused runs are labelled.
**AP** summarises precision across recall levels and is useful for rare melanoma;
**ROC-AUC** measures overall ranking. Higher is better for both; their guides
are malignant-image prevalence and chance AUC 0.5. **Log loss** penalises confident
wrong scores and **Brier** averages squared score errors; lower is better. A low
loss can still accompany poor melanoma detection, so read the ranking panels too.

Hollow rings mark the AP-selected epoch across all panels. The dotted vertical
line separates warmup from whole-network fine-tuning. The next figure plots
fitting BCE (binary cross-entropy, the training objective) against epoch, grouped
by weighting. Falling fitting loss should be checked against validation trends.
Weighted and unweighted fitting-loss magnitudes are not comparable. Six epochs
show early progress, not each family's full learning potential.
''',
8: '''## Compare completed matching recipes

Only slots completed for **all three families** enter these tables and plots.
All scores concern the same validation images and patients. AP is primary;
log loss and Brier describe score errors, and ROC-AUC supplies another ranking
view. Raw scores remain uncalibrated.

**Read the horizontal bars:** each row is a network/recipe. The three panels
show selected-checkpoint AP (longer is better), total measured training minutes
and peak allocated GPU memory in GiB (shorter is cheaper). Time sums the recorded
epochs despite the figure's "epoch time" title and excludes downloads/setup;
memory is allocated model memory, not total device usage. Assess the trade-off
without treating the small AP difference as final evidence.

**Read each matched-slot curve pair:** each line is one family. The left ROC
panel plots benign false-positive rate horizontally against melanoma recall
vertically; nearer the upper left is better than the chance diagonal. The right
precision–recall panel plots recall horizontally against precision (malignancy
among flags) vertically; higher at the same recall is better. The horizontal
guide is prevalence. Moving along a curve changes the cutoff; the legend names
the selected epoch and metric. AP is not the trapezoidal area under the plotted
steps. No curve chooses a referral or classification cutoff here.
''',
10: '''## Fixed-cutoff errors and descriptive reliability

Flag a raw score `>= 0.5`; the stored `base_probability` column is an uncalibrated
score despite its name. The cutoff is declared rather than fitted to these labels.
Recall/sensitivity measures malignant images found; specificity measures benign
images correctly unflagged. Precision measures malignancy among flags; F1
balances precision and recall. Undefined precision remains missing.

**Read the confusion matrices:** one panel per family, rows are true class and
columns predicted class. Diagonal cells are correct; bottom left is missed
melanoma, top right is benign images flagged. Text gives counts and percentages
within each true-class row; darker blue means a larger row fraction. Compare
missed malignancies and false alarms alongside AP.

**Read the reliability panels:** horizontal position is mean score in a bin;
vertical position is observed melanoma fraction. Points on the diagonal agree;
below it scores exceed the observed frequency, above it scores fall below it.
Panels share the observed-range axes and the table gives bin support. Few
malignant images and shared patients make bins noisy. This describes reliability
without fitting a calibrator or validating clinical risk. An 80%-recall cutoff
requires later training-only OOF predictions.
''',
12: '''## Incomplete records and next steps

The table below lists any completed unmatched trials separately. They cannot
rank families. Paused fits preserve model and training states for compatible
resumption at epoch boundaries; reaching a time budget is not evidence against
their performance. Complete matching comparisons before expanding conclusions.

Next, validate every family with longer training and three inner patient folds.
Choose augmentation and preparation separately for each architecture, then
repeat the complete selection procedure within each outer-training partition.
The current [uncertainty notebook](13_image_uncertainty.ipynb) supplies inference
diagnostics only. Fusion, final outer evaluation, any eligible Swin experiment
and reserved-test evaluation remain separate work.
''',
14: '''## Limitations

- Six epochs on one inner split measure early learning rather than final potential.
- Choosing recipes and epochs on these validation labels makes the results optimistic.
- The shared weak preparation controls this comparison; it does not establish each family's best preparation.
- Weighted fitting objectives and raw score scales differ; 0.5 is an untuned reporting cutoff.
- Paused or unmatched trials cannot lose a comparison; short results cannot rule out slower families.
- Longer confirmation, calibration, full outer evaluation, fusion and the reserved test remain pending.
''',
15: '''## Summary

The saved comparison shows early ranking, fixed-cutoff errors, reliability and
cost on matching patients. The family observations below are exploratory.
No final architecture is selected; confirming the selection process on independent
outer patients remains necessary.
'''
}

condition_intro = '''This notebook reads saved results for a controlled B0 experiment. B0 is a
pretrained image network; the fixed control is a reference, not an established
optimum. **Augmentation** varies fitting images to discourage memorising their
orientation or appearance. **Preprocessing** applies a fixed preparation rule.
Changing one factor at a time helps isolate its effect.

An **epoch** is one pass through fitting data; a **checkpoint** stores the model
weights at an epoch. A complete fit runs twelve epochs, while its selected
checkpoint is the epoch with the highest validation average precision (AP).
Training defaults to disabled.

This is an **inner-validation pilot**: held-out fitting-pool patients guide
development. Their labels also select checkpoints, so the results are exploratory.
Full out-of-fold (OOF) evaluation would test the entire selection process on
independent outer patients. Outer fold 0 and the reserved test are excluded here.
No calibration is fitted. See the [concept guide](../concepts.md) for definitions.
'''
condition_controls = '''## Controls and patient separation

The saved first inner fold lies within outer-training folds 1–4: 3,660 fitting
images from 225 patients and 1,834 validation images from 113 patients. All images
from a patient stay together, preventing direct training leakage.

Both conditions start from fresh pretrained weights and seed 42. The fixed B0
recipe uses dropout 0.2, fitting-only square-root positive weighting (to give
rare malignant examples more influence), backbone/classifier learning rates
3e-5/1e-3, AdamW weight decay 1e-4 and effective batch 64. BatchNorm is frozen
and automatic mixed precision (AMP) reduces GPU memory use. These settings stay
fixed so any completed comparison isolates the transformation change.

Each admitted fit runs twelve epochs with no early stopping, including one
classifier-only warmup; the fixed learning-rate schedule spans thirty epochs.
Highest AP selects the checkpoint, with exact ties retaining the earlier epoch.
Validation preparation is deterministic (repeatable), without random augmentation.
The shared runner owns saved settings, training, resume and time budgets; excluded
patients' labels do not influence those operations. Read the ledger's completion
status before interpreting a comparison.
'''
condition_results = '''## Saved learning and prediction results

Only committed epochs and complete twelve-epoch prediction records are shown.
Each line names an available condition; a missing condition has no performance
result. The current saved figures show the weak control A only, so they cannot
establish an A/B or A/C winner.

**Read the figures in order:**

1. **Learning curves:** completed epoch is horizontal. Upper panels show AP
   (precision across recall levels) and ROC-AUC (overall ranking), where higher
   is better. Guides mark malignant prevalence and chance AUC 0.5. Lower panels
   show log loss (penalty for confident wrong scores) and Brier (mean squared
   score error), where lower is better. Hollow rings mark the AP-selected epoch
   in all panels; the vertical dotted line separates warmup from fine-tuning.
2. **Fitting loss:** epoch is horizontal and binary cross-entropy (BCE, the
   training objective) vertical. Falling fitting loss means a closer fit to
   training labels; compare it with validation trends. All conditions here use
   the same square-root weighting; differently weighted historical losses are
   not directly comparable.
3. **Best-checkpoint ROC/precision–recall:** each line is a condition. ROC plots
   benign false-positive rate horizontally against melanoma recall vertically;
   closer to the upper left beats the chance diagonal. Precision–recall plots
   recall horizontally against precision (malignancy among flags) vertically;
   higher at the same recall is better, with prevalence as a reference. Moving
   along a curve changes the score cutoff. AP is not its trapezoidal area.
4. **Confusion matrices at 0.5:** rows are true class, columns predicted class.
   Diagonal cells are correct; bottom left is missed melanoma and top right
   benign images flagged. Text gives counts and percentages within each row;
   darker blue means a larger row fraction. Large benign counts alone are not
   evidence of good melanoma detection.
5. **Reliability:** mean score within each bin is horizontal, observed melanoma
   fraction vertical. The diagonal means agreement; below it scores exceed
   observed frequency, above it the reverse. Check bin support before interpreting
   sparse positives. These raw scores are not calibrated probabilities.
6. **Epoch-12 ROC/precision–recall:** read the axes as in figure 3. These use the
   final scheduled checkpoint rather than the AP-selected checkpoint, helping
   show whether later fitting improved the same patients' ranking.

The two metric tables keep best and epoch-12 scores separate. AP is primary
because malignancy is rare; loss and Brier can look small even with poor recall.
Reliability is descriptive and does not fit a calibrator. The resource table
sums measured epoch time and reports peak allocated GPU memory, excluding setup.
'''
condition_threshold = '''## Classification at the declared cutoff

The **raw score ≥0.5** cutoff is a fixed reporting scenario. Recall/sensitivity
measures malignant images found; specificity measures benign images correctly
unflagged. Precision measures malignancy among flags; F1 balances precision
and recall. TP/TN are correct malignant/benign decisions; FN/FP are missed
malignancies/false alarms. Undefined precision remains missing.

Scores are uncalibrated, and no cutoff is selected using these validation labels.
The later 80%-recall scenario and any fitted score-to-probability mapping require
training-only OOF predictions. Good ranking alone does not make 0.5 useful.
'''
condition_next = '''## Compare changes only when both fits are complete

The table reports candidate AP minus control AP: positive favours the candidate.
Compare B (stronger augmentation) or C (lighting correction) with fresh A only
when both completed the same twelve-epoch budget and passed quality checks.
A missing AP difference is an unavailable comparison, not zero improvement.

The predeclared AP gain of 0.005 is a screening margin for a promising B0
hypothesis, not statistical proof, a clinical rule or an established optimum.
Smaller gains retain weak/unchanged as the reference. Quality-blocked, paused
or deferred conditions cannot be called worse-performing. B0 findings cannot
select transformations for DenseNet or ConvNeXt. A later combined augmentation
and CLAHE experiment would need individual quality gates and a matched comparison
against the better individual change.
'''
condition_roadmap = '''## Later work

First inspect a safer augmentation profile and complete a fresh matching
comparison. CLAHE still requires its admission gate. Hair removal, combined
changes, longer training and three-inner-fold confirmation remain separate.
Every architecture needs its own recipe and preparation choices.

The entire selection process must later repeat within each outer-training
partition before full OOF evaluation, calibration and any threshold fitting.
Current [uncertainty diagnostics](13_image_uncertainty.ipynb) do not replace that
evaluation. Fusion and the single reserved-test evaluation remain pending.
'''
condition_end = '''## Limitations

One inner split contains only 33 malignant validation images, so results can
vary with the patient sample. Control and epoch choices reuse development labels;
these scores are not unbiased outer-fold evidence. A finite quality preview
cannot certify every lesion, and an untrained condition has no performance
result. Class-weighted scores are uncalibrated; 0.5 is a reporting cutoff rather
than a clinical rule. B0 findings do not establish another architecture's choices.

## Summary

Read completion and image-quality status before the curves. The saved control
provides a reference, but the unavailable candidate comparison supports no
performance winner. The observations below guide the next safe, matched
experiment; full outer evaluation remains necessary.
'''
for name, title, status in [
    ('11_image_augmentation.ipynb', '11 — Weak versus stronger B0 augmentation',
     '**Current saved status:** A (weak augmentation) completed twelve epochs. B\n(stronger augmentation) failed fitting-image quality review and was not trained.\nThe A/B comparison is incomplete; no performance winner is declared.'),
    ('12_image_lighting.ipynb', '12 — Optional lighting correction with CLAHE',
     '**Current saved status:** C (lighting correction) was deferred because its\nrequired complete, safe A/B comparison was unavailable. No correction previews\nor C training were performed. The figures show saved control A only.')]:
    updates[name] = {0:f'# {title}\n\n## Purpose\n\n{condition_intro}\n{status}\n',
        2:condition_controls,6:condition_results,8:condition_threshold,
        10:condition_next,12:condition_roadmap,13:condition_end}

updates['11_image_augmentation.ipynb'][4] = '''## Inspect fitting-image quality before training

The frozen quality sample contains twelve malignant and twelve benign images
from 24 fitting patients, covering framing, hair and lighting variation. It is
a deliberate quality sample, not a prevalence estimate. Each saved page has
originals at left, four A (weak) variants, then four B (stronger) variants.
Each row is one image; `target=1` means melanoma and `target=0` benign. Seeds
make the random variants repeatable and are restored before training.

**Read across each row:** compare the complete lesion edge, texture and pigment
with the original. Look for cropping, distorted boundaries, lost texture,
implausible colour or reflected copies of the lesion at the border. A changed
orientation alone is expected; lost lesion information is not. The recorded
severity scale is 0 (no defect), 1 (minor background-only artifact), 2 (useful
lesion information lost or implausible appearance) and 3 (major damage).
Any severity ≥2 blocks that candidate. An unfinished review is not approval.

A uses flips, ±20° nearest-neighbour rotation and black fill. B adds ±45°
bilinear rotation, ±5% translation, scale 0.9–1.05, reflected padding and
brightness/contrast ±20%, saturation ±10%, with hue unchanged. It has no random
resized crop or masking. The saved review blocked B; this is a quality decision,
not evidence that a stronger model would perform worse. The finite sample cannot
certify every development image.
'''
updates['12_image_lighting.ipynb'][4] = '''## Inspect lighting correction before training

**CLAHE** (contrast-limited adaptive histogram equalisation) adjusts local image
contrast while limiting strong amplification. Condition C would apply it to the
CIELAB lightness channel only (clip limit 2, 8 × 8 tiles), aiming to change
brightness contrast while limiting colour changes. Original files are preserved.
This fixed correction would precede weak fitting augmentation and also apply
before prediction; it differs from B's random brightness/contrast augmentation.

If admitted, original/corrected pairs from the same 24 fitting patients would
be displayed. Read each pair side by side: the full lesion boundary, fine
texture and pigmentation should remain visible, with plausible colour and no
amplified artifacts. Any lost lesion information or implausible colour blocks C.
This finite preview would not certify every image.

Admission requires complete safe A/B fits, passed numerical checks and enough
time, regardless of A/B AP. That gate was not met: there are no saved C preview
pages, and the following missing-preview message is expected. It is not evidence
that CLAHE helps or harms performance.
'''

updates['13_image_uncertainty.ipynb'] = {
0: '''# 13 — Does image uncertainty help prioritise review?

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
''',
1: '''## 1. Choose a saved checkpoint explicitly

The shared loader rebuilds the trained network and exact deterministic image
preparation. Checks verify the run, recipe (training settings), best epoch,
checkpoint hash (file fingerprint), patient memberships and saved predictions.

The defaults below select the experimental B0 control. To inspect another
supported checkpoint, set `SOURCE_RUN`, `ARCHITECTURE` and `RECIPE_ID` together:
for example `architecture_pilot_3h_v1`, `densenet121`, `low_none`, or
`architecture_pilot_3h_v1`, `convnext_tiny`, `high_sqrt`. This is an explicit
diagnostic choice, not automatic architecture selection. `CONDITION_ID` applies
only to a controlled B0 condition. Missing or incompatible inputs stop execution.
''',
5: '''## 3. Check a small sample before full inference

The smoke check uses 16 images to catch a mismatch before a costly full run.
Deterministic predictions must repeat and agree with the saved export. Two
seeded dropout passes must vary under different seeds and repeat under the same
seed. Only explicit dropout may activate; BatchNorm buffers and all model state
must stay unchanged. Identity TTA (no image change) must reproduce the ordinary
score. Any failure stops execution before the 30-pass run.
''',
7: '''### Inspect the four conservative views

**Read the preview:** each row is one prepared image; columns are identity,
horizontal flip, vertical flip and both flips. `target=0` is benign, `target=1`
malignant. Check that the whole lesion, texture and colour remain visible; only
orientation should change. These operations rearrange pixels without deleting
them or creating reflected border fragments. One example from each class shows
framing, not representative prevalence. Labels do not choose transformations.
''',
9: '''## 4. Keep dropout variation and image-view variation separate

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
''',
13: '''## 6. Check stability as the number of passes increases

Compare the first 5, 10 and 20 seeded passes with the same 30-pass reference;
labels do not select the pass count. **Read the two panels:** pass count is
horizontal. Left, mean absolute difference from the reference is vertical,
with separate lines for mean score and SD; closer to zero means more stable
values. Right, SD rank correlation is vertical; closer to 1 means a more similar
ordering of images. The table also gives overlap among the highest-SD 20%.

Stable scores need not imply stable rankings. The finite 30-pass reference is
not an exact expectation, so agreement is evidence of stability rather than
proof that all uncertainty has been captured.
''',
15: '''## 7. Compare ranking and descriptive reliability

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
''',
17: '''## 8. Examine what the two SD measures capture

**Read the histograms:** the left panel bins MC score SD and the middle bins
TTA score SD horizontally; vertical height counts images. Mass near zero means
little change under that procedure. Each panel has its own SD scale because
dropout masks and image orientations perturb different things.

**Read the scatterplot:** each dot is an image, with MC SD horizontally and
TTA SD vertically. A rising pattern suggests the methods prioritise similar
images; scatter suggests different sensitivities. There is no equal-SD target
because the measures have different meanings. Variation does not prove malignancy
or error, and a model can be consistently, confidently wrong.
''',
19: '''## 9. Does SD improve the referral order?

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
''',
22: '''## 10. Estimate paired differences with patient bootstrap intervals

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
'''
}

updates['14_fusion_model.ipynb'] = {
0: '''# 14 — Combine image and metadata predictions

**Fusion** means combining predictions from different information sources.
Image models learn visual patterns; metadata models use recorded attributes
such as age and lesion location. The question is whether their combined score
improves on the stronger model alone, especially when they disagree.

This follows [metadata robustness](08_metadata_robustness.ipynb), the
[image pilot](09_image_b0_pilot.ipynb) and
[uncertainty diagnostics](13_image_uncertainty.ipynb). The comparison needs
predictions for matching images made without training on those patients.
Full **out-of-fold (OOF)** predictions come from holding out each outer patient
fold while fitting and choosing settings within the remaining folds. Current
image results are inner-validation diagnostics and are not those complete inputs.

## Current status

This placeholder contains no fusion code, tables or figures. A planned first
experiment would combine the two scores with a mixing weight chosen inside
training folds, then compare held-out performance with each model alone.
Calibration would map scores to estimated probabilities using training-only
data; raw score scales must be checked before combining them.
''',
1: '''## Limitations

No combined model has been fitted or evaluated, so this file provides no
evidence of improvement and cannot choose a mixing weight or referral rule.
Mismatched image coverage, incompatible score scales or predictions made on
training patients would make the comparison misleading. These checks must pass
before fitting the combination.
''',
2: '''## Summary

Fusion remains planned. First obtain matching patient-safe OOF inputs, choose
any mixing weight within training folds, and compare held-out results with each
model alone. This file has no findings or final model selection.
'''
}
updates['15_final_evaluation.ipynb'] = {
0: '''# 15 — Final evaluation on reserved patients

The **reserved test** contains patients kept out of model development. It asks
how the completed workflow performs after choices have been fixed, without
letting its labels choose the model or decision rules.

Earlier notebooks compare candidates using development data, including the
planned [fusion experiment](14_fusion_model.ipynb). Before this stage, record
the selected model, image preparation, any calibration (a score-to-probability
mapping), and classification/referral rules. A metadata-only final test is also
possible if that is the workflow chosen before viewing test results.

## Current status

This placeholder contains no final-evaluation code, test results, tables or
figures. It has not evaluated reserved patients. The test should be used once
after the complete workflow is fixed. Report malignant cases found and missed,
false alarms, ranking and reliability, with intervals that keep each patient's
images together; a single accuracy number would hide the rare malignant cases.
''',
1: '''## Limitations

Without a test run this file cannot establish final performance or support a
deployment decision. Even a completed test would describe this dataset and
patient mix, not prove clinical safety or performance elsewhere. Few malignant
cases would make recall and model comparisons uncertain. Changing choices after
seeing test results would turn the test into further development data.
''',
2: '''## Summary

Final evaluation remains pending. Record the chosen workflow and rules first,
then evaluate reserved patients once and report patient-based uncertainty and
the costs of missed malignancies and false alarms. This file has no findings
and cannot select a final model.
'''
}

builders = {
    '09_image_b0_pilot.ipynb':'scripts/build_image_notebook.py',
    '10_image_architecture_comparison.ipynb':'scripts/build_image_architecture_notebook.py',
    '11_image_augmentation.ipynb':'scripts/build_image_condition_notebooks.py',
    '12_image_lighting.ipynb':'scripts/build_image_condition_notebooks.py',
    '13_image_uncertainty.ipynb':'scripts/build_image_uncertainty_notebook.py',
}
sources={relative:(ROOT/relative).read_text(encoding='utf-8') for relative in set(builders.values())}
done_replacements=set()
for name, cells in updates.items():
    path=ROOT/'notebooks/Subset'/name
    doc=json.loads(path.read_text(encoding='utf-8'))
    for index, new in cells.items():
        cell=doc['cells'][index]
        assert cell['cell_type']=='markdown'
        old=''.join(cell['source']).strip()
        new=new.strip()+'\n'
        if name in builders:
            relative=builders[name]
            if name.startswith(('11_', '12_')) and index==0:
                pass  # This shared f-string is replaced once below.
            elif (relative,old) not in done_replacements:
                assert old in sources[relative], (name,index,'template mismatch')
                sources[relative]=sources[relative].replace(old,new.strip())
                done_replacements.add((relative,old))
        cell['source']=new.splitlines(keepends=True)
    path.write_text(json.dumps(doc,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')

relative='scripts/build_image_condition_notebooks.py'
start=sources[relative].index('    title = ')
end=sources[relative].index('    code(f"""',start)
sources[relative]=sources[relative][:start]+'''    title = "12 — Optional lighting correction with CLAHE" if lighting else "11 — Weak versus stronger B0 augmentation"
    status = ("**Current saved status:** C (lighting correction) was deferred because its\\n"
        "required complete, safe A/B comparison was unavailable. No correction previews\\n"
        "or C training were performed. The figures show saved control A only." if lighting else
        "**Current saved status:** A (weak augmentation) completed twelve epochs. B\\n"
        "(stronger augmentation) failed fitting-image quality review and was not trained.\\n"
        "The A/B comparison is incomplete; no performance winner is declared.")
    md(f"""
# {title}

## Purpose

'''+condition_intro+'''
{status}
""")
'''+sources[relative][end:]

for relative,source in sources.items():
    compile(source,relative,'exec')
    (ROOT/relative).write_text(source,encoding='utf-8')

print('Updated markdown in',len(updates),'notebooks and',len(sources),'builder scripts')
