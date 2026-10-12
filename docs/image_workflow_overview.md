# Image modelling workflow overview

**Current v2 reporting contract.** Unweighted mean outer-fold **ROC-AUC** is the primary ranking summary; mean-fold AP, pooled metrics, log loss and Brier score remain supporting measures. Each fold has equal weight, and patient-bootstrap intervals resample whole patients within folds. A single selected inner split is exploratory and cannot supply a five-fold estimate. Scores describe individual images; patient grouping does not turn them into patient-level diagnoses.

The metadata v2 experiment selects settings by inner ROC-AUC. Existing image recipes, checkpoints, durations and transformation-admission gates retain their historical AP-based rules. Reporting their saved ROC-AUC does not retrospectively make them ROC-selected. Calibration and recall cutoffs remain attached to their original procedures. All active report notebooks read saved artifacts; they start no training, image smoke or checkpoint inference and never open the reserved test. Final fitting and independent testing wait until the entire procedure is frozen.

Presentation snapshots are preserved under `results/metadata_nn_tuning/v2/archive_before_update/`. New image figures/tables use `results/metadata_nn_tuning/v2/report_verification/image_reports/<condition>/<stage>/`; current metadata assessment tables and figures use the explicit v2 `roc_auc_grid36_v1` run's `report/` directory. v1 AP results and historical image outputs retain their original identities.


## Workflow at a glance

The objective is to use lesion pixels to rank malignant images above benign ones, then investigate whether variation in predictions helps prioritise review. The completed modelling uses **Subset** data. An image is the prediction unit; a patient is the unit used to separate training and evaluation.

The current workflow is:

1. Audit the images and establish separate fitting, inner-validation and excluded outer patients.
2. Train four short EfficientNet-B0 recipes through a shared, reproducible pipeline.
3. Compare DenseNet121 and ConvNeXt-Tiny on the same patient split.
4. Extend the weak B0 control and inspect stronger transformations before admitting them to training.
5. Use the control's recorded best checkpoint for Monte Carlo dropout and test-time augmentation.
6. Check prediction reliability and compare referral rankings fairly.
7. Run the bounded B0 selection process separately inside five outer folds and
   report performance only after every fold completes.

The first four stages produced training histories, checkpoints, predictions and reports. The uncertainty stage used existing checkpoints and produced separate inference results. Stronger augmentation was blocked after inspection; lighting correction was deferred. The bounded five-fold raw B0 stage and separate training-only calibration stage are complete. The larger multi-architecture outer search, fusion, Full-data modelling and the reserved final test remain planned.

Start with [notebook 09](../notebooks/Subset/09_image_b0_pilot.ipynb), then follow [10](../notebooks/Subset/10_image_architecture_comparison.ipynb), [11](../notebooks/Subset/11_image_augmentation.ipynb), [12](../notebooks/Subset/12_image_lighting.ipynb), [13](../notebooks/Subset/13_image_uncertainty.ipynb) and the new [14 image evaluation](../notebooks/Subset/14_image_nested_evaluation.ipynb). These notebooks report saved work; the shared code owns training and checkpoint loading. The [image methodology](image_methodology.md) gives the detailed experiment rules. Historical `d1` names in result folders identify existing Subset experiments and have been retained for traceability.

## 1. Audit the inputs and separate patients

### What goes in, and why grouping matters

The inputs are cleaned development records, their saved patient folds and the corresponding JPEGs. Labels indicate whether each image is malignant. Patient identifiers determine membership; they are not predictors. Several images can belong to one patient, so splitting images independently could let a model learn from someone later counted as an unseen evaluation case.

[Patient splitting](../notebooks/Subset/03_patient_splitting.ipynb) created five development **folds**, or patient groups. Current image pilots set outer fold 0 aside, then construct three stratified patient groups inside the remaining data. Stratification attempts to distribute malignant cases while keeping each patient intact. The first inner group supplies validation; the other two supply fitting data.

| Current pilot partition | Images | Patients | Malignant images | Role |
|---|---:|---:|---:|---|
| Fitting | 3,660 | 225 | 70 | Update model weights |
| Inner validation | 1,834 | 113 | 33 | Compare settings and select checkpoint epochs |
| Excluded outer fold 0 | 1,370 | 84 | 25 | Not fitted or scored by current pilots |

**Inner validation** guides development choices using patients excluded from weight fitting. **Outer evaluation** tests those choices on another patient group excluded from both fitting and selection. Repeating the complete selection-and-fitting procedure across all five outer folds produces **out-of-fold (OOF) predictions**: each patient is predicted by a workflow fitted and selected without that patient in this run. Earlier image reports cover one inner split; the new bounded run repeats selection inside all five outer training pools. Prior development use of these patients still limits independence.

For example, suppose patient A has three images and patient B has one. All three A images must move together. If A belongs to inner validation, none can update weights, although A's labels may guide checkpoint selection. If A belongs to outer evaluation, its labels must also stay outside selection. The separately reserved test serves a later final check after development decisions are fixed; it is not another name for outer fold 0.

### Checks performed and outputs

The saved audit decoded and fingerprinted all **6,864 development JPEGs**. A fingerprint is a reproducible identifier for image content. The audit found **158 exact-duplicate groups**, all within patients, with no cross-patient duplicate groups or conflicting duplicate labels. Repeats were retained so later experiments used comparable records. Saved memberships, image fingerprints and manifests bind a result to its actual inputs.

The controlled-stage preparation additionally isolates excluded outer labels. A saved check changed those labels and reproduced the control's predictions unchanged. This tests that excluded labels did not influence that preparation and prediction path; it does not mean no earlier project audit ever inspected them.

These checks establish readable files, traceable content and patient separation. They do not establish clinical label correctness or detect every visually similar image. Retained duplicates can give repeated views extra influence. The source images are 512×512 pixels; reducing them to 224×224 also sacrifices fine detail. See [the implemented workflow](image_methodology.md#implemented-workflow) and the [pilot audit](../results/image_d1_runs/b0_pilot_v1/image_audit.json).

## 2. Establish a reproducible EfficientNet-B0 reference

### From an image to a prediction score

EfficientNet-B0 is a convolutional neural network: it learns visual patterns through small filters applied across images. Starting from weights learned on general images provides useful initial features before adapting to this smaller lesion dataset. The original classification output is replaced with an explicit dropout layer and one numerical output, called a **logit**. A sigmoid maps that output into a score between 0 and 1. This range alone does not make the score a calibrated malignant probability.

The deterministic preparation decodes RGB colour, resizes the **whole image** to 224×224 using bilinear interpolation and applies ImageNet numeric normalisation. Normalisation places colour values on the scale expected by the pretrained model. Whole-image resizing avoids silently introducing a centre crop that could remove lesion boundaries.

During fitting only, weak augmentation adds random horizontal/vertical flips and rotations up to 20 degrees with black fill. **Augmentation** creates varied presentations of existing fitting images; it does not add independent patients. Validation uses the deterministic preparation without these random changes. This separation allows repeated predictions to reproduce the same input.

### What the recipe comparison tested

[Notebook 09](../notebooks/Subset/09_image_b0_pilot.ipynb) reports four completed six-epoch fits. An **epoch** is one pass through the fitting images. The recipes combine two backbone learning rates with ordinary or moderately class-weighted binary loss. The backbone extracts visual features; the output head turns those features into a score. A learning rate controls the size of weight updates; loss measures prediction error during fitting.

| Backbone learning rate | Malignant loss weighting | Highest-AP epoch | Inner-validation AP |
|---|---|---:|---:|
| 0.00003 | None | 5 | 0.1155 |
| 0.00010 | None | 4 | 0.1119 |
| 0.00003 | Square-root weighting | 6 | 0.1383 |
| 0.00010 | Square-root weighting | 6 | 0.1097 |

**Worked example: class weighting.** Fitting contains 3,590 benign and 70 malignant images. The positive weight is `sqrt(3590 / 70) ≈ 7.16`, computed from fitting data alone. The same malignant error therefore contributes about 7.16 times its ordinary loss. This gives scarce positive cases more influence without multiplying them into independent observations. It also changes the fitting objective, so weighted and unweighted loss curves are not directly interchangeable, and scores may need later calibration.

All recipes use seed 42, the same pretrained starting weights and explicit dropout rate 0.2. During epoch 1, only the new output linear layer learns; subsequent epochs fine-tune the backbone. BatchNorm, which stores feature-normalisation statistics, remains frozen. AdamW updates the weights, with a larger learning rate for the new head. The declared learning-rate schedule spans 30 epochs, even though these pilots execute only six; a short pilot does not compress the whole schedule into six epochs.

The shared trainer saves both the **best checkpoint**, selected by highest inner-validation AP, and the **latest checkpoint**, representing the last completed epoch and its continuation state. It also saves histories, deterministic predictions, manifests and figures. A manifest records the recipe, preparation, data membership, seeds and relevant fingerprints. These outputs allow later reporting and prediction without repeating training. The implementation is in [the shared image model code](../src/melanoma_detection/image_model.py).

### What the results mean

**Average precision (AP)** summarises precision across a malignant-recall ranking. Precision asks what fraction of flagged images are malignant; recall asks what fraction of malignant images are found. Higher AP is preferable. The strongest observed B0 AP was **0.1383**, compared with approximately **0.0180** malignant prevalence on this validation split. AP 0.1383 does not mean 13.83% accuracy or a selected operating recall.

**Illustrative AP example:** four images, ordered by scores `0.4, 0.3, 0.2, 0.1`, have labels malignant, benign, benign, malignant. Precision when each malignant image is reached is `1/1` and `2/4`. AP is their average, `(1 + 0.5)/2 = 0.75`. Yet a cutoff of 0.5 flags no image and gives zero recall. The ordering can contain information even when the numerical score scale does not suit that cutoff.

The distinction appears in the actual fixed-cutoff report. At raw score `>= 0.5`, that B0 checkpoint flagged 18 images: **4 malignant and 14 benign**, while missing **29 of 33 malignant images**. Thus recall was `4/33 = 12.1%`, despite useful ranking. The cutoff is an exploratory illustration; it was not selected to meet a clinical objective.

Read precision–recall curves by comparing precision at the same recall; the prevalence line is the no-information reference. ROC curves compare recall against the benign false-positive rate; their area, **ROC-AUC**, measures another aspect of ranking. In confusion matrices, rows show actual classes and columns predicted classes: missed malignant images are false negatives; flagged benign images are false positives. Learning curves distinguish fitting from validation patients: declining fitting loss alongside worsening validation AP can suggest overfitting. The [saved B0 metric table](../results/image_d1_runs/b0_pilot_reporting_v2/tables/exploratory_metrics_threshold_0_5.csv) records these outcomes. Selecting settings and epochs on these same validation labels makes them development diagnostics rather than unbiased final estimates.

## 3. Compare architectures on matching patients

An **architecture** is the network's arrangement of layers. [Notebook 10](../notebooks/Subset/10_image_architecture_comparison.ipynb) adds DenseNet121 and ConvNeXt-Tiny through the shared pipeline. Each family receives four learning-rate/loss recipes and six epochs. All **eight new fits completed**, using the same fitting patients, validation patients, whole-image preparation and weak augmentation as the B0 reference.

DenseNet tests backbone rates 0.00003/0.00010; ConvNeXt tests 0.00001/0.00003. Each family crosses those with both loss settings. They retain their appropriate pretrained normalisation: DenseNet BatchNorm stays frozen; ConvNeXt LayerNorm is trainable during full fine-tuning. These choices support adapting each family without treating B0's provisional recipe as universally best.

Smaller GPU batches preserve an effective batch of 64 through **gradient accumulation**, which combines several small batches before updating weights. For example, DenseNet's two batches of 32 images contribute to one 64-image update; ConvNeXt uses four batches of 16. This makes the update size comparable while accommodating memory limits.

| Family | Best observed recipe | Selected epoch | AP | ROC-AUC |
|---|---|---:|---:|---:|
| B0 reference | 0.00003, weighted | 6 | 0.1383 | 0.8663 |
| DenseNet121 | 0.00003, unweighted | 2 | 0.1452 | 0.7730 |
| ConvNeXt-Tiny | 0.00003, weighted | 6 | 0.1495 | 0.8642 |

The [matched comparison table](../results/image_d1_runs/architecture_pilot_3h_v1/tables/matched_metrics_raw_threshold_0_5.csv) includes every recipe, not just these maxima. DenseNet's best AP with lower ROC-AUC illustrates why metric roles matter: the two summaries weight ranking behaviour differently. ConvNeXt has the highest selected pilot AP, but neither one split nor selected maxima establish a final architecture.

Outputs also include learning curves, raw reliability checks, epoch times and GPU memory. Mean measured epochs took about **33.7 seconds for B0, 43.1 for DenseNet and 45.3 for ConvNeXt** on this machine. Peak allocated memory was 2.81, 2.19 and 1.48 GiB respectively. These are measurements under different microbatches, not general claims about inherent architecture efficiency. Prediction-only checks reproduced saved scores and preserved training artifacts.

## 4. Extend the control and inspect transformations

### The completed twelve-epoch weak control

[Notebook 11](../notebooks/Subset/11_image_augmentation.ipynb) reports a fresh matched B0 control, condition A, with weak augmentation and unchanged deterministic preparation. It uses the same patients, seed 42, lower backbone rate and square-root weighting. It completed **12 epochs**, with no early stopping, under the same 30-epoch schedule.

Its best checkpoint remained **epoch 6, AP 0.1383**; epoch-12 AP fell to **0.1005**. More training did not improve AP in this run. The latest checkpoint therefore answers “where did execution stop?”, whereas the best checkpoint answers “which recorded epoch ranked best on inner validation?”. The later uncertainty analysis explicitly uses the latter. Other metrics can move differently: epoch-12 Brier error fell slightly, so the AP finding should not be generalised to every measure of score quality.

### The completed quality review, and the blocked comparison

Condition B proposed stronger affine and colour transformations, including larger rotations, small translations/scaling and reflection padding. Before training, the review used **24 fitting examples from distinct patients**, with four seeded variants per example and matched originals. The examples covered benign/malignant labels and varied framing, hair and lighting.

All twelve preview pages were inspected. **Seven stronger variants contained useful boundary clipping or reflected lesion fragments.** Reflection padding can copy image content into a border; near a lesion, that may create an artificial fragment. Condition B failed the quality gate and received no training. This completed review established a data-preservation problem, not evidence that a safely trained stronger model would perform worse.

Read the grids by comparing lesion edges, pigment, texture and plausible colour against the original. Extra visual variety is useful only if the label-relevant content remains intact. The [quality record](../results/image_d1_runs/b0_augmentation_12e_v1/quality_review.json) preserves the decision. The sampled review cannot guarantee safety for every fitting image.

Optional condition C would apply **CLAHE**, local contrast correction on the lightness channel while preserving colour channels. [Notebook 12](../notebooks/Subset/12_image_lighting.ipynb) records its deferred status and shows the exact A reference. C required a complete safe A/B pair; no corrected previews or C training were performed. Therefore neither a stronger-versus-weak AP difference nor a lighting improvement has been measured.

## 5. Measure prediction variation without changing training

### Load exactly the model that produced the reference

[Notebook 13](../notebooks/Subset/13_image_uncertainty.ipynb) selects the recorded best checkpoint from run `b0_augmentation_12e_v1`, condition A (`A_weak_none`), recipe `lr3e5_sqrt`, **epoch 6**. It uses the current shared checkpoint loader and reconstructs the checkpoint's exact deterministic preparation and patient membership. Choosing this weak control provides a declared reference; it does not declare B0 the final architecture.

Explicit DenseNet121 and ConvNeXt-Tiny recipes are also supported. Their checkpoint smoke checks completed, but the full 30-pass uncertainty results described here belong to B0. The loader rejects incompatible manifests, recipes or incomplete conditions rather than substituting another checkpoint. See [the inference code](../src/melanoma_detection/image_uncertainty.py).

A short **smoke check**, a small preliminary execution, preceded the full run. Saved checks verified checkpoint compatibility, separated patients, repeated deterministic preparation/predictions, active dropout, unchanged normalisation state and identity-TTA agreement. This protects the meaning of repeated inference: differences should come from the declared procedure, not an accidentally changed model or split.

### Monte Carlo dropout

Ordinary prediction disables dropout. **Monte Carlo (MC) dropout** instead makes repeated predictions while randomly switching off units in explicit dropout layers, using their trained rate **0.2**. Only those layers become active. BatchNorm remains in evaluation mode, and stochastic depth, which randomly skips network paths, also remains off. Calling the entire model's training mode would activate unwanted behaviour and invalidate the intended comparison.

The completed run uses **30 passes** with recorded seeds, beginning at 20261004. For every image it calculates a mean score and **sample standard deviation (SD)**, a measure of spread across passes.

**Worked example: mean versus variation.** Suppose three illustrative passes give scores `0.10, 0.20, 0.30`. Their mean is `(0.10 + 0.20 + 0.30)/3 = 0.20`. The sample SD is `sqrt((0.10² + 0² + 0.10²)/(3−1)) = 0.10`. A second image with `0.19, 0.20, 0.21` has the same mean but SD 0.01. Its score is less sensitive to these dropout draws. Neither SD tells us the probability that the prediction is wrong; a model can be consistently wrong.

Convergence checks reuse the first 5, 10 and 20 passes and compare them with the 30-pass summary. At 20 passes, mean scores differed by about **0.0010** on average, SD rankings had correlation **0.9949**, and the highest-SD fifth of images had **95.1% overlap** with the 30-pass set. This shows improving stability within this run, while some referral candidates still change. It does not prove that 30 passes suffice for every seed or checkpoint. The [convergence table](../results/image_uncertainty_runs/efficientnet_b0_lr3e5_sqrt_mc30_tta4_20261003T235530258956Z/mc_convergence.csv) records all three checks.

### Test-time augmentation stays separate

**Test-time augmentation (TTA)** evaluates several views of the same image with dropout off. This run uses identity, horizontal flip, vertical flip and both flips. They operate after exact deterministic preparation, preserving the full canvas without new crops, padding or interpolation. The identity view must reproduce ordinary predictions.

TTA mean scores summarise those four predictions; TTA SD describes orientation sensitivity. MC SD describes sensitivity to dropout draws. Their raw predictions and summaries are kept separate so these distinct sources of variation are not mixed together. The conservative flips also avoid the clipping and reflected fragments found in the stronger training candidate.

In SD histograms, farther right means greater variation, and bar height shows how many images have that spread. The scatterplot places MC SD horizontally and TTA SD vertically, with one dot per image. Similar ordering suggests the procedures flag similar images; scattered ordering suggests different sensitivities. Their numerical SDs need not match because the perturbations differ.

### Validate raw rows before averaging

Every raw score must be finite and within `[0,1]`. Each image/pass key must be unique; all expected images and passes must appear, and patient/label metadata must agree. Checking the total row count alone is insufficient: a duplicated image/pass could conceal a missing one. For example, two images with three passes require all six distinct combinations, not merely six rows.

The saved run contains **55,020 MC rows = 1,834×30** and **7,336 TTA rows = 1,834×4**, with complete validated coverage. Checks also confirmed identity-TTA agreement, the trained dropout rate, unchanged model state and **147 unchanged BatchNorm buffers**. Dedicated tests verified rejection of invalid scores, duplicate keys, missing passes and incomplete image coverage. Preservation receipts record unchanged checkpoints, histories and existing results.

The separate [run directory](../results/image_uncertainty_runs/efficientnet_b0_lr3e5_sqrt_mc30_tta4_20261003T235530258956Z/) contains raw predictions, image summaries, inference checks and provenance. **Provenance** records where a result came from: checkpoint identity, membership, preparation, seeds, pass counts, package versions and inference settings. This makes the output auditable without overwriting earlier experiments.

## 6. Assess score reliability and added referral value

### Ranking and probability quality answer different questions

The completed score-quality comparison reports:

| Predictions | AP | Brier error |
|---|---:|---:|
| Ordinary deterministic scores | 0.1383 | 0.02103 |
| MC mean scores | 0.1375 | 0.02109 |
| TTA mean scores | 0.1413 | 0.02090 |
| Constant fitting-prevalence score | 0.0180 | 0.01767 |

**Brier error** averages `(score − label)²`; lower is better. For a benign image with score 0.2 the error is 0.04, while for a malignant image with the same score it is 0.64. **Log loss**, also reported, penalises confidently wrong scores strongly. The constant baseline's lower Brier error despite poor AP shows that better ranking does not guarantee better probability quality, especially when benign images dominate.

A **reliability plot** groups images by score, compares mean score with observed malignant fraction and displays bin counts. Points below the diagonal suggest overprediction. In the highest deterministic-score group, the mean was about **0.181**, while **22/367 ≈ 0.060** images were malignant. This is a descriptive mismatch, with uncertainty from limited positives; it is not a fitted calibration correction. No calibrator or decision threshold was selected from these pilot labels. See [the score-quality table](../results/image_uncertainty_runs/efficientnet_b0_lr3e5_sqrt_mc30_tta4_20261003T235530258956Z/prediction_quality.csv).

### Keep the prediction rule fixed when comparing referral rankings

**Referral** here means setting images aside for simulated review. The remaining images form the **retained set**. Fractions count images, so one patient can contribute both referred and retained images. The comparison asks whether uncertainty helps choose which images to remove, beyond information already present in ordinary scores.

The tested rankings are high MC SD, high TTA SD, high ordinary prediction score, high **predictive entropy** and repeated random referral. Entropy is highest near score 0.5, where the binary score is least decisive, and lowest near 0 or 1. It is another score-derived statistic, not proof of calibrated uncertainty. Random referral uses **100 seeded repeats**, avoiding dependence on one lucky draw.

All rankings use the same images and fractions: 0%, 5%, 10%, 20%, 30%, 40% and 50%. The main comparison keeps the same deterministic scores and raw classification cutoff 0.5 for retained error. Changing both the ranking and the underlying prediction scores would confound two effects. The MC/TTA mean-quality table above answers the separate question of whether averaging changes predictions.

**Worked example using the saved 20% comparison.** The implementation refers `floor(0.20×1834) = 366` images and retains 1,468. MC SD refers **25 of 33 malignant images**, versus **22** for ordinary score ranking and **24** for TTA SD; random referral averages **6.7**. MC's retained classification error is **8/1468 = 0.545%**, versus **11/1468 = 0.749%** for score ranking. TTA also has eleven retained errors: nine missed malignant images and two falsely flagged benign images. Thus a higher malignant-referral count need not reduce total error. This is a point comparison; it does not show what a reviewer would diagnose or justify selecting 20% as a policy.

Read referral curves at the same fraction. Lower retained error means fewer mistakes among images left with the model; malignant-referral counts show who was removed. A lower error can arise partly because difficult or malignant images have left the retained population. The reports also include retained Brier error and patient counts, so a single error curve is insufficient.

### Account for related images and unresolved differences

The analysis uses **2,000 paired patient-bootstrap samples**. Bootstrapping resamples patients with replacement, carrying all their images together. Both rankings are evaluated on the same sampled patients, so their difference is paired. Resampling individual images would ignore within-patient dependence.

At 20%, MC-minus-score retained error was **−0.204 percentage points**, with a 95% interval **[−0.359, 0.000]**. The malignant-referral fraction difference was **+9.1 percentage points**, with interval **[0.0, 16.7]**. Both reach zero, so they do not clearly resolve an improvement over score ranking. MC SD outperformed random referral on these diagnostics, but beating randomness does not establish added value over ordinary predictions. TTA retained error was worse than score ranking at 5% and 10%; there is no consistent SD advantage across fractions.

In the difference plots, negative retained-error differences favour the uncertainty ranking; positive malignant-referral differences favour it. The [paired interval table](../results/image_uncertainty_runs/efficientnet_b0_lr3e5_sqrt_mc30_tta4_20261003T235530258956Z/paired_patient_bootstrap_differences.csv) gives estimates, intervals and valid sample counts. These intervals hold the trained checkpoint and inference draws fixed: they exclude retraining, recipe selection and new-hospital variation. Reviewing many fractions is exploratory, not threshold selection or independent confirmation.

## 7. Evaluate the bounded B0 procedure on five outer folds

[Notebook 14](../notebooks/Subset/14_image_nested_evaluation.ipynb) reads the new
`b0_nested_5fold_5h_v1` run. Within each outer training pool, the runner builds
three inner patient folds, screens all four B0 learning-rate/loss recipes for
six epochs on the first inner fold, and selects by screening AP. It confirms
only the selected recipe for twelve epochs on each of the three inner folds.
The median best-AP epoch fixes a fresh refit duration on all outer training
patients. Held-out outer labels are opened for scoring only after that refit.

All five raw folds completed on 6,864 images from 422 patients, including 128
melanoma images. Secondary mean fold AP is **0.1447** (95% conditional
patient-bootstrap interval **0.1033–0.2225**); mean fold ROC-AUC is **0.8408**.
These are development OOF results after earlier pilot use, including archived
fold 0. The interval holds fitted models and selections fixed and excludes
retraining variability and prior adaptation. It is not an untouched-test estimate.

The persistent five-hour clock includes implementation and reserves twenty
minutes for reporting. An absolute watchdog cutoff and compatible saved
epoch-boundary state support resumption without resetting the allocation.
Use the declared audit and resume commands:

```powershell
.venv/Scripts/python.exe scripts/run_image_nested.py --audit-only
.venv/Scripts/python.exe scripts/run_image_nested.py --resume
```

Final performance requires all five folds and exact canonical image coverage.
Mean fold ROC-AUC is now primary; mean-fold AP and pooled metrics are secondary because independently fitted
models can have different score scales. Patient-bootstrap intervals resample
whole patients within each fold and record invalid single-class draws. The
report adds raw reliability, fixed-0.5 confusion counts and verified paired
metadata comparisons when available. The raw stage fits no calibrator or cutoff.
Its intervals condition on saved fits and choices.

A separate calibration stage completed after the raw folds, with three fresh
fixed-duration inner refits per outer pool inside the same remaining deadline.
These training-only OOF scores fit sigmoid calibration
(`C=1`, rejecting non-positive slope) and an academic 80%-recall cutoff. Frozen
maps and cutoffs apply to unchanged raw outer predictions. Notebook 14 shows
raw/calibrated probability quality and actual outer recall, precision, F1,
specificity and counts after all five calibration folds complete. Its
training recall target does not guarantee the same outer recall. Neither stage
starts new MC uncertainty, fusion or reserved-test evaluation.

Calibration reduced mean fold log loss **0.0894 → 0.0791** and Brier
**0.0209 → 0.0174**, while preserving AP and ROC-AUC within every fold. The
training-selected 80%-recall scenario flagged **106/128 melanoma images
(82.81% outer recall)**, with **3.95% precision**. It flagged **2,684/6,864
images (39.10%)**: 106 melanoma and 2,578 benign images. It missed 22 melanoma
images and left 4,158 benign images unflagged. Raw and calibrated cutoffs gave
identical decisions because calibration preserved ordering. These are
development outcomes for an academic scenario; conditional intervals omit
retraining variability, and the workload/precision figures require attention
alongside recall. No clinical benefit has been established.

The larger 235-fit architecture/transformation procedure remains deferred;
bounded B0 evaluation cannot establish a final architecture winner. Earlier
pilot choices and the archived fold-0 scores also limit claims of independent
generalisation. The reserved test remains unopened.

## Historical experiments and work still planned

The [historical three-epoch B0 notebook](../notebooks/archive/image_model_three_epoch_fold0.ipynb) established an executable image prototype. It evaluated **1,370 images from 84 patients in outer fold 0**, including 25 malignant images, and recorded AP **0.1415** and ROC-AUC **0.8452**. It covered one fold even where historical filenames contain `oof`; it was not a complete five-fold OOF evaluation.

The [historical uncertainty notebook](../notebooks/archive/image_uncertainty_three_epoch_fold0.ipynb) tried 30 dropout passes and eight augmented views. It demonstrated repeated inference, but lacked the current ordinary-score/random baselines and paired patient intervals. Those results remain archived. Their scores cannot be used as a controlled comparison with today's inner-validation results because the patients, training and inference procedures differ.

The larger future search must declare and inspect safer transformations, then confirm each architecture's settings inside every outer training partition. Calibration and any recall/referral rule need training-only predictions and separate evaluation. Fusion and final reserved-test evaluation remain planned outside the active report sequence. Earlier metadata OOF and image inner-validation AP use different populations; the new bounded evaluation compares only verified matching OOF images.

## Main findings and limitations

The completed work provides a reproducible image pipeline, several viable architecture candidates and a documented weak B0 reference. It also caught unsafe stronger augmentation before training. Longer B0 training did not improve AP on the current split, and MC/TTA variation has not shown consistent referral benefit beyond ordinary score ranking.

The principal limits are one repeatedly used selection split, only 33 validation malignant images, retained repeated views and uncertainty intervals conditional on an existing fit. Potential near duplicates, clinical label errors and unfamiliar image sources remain unresolved. A score of 0.7 is therefore a prediction score, not a validated 70% risk; an SD is procedure-specific variation, not an error probability.

These results support further patient-safe confirmation. Bounded raw five-fold B0 evaluation and separate training-only calibration are complete and verified, with mean fold AP 0.1447 and improved probability quality. The larger architecture search, final model selection and reserved test remain deferred. The reserved test should be evaluated once after the modelling and decision procedure is fixed; simulated referral needs additional evidence before any claim about reviewer or clinical benefit.


## Read-only v2 presentation refresh

Regenerate the editable reports from the verified snapshots without executing model code:

```powershell
.venv/Scripts/python.exe -B scripts/build_metadata_tuning_v2_notebooks.py --condition both --run-id roc_auc_grid36_v1
.venv/Scripts/python.exe -B scripts/build_image_roc_reports.py
.venv/Scripts/python.exe -B scripts/execute_metadata_roc_reports.py
```

The image builder owns original09–15 and clean09–12. Original13 reads the completed uncertainty run `efficientnet_b0_lr3e5_sqrt_mc30_tta4_20261003T235530258956Z`; saved MC/TTA scores, referral results and previews are inspected rather than recalculated. Execute reports only after the v2 experiment has complete verified receipts. The execution and acceptance stages create new receipts, previews and PNG exports under `results/metadata_nn_tuning/v2/report_verification/`. No historical image result, AP fitting source, dataset assignment or reserved-test file is changed.

The preserved five-fold image procedure has primary mean-fold ROC-AUC **0.8408** on original development records (conditional 95% patient interval **0.7953–0.8747**) and **0.8308** on retained records (**0.7871–0.8651**). Secondary mean-fold AP is 0.1447 and 0.1316. These values come from the unchanged saved outer predictions and existing bootstrap artifacts. They assess the historical AP-selected image procedure; the new metadata search has a separate ROC selection identity.
