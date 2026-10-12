# Deduplicated development workflow

**Current v2 reporting contract.** Unweighted mean outer-fold **ROC-AUC** is the primary ranking summary; mean-fold AP, pooled metrics, log loss and Brier score remain supporting measures. Each fold has equal weight, and patient-bootstrap intervals resample whole patients within folds. A single selected inner split is exploratory and cannot supply a five-fold estimate. Scores describe individual images; patient grouping does not turn them into patient-level diagnoses.

The metadata v2 experiment selects settings by inner ROC-AUC. Existing image recipes, checkpoints, durations and transformation-admission gates retain their historical AP-based rules. Reporting their saved ROC-AUC does not retrospectively make them ROC-selected. Calibration and recall cutoffs remain attached to their original procedures. All active report notebooks read saved artifacts; they start no training, image smoke or checkpoint inference and never open the reserved test. Final fitting and independent testing wait until the entire procedure is frozen.

Presentation snapshots are preserved under `results/metadata_nn_tuning/v2/archive_before_update/`. New image figures/tables use `results/metadata_nn_tuning/v2/report_verification/image_reports/<condition>/<stage>/`; current metadata assessment tables and figures use the explicit v2 `roc_auc_grid36_v1` run's `report/` directory. v1 AP results and historical image outputs retain their original identities.


The active workflow includes exact-copy removal in development-data cleaning and uses the same retained image IDs throughout metadata modelling, image modelling and evaluation. Existing data, images, notebooks, checkpoints and results remain reproducible. The reserved test is outside these entrypoints.

Start with `notebooks/Subset_dedup/01_data_cleaning.ipynb` and read the twelve clean notebooks in numerical order. Every notebook explicitly loads the frozen manifest; executing them starts no fitting or training. Stages with unfinished image work show pending results. `notebooks/Subset_dedup/12_combined_evaluation.ipynb` is the main clean-model comparison.

The original `notebooks/Subset` notebooks are the presentation templates for this sequence. The adapted reports preserve their Markdown cells, section order, worked explanations, model colours, chart composition and plotting defaults. Findings are updated to the retained development records and matching saved clean fits. Each changed experiment scope or estimand is labelled beside the relevant section; original results are never relabelled as clean results. The generator enforces a minimum of 90% of the mapped originals' Markdown word count and records both counts in `METHOD_COVERAGE.md`'s accompanying coverage receipt.

The earlier v1 regeneration command is retained below for historical traceability; it does not build the current v2 presentation. Use the explicit v2 builders documented at the end of this guide.

```powershell
.venv\Scripts\python.exe -B scripts/build_dedup_workflow_notebooks.py --execute
```

This command reads existing results, makes tables and figures, and saves executed notebook outputs. It performs no model fitting, image uncertainty inference or reserved-test evaluation. The earlier clean presentation is archived under `results/dedup_v1/clean_workflow/presentation_revisions/pre_template_v2/`.

The earlier additional metadata suite is complete: 695 new CPU fits took about 175 seconds, with 60 completed clean fits reused. Every baseline, feature ablation and tuned model has the same 6,706 OOF records. Metadata tuning memberships were reconstructed from original CSV row ordering and seed `42 + outer_fold`, then filtered; no historical membership receipt exists for that notebook, so these memberships are explicitly labelled reconstructed.

The completed AP-selected v1 stages 06 and 07 used the same report template in the original and clean workflows. Stage 06 remains the fixed NN; stage 07 adds the frozen 36-setting three-seed NN search to the existing spline, Random Forest and XGBoost grids. Both five-fold tuning conditions completed: original data required 3,960 new fits, while the clean condition required 3,270 new fits and reused 690 verified historical clean fits. This additional development experiment uses the saved metadata-suite memberships, rather than the image memberships. Its run configuration, execution commands, selection protocol and preservation boundaries are documented in [metadata NN tuning](metadata_nn_tuning.md). New reports use separate versioned `metadata_nn_tuning` directories; all affected notebooks were archived before editing. Five-fold completion receipts govern whether findings may be displayed.

The existing image controller owns five-fold B0, image calibration and both bounded pilots. The historical controller waited for that queue, then completed clean MC/TTA uncertainty inference, regenerated evaluation and executed the twelve notebooks. This describes the saved experiment, not current v2 report execution. It never starts a duplicate B0 job or resets an image deadline.

```powershell
.venv\Scripts\python.exe -B scripts/continue_clean_workflow.py
```

The controller holds an exclusive operating-system lock. Reinvoking a completed metadata or uncertainty runner verifies its receipt and reuses outputs. A training-budget stop keeps the overall workflow incomplete and preserves resumable checkpoints; it does not silently extend the allocation.

For a report-only refresh, without training or inference:

```powershell
.venv\Scripts\python.exe -B scripts/continue_clean_workflow.py --report-only
```

Current clean status is saved in `results/dedup_v1/clean_workflow/status.json`. Full JPEG re-verification is recorded under `clean_workflow/cleaning/`; metadata outputs are under `metadata_suite/metadata_full_dedup_v1/`; inference is under `uncertainty/b0_uncertainty_dedup_v1/`. Evaluation exports are immutable directories selected by input fingerprints, with `clean_workflow/evaluation/latest.json` pointing to the current result.

The primary model contrasts remain raw B0 versus fixed raw spline, raw B0 versus the prescribed NN mean, and NN mean versus fixed raw spline. Wider model and ablation comparisons are secondary. Training-only spline/image calibration and 80%-recall cutoff scenarios are separate from ranking comparisons. Fixed 0.5 diagnostics are descriptive; cutoffs selected on the same scored OOF labels are excluded from primary decision claims.

Clean uncertainty uses each patient's held-out B0 checkpoint, 30 MC-dropout passes and four deterministic tensor flips. Batch normalisation remains frozen. Referral uses the saved deterministic base score and paired whole-patient bootstrap draws within outer folds; MC/TTA mean scores do not automatically replace calibrated probabilities or cutoffs. The 20% referral point is a descriptive diagnostic, not clinical validation.

All-development final fitting, reserved-test evaluation, full DenseNet/ConvNeXt OOF experiments, stronger augmentation, CLAHE, resolution changes and fusion remain deferred. Pilots cannot automatically become the chosen final model.

The earlier `notebooks/Subset/15_deduplicated_workflow.ipynb` remains a supporting audit/sensitivity report. The active sequence and main conclusions live in `Subset_dedup`. Both report sets check saved artifacts rather than starting training. Saved executions reflect the time they were run; refresh after later stages finish.

Use `.venv\Scripts\python.exe scripts/execute_dedup_report.py` to execute and save the report with the project kernel. Add `--wait-workflow` to wait for a terminal controller state, then refresh the report. This executor does not register a global kernel or invoke training. It exports an HTML preview when `nbconvert` is available.

## What is complete and what comes next

The audit confirms 158 exact pixel-identical pairs. Retaining the smallest image ID excludes 157 benign copies and one malignant copy, leaving **6,706 images, 127 malignant images, 422 patients and 88 malignant patients**. Source images remain intact. Outer patient folds and the original image-model inner memberships are preserved through filtering.

The saved-prediction sensitivity report, clean CPU metadata replication, five-fold B0 evaluation, training-only image calibration/cutoff refresh and clean uncertainty inference have complete saved summaries. The duration and ConvNeXt pilots do not yet have completed summaries. The upstream controller still records an incomplete queue after its sensitivity stage stopped; the reports verify each stage's own artifacts and distinguish this queue status from completed model evidence.

`saved_oof_v2` supersedes the first sensitivity report's missing-site diagnostic. Version 1 failed to recognise the cleaned `Unknown` sentinel. Version 2 reports 196 original and 195 retained missing-site images, each population containing 53 patients and two malignant patients. Original rankings, paired intervals and cutoff results did not change. Both versions remain preserved; use version 2 for reporting.

The audit also records age variation, missing-site concentration, unequal patient image counts and deterministic image-quality examples. Multiple ages/sites do not automatically indicate wrong records. Quality and coarse similarity measurements create review candidates, not exclusions. The baseline keeps the historical weak rotation so its black corners and possible edge clipping remain a documented limitation rather than a simultaneous preparation change.

The supplied `Melanoma_Project_main.ipynb` is a methodological reference, not a source of runner instructions. It manually merged the non-identical pair `ISIC_2887033` / `ISIC_5194874`, which would produce 6,705 rows. Both records remain retained under the exact-copy rule. Borrow its matched image-only versus image-plus-metadata fusion control later; do not transfer its clinical or clean-OOF claims to these experiments.

## Supporting duplicate-sensitivity evidence

The following original-versus-retained-versus-clean comparison is retained as an appendix. It is not required as the main side-by-side presentation; the active workflow compares clean models against clean models.

| Condition | Models were fitted on | Evaluation records | What the comparison establishes |
|---|---|---|---|
| Original | Original development data | Original records | Historical baseline |
| Retained | Original development data | Retained records | Evaluation weighting sensitivity relative to original |
| Clean | Retained development data | Same retained records | Combined fitting and training-only selection influence relative to retained |

Filtering preserves each retained image's historical score. It does not average copies or remove their influence on historical preprocessing, training, recipe/epoch selection, calibration or cutoff fitting.

Primary metrics are unweighted means of the five outer-fold AP and ROC-AUC values. Pooled metrics are secondary. Paired intervals use 2,000 whole-patient bootstrap draws within outer folds, seed 42, with common patient multiplicities across models. These intervals condition on the saved fits: they omit retraining variability and previous development-set adaptation. An interval containing zero does not demonstrate equivalence.

Equal-patient weighting is a separate sensitivity diagnostic, assigning each patient's retained images weights summing to one. Subgroups include support counts; undefined metrics stay missing. Frozen cutoff reports show malignant images detected/missed, benign images flagged, precision and workload. The 80% training recall target is an academic scenario, not a clinical validation claim.

## Run the stages explicitly

Run commands from the project root with the existing project environment. They are documented here for reproducibility; the reporting notebook never invokes them.

### 1. Audit and saved-prediction sensitivity

```powershell
.venv\Scripts\python.exe scripts/run_dedup_audit.py
.venv\Scripts\python.exe scripts/run_dedup_sensitivity.py --run-id saved_oof_v2
```

The manifest and retention ledger live in `data/subset/processed/dedup_v1/`. Verification enforces canonical-source and artifact hashes, deterministic representatives, exact counts and preserved patient memberships. A completed sensitivity invocation verifies its saved receipt instead of rewriting results. Changed inputs or methods require a new run identity.

### 2. Clean metadata and B0 replication

```powershell
.venv\Scripts\python.exe scripts/run_dedup_metadata.py --config configs/metadata_dedup.yaml
.venv\Scripts\python.exe scripts/run_image_nested_calibration.py --config configs/image_nested_calibration_dedup.yaml --declare-only
.venv\Scripts\python.exe scripts/run_image_nested.py --config configs/image_nested_dedup.yaml --resume
.venv\Scripts\python.exe scripts/finalize_image_nested.py --run-id b0_nested_dedup_v1
```

The CPU metadata run contains five raw spline fits and 30 neural-network fits (15 stopping fits plus 15 fresh outer refits). Its optional spline calibration is enabled and adds 20 inner spline fits and five maps. The NN reproduces the three prescribed seeds, stopping rule and equal-mean ensemble. Its stopping patients come directly from the original outer-fold assignments; no splitter is rerun on retained rows. Fitting alone learns preprocessing, and outer labels cannot influence stopping or calibration.

B0 runs the existing bounded selection procedure: 35 GPU fits, including recipe screening, confirmation and fresh outer refits. All distinct fits start from original `IMAGENET1K_V1` weights. Historical melanoma checkpoints cannot initialise these experiments. Configured persistent budgets and checkpoint compatibility remain enforced; incomplete runs do not produce final comparisons.

### 3. Image calibration, cutoffs and three-condition comparison

```powershell
.venv\Scripts\python.exe scripts/run_image_nested_calibration.py --config configs/image_nested_calibration_dedup.yaml --resume
.venv\Scripts\python.exe scripts/finalize_image_nested_calibration.py --run-id b0_nested_calibration_dedup_v1
.venv\Scripts\python.exe scripts/run_dedup_sensitivity.py --run-id clean_three_condition_v1 --clean-image-run b0_nested_dedup_v1 --clean-metadata-receipt results/dedup_v1/metadata/comparator_receipt.json --clean-calibration-run b0_nested_calibration_dedup_v1
```

Declare the existing calibration protocol before new outer predictions. Fifteen fresh image inner fits generate training-only fixed-duration scores for maps and cutoff selection. The calibration runner shares the original raw allocation rather than silently extending it. Raw and calibrated reports require five complete folds and verified predictions/receipts.

The three-condition report is a new output under `results/dedup_v1/sensitivity/clean_three_condition_v1/`. It requires complete matching clean image and metadata outputs. The raw and calibrated image runs have separate identities under `results/image_d1_runs/`.

### 4. Bounded optimisation after the completed baseline

```powershell
.venv\Scripts\python.exe scripts/run_dedup_image_pilots.py --config configs/image_dedup_pilots.yaml --stage duration
.venv\Scripts\python.exe scripts/run_dedup_image_pilots.py --config configs/image_dedup_pilots.yaml --stage challenger
```

Both pilots use outer fold 0's training pool and its preserved three inner patient folds. The duration pilot fits three B0 trajectories to 24 epochs, retaining epoch-12 and epoch-24 snapshots. ConvNeXt screens four six-epoch recipes, continues the winner and fits the remaining inner folds: six distinct fits and 90 epochs. Effective batch, preparation and the 30-epoch scheduler stay comparable. Best-in-cap results are explicitly selected inner-validation evidence; fixed endpoints are reported separately. A pilot winner is not automatically promoted to the final architecture or duration.

The pilot identities are `b0_duration_dedup_v1` and `convnext_duration_dedup_v1`. They retain their own clocks and budgets. Any promoted procedure needs a further patient-safe five-fold evaluation; resolution changes, stronger augmentation, CLAHE, more architectures and fusion training remain deferred.

## Reuse, regeneration and final checks

The available workload records provide estimates rather than completion guarantees:

| Stage | Distinct fits | Workload basis |
|---|---:|---|
| Clean metadata, including spline recalibration | 60 CPU fits | 36.9 seconds of recorded fold work in this execution, plus setup and reporting |
| Clean bounded B0 | 35 GPU fits | About 172 minutes of historical training |
| Image calibration | 15 GPU fits | About 54 minutes of historical training; shares B0's five-hour session |
| B0 duration pilot | 3 GPU fits, 72 epochs | About 40 minutes of historical training speed; separate 90-minute session |
| ConvNeXt pilot | 6 GPU fits, 90 epochs | About 68 minutes of historical pilot speed; separate 120-minute session |

Allow roughly 4.5–5.5 hours for B0, image calibration and verification, subject to the configured five-hour session and its reporting reserve. Current B0 screening epochs have taken roughly 35–54 seconds. No runner silently resets a clock or extends a limit; a partial stage remains incomplete and blocks later experiments.

The running continuation queue is `scripts/continue_dedup_workflow.py`. Its current phase and completed stages are recorded in `results/dedup_v1/implementation/workflow_status.json`. A separate report executor waits for a terminal workflow status before refreshing the notebook, including when a budget stops the queue.

Reuse original image files, pretrained weight caches, patient assignments, model definitions and verified historical predictions for the sensitivity stage. Regenerate fitted preprocessing, model weights, clean OOF scores, calibration maps, cutoffs, subgroup figures and numerical reports from matching clean runs.

Historical uncertainty, MC/TTA, referral and final bundle outputs retain their original meaning. They do not describe the new clean checkpoints until regenerated. Later fusion needs matching clean training-only predictions and a matched image-only versus image-plus-metadata control. Final all-development fitting waits until the procedure is settled; the reserved test remains untouched here.

Verify exact retained prediction coverage across modalities, training/validation patient isolation, unchanged original artifacts, all five outer folds, positive-slope training-only maps and shared paired records. Outer-label/feature perturbation tests confirm that fitting and selection cannot depend on outer validation values. Source, runtime, manifest and output hashes guard resume and reporting.


## Read-only v2 presentation refresh

Regenerate the editable reports from the verified snapshots without executing model code:

```powershell
.venv/Scripts/python.exe -B scripts/build_metadata_tuning_v2_notebooks.py --condition both --run-id roc_auc_grid36_v1
.venv/Scripts/python.exe -B scripts/build_image_roc_reports.py
.venv/Scripts/python.exe -B scripts/execute_metadata_roc_reports.py
```

The image builder owns original09–15 and clean09–12. Original13 reads the completed uncertainty run `efficientnet_b0_lr3e5_sqrt_mc30_tta4_20261003T235530258956Z`; saved MC/TTA scores, referral results and previews are inspected rather than recalculated. Execute reports only after the v2 experiment has complete verified receipts. The execution and acceptance stages create new receipts, previews and PNG exports under `results/metadata_nn_tuning/v2/report_verification/`. No historical image result, AP fitting source, dataset assignment or reserved-test file is changed.

The preserved five-fold image procedure has primary mean-fold ROC-AUC **0.8408** on original development records (conditional 95% patient interval **0.7953–0.8747**) and **0.8308** on retained records (**0.7871–0.8651**). Secondary mean-fold AP is 0.1447 and 0.1316. These values come from the unchanged saved outer predictions and existing bootstrap artifacts. They assess the historical AP-selected image procedure; the new metadata search has a separate ROC selection identity.
