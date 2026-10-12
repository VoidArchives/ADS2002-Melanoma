# ROC-AUC-first metadata workflow

The original and deduplicated workflows compare models primarily by **unweighted mean outer-fold ROC-AUC**. Mean outer-fold AP, log loss and Brier score remain secondary. Pooled ROC-AUC and AP are supporting summaries because pooling permits comparisons between scores produced by different outer models. A pooled winner need not be the mean-fold winner.

Read the notebooks in their existing order: **05 metadata models → 06 fixed neural network → 07 tuning including NN → 08 robustness**. Stage 06 explains the fixed network; stage 07 compares four setting-selection procedures. The four tuning bars remain in this order:

| Chart | Role |
|---|---|
| Mean outer-fold ROC-AUC | Primary model-comparison metric |
| Mean outer-fold AP | Secondary ranking metric, sensitive to performance among rare malignant images |
| Pooled ROC-AUC | Supporting summary of saved OOF scores |
| Pooled AP | Supporting summary of saved OOF scores |

Fixed bars remain blue `#2563EB`, tuned bars gold `#C58A00`; fixed curves are dashed and tuned curves solid. Original/clean reports share presentation templates, including explanatory examples, table formatting, figure composition and spacing. At least 90% of pre-update Markdown coverage is required, together with substantive visual review.

## Two distinct completed-development experiment identities

The earlier `nn_grid36_v1` experiment selected settings using **AP**. Its source, configuration, predictions, models, metrics, figures, memberships and receipts remain historical evidence. Displaying its ROC-AUC does not make it ROC-tuned. The new experiment has `experiment_version: 2`, `run_id: roc_auc_grid36_v1` and `selection_metric: roc_auc`. These identities are passed explicitly to report readers; AP receipts cannot satisfy a ROC report request.

This is an **additional development experiment after earlier outer-fold results were examined**. Its objective, search and source identities were frozen before execution. The grid is not expanded using the new outer scores. Nested patient separation protects each outer fold from its own fitting/selection, but does not erase prior use of the development dataset. Deduplicated evidence remains the main clean-data evidence; original evidence describes the duplicate-bearing comparison. Neither run fits an all-development final model or accesses the reserved test.

Original data contains 6,864 images; the frozen `dedup_v1` manifest retains 6,706 images from the same 422 patients. Both use age, sex and anatomical site, unchanged preprocessing and the original five patient outer folds. Metadata-suite inner memberships were **reconstructed from the original notebook protocol**, rather than captured contemporaneously. Clean memberships must exactly equal original memberships filtered to retained image IDs. Image-workflow inner memberships are different and are not substituted.

## Frozen setting and duration selection

The conventional spline logistic, Random Forest and XGBoost grids remain unchanged. The network retains one ReLU hidden layer, AdamW weight decay 0.001, unweighted binary cross-entropy, batch size 128, maximum 100 epochs, and its existing loss-based stopping rules.

| NN setting | Frozen values | Meaning |
|---|---|---|
| Hidden units | 8,16,32,64 | Width controls the number of learned hidden representations. |
| Learning rate | 0.0003,0.001,0.003 | Controls the size of each optimizer update. |
| Dropout | 0.0,0.2,0.4 | Masks this proportion of hidden activations during training; disabled for prediction. |

There are 36 settings. Current width 32 / learning rate 0.001 / dropout 0.2 is first (`r00`), followed by remaining combinations in ascending width, learning rate and dropout order. Each setting uses seeds 42,142,242. **Average their probabilities before calculating each inner-fold ROC-AUC**, then take the unweighted mean of the three inner-fold scores. Do not average seed AUCs or select the best seed. Exact score ties follow the frozen candidate order. Conventional ties follow their preserved ParameterGrid order.

Within outer training, each of the three saved inner patient groups is held out in turn. For inner validation group `v`, remaining groups are `[(v+1)%3,(v+2)%3]`. Seeds with indices 0 and 2 stop on the first remaining group and fit on the second; seed index 1 reverses these roles. Fit preprocessing only on epoch-fitting patients; choose duration by stopping loss; discard that network and preprocessor. Then fit a fresh preprocessor and network on both inner-training groups for that duration, and predict the excluded inner group.

After selecting a setting, use stopping fold `(outer_fold+1+seed_index)%5` and training seed `base_seed+outer_fold`. Duration selection stays within outer training, followed by a fresh refit on all outer-training patients and prediction of the excluded outer fold. Outer labels cannot influence settings, duration, preprocessing or predictions. Changing the comparison objective does not require changing BCE training loss.

## Verified reuse and bounded execution

The v2 wrapper imports the byte-identical v1 fitting implementation instead of changing its compatibility rules. Parent AP completion receipts bind source/configuration bytes, inputs, memberships, preprocessing, training protocol, versions and prediction coverage. NN candidate probabilities are reused for both conditions; original conventional candidate probabilities are also reusable. Both ROC-AUC and AP are recomputed from probabilities, never converted from saved AP values.

The 630 absent clean conventional inner fits are reconstructed inside their proper inner-training patients. No original-condition predictions or outer-trained models replace them. Selected outer procedures may be reused only when parameters, patients, seeds and stopping/refit rules match exactly. Cross-run reuse records explicit prediction and parent-receipt hashes; it never edits an AP checkpoint's compatibility digest. Independent stopping and refit receipts permit deterministic recovery after interruption.

Counters separately describe newly executed fit phases, externally reused fit phases, resumed native phases, executed phases in the current invocation, and candidate-rescoring operations. Resumed phases are a subset of native phases already executed and are not additional fits. The approved ceiling is 725 new calls:630 clean inner fits, up to 90 selected outer calls and five original baseline logistic fits. Dummy scores use each outer-training prevalence without fitting. Verified historical age and ablation figures are preserved; no age model is fitted on all development records.

Missing or incompatible expected caches fail closed, rather than silently launching a larger experiment. The five baseline fits have their own versioned predictions, patient roles, models and receipts. They are reused in original 04/05; clean baseline scores come from their verified historical suite.

## Commands and output boundaries

```powershell
.venv/Scripts/python.exe -B scripts/run_metadata_tuning_v2.py --plan
.venv/Scripts/python.exe -B scripts/run_metadata_tuning_v2.py --config configs/metadata_nn_tuning_roc_auc.yaml --resume
.venv/Scripts/python.exe -B scripts/verify_metadata_tuning_v2_results.py
.venv/Scripts/python.exe -B scripts/publish_metadata_tuning_v2.py
.venv/Scripts/python.exe -B scripts/build_metadata_tuning_v2_notebooks.py
.venv/Scripts/python.exe -B scripts/build_image_roc_reports.py
.venv/Scripts/python.exe -B scripts/execute_metadata_roc_reports.py
.venv/Scripts/python.exe -B scripts/finalize_metadata_roc_visual_review.py
.venv/Scripts/python.exe -B scripts/verify_metadata_roc_reports.py --require-complete
```

Use `--condition original` or `--condition dedup_v1` for one runner condition. The runner also supports `average_precision` as an explicit selection objective under a separately named/configured experiment. Changing metric/configuration/memberships cannot resume a different experiment. The current reports require version 2 plus the exact ROC experiment identity.

| Purpose | Output |
|---|---|
| Original ROC experiment | `results/metadata_nn_tuning/v2/original/roc_auc_grid36_v1/` |
| Clean ROC experiment | `results/dedup_v1/metadata_nn_tuning/v2/roc_auc_grid36_v1/` |
| Original 04/05 baseline reconstruction | `results/metadata_nn_tuning/v2/baseline_diagnostics/original_baseline_v1/` |
| Execution, visual and acceptance receipts | `results/metadata_nn_tuning/v2/report_verification/` |
| Pre-update source/notebook snapshots and protected hashes | `results/metadata_nn_tuning/v2/archive_before_update/` |

V1 latest pointers and outputs remain separate. New reports write only to v2 destinations, even when they read historical image or metadata artifacts. The 21 affected active notebooks are original 04–15 and clean 04–12. Preparation 01–03, Full notebooks and archives are excluded from execution. Kernel guards block fitting, inference dispatch, reserved-test reads and writes outside new v2 outputs. Original uncertainty reporting reads the pinned existing saved inference run and six saved figures without loading a network or rerunning inference.

## Robustness and interpretation

For ROC-selected NN versus fixed NN, fixed spline and ROC-selected spline, 2,000 paired bootstrap replicates resample whole patients **within outer folds**. Every model receives the same draw. Lead with mean-fold ROC-AUC differences and 95% intervals; retain AP, log loss, Brier and pooled supporting comparisons. Intervals condition on saved predictions and exclude repeating the complete fitting and selection process. A zero-crossing interval leaves the difference unresolved and does not establish equivalence.

AP describes a different ranking objective from ROC-AUC; a model may improve one and worsen the other. Log loss and Brier assess probability quality. A ROC point winner with a broad interval is a candidate supported by the chosen objective, rather than proven superior. Existing calibration maps and recall cutoffs stay attached to their original procedures and do not establish calibrated probabilities or decision performance for new ROC-selected models.

Image and downstream reports lead with ROC-AUC where their saved evaluation population permits it. Historical image recipes, selected epochs and admission gates remain explicitly AP-selected. Inner-validation pilots remain exploratory; reports never invent outer-fold summaries for them. Model-choice evidence does not automatically replace the final spline procedure.

## Acceptance evidence

The independent result verifier reconstructs all 1,170 candidate inner scores per condition:540 NN ensembles and 630 conventional candidates. It checks the frozen grids, deterministic winners/margins, membership coverage, selected outer predictions, ensemble averaging, OOF metrics, patient roles and fit counts. Tests cover different AP/ROC winners, class support, held-out-label independence, training-only preprocessing, phase recovery, incompatible identities and strict archived-source/path rejection.

Report acceptance checks complete top-to-bottom execution of all 21 notebooks, no errors, valid local links, preserved Markdown coverage, matching metadata heading structure and visual fingerprints for all rendered figures. Historical hash verification checks 31,655 protected files and all 131 pre-update source/notebook snapshots. Results are published only after both five-fold completion receipts pass verification. Final results and counts are recorded below after execution and independent acceptance.

## Completed ROC-selected experiment: `roc_auc_grid36_v1`

Both conditions passed all five outer folds. These are additional development findings after earlier results were examined.

| Dataset | Procedure | Mean-fold ROC-AUC (primary) | Mean-fold AP | Log loss | Brier |
|---|---|---:|---:|---:|---:|
| Original | ROC-selected NN | 0.6809 | 0.0674 | 0.08863 | 0.01808 |
| Original | Fixed NN | 0.6788 | 0.0581 | 0.08896 | 0.01810 |
| Original | ROC-selected spline | 0.6739 | 0.0685 | 0.08943 | 0.01810 |
| Original | Fixed spline | 0.6629 | 0.0649 | 0.08974 | 0.01814 |
| Original | ROC-selected Random Forest | 0.6084 | 0.0507 | 0.09104 | 0.01821 |
| Original | Fixed Random Forest | 0.6050 | 0.0554 | 0.09075 | 0.01815 |
| Original | Fixed XGBoost | 0.5969 | 0.0586 | 0.09255 | 0.01822 |
| Original | ROC-selected XGBoost | 0.5628 | 0.0524 | 0.09433 | 0.01832 |
| Deduplicated | ROC-selected NN | 0.6774 | 0.0688 | 0.08981 | 0.01836 |
| Deduplicated | Fixed NN | 0.6767 | 0.0596 | 0.09017 | 0.01838 |
| Deduplicated | ROC-selected spline | 0.6696 | 0.0691 | 0.09077 | 0.01839 |
| Deduplicated | Fixed spline | 0.6556 | 0.0648 | 0.09101 | 0.01842 |
| Deduplicated | ROC-selected Random Forest | 0.6107 | 0.0522 | 0.09185 | 0.01847 |
| Deduplicated | Fixed Random Forest | 0.6032 | 0.0548 | 0.09195 | 0.01845 |
| Deduplicated | Fixed XGBoost | 0.5930 | 0.0591 | 0.09385 | 0.01851 |
| Deduplicated | ROC-selected XGBoost | 0.5624 | 0.0535 | 0.09618 | 0.01864 |

Actual new fitting calls: **710**, comprising 43 original core fits, 662 clean core fits, and five original baseline logistic fits. The core ledger explicitly reuses **7215** historical fit phases and performs **2340** candidate rescoring operations. Resumed native phases are recorded separately and are not added to executed or external-reuse totals. Dummy scores require zero fits.

| Clean-data comparison | Mean-fold ROC-AUC difference | 95% conditional paired patient interval | AP difference | 95% conditional paired patient interval |
|---|---:|---|---:|---|
| nn tuned minus fixed nn | +0.0007 | [-0.0083, +0.0098] | +0.0092 | [-0.0014, +0.0278] |
| nn tuned minus fixed spline | +0.0218 | [-0.0057, +0.0411] | +0.0039 | [-0.0062, +0.0159] |
| nn tuned minus tuned spline | +0.0079 | [-0.0204, +0.0290] | -0.0004 | [-0.0120, +0.0086] |

The clean-data ROC-AUC point leader is **ROC-selected NN** (0.6774). Its AP is 0.0688, compared with fixed spline AP 0.0648. The paired intervals above indicate whether the NN advantage is resolved against each specified comparator; zero-crossing intervals leave it uncertain. These intervals condition on saved predictions and omit repeating fitting and selection. The chosen primary metric supports consideration of the ROC leader, with the AP and probability-error trade-offs retained; no final model is automatically replaced.

All bootstrap analyses use 2,000 whole-patient replicates within outer folds. Historical image selection remains AP-based. No calibration maps or recall cutoffs were transferred to new models; no image model was retrained, no all-development final fit was performed, and reserved-test data was not accessed.

The earlier **AP-selected** clean NN had mean-fold ROC-AUC 0.6804 and AP 0.0681; the new **ROC-selected** NN has ROC-AUC 0.6774 and AP 0.0688. Changing the inner selection objective does not guarantee a higher outer score. These remain separate development experiments, with their actual objectives preserved; a historical ROC score is not evidence of ROC-based tuning.

All five clean selected NN settings use eight units and learning rate 0.0003, with varying dropout. Their small inner margins indicate selection sensitivity rather than universal optimality. The evidence concerns the prescribed three-seed selection procedure; it does not justify choosing seed 42 or a single global recipe from outer results. Original and clean results share patients, so their agreement is sensitivity to deduplication rather than independent replication. The near tie with fixed NN and the zero-crossing primary intervals leave the benefit of tuning unresolved.

The 7,215 reuse count refers to the four-family tuning ledger. Historical clean dummy and categorical-logistic baseline predictions also appear in 04/05; their ten old fitting calls are retained diagnostic evidence, rather than new ROC-experiment fits. Other historical age, ablation, calibration and image analyses are preserved separately. No historical fit is counted twice across the current tuning and baseline-reconstruction totals.

## Completed verification

Both five-fold result reconstructions and the guarded completed-run resume checks passed. All **50 focused protocol, provenance, reporting and entry-point tests** passed. All **21 affected reports** completed top to bottom, and all **178 rendered figures** passed inspection with current notebook/figure hashes. Markdown coverage exceeds the 90% requirement, shared metadata headings match, and both robustness appendices preserve their archived heading levels.

The [final acceptance receipt](../results/metadata_nn_tuning/v2/report_verification/report_acceptance.json) records no errors and verifies **31,655 unchanged historical files**, 21 archived notebooks, 110 archived supporting files, and unchanged AP fitting dependencies. The [fit ledger](../results/metadata_nn_tuning/v2/report_verification/combined_fit_ledger.json) records 710 newly executed fitting calls, 7,215 externally reused core fit phases, zero resumed phases, and 2,340 rescoring operations. All paired evaluations retain 2,000 valid patient-bootstrap replicates.

Historical bounded image pilots without saved completed runs remain explicitly pending in clean stage 10. Its single saved preparation figure and historical evaluation scope are preserved; report completion does not imply that those pilots were trained. No image training/inference, reserved-test access or final all-development fitting occurred during report execution.
