# Method and figure coverage

Every plot uses clean development records or verified clean run artifacts. Historical model figures are not relabelled as clean results.

Opening or executing a notebook starts no training. Saved estimators may be used for prediction-only explanations. GPU work remains owned by its separate runners and deadlines.

| Clean stage | Original notebook(s) | Retained analytical depth and figures | Declared adaptation |
|---|---|---|---|
| [01_data_cleaning.ipynb](01_data_cleaning.ipynb) | [01_data_cleaning.ipynb](../Subset/01_data_cleaning.ipynb) | Schema, missingness, label/category/age checks, source agreement; exact-copy ledger and image-quality review | Development IDs only; read frozen cleaning artifacts instead of rewriting source; no test source loaded |
| [02_clean_data_eda.ipynb](02_clean_data_eda.ipynb) | [02_data_exploration.ipynb](../Subset/02_data_exploration.ipynb) | Image/patient class balance, images per patient, sex/age/site share and malignancy-rate panels; diagnoses | Regenerate all figures on retained records; patients counted within each category as in the original |
| [03_patient_fold_verification.ipynb](03_patient_fold_verification.ipynb) | [03_patient_splitting.ipynb](../Subset/03_patient_splitting.ipynb) | Fold support table and malignancy-rate figure; outer/inner separation; malignant-patient support | Verify existing patient folds instead of running a splitter; original test-ID check omitted under no-test-access instruction |
| [04_metadata_baselines.ipynb](04_metadata_baselines.ipynb) | [04_dummy_logistic_baseline.ipynb](../Subset/04_dummy_logistic_baseline.ipynb) | Dummy/categorical/spline protocols, fold and pooled AP/AUC, ROC/PR, confusion, cutoff diagnostics and coefficients | Mean-fold ranking primary; OOF-selected cutoff explicitly exploratory |
| [05_metadata_models.ipynb](05_metadata_models.ipynb) | [05_metadata_models.ipynb](../Subset/05_metadata_models.ipynb) | Five-model ranking/curves, fold variability, nonlinear age profiles, seven feature sets, coefficients/importances | Clean fit artifacts only; ablation/model expansion secondary |
| [06_metadata_neural_network.ipynb](06_metadata_neural_network.ipynb) | [06_metadata_neural_network.ipynb](../Subset/06_metadata_neural_network.ipynb) | Fixed NN architecture/preprocessing/stopping, histories/epochs, three-seed stability, equal mean, AP/AUC panels with primary mean-fold and secondary pooled views | Shared original/clean report template; verified existing clean fixed fits; no best-seed selection |
| [07_metadata_tuning.ipynb](07_metadata_tuning.ipynb) | [07_metadata_tuning.ipynb](../Subset/07_metadata_tuning.ipynb) | Four-family nested grids, 36 NN recipes, patient support, candidates/margins, selected settings and parameter frequencies, duration, blue/gold bars, four-row ROC/PR and per-fold gains | Reconstructed metadata-suite memberships, then exact clean filtering; inner AP selection separate from outer evaluation; additional development experiment |
| [08_metadata_robustness.ipynb](08_metadata_robustness.ipynb) | [08_metadata_robustness.ipynb](../Subset/08_metadata_robustness.ipynb) | Paired patient intervals, reliability/distributions, Brier/log loss, historical training-only decisions, subgroup support; new tuned/fixed NN and spline contrasts | 2,000 within-fold patient draws; historical spline calibration/cutoffs do not apply to tuned NN; no final fit |
| [09_image_modelling.ipynb](09_image_modelling.ipynb) | [09_image_b0_pilot.ipynb](../Subset/09_image_b0_pilot.ipynb)<br>[14_image_nested_evaluation.ipynb](../Subset/14_image_nested_evaluation.ipynb) | Image preparation/weak previews, recipes/nested protocol, histories/cost, OOF ranking/curves/confusion/reliability and calibration | Five clean outer folds replace the original one-split pilot as primary evidence; results pending until verified |
| [10_bounded_image_pilots.ipynb](10_bounded_image_pilots.ipynb) | [10_image_architecture_comparison.ipynb](../Subset/10_image_architecture_comparison.ipynb)<br>[11_image_augmentation.ipynb](../Subset/11_image_augmentation.ipynb)<br>[12_image_lighting.ipynb](../Subset/12_image_lighting.ipynb) | Three-inner-fold duration curves, 12/24 endpoint vs selected-epoch results, matched challenger curves/intervals/cost | Bounded B0 and ConvNeXt pilots only; DenseNet, stronger augmentation and CLAHE remain deferred, not losing conditions |
| [11_image_uncertainty.ipynb](11_image_uncertainty.ipynb) | [13_image_uncertainty.ipynb](../Subset/13_image_uncertainty.ipynb) | MC/TTA views, provenance/pass coverage, convergence, distributions/scatter/entropy, ranking/reliability/referral and intervals | Inference from matching five clean held-out checkpoints; frozen base scores for referral; full inference pending until ready |
| [12_combined_evaluation.ipynb](12_combined_evaluation.ipynb) | [08_metadata_robustness.ipynb](../Subset/08_metadata_robustness.ipynb)<br>[14_image_nested_evaluation.ipynb](../Subset/14_image_nested_evaluation.ipynb) | Completion, mean-fold/pooled ranking and curves, paired intervals, equal-patient weights, decisions/subgroups and conclusions; separate tuned-NN versus fixed NN and fixed/tuned spline appendix | Clean models are main comparison; historical image/fixed-metadata contrasts and duplicate-sensitivity appendix retained; new metadata experiment explicitly labelled; no reserved-test claim |

## Markdown coverage

Counts exclude code and rendered outputs. Combined stages sum the Markdown of their mapped original templates.

| Stage | Original template words | Adapted clean words |
|---|---:|---:|
| 01 | 692 | 1,638 |
| 02 | 1,160 | 1,576 |
| 03 | 837 | 1,750 |
| 04 | 1,062 | 1,344 |
| 05 | 2,231 | 2,592 |
| 06 | 1,215 | 1,360 |
| 07 | 1,247 | 2,210 |
| 08 | 1,678 | 2,518 |
| 09 | 2,407 | 3,021 |
| 10 | 4,107 | 5,205 |
| 11 | 2,031 | 2,394 |
| 12 | 2,751 | 4,557 |

The original Markdown is the template: preserve its heading hierarchy, teaching examples, reading instructions, notebook-specific limitations and conclusions. Adapt source paths, clean values and experiment scope explicitly. Source Markdown must retain at least 90% of the mapped originals' word count; the coverage receipt records both counts. This length check complements review of actual explanatory content and figure correspondence.

Original plot colours, markers, composition, labels and typography are retained where the analytical question is unchanged. Added clean diagnostics are separately labelled. A changed experiment or estimand is documented rather than silently substituted. Selection methods remain confined to training patients. Pending plots appear only when matching run receipts become complete.

All-development fitting, full OOF evaluation of image challengers, stronger augmentation, CLAHE, resolution changes, fusion and reserved-test evaluation require their separate later stage.
