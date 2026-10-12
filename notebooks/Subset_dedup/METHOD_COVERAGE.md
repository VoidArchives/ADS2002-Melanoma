# Method and figure coverage

Every plot uses clean development records or verified clean run artifacts. Historical model figures are not relabelled as clean results.

Opening or executing a notebook starts no training or new image inference. Preparation 01–03 remains unchanged. Active 04–12 reports lead with mean-fold ROC-AUC, retain AP/error metrics as secondary and pooled results as supporting. Historical AP-era diagnostics are explicitly labelled; image selection objectives remain unchanged.

| Clean stage | Original notebook(s) | Retained analytical depth and figures | Declared adaptation |
|---|---|---|---|
| [01_data_cleaning.ipynb](01_data_cleaning.ipynb) | [01_data_cleaning.ipynb](../Subset/01_data_cleaning.ipynb) | Schema, missingness, label/category/age checks, source agreement; exact-copy ledger and image-quality review | Development IDs only; read frozen cleaning artifacts instead of rewriting source; no test source loaded |
| [02_clean_data_eda.ipynb](02_clean_data_eda.ipynb) | [02_data_exploration.ipynb](../Subset/02_data_exploration.ipynb) | Image/patient class balance, images per patient, sex/age/site share and malignancy-rate panels; diagnoses | Regenerate all figures on retained records; patients counted within each category as in the original |
| [03_patient_fold_verification.ipynb](03_patient_fold_verification.ipynb) | [03_patient_splitting.ipynb](../Subset/03_patient_splitting.ipynb) | Fold support table and malignancy-rate figure; outer/inner separation; malignant-patient support | Verify existing patient folds instead of running a splitter; original test-ID check omitted under no-test-access instruction |
| [04_metadata_baselines.ipynb](04_metadata_baselines.ipynb) | [04_dummy_logistic_baseline.ipynb](../Subset/04_dummy_logistic_baseline.ipynb) | Dummy/categorical protocols, primary mean-fold ROC-AUC, secondary AP/error metrics, supporting ROC/PR; historical confusion, cutoff diagnostics and coefficients | Verified versioned baseline inputs; historical OOF-selected cutoff explicitly exploratory; no fitting in report cells |
| [05_metadata_models.ipynb](05_metadata_models.ipynb) | [05_metadata_models.ipynb](../Subset/05_metadata_models.ipynb) | Five-model ROC-AUC-first ranking/curves, fold variability; historical nonlinear age profiles, seven feature sets, coefficients/importances | Current fixed-family probabilities are versioned; saved historical ablation/age outputs are labelled supporting and never recreated by all-development fitting |
| [06_metadata_neural_network.ipynb](06_metadata_neural_network.ipynb) | [06_metadata_neural_network.ipynb](../Subset/06_metadata_neural_network.ipynb) | Fixed NN architecture/preprocessing/stopping, histories/epochs, three-seed stability, equal mean, ROC-AUC/AP panels with primary mean-fold ROC-AUC and supporting pooled views | Shared pre-ROC original/clean report template; verified existing fixed fits; no best-seed selection |
| [07_metadata_tuning.ipynb](07_metadata_tuning.ipynb) | [07_metadata_tuning.ipynb](../Subset/07_metadata_tuning.ipynb) | Four-family nested grids, 36 NN recipes, patient support, candidates/margins, selected settings and parameter frequencies, duration, blue/gold bars, four-row ROC/PR and per-fold gains | Version-2 inner ROC-AUC selection separate from outer ROC-AUC evaluation; historical AP-selected version preserved; reuse, resume and rescoring counters distinguished |
| [08_metadata_robustness.ipynb](08_metadata_robustness.ipynb) | [08_metadata_robustness.ipynb](../Subset/08_metadata_robustness.ipynb) | Paired patient intervals, reliability/distributions, Brier/log loss, historical training-only decisions, subgroup support; new tuned/fixed NN and spline contrasts | 2,000 within-fold patient draws; historical spline calibration/cutoffs do not apply to tuned NN; no final fit |
| [09_image_modelling.ipynb](09_image_modelling.ipynb) | [09_image_b0_pilot.ipynb](../Subset/09_image_b0_pilot.ipynb)<br>[14_image_nested_evaluation.ipynb](../Subset/14_image_nested_evaluation.ipynb) | Image preparation/weak previews, recipes/nested protocol, histories/cost, ROC-first OOF ranking/curves/confusion/reliability and calibration | Five clean outer folds replace the original one-split pilot as primary evidence; historical AP-based image selection unchanged |
| [10_bounded_image_pilots.ipynb](10_bounded_image_pilots.ipynb) | [10_image_architecture_comparison.ipynb](../Subset/10_image_architecture_comparison.ipynb)<br>[11_image_augmentation.ipynb](../Subset/11_image_augmentation.ipynb)<br>[12_image_lighting.ipynb](../Subset/12_image_lighting.ipynb) | Three-inner-fold duration curves, 12/24 endpoint vs selected-epoch results, matched challenger curves/intervals/cost | Bounded B0 and ConvNeXt pilots only; DenseNet, stronger augmentation and CLAHE remain deferred, not losing conditions |
| [11_image_uncertainty.ipynb](11_image_uncertainty.ipynb) | [13_image_uncertainty.ipynb](../Subset/13_image_uncertainty.ipynb) | MC/TTA views, provenance/pass coverage, convergence, distributions/scatter/entropy, ROC-first ranking/reliability/referral and intervals | Read matching saved five-fold uncertainty artifacts; frozen base scores for referral; no new image inference during report execution |
| [12_combined_evaluation.ipynb](12_combined_evaluation.ipynb) | [08_metadata_robustness.ipynb](../Subset/08_metadata_robustness.ipynb)<br>[14_image_nested_evaluation.ipynb](../Subset/14_image_nested_evaluation.ipynb) | Completion, ROC-first mean-fold/pooled ranking and curves, paired intervals, equal-patient weights, decisions/subgroups and conclusions; separate v2 tuned-NN versus fixed NN and fixed/tuned spline appendix | Clean models are main comparison; historical image/fixed-metadata contrasts and duplicate-sensitivity appendix retained; current ROC-selected metadata and historical AP-selected image procedures distinguished; no reserved-test claim |

## Markdown coverage

Counts exclude code and rendered outputs. Active sources are bound to verified pre-ROC snapshots rather than the edited live notebooks. Mapped original source counts are retained in the receipt; metadata reports may preserve the longer preceding clean report as their immediate template.

| Stage | Required pre-ROC source words | Adapted clean words |
|---|---:|---:|
| 01 | 692 | 1,638 |
| 02 | 1,160 | 1,576 |
| 03 | 837 | 1,750 |
| 04 | 1,344 | 1,616 |
| 05 | 2,592 | 2,933 |
| 06 | 1,360 | 1,438 |
| 07 | 2,210 | 2,298 |
| 08 | 2,518 | 2,762 |
| 09 | 2,407 | 3,135 |
| 10 | 4,107 | 5,321 |
| 11 | 2,031 | 2,508 |
| 12 | 2,751 | 5,018 |

The original Markdown is the template: preserve its heading hierarchy, teaching examples, reading instructions, notebook-specific limitations and conclusions. Adapt source paths, clean values and experiment scope explicitly. Source Markdown must retain at least 90% of the mapped originals' word count; the coverage receipt records both counts. This length check complements review of actual explanatory content and figure correspondence.

Original plot colours, markers, composition, labels and typography are retained where the analytical question is unchanged. Added clean diagnostics are separately labelled. A changed experiment or estimand is documented rather than silently substituted. Selection methods remain confined to training patients. Pending plots appear only when matching run receipts become complete.

All-development fitting, full OOF evaluation of image challengers, stronger augmentation, CLAHE, resolution changes, fusion and reserved-test evaluation require their separate later stage.
