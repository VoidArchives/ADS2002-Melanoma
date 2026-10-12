# Metadata modelling workflow

The objective is to predict whether **one lesion image is malignant** using age, sex and anatomical site, the information recorded alongside the image. This branch does not examine image pixels. A patient may contribute several images, so predictions concern images while splitting and uncertainty calculations keep patients together.

The completed workflow is:

1. Clean the records and explore what they contain.
2. Create shared patient groups for development evaluation.
3. Establish dummy and logistic baselines, then compare models and feature sets.
4. Test a small neural network and tuning alternatives.
5. Check sampling uncertainty, calibration, decision cutoffs and subgroups.
6. Save the retained model and cutoff for later final testing.

Modelling has used **Subset** development data. The [Full notebooks](../notebooks/README.md) cover preparation and exploration; they do not provide a second completed metadata-model evaluation. Below, findings come from saved notebook outputs, result tables and the frozen-model manifest. Illustrative examples explain the methods and are separate from those findings.

## 1. Clean records and understand the dataset

[Notebook 01](../notebooks/Subset/01_data_cleaning.ipynb) reads the source metadata and produces cleaned copies. It trims text, standardises missing values and checks expected labels. It preserves image rows rather than dropping inconvenient records. Missing anatomical site becomes `Unknown`, which preserves the fact that information was absent rather than inventing a site. Age remains in its recorded five-year buckets; later training pipelines decide how to encode it.

The cleaned development data contains **6,864 images from 422 patients**, including **128 malignant images (1.86%)**. There were 196 missing anatomical-site entries, all represented as `Unknown`. Targets are checked in their original form before conversion to integers. For example, `1.5` is invalid: converting it to `1` first would conceal the problem. Validation also rejects missing, non-numeric and infinite targets.

[Notebook 02](../notebooks/Subset/02_data_exploration.ipynb) produces descriptive tables and plots before fitting. Its saved checks show no duplicate image identifiers, exact duplicate rows, missing patient identifiers or target/benign–malignant disagreements. All 6,864 development records have corresponding JPEG filenames, with no extra JPEGs in that directory. A filename check does not establish that an image decodes correctly or has the right diagnosis.

The exploration identifies a crucial distinction: **1.86% of images are malignant, but 20.85% of patients have at least one malignant image**. The median patient contributes 12 images and the maximum is 115. These are different outcomes and denominators; a model predicting image malignancy should be assessed against the image outcome.

Read distribution plots for which ages, sites and sexes are represented, and the images-per-patient plot for unequal contributions. Larger observed malignant percentages describe this collection, not a causal effect. Patients can appear in several age or site groups: the saved exploration identifies multiple age buckets for 204 patients. The records do not establish why those ages differ, and category-level patient counts should not be summed as though the groups were disjoint.

These checks address detectable formatting and consistency problems. They cannot confirm source-record accuracy, explain missing-information bias, or rule out visually duplicated lesions stored under different identifiers. See [the validation methodology](metadata_methodology.md#1-check-the-values-before-changing-their-type).

## 2. Keep patients separate during development

Randomly dividing images would let one patient's images appear in both training and evaluation. That can make evaluation easier because the model has already seen information associated with that person. [Notebook 03](../notebooks/Subset/03_patient_splitting.ipynb) instead creates five **folds**, or groups of patients, using the same saved membership for later comparisons.

The split aims for similar malignant-image proportions while keeping every patient's images together. The saved folds contain 25 or 26 malignant images each, with malignant proportions of 1.82–1.90%. Checks found no unassigned rows or patients spanning folds. The saved preparation check also records no patient overlap between development and the reserved test; it does not report test performance.

**Outer evaluation** holds one fold aside and fits on the other four. Repeating this five times gives every development image an **out-of-fold (OOF) prediction** from a model that excluded its patient during fitting. Combining those predictions allows models to be compared on the same images.

**Inner validation** is a separate selection step inside the outer training portion. It chooses settings, training length or cutoffs without consulting the outer patients. The implementations use different inner arrangements for different tasks: tuning uses three newly grouped inner folds, neural-network stopping uses one of the four available folds, and the later spline cutoff procedure rotates those four folds.

**Illustrative example:** suppose fold 0 is the outer evaluation group. For neural-network stopping, fit on folds 2–4 and monitor fold 1 to choose the training length. Then start a fresh network, fit on folds 1–4 for that chosen length, and predict fold 0. Fold 0 influences neither stopping nor fitting.

Preprocessing follows the same boundary. A **pipeline** bundles data transformations with the model so their fitted values change whenever the training patients change. If illustrative training ages are 30, 40 and 50, the median used to fill a missing age is 40; an outer patient's age of 90 cannot alter it. Category encoders and scaling also learn only from training data. Image ID, patient ID, fold, diagnosis, benign/malignant description and target are excluded from predictor inputs.

Grouping protects the supplied patient identities. It cannot detect incorrect identities, related patients or near-duplicate images across identities. Reusing outer scores for broader model choices also introduces selection optimism; patient separation does not erase earlier exploration.

## 3. Establish baselines and read the metrics correctly

[Notebook 04](../notebooks/Subset/04_dummy_logistic_baseline.ipynb) compares a **prevalence dummy** with logistic regression. The dummy assigns every held-out image the training malignant proportion. It provides a reference for what happens without useful individual ranking. Logistic regression learns a weighted combination of encoded metadata and maps it to a score between zero and one. Here, the ordinary baseline treats age buckets as categories alongside sex and site.

**One-hot encoding** creates yes/no columns for categories, such as whether the site is torso. Encoders are fitted within training and tolerate previously unseen categories without learning from evaluation data. The saved ordinary-logistic AP is **0.0418**, versus **0.0184** for the dummy: metadata contains some ranking information.

Three distinctions help interpret the outputs:

- **Recall**, also called sensitivity, is the fraction of malignant images found. **Precision** is the fraction of flagged images that are malignant. A precision–recall plot puts recall horizontally and precision vertically; higher precision at the same recall is preferable. **Average precision (AP)** summarises this ranking trade-off. It is not the percentage of predictions that are correct.
- A **receiver operating characteristic (ROC)** curve compares malignant recall with the benign false-positive rate. Its **area under the curve (AUC)** summarises ranking; 0.5 indicates chance-level ordering. The dummy has AUC 0.5 within every fold. Its pooled AUC is 0.4952 because different training folds give slightly different constant scores.
- A **cutoff** turns a score into a flagged/unflagged decision. Ranking metrics examine many cutoffs; a confusion matrix describes one cutoff and its actual errors.

**Illustrative imbalance example:** in 1,000 images containing 20 malignant images, predicting every image benign achieves 98% accuracy while finding none of the malignant images. This is why accuracy alone is unhelpful here. In the saved baseline run, a cutoff of 0.50 leaves every image unflagged for both models.

Read confusion matrices by their axes: rows are actual outcomes and columns are predicted outcomes. False negatives are malignant images missed; false positives are benign images flagged. Row percentages answer questions about each actual class, whereas precision uses all flagged images as its denominator. The exploratory cutoff in notebook 04 demonstrates this trade-off, but was selected and assessed using the same OOF labels. It is not an independently evaluated policy.

## 4. Compare flexibility and identify useful features

[Notebook 05](../notebooks/Subset/05_metadata_models.ipynb) expands the comparison to five fixed methods. A **random forest** averages many decision trees; **XGBoost** builds trees sequentially to improve earlier predictions. They provide flexible alternatives to logistic regression. The spline version changes how age is represented while retaining a relatively small logistic model.

A **spline** joins smooth curve pieces, allowing age's relationship with the model score to bend. The fixed model uses a four-knot cubic spline: four age locations learned from training anchor the curve. Regularisation discourages unnecessarily large coefficients. Sex and site remain category indicators.

**Illustrative spline example:** ordinary category encoding treats ages 45 and 50 as separate buckets. A spline uses their numerical proximity to learn a smooth relationship, which may rise more steeply at older ages. It does not require risk to increase everywhere, and it does not establish that changing age causes a change in malignancy.

The saved pooled OOF comparison is:

| Fixed model | AP | ROC-AUC |
|---|---:|---:|
| Prevalence dummy | 0.0184 | 0.4952 |
| Ordinary logistic regression | 0.0418 | 0.5945 |
| Spline logistic regression | **0.0484** | **0.6499** |
| Random forest | 0.0443 | 0.6060 |
| XGBoost | 0.0439 | 0.5896 |

The spline ranks best on AP in this comparison. The notebook also performs **feature ablation**: repeat the same spline procedure with selected fields omitted to see what information contributes. Age alone gives AP **0.0410**, site alone **0.0264**, and sex alone **0.0196**. Age plus site reaches **0.0479**, close to all three fields at **0.0484**. Age supplies much of the available signal; these comparisons do not prove that the other fields have no value elsewhere.

In the ablation bar chart, farther right means higher AP, and the dashed dummy line provides context. Read the age plot differently: points show observed malignant proportions, with larger points representing more images. The smooth line comes from a model fitted to all development data, assigning the same hypothetical age to every record while retaining its other fields and averaging predictions. This is a descriptive model view, not additional held-out evaluation, a calibration test or a causal age effect. Sparse age groups make observed rates unstable.

## 5. Test a small neural network without choosing the luckiest run

[Notebook 06](../notebooks/Subset/06_metadata_neural_network.ipynb) asks whether a small network improves on the spline using the same three fields. Its numerical inputs pass through one hidden layer of 32 units, then a sigmoid maps the output to [0,1]. **Dropout** randomly masks 20% of hidden activations during training and is disabled for ordinary prediction. Weight decay discourages large weights. These help limit overfitting, learning details that fail on unseen patients.

Training length is chosen with the patient-separated stopping procedure described above. An **epoch** is a pass through the training records. The monitored binary cross-entropy measures prediction error; smaller is better. After selecting the best epoch count, preprocessing and a fresh network are fitted on all four outer training folds for that count. Outer labels never select the epoch.

The notebook prescribes seeds **42, 142 and 242**, changes the stopping fold between runs, and reports every run plus their equal prediction average. A seed makes random choices repeatable; averaging does not select whichever run scores highest. Saved outputs include complete OOF predictions, selected-epoch records, metric tables and a stability figure.

Individual AP values are **0.0458, 0.0486 and 0.0461**. Their average predictions give **AP 0.0465 and ROC-AUC 0.6723**, versus the spline's 0.0484 and 0.6499. Thus the network improves AUC but does not improve the main AP measure. The later paired comparison investigates whether these differences are resolved by the available patients.

Read seed plots as sensitivity to fitting choices, not three independent studies. The random seed and stopping partition change together, so their effects cannot be separated. In any learning curve, falling training loss alongside worsening validation loss suggests overfitting. Only one small architecture was tested; this is not evidence about every possible neural network. The [run configuration](../results/metrics/metadata_nn_run_config_d1.json) records the exact procedure.

## 6. Evaluate settings searches inside training

[Notebook 07](../notebooks/Subset/07_metadata_tuning.ipynb) tests whether changing spline and tree settings helps. **Hyperparameters** are choices made before fitting, such as tree depth, regularisation strength or spline complexity. The search scores candidate settings using three inner patient groups, selects them there, refits on the outer training patients, and evaluates the outer fold.

The [saved tuning table](../results/metrics/metadata_nested_tuning_metrics_d1.csv) shows spline AP rising from **0.0484 to 0.0521**, while its AUC falls from **0.6499 to 0.6440**. Random-forest AP falls from **0.0443 to 0.0373** and XGBoost AP from **0.0439 to 0.0383**. The searches include the fixed settings, so this illustrates that selecting the inner winner can still worsen outer performance.

Selected settings are recorded per outer fold. Read paired tuned/untuned bars within each model before comparing architectures, and use the curves to see where their trade-offs differ. The small spline AP gain varies across folds and has no paired uncertainty interval in this notebook. It remains a sensitivity check; the primary model stays the fixed spline. Nested selection protects each outer fold from its own settings search, but the outer folds have already informed broader development choices.

## 7. Quantify patient-sampling uncertainty

[Notebook 08](../notebooks/Subset/08_metadata_robustness.ipynb) combines saved network predictions with the fixed spline and adds stronger checks. Predictions are matched to the development records and their recorded fingerprints. Scores must be finite and between zero and one. The saved split audit records zero patient overlap in all 20 inner-split rows.

A **patient bootstrap** samples patients with replacement, including all images of every selected patient. It estimates how the reported metrics vary with patient sampling while keeping predictions fixed. The implementation uses **2,000 draws** and compares models on the same draws, making differences **paired**. The middle 95% of resampled metric values defines each reported interval.

**Illustrative paired example:** patients A, B and C have three, two and one images. Drawing A, A and C includes A's three images twice and C's image once: seven image rows from three patient selections. Both models are scored on exactly those rows, then their AP values are subtracted. Comparing this difference directly is more informative than treating two separate confidence intervals as independent evidence.

The [bootstrap table](../results/metrics/metadata_patient_bootstrap_d1.csv) gives spline AP **0.0484**, with a 95% interval **[0.0325, 0.0813]**. Network-minus-spline AP is **−0.0019**, with interval **[−0.0190, +0.0059]**. Because zero lies inside it, the current patients do not clearly resolve an AP advantage. The network's AUC difference is positive with an interval narrowly above zero; that is a different metric, not evidence that AP improved.

Read difference plots using the stated subtraction order. Positive AP/AUC differences favour the first model; negative loss differences favour it because lower error is better. These intervals describe sampling variation conditional on the saved predictions. They exclude retraining, seed variation, earlier model selection and a different hospital.

## 8. Separate calibration from ranking and decision selection

**Calibration** asks whether predicted scores agree with observed malignant proportions. For an illustrative group assigned about 3% risk, approximately 30 malignant images among 1,000 would indicate agreement. Good ranking alone does not guarantee this.

Notebook 08 compares the raw spline with a fixed logistic recalibration learned from **inner OOF scores and labels**. Those scores come from models that excluded their own patients during fitting. The recalibrator adjusts score level and spread without seeing outer labels. Both procedures are reported; the raw spline remains primary.

The reliability figure groups scores into five approximately equal-count bins. Horizontal position is mean predicted score and vertical position is observed malignant proportion. The diagonal indicates agreement: points below it suggest overprediction and points above it underprediction. Check image counts, patient counts and the vertical bootstrap intervals. A five-bin plot can hide differences within bins. Log loss and Brier score assess probability quality, with lower values better, but combine calibration and ranking effects.

Saved log loss changes from **0.089742 to 0.089545**, and Brier score from **0.018138 to 0.018121**. These small changes do not establish a decisive benefit. A sigmoid output or probability-shaped score should therefore not be treated automatically as a reliable clinical probability.

The later cutoff procedure also uses inner OOF scores. It chooses the highest cutoff attaining **80% inner recall**, refits the model on all four training folds, and applies that fixed cutoff to outer patients.

**Illustrative cutoff example:** five malignant training scores are 0.09, 0.07, 0.04, 0.02 and 0.01. A cutoff of 0.02 finds four of five, reaching 80%; every benign score at least 0.02 is also flagged. This explains why high recall can require many false positives. The 80% target is an academic scenario, not a clinical requirement or a guarantee for new patients.

The [saved outer decisions](../results/metrics/metadata_nested_threshold_metrics_d1.csv) find **102 of 128 malignant images (79.7%)**, miss 26, and flag **4,599 of 6,864 images (67.0%)**. Only **2.2% of flagged images are malignant**. Patient-bootstrap sensitivity intervals span about 71.3–87.9%. Raw and recalibrated procedures flag the same images here because the positive-slope calibration preserves score ordering within each fold and both use the same recall rule.

Earlier OOF-selected cutoffs in notebooks 04–05 remain historical exploratory demonstrations. Their labels were used both to choose and assess the cutoff; notebook 08 evaluates the stronger training-selected procedure. See [the methodology's cutoff and calibration sections](metadata_methodology.md#3-choose-a-cutoff-before-evaluating-its-performance).

## 9. Inspect subgroups, then freeze the retained model

Notebook 08 also reports fixed age bands, sex and anatomical-site groups using outer spline scores and the same training-selected cutoff policy. It does not tune separate subgroup cutoffs. The [subgroup table](../results/tables/metadata_subgroup_checks_d1.csv) records images, patients, malignant images and patients with malignancy alongside performance.

Sensitivity is **43.8% below age 50**, versus **91.7% at 50 or above**, with 23 and 65 patients contributing malignant images respectively. This is a material descriptive gap. Different case mix and prevalence affect comparisons; the table cannot establish its cause or a fairness conclusion.

Three of eleven groups have fewer than ten patients with a malignant image. Their intervals are suppressed and labelled sparse; palms/soles has no malignant images, so AP and sensitivity are undefined, rather than zero performance. In subgroup plots, compare AP positions together with counts and interval lengths. A large image count is not a large independent positive-patient sample, and age/site patient counts can overlap.

Finally, the fixed raw spline is fitted on all development patients. Its academic cutoff is selected from five-fold development OOF scores. The model bundle stores the fitted preprocessing, classifier, cutoff and training identities; the [manifest](../models/metadata_d1/frozen_spline.json) records input/model/code fingerprints and package versions. Its status is **development-only fitting; final test not evaluated**.

The prepared final-test script checks those artifacts, rejects development-patient or image overlap, uses the stored cutoff and records one attempted evaluation beside the bundle. Changing output folders cannot silently permit another attempt. This safeguards a future final evaluation; saving the bundle does not perform that evaluation.

## Main findings, limitations and next steps

The completed work establishes a modest metadata baseline: a smooth age model ranks better than the dummy, but a high-recall decision rule flags about two-thirds of images. Neither the prescribed network average nor the tree searches justify replacing the fixed spline on AP. Calibration and subgroup checks reveal why ranking scores alone are insufficient.

Remaining limitations include few malignant images, sparse subgroups, uncertain record quality, repeated development comparisons and no evidence from a new clinical setting. Bootstrap intervals do not resolve all those uncertainties. All reported performance is development evidence; the reserved test has not been evaluated.

Keep the frozen spline as the current primary metadata method. Image modelling is a separate completed branch; combining metadata and image scores (**fusion**) and final evaluation remain planned. Complete and freeze the remaining modelling decisions before using the reserved test once. The [detailed methodology](metadata_methodology.md) documents the implementation and safeguards; the [notebook index](../notebooks/README.md) provides the reading order.
