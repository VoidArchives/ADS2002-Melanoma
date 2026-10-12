# Metadata modelling: purpose, evidence and interpretation

The aim is to predict whether **one lesion image is malignant** from age, sex and anatomical site. A patient may contribute several images. We therefore predict and score individual images, but keep each patient's images together whenever we split or resample data.

For a first read, follow [Subset notebooks 01–04](../notebooks/README.md): prepare
records, explore them, separate patients and establish a simple baseline. Notebooks
05–07 compare model choices. [Notebook 08, metadata robustness](../notebooks/Subset/08_metadata_robustness.ipynb)
then asks whether the apparent differences and decision rules hold up to stronger
checks. The [concept guide](../notebooks/concepts.md) defines shared terms.

These models read patient-grouped Subset development records and produce
per-image predictions, comparison tables and figures. Full data has preparation
and exploration notebooks, without new model evaluation. Dataset folder names are
now `Subset` and `Full`; recorded data/model/result identifiers such as
`models/metadata_d1` stay unchanged. The updated reports retain **05 models → 06
fixed NN → 07 tuning including NN → 08 robustness** in both original and clean
workflows. To create the new versioned prediction files, use the explicit runner
described in [ROC-AUC-first metadata tuning](metadata_roc_auc_workflow.md). Executing these updated
reports reads completed artifacts; it starts no training or final fit.

## Read the figures and tables

| Output | How to read it | Why it matters |
|---|---|---|
| Precision–recall curve / AP | Recall is the fraction of malignant images found; precision is the fraction of flagged images that are malignant. Higher precision at the same recall is useful; AP summarises the curve. Compare against the malignant prevalence. | Malignant images are rare, so accuracy alone can hide missed cases. |
| ROC curve / ROC-AUC | The horizontal axis is the benign false-positive rate; the vertical axis is malignant recall. Curves nearer the top-left rank better. | This is a ranking view; it does not tell us the number of false referrals at a chosen cutoff. |
| Confusion matrix and cutoff table | Read actual versus predicted labels from the axes. Missed malignant images are false negatives; benign flags are false positives. Compare recall with precision and the flagged fraction. | A high recall can require reviewing many benign images. |
| Calibration / reliability plot | Mean predicted score is horizontal; observed malignant fraction is vertical. The diagonal indicates agreement. Below it suggests overprediction, above it underprediction. Read bin counts and patient-bootstrap intervals too. | A useful ranking need not provide accurate risk estimates; small bins are uncertain. |
| Bootstrap difference plot | Read the stated model order in the subtraction. For AP/AUC, positive means the first model ranks better; for losses, negative means lower error. A 95% interval crossing zero does not clearly resolve the difference. | Paired resampling keeps the same patients in each model comparison. |
| NN learning curves / seed comparison | Epoch counts training passes; loss measures prediction error. Improving training loss with worsening validation loss suggests overfitting. Report all fixed seeds, without choosing the luckiest run. | Random fitting and the stopping split can change apparent performance. |
| Subgroup table / figure | Compare rates only alongside patient counts, malignant-patient counts and intervals. Missing AP or recall can mean no malignant cases; it is not zero performance. | Sparse groups and different case mix limit conclusions about fairness or clinical risk. |

The sections below explain how these outputs are computed. Development labels
have already informed model exploration; the reported estimates are development
evidence, rather than a final performance claim.

The numerical examples in sections 3–7 describe the archived original metadata
evaluation. Its AP examples are pooled secondary scores, not the primary
mean-fold ROC-AUC from the new original and deduplicated searches. The [completed AP-selected NN comparison](metadata_nn_tuning.md#completed-experiment-nn_grid36_v1) remains historical. Read the [ROC-AUC-first workflow](metadata_roc_auc_workflow.md) and active stages07/08 for current versioned findings and paired intervals.

## 1. Check the values before changing their type

The target must be exactly 0 or 1. Previously, converting a target such as 1.5 to an integer could turn it into 1 before validation, hiding the error.

The cleaning and EDA notebooks now read targets as text, validate their original values, and only then convert them to integers. Reading as text also protects values such as `1.000000000000000001`, which normal CSV number conversion could round to 1. Missing, infinite, fractional and non-numeric labels are rejected. The shared modelling validator additionally rejects duplicate images, invalid ages and patients appearing in multiple folds.

These checks stop bad inputs; they do not invent replacement labels or change valid data. Existing processed training CSVs were not rewritten.

## 2. Keep the metadata model small and fixed

The existing final-method choice remains spline logistic regression until an explicit model-choice decision. The new development comparison ranks procedures primarily by mean outer-fold ROC-AUC:

- Age passes through a median imputer, a four-knot cubic spline and scaling. The spline lets the relationship between age and risk bend smoothly.
- Sex and site become indicator columns, including a missing category where needed. An unseen category at prediction time is handled without refitting the encoder.
- Logistic regression combines these inputs into a score on a probability scale, using the existing regularisation setting C=1. Regularisation limits large coefficients to reduce overfitting; the score still needs calibration checks.

All of these steps are fitted inside each training pipeline. A held-out patient's age cannot influence the training median, age curve or scaling. Patient ID, image ID, fold, diagnosis and target never enter the predictor list.

The small neural network (NN) remains an alternative: encoded metadata → 32
hidden units → 20% training dropout → one sigmoid score between 0 and 1. ReLU
allows the hidden layer to learn nonlinear relationships; dropout randomly
omits units during fitting to limit reliance on any one unit. A sigmoid output
does not by itself make the score a calibrated probability. Stage 06 retains this
fixed comparison. Stage 07 additionally searches 36 combinations of width,
learning rate and dropout using inner patient groups. It compares the resulting
selection procedure on excluded outer patients, with mean-fold ROC-AUC primary and
paired patient-bootstrap intervals. This added development experiment supplies
evidence for model choice; it does not automatically replace the spline.

## 3. Choose a cutoff before evaluating its performance

A probability needs a cutoff if we want a flagged/unflagged decision. Finding the most favourable cutoff from the same labels used to report its performance gives an optimistic estimate.

The new procedure has two levels:

1. Hold out one of the five patient folds for evaluation.
2. Inside the four remaining folds, fit on three and predict the fourth, four times. Each training patient receives an **out-of-fold (OOF)** score from a model that did not see them.
3. Choose a cutoff from those inner scores.
4. Fit on all four training folds, predict the outer patients and apply that already chosen cutoff.

Repeat this for every outer fold. The combined outer decisions evaluate the whole cutoff-selection procedure. Outer labels are never used to choose the cutoff.

For a clear academic example, we request 80% recall on the inner scores. Suppose five malignant training cases have scores 0.09, 0.07, 0.04, 0.02 and 0.01. A cutoff of 0.02 captures four of five, reaching 80%. Benign cases scoring at least 0.02 are also flagged. We choose the highest cutoff meeting the requested recall, keeping tied scores together.

The 80% request is a demonstration, not a clinical requirement. It does not guarantee 80% recall on new patients. In the historical original outer evaluation, recall was **79.7%**, but **67.0%** of images were flagged and precision was only **2.2%**. That large false-positive burden is an important limitation of metadata alone.

The older OOF-selected cutoffs in notebooks 04 and 05 remain exploratory examples. Use notebook 08 for the evaluation of a training-selected cutoff policy.

## 4. Check whether probabilities match observed risk

Calibration asks whether cases assigned, for example, about 3% risk actually include about 3% malignant lesions. A model can rank cases reasonably while estimating their probabilities poorly.

We compare two fixed procedures: the raw spline and a logistic recalibration. The recalibrator learns two coefficients from inner OOF scores and their labels. It adjusts the score's level and spread without seeing outer labels. Its regularisation is fixed at C=1; there is no search for the most favourable calibration method.

The notebook plots five score bins, showing average predicted risk against observed malignancy, with patient-bootstrap intervals. It also reports log loss and Brier score. Those two losses assess probability quality but do not isolate calibration from discrimination. See the [scikit-learn calibration explanation](https://scikit-learn.org/stable/modules/calibration.html).

The recalibration changes probabilities while preserving ordering within each fold. Consequently, the recall-selected raw and recalibrated cutoffs flag the same cases in the historical original run. Their pooled AP can differ slightly because each fold learns a different recalibration. Historical original log loss changes from **0.089742 to 0.089545**, and Brier score from **0.018138 to 0.018121**; these small development changes are not decisive evidence to replace the raw baseline. These maps and cutoffs do not evaluate the new tuned NN.

## 5. Show uncertainty in scores and model differences

The historical original robustness analysis used **2,000 whole-patient bootstrap draws**. Each draw samples the original number of patients with replacement and includes every image of each selected patient. The current NN tuning assessment additionally preserves outer-fold membership by resampling patients within each fold.

For example, if A has three images, B has two and C has one, drawing A, A and C includes all three A images twice and C's image once. The draw contains seven images from three patient selections. This keeps the dependence between images from the same person.

Every model is assessed on the same draw. We then subtract the spline score from the NN score. This paired difference directly measures uncertainty in their comparison.

In the historical original pooled comparison, spline AP is **0.04837**, with a percentile 95% interval of **0.03252–0.08134**. The mean-of-three NN AP is **0.04647**. Its AP difference from the spline is **−0.00190**, with interval **−0.01905 to +0.00589**. Because the difference interval includes zero, those historical predictions do not give clear evidence that the fixed NN improves pooled AP.

These intervals hold predictions fixed. They do not retrain models, include random-seed variation, correct previous model selection, or prove performance in a different hospital.

## 6. Repeat NN training without choosing the luckiest run

Notebook 06 uses three fixed seeds: 42, 142 and 242. It also rotates the patient fold used for epoch stopping. Each run keeps the outer patients outside preprocessing, stopping and fitting. After selecting the epoch count, the network is reinitialised and fitted on all outer training patients for that count.

All runs are reported. Their historical original pooled AP values are **0.04575, 0.04865 and 0.04613**. The current versioned notebooks display both primary mean-fold and secondary pooled metrics for their own condition. The mean prediction gives every run equal weight; it does not select the best seed. This spread measures the combined effect of random training and stopping-partition choice. It is not a confidence interval or three independent studies.

## 7. Look for weak or unsupported subgroups

We report fixed age bands (<50 and ≥50), sex and anatomical site. Each group uses the held-out spline scores and the same training-selected cutoff policy; there is no subgroup-specific tuning.

Tables include images, patients, malignant images and patients with malignancy. A large number of images does not imply many independent positive patients. Patients can contribute to multiple age/site groups when per-image metadata differs, so those patient counts must not be added across groups.

Groups with at least ten positive patients receive exploratory patient-bootstrap intervals. Below that count, intervals are suppressed and the evidence is marked sparse. Palms/soles has no malignant images in Subset development data, so its AP and sensitivity are undefined, rather than zero.

The historical original held-out sensitivity is **43.8% below age 50** versus **91.7% at age 50 or above** under this policy. This is a material descriptive gap. Case mix, prevalence and limited samples affect comparisons; the result does not establish a causal explanation or fairness conclusion.

## 8. Freeze decisions before the final test

The archived original Notebook 08 fitted the fixed **raw spline** on all original development patients and stored its weights and training-OOF-selected cutoff in `models/metadata_d1/frozen_spline.joblib`. Its manifest records the specification, training identities, input and code fingerprints, Python version and package versions. That historical bundle remains preserved. The current original and deduplicated Notebook 08 reports read saved evidence and fit no final model. Final all-development fitting for the current workflow remains deferred until the complete procedure is settled. Model files are locally generated and excluded from Git by the repository's existing rules.

The historical prepared final-test command is retained for reference:

```powershell
.venv/Scripts/python.exe scripts/evaluate_metadata_holdout.py --bundle models/metadata_d1/frozen_spline.joblib --data PATH_TO_CLEANED_HELDOUT_CSV --output-dir results/metadata_final_holdout
```

Any future final evaluation requires settled modelling decisions and the matching frozen bundle. The script checks the manifest, rejects development-patient/image overlap, uses the stored cutoff and records one attempted evaluation beside the bundle. An existing output directory or recorded attempt stops another run. Failed attempts remain recorded for investigation; the script does not silently retry. The archived robustness fitting code also refuses to replace a bundle after a recorded holdout attempt.

**The actual reserved test was not evaluated.** Synthetic-data tests exercised the final-test workflow. Repeated development-set exploration cannot be undone by new code; the single frozen final evaluation remains necessary before making a final generalisation claim.

## Files and verification

- [notebooks/Subset/06_metadata_neural_network.ipynb](../notebooks/Subset/06_metadata_neural_network.ipynb): three NN repetitions, explanations and saved OOF predictions.
- [notebooks/Subset/07_metadata_tuning.ipynb](../notebooks/Subset/07_metadata_tuning.ipynb): completed classical and 36-setting NN tuning comparisons from verified versioned outputs.
- [notebooks/Subset/08_metadata_robustness.ipynb](../notebooks/Subset/08_metadata_robustness.ipynb): saved cutoff/calibration evaluation, patient intervals, subgroup checks and separate tuned-NN diagnostics; report execution fits no model.
- `src/melanoma_detection/metadata_validation.py`: original-value label and patient-fold checks.
- `src/melanoma_detection/metadata_evaluation.py`: metrics, thresholds, paired bootstrap and calibration bins.
- `src/melanoma_detection/metadata_model.py`: the fixed spline and training-only decision procedures.
- `scripts/evaluate_metadata_holdout.py`: the later frozen final evaluation.
- `requirements-metadata.txt`: the recorded dependencies for notebooks 06 and 08. The executed Python version is 3.14.6; NumPy requires at least Python 3.12.

Both modelling notebooks executed fully and their charts were inspected. Both EDA notebooks also executed fully with unchanged charts. The cleaning functions were tested directly and reproduced valid training outputs; their full file-writing notebooks were not rerun. At the original metadata evaluation step, **46 tests passed**, including checks that changing outer labels cannot change model predictions, calibration or cutoffs, that paired patient bootstrap matches independently resampled rows, and that the final-test script rejects altered artifacts and repeated attempts.
