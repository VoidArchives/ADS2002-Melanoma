# How the metadata evaluation now works

The aim is to predict whether **one lesion image is malignant** from age, sex and anatomical site. A patient may contribute several images. We therefore predict and score individual images, but keep each patient's images together whenever we split or resample data.

Start with `notebooks/d1/08_metadata_robustness.ipynb`. It contains the explanations, calculations, tables and figures. Run `d1/06_metadata_neural_network.ipynb` first if its prediction files need refreshing. These experiments use d1 development data only; d2 received the same label-validation fix, without new modelling.

## 1. Check the values before changing their type

The target must be exactly 0 or 1. Previously, converting a target such as 1.5 to an integer could turn it into 1 before validation, hiding the error.

The cleaning and EDA notebooks now read targets as text, validate their original values, and only then convert them to integers. Reading as text also protects values such as `1.000000000000000001`, which normal CSV number conversion could round to 1. Missing, infinite, fractional and non-numeric labels are rejected. The shared modelling validator additionally rejects duplicate images, invalid ages and patients appearing in multiple folds.

These checks stop bad inputs; they do not invent replacement labels or change valid data. Existing processed training CSVs were not rewritten.

## 2. Keep the metadata model small and fixed

The primary model remains the existing spline logistic regression:

- Age passes through a median imputer, a four-knot cubic spline and scaling. The spline lets the relationship between age and risk bend smoothly.
- Sex and site become indicator columns, including a missing category where needed. An unseen category at prediction time is handled without refitting the encoder.
- Logistic regression combines these inputs into a malignancy probability, using the existing regularisation setting C=1.

All of these steps are fitted inside each training pipeline. A held-out patient's age cannot influence the training median, age curve or scaling. Patient ID, image ID, fold, diagnosis and target never enter the predictor list.

The small NN remains an alternative: encoded metadata → 32 hidden units with ReLU → 20% training dropout → one output score → sigmoid probability. We did not expand the architecture search, because better evaluation was the main missing evidence.

## 3. Choose a cutoff before evaluating its performance

A probability needs a cutoff if we want a flagged/unflagged decision. Finding the most favourable cutoff from the same labels used to report its performance gives an optimistic estimate.

The new procedure has two levels:

1. Hold out one of the five patient folds for evaluation.
2. Inside the four remaining folds, fit on three and predict the fourth, four times. Each training patient receives an **out-of-fold (OOF)** score from a model that did not see them.
3. Choose a cutoff from those inner scores.
4. Fit on all four training folds, predict the outer patients and apply that already chosen cutoff.

Repeat this for every outer fold. The combined outer decisions evaluate the whole cutoff-selection procedure. Outer labels are never used to choose the cutoff.

For a clear academic example, we request 80% recall on the inner scores. Suppose five malignant training cases have scores 0.09, 0.07, 0.04, 0.02 and 0.01. A cutoff of 0.02 captures four of five, reaching 80%. Benign cases scoring at least 0.02 are also flagged. We choose the highest cutoff meeting the requested recall, keeping tied scores together.

The 80% request is a demonstration, not a clinical requirement. It does not guarantee 80% recall on new patients. In the executed outer evaluation, recall was **79.7%**, but **67.0%** of images were flagged and precision was only **2.2%**. That large false-positive burden is an important limitation of metadata alone.

The older OOF-selected cutoffs in notebooks 03 and 03a remain exploratory examples. Use notebook 04b for the evaluation of a training-selected cutoff policy.

## 4. Check whether probabilities match observed risk

Calibration asks whether cases assigned, for example, about 3% risk actually include about 3% malignant lesions. A model can rank cases reasonably while estimating their probabilities poorly.

We compare two fixed procedures: the raw spline and a logistic recalibration. The recalibrator learns two coefficients from inner OOF scores and their labels. It adjusts the score's level and spread without seeing outer labels. Its regularisation is fixed at C=1; there is no search for the most favourable calibration method.

The notebook plots five score bins, showing average predicted risk against observed malignancy, with patient-bootstrap intervals. It also reports log loss and Brier score. Those two losses assess probability quality but do not isolate calibration from discrimination. See the [scikit-learn calibration explanation](https://scikit-learn.org/stable/modules/calibration.html).

The recalibration changes probabilities while preserving ordering within each fold. Consequently, the recall-selected raw and recalibrated cutoffs flag the same cases in this run. Their pooled AP can differ slightly because each fold learns a different recalibration. Log loss changes from **0.089742 to 0.089545**, and Brier score from **0.018138 to 0.018121**; these small development changes are not decisive evidence to replace the raw baseline.

## 5. Show uncertainty in scores and model differences

We use **2,000 whole-patient bootstrap draws**. Each draw samples the original number of patients with replacement and includes every image of each selected patient.

For example, if A has three images, B has two and C has one, drawing A, A and C includes all three A images twice and C's image once. The draw contains seven images from three patient selections. This keeps the dependence between images from the same person.

Every model is assessed on the same draw. We then subtract the spline score from the NN score. This paired difference directly measures uncertainty in their comparison.

The spline AP is **0.04837**, with a percentile 95% interval of **0.03252–0.08134**. The mean-of-three NN AP is **0.04647**. Its AP difference from the spline is **−0.00190**, with interval **−0.01905 to +0.00589**. Because the difference interval includes zero, the current predictions do not give clear evidence that the NN improves AP.

These intervals hold predictions fixed. They do not retrain models, include random-seed variation, correct previous model selection, or prove performance in a different hospital.

## 6. Repeat NN training without choosing the luckiest run

Notebook 03b now uses three fixed seeds: 42, 142 and 242. It also rotates the patient fold used for epoch stopping. Each run keeps the outer patients outside preprocessing, stopping and fitting. After selecting the epoch count, the network is reinitialised and fitted on all outer training patients for that count.

All runs are reported. Their AP values are **0.04575, 0.04865 and 0.04613**. The mean prediction gives every run equal weight; it does not select the best seed. This spread measures the combined effect of random training and stopping-partition choice. It is not a confidence interval or three independent studies.

## 7. Look for weak or unsupported subgroups

We report fixed age bands (<50 and ≥50), sex and anatomical site. Each group uses the held-out spline scores and the same training-selected cutoff policy; there is no subgroup-specific tuning.

Tables include images, patients, malignant images and patients with malignancy. A large number of images does not imply many independent positive patients. Patients can contribute to multiple age/site groups when per-image metadata differs, so those patient counts must not be added across groups.

Groups with at least ten positive patients receive exploratory patient-bootstrap intervals. Below that count, intervals are suppressed and the evidence is marked sparse. Palms/soles has no malignant images in d1 development data, so its AP and sensitivity are undefined, rather than zero.

The held-out sensitivity is **43.8% below age 50** versus **91.7% at age 50 or above** under this policy. This is a material descriptive gap. Case mix, prevalence and limited samples affect comparisons; the result does not establish a causal explanation or fairness conclusion.

## 8. Freeze decisions before the final test

Notebook 04b fits the fixed **raw spline** on all development patients and stores its weights and training-OOF-selected cutoff in `models/metadata_d1/frozen_spline.joblib`. Its manifest records the specification, training identities, input and code fingerprints, Python version and package versions. Model files are locally generated and excluded from Git by the repository's existing rules.

The prepared final-test command is:

```powershell
.venv/Scripts/python.exe scripts/evaluate_metadata_holdout.py --bundle models/metadata_d1/frozen_spline.joblib --data PATH_TO_CLEANED_HELDOUT_CSV --output-dir results/metadata_final_holdout
```

Run it only after all modelling decisions are settled. It checks the manifest, rejects development-patient/image overlap, uses the stored cutoff and records one attempted evaluation beside the bundle. An existing output directory or recorded attempt stops another run. Failed attempts remain recorded for investigation; the script does not silently retry. The robustness notebook also refuses to replace a bundle after a recorded holdout attempt.

**The actual reserved test was not evaluated.** Synthetic-data tests exercised the final-test workflow. Repeated development-set exploration cannot be undone by new code; the single frozen final evaluation remains necessary before making a final generalisation claim.

## Files and verification

- `d1/06_metadata_neural_network.ipynb`: three NN repetitions, explanations and saved OOF predictions.
- `d1/08_metadata_robustness.ipynb`: cutoff/calibration evaluation, patient intervals, subgroup checks and frozen model fitting.
- `src/melanoma_detection/metadata_validation.py`: original-value label and patient-fold checks.
- `src/melanoma_detection/metadata_evaluation.py`: metrics, thresholds, paired bootstrap and calibration bins.
- `src/melanoma_detection/metadata_model.py`: the fixed spline and training-only decision procedures.
- `scripts/evaluate_metadata_holdout.py`: the later frozen final evaluation.
- `requirements-metadata.txt`: the recorded dependencies for notebooks 03b and 04b. The executed Python version is 3.14.6; NumPy requires at least Python 3.12.

Both modelling notebooks executed fully and their charts were inspected. Both EDA notebooks also executed fully with unchanged charts. The cleaning functions were tested directly and reproduced valid training outputs; their full file-writing notebooks were not rerun. **46 tests passed**, including checks that changing outer labels cannot change model predictions, calibration or cutoffs, that paired patient bootstrap matches independently resampled rows, and that the final-test script rejects altered artifacts and repeated attempts.
