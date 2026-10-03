# Implementation audit — 3 October 2026

**Metadata follow-up:** The target-validation defect below has now been fixed in both cleaning and EDA workflows. New metadata validation adds patient-bootstrap intervals, training-only cutoff/calibration selection, three fixed NN repetitions, subgroup checks and frozen-model provenance. See the [methodology guide](../../docs/metadata_methodology.md) and [executed robustness notebook](../../notebooks/04b_metadata_robustness_d1.ipynb). The actual reserved test remains unevaluated. The findings below describe the original audit; image-related findings were outside this follow-up's scope.

The current metadata experiments use a defensible patient-grouped design and their fixed-model results reproduce. The project as a whole is not yet a complete, robust evaluation pipeline. Several input and artifact checks need strengthening, the image probabilities need calibration before risk interpretation, and fusion and final evaluation are unimplemented.

This review inspected all 16 notebooks, the Python modules, dependency/configuration files, available tests, and saved d1 model outputs. Checks ran in the existing `.venv`. Model source, datasets, checkpoints, and existing result files were not edited. Temporary audit scripts recomputed results in memory.

## Findings requiring changes

### 1. P1 — Checkpoint reuse does not establish data or training provenance

Location: `notebooks/05_image_model_d1_wip.ipynb`, cell 16 (`save_checkpoint` / `load_checkpoint`); `notebooks/05a_image_uncertainty_d1_wip.ipynb`, cell 10.

The loader validates model name, fold number, image size, epoch count, and seed. The checkpoint does not identify its training images/patients, input fold-file hash, learning rates, batch size, transforms, or code/environment version. Invoking the current loader with changed learning rates succeeds.

Consequently, a checkpoint can be reused after changing the training recipe or regenerating folds. In the latter case, some currently held-out patients could have belonged to that checkpoint's original training set even though the current fold file passes its overlap checks. This is a prospective failure mode; this audit did not find evidence that it occurred in the current fold-0 run.

Record a training configuration and dataset/fold fingerprint in each checkpoint and run manifest, and reject reuse on a mismatch. Record training and validation membership so patient separation can be independently checked. Use the same provenance checks in both image notebooks.

### 2. P2 — Invalid repeated-inference values can be hidden by aggregation

Location: `notebooks/05a_image_uncertainty_d1_wip.ipynb`, cell 14.

Checks validate the aggregated means and SDs, rather than every MC/TTA prediction before aggregation. Pandas ignores NaN when calculating means/SDs, while the `size` aggregation still counts that record as a pass. Replacing one saved MC prediction with NaN allowed the actual aggregation and validation code to complete successfully.

Check that every raw probability is finite and lies in [0, 1] before grouping. Also require unique `(image_id, mc_pass)` / `(image_id, augmentation_id)` keys and the complete expected pass identities per image. The current saved raw logs pass those stricter checks, and their statistics recompute correctly.

### 3. P2 — Binary targets are cast before their original values are validated

Location: both `00_data_cleaning_*.ipynb` notebooks, cell 8; the same cast-before-check pattern appears in the EDA notebooks.

`astype('int8')` can silently turn a fractional label into an integer. The current cleaning function converts target `1.5` into `1`, after which a binary-value check no longer detects the invalid source label.

Require non-missing numeric targets whose original values are exactly 0 or 1 before casting. This is a robustness defect for future inputs; no such invalid targets were found in the current d1 fold data.

### 4. P2 — Reusing image checkpoints overwrites the training history with an empty table

Location: `notebooks/05_image_model_d1_wip.ipynb`, cells 21 and 23.

With `TRAIN_MODELS=False`, no training histories are appended. The notebook builds an empty history table and then unconditionally writes it to the existing training-history path. A normal prediction-only rerun would therefore remove the previous loss/time records. The current saved histories have not been modified by this audit.

Write training history only after training, or load and preserve the existing history when reusing checkpoints.

### 5. P2 — The declared installation does not reproduce the working environment

Location: `requirements.txt`, `pyproject.toml`, and notebook setup cells.

The requirements omit directly used neural-network packages such as PyTorch, torchvision, Pillow, and tqdm. The package metadata does not declare dependencies, and installing requirements alone does not install the repository package required by the tests and inventory notebook. Versions are unpinned. Several notebooks also contain absolute paths tied to this user's checkout.

Document an installation path including the repository package and a compatible torch/torchvision pair, record tested versions, and use project-root discovery consistently. A functioning existing `.venv` is not evidence that a fresh installation will reproduce the workflow. This audit did not install into a fresh environment.

### 6. P2 — The metadata download helper treats a CSV as a ZIP archive

Location: `src/melanoma_detection/data.py`, `load_train_metadata`.

The helper requests `train.csv` and then forces `compression='zip'`. Pointing the helper at the project's actual uncompressed `data/raw/train.csv` reproduces `BadZipFile: File is not a zip file`.

Use inferred compression for the returned file, or explicitly select a file from an archive when the returned path really is a ZIP. This helper is not used by the active modelling notebooks, so the defect does not alter their saved results.

### 7. P2 — The classifier-only warm-up also updates the backbone state

Location: `notebooks/05_image_model_d1_wip.ipynb`, `build_model` and `train_fixed_recipe`.

Setting `requires_grad=False` freezes the feature parameters, but the subsequent `model.train()` leaves backbone BatchNorm and stochastic-depth modules in training mode. A forward pass with frozen feature parameters changed running means in 49 BatchNorm layers. Thus the documented first epoch is not a fully frozen-backbone warm-up.

If full freezing is intended, keep `model.features.eval()` during that epoch while training the classifier, then restore feature training mode when unfreezing. Otherwise explicitly document that BatchNorm state adapts during warm-up. This is not patient leakage: the updates occur on training images.

## Methodological limitations

### Raw image scores are poorly calibrated

The image model trains with a positive-class weight of 52.3398. Its raw sigmoid scores on fold 0 average **0.4233**, while the observed malignant proportion is **0.01825**. The Brier score is **0.22936**, versus **0.01792** for an illustrative constant prediction equal to the evaluation-fold prevalence. That constant is a descriptive reference, not a fitted deployment model.

Class weighting changes the loss being optimised, so these raw scores cannot be assumed to estimate the natural melanoma probability. Their ranking metrics can still be useful. The notebook correctly plots descriptive calibration and defers fitting a calibrator, but downstream fusion, probability-error interpretation, and referral decisions must account for this limitation. Calibration must use appropriate patient-disjoint development data; its final assessment must not reuse the calibration labels.

### Uncertainty benefit has not been established

MC dropout is implemented with the trained classifier dropout active and BatchNorm frozen. TTA is kept separate, and both rankings are compared using the same MC-mean prediction. Those are sound implementation choices.

The fold-0 risk-coverage comparison is descriptive. It lacks random-referral and ordinary prediction-score/entropy comparators and patient-level uncertainty intervals. In a dataset dominated by benign images, absolute error `abs(y - p)` equals the raw prediction for nearly every row. An association between dropout SD and this error can partly reflect association with the prediction score itself. Compare against those simple referral rankings and use patient-bootstrap uncertainty before claiming that the extra uncertainty calculation adds value. No clinical referral policy is validated by the present prototype.

### Development comparisons remain exploratory

The neural network is one prespecified architecture with one seed per fold. It selects its epoch within outer-training patients, refits on all outer-training patients, and scores only the untouched outer fold. The spline/tree tuning grids likewise stay inside patient-grouped inner folds. No direct target or patient leakage was found in these fitted metadata pipelines.

Repeated use of the outer OOF results for model-family selection, feature ablation, and threshold inspection still makes them development evidence. Small AP differences are not established improvements without patient-aware uncertainty estimates and a frozen final evaluation. The existing threshold explanations correctly acknowledge that choosing and evaluating a threshold on the same OOF labels is descriptive.

### The full pipeline is incomplete

`05_image_model_d1.ipynb`, `06_fusion_model_d1.ipynb`, and `07_final_evaluation_d1.ipynb` are placeholders. The implemented image WIP has saved results and a checkpoint for fold 0 only: 1,370/6,864 development images. Reusable preprocessing/image/fusion/evaluation modules and the YAML configurations are also empty placeholders. They should not be represented as implemented or tested components.

## Verification completed

- All **10 existing unit/integration tests passed**, including the real d1 patient-split integration test. The imported module path points to this checkout.
- All **16 notebooks** passed current format validation and every code cell parsed. Several older notebooks have missing cell IDs and trigger a format warning; add IDs for future notebook-format compatibility.
- d1 has **6,864 images, 422 patients, 128 malignant images, and 88 patients with a malignant image**. Patient and image identifiers do not overlap with the reserved test split. Test data were inspected only for identifiers and image-file integrity, not target performance.
- Regenerating the folds with the current splitter reproduces the saved fold labels. Every outer fold keeps patients together. All **15 inner tuning splits** have disjoint training/validation patients and both classes.
- The fixed dummy, logistic, spline, random-forest, and XGBoost predictions were recomputed from the active notebook code. Their metrics match the saved values exactly.
- The metadata neural-network code was rerun end to end in memory. It reproduces **AP 0.0457505 / ROC-AUC 0.6627125** and epoch selections **20, 38, 77, 11, 37**. The shared spline reference reproduces **AP 0.0483686 / ROC-AUC 0.6498641**.
- Both EDA notebook computations ran successfully in memory. The d2 inventory check found exactly **33,126 CSV identifiers, JPEG files, and DICOM files**, with a strict one-to-one match.
- Exact byte hashes were checked for **8,852 d1 train/test JPEGs**. No file was missing, and no exact byte-duplicate group crosses training folds or the train/test boundary. There are 160 duplicate groups contained within a single split/fold. This does not rule out near duplicates or differently encoded copies of the same lesion.
- The **41,100 MC predictions** and **10,960 TTA predictions** have valid raw probabilities and unique complete pass keys. Recomputed means/SDs agree with the summary within **1.36e-16**. All saved uncertainty prediction metrics also recompute.
- A fresh GPU forward pass for **128 held-out images** agrees with saved deterministic scores within **9.1e-17**. MC dropout varies the predictions while preserving BatchNorm buffers.
- Targeted reproductions confirmed the checkpoint-configuration, NaN-aggregation, fractional-target, CSV-loader, and warm-up BatchNorm issues above.

The audit did not retrain the image model, rerun the entire nested grid search, inspect every rendered notebook figure, verify near-duplicate lesions, or evaluate the reserved test labels. It validates the inspected code paths and current artifacts, not an unimplemented final system.

## Suggested order of work

1. Add checkpoint provenance and stricter raw-prediction/target guards; preserve training histories on reuse.
2. Fix the loader, declare reproducible dependencies, and clarify or fix the frozen-backbone warm-up.
3. Freeze the image recipe and generate all five outer folds; regenerate uncertainty from those matching checkpoints.
4. Implement patient-safe fusion, calibration, and any referral-policy selection, with patient-bootstrap comparisons.
5. Freeze the complete specification, then implement and run the reserved final evaluation once.
