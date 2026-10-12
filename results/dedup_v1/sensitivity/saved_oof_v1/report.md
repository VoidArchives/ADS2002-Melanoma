# Exact-copy retention sensitivity

Filtered historical predictions check duplicate influence on evaluation weighting only. Duplicates still influenced historical training, preprocessing, recipe/epoch selection and calibration fitting. Models, maps and cutoffs stay fixed. Patient bootstrap intervals condition on saved fits and omit retraining/selection variability and earlier pilot adaptivity. These are development results, not reserved-test or clinical evidence.

Original: 6,864 images, 128 malignant; retained: 6,706 images, 127 malignant.
Both populations contain the same patients and preserve outer and inner patient assignments.

## Primary mean-fold metrics

| Population | Model | AP | ROC-AUC |
|---|---|---:|---:|
| original | b0_raw | 0.144716 | 0.840783 |
| original | metadata_nn_mean | 0.058193 | 0.678797 |
| original | spline_raw | 0.064855 | 0.662886 |
| retained | b0_raw | 0.145637 | 0.839361 |
| retained | metadata_nn_mean | 0.058708 | 0.677243 |
| retained | spline_raw | 0.065266 | 0.661317 |

Bootstrap: 2000/2000 valid paired patient draws, seed 42; invalid draws are shared across all comparisons.

See tables/paired_comparisons.csv for the three predeclared model comparisons and retained-minus-original changes.
Equal-patient metrics are secondary: each patient's retained image weights sum to one. Sparse subgroup metrics remain missing when undefined.
Calibration maps and training-selected cutoffs were held fixed; no cutoff or calibrator was fitted using development OOF labels.

Clean retraining is a separate stage. The three-condition report can be extended only with explicit matching clean run and metadata receipts.
