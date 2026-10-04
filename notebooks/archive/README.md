# Reading the historical notebooks

These files preserve earlier experiments exactly, including their old titles,
paths and saved outputs. Read them as records of how the project developed.
Use the [current notebook index](../README.md) for work with the current pipeline.
The [concept guide](../concepts.md) explains the shared terminology.

## Original image model

[image_model_three_epoch_fold0.ipynb](image_model_three_epoch_fold0.ipynb)
contains the original three-epoch EfficientNet-B0 training recipe. A fold is a
group of held-out patients; this prototype evaluated only fold 0. It has no saved
cell outputs or figures. Its recorded metrics are cited inside the notebook from
existing result files, so they do not represent a fresh execution of this copy.

Read the fold table as counts of images, patients and malignant images. Read AP
and ROC-AUC as ranking measures, with higher values indicating better ranking.
Lower Brier score and log loss indicate better agreement between scores and
labels. Class-weighted training gives malignant images more influence; the
resulting scores need calibration checks before they can describe risk.

For the current workflow, start with
[09 Image B0 pilot](../Subset/09_image_b0_pilot.ipynb). It reports checkpoints from
the shared training pipeline, recorded recipes and a separate inner-validation
patient split.

## Original image uncertainty

[image_uncertainty_three_epoch_fold0.ipynb](image_uncertainty_three_epoch_fold0.ipynb)
contains the earlier 30-pass dropout and eight-view TTA analysis of 1,370 images
from 84 held-out patients. Its results concern the old checkpoint and fold only.
The word `probability` in historical column and axis names denotes a prediction
score; the reliability check shows that those scores are poorly calibrated.

Read its four figures as follows:

- **MC convergence:** the horizontal axis counts dropout passes; the vertical
  axis is the median absolute difference from the 30-pass mean or SD. Falling lines mean
  closer agreement with that reference. Zero at 30 is guaranteed by comparison
  with itself and does not prove convergence beyond 30 passes.
- **Four-panel uncertainty comparison:** the upper histograms count images at
  each SD. The lower-left scatterplot compares dropout SD with TTA SD for each
  image. The lower-right panel compares mean absolute score error across groups ordered
  by SD. An upward line associates higher variation with larger error, but benign
  images dominate and their absolute error equals their score.
- **Reliability diagram:** each point places a group's mean score on the
  horizontal axis and its observed malignant fraction on the vertical axis.
  Points below the diagonal indicate scores above the observed frequency.
  There are only 25 malignant images and no interval around those points.
- **Risk–coverage curve:** start at 100% retained on the right and move left as
  more images are set aside. A lower line means lower mean score error among
  retained images at the same coverage. It does not measure reviewer outcomes.

The archive lacks score-ranking and random-referral baselines and patient-based
intervals for key differences. Those comparisons are provided in
[13 Image uncertainty](../Subset/13_image_uncertainty.ipynb) for the current
checkpoint pipeline.

## Legacy preparation experiments

[legacy_training.ipynb](legacy_training.ipynb) records early attempts to select a
patient subset, assign folds and count missing metadata. It has no figures or
model evaluation. In its fold table, `count` is the number of images, `sum` the
number malignant and `mean` their fraction. Similar fractions suggest comparable
class balance, rather than model quality. A missing-value count of zero after
replacement means the rows now carry `Unknown`; it does not recover the original
information. The old subset and paths differ from the current inputs.

Use [Subset cleaning](../Subset/01_data_cleaning.ipynb) or
[Full cleaning](../Full/01_data_cleaning.ipynb), followed by the matching exploration
notebook, for current data preparation.

## Limitations

Historical paths and settings are retained for provenance, so these notebooks
are outside the current run sequence. Their limited checks and saved results
cannot establish final model performance or a clinical referral rule.

## Summary

The archives explain earlier choices and their limitations. Follow the current
Subset or Full sequence to reproduce the present workflow.
