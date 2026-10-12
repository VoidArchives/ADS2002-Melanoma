# Melanoma Detection

This project explores whether lesion images and their recorded age, sex and body
location can help distinguish benign from malignant images. The notebooks show
data preparation, model comparisons and checks of what those comparisons can
establish. The work is exploratory; it does not establish a clinical screening tool.

Start with the [notebook guide](notebooks/README.md) for a reading route and the
status of each step. [Subset](notebooks/Subset) contains preparation and modelling
on the smaller dataset; [Full](notebooks/Full) contains preparation and exploration
of the full collection. The [short concept guide](notebooks/concepts.md) explains
the metrics and validation terms used throughout.

For method details, read the [metadata guide](docs/metadata_methodology.md) or
[image guide](docs/image_methodology.md). Earlier image pilots use an inner
validation split: patients separate from fitting, whose labels helped choose
pilot settings. The new [five-fold image evaluation](notebooks/Subset/14_image_nested_evaluation.ipynb)
reports the completed bounded B0 workflow across all five outer folds. Each outer
training pool repeats recipe selection across an inner patient split and confirms
duration across three inner folds before refitting. This run has a persistent
five-hour session budget, including implementation and a 20-minute reporting
reserve. Its notebook reads saved results. Mean fold AP is **0.1447** (95%
conditional patient-bootstrap interval **0.1033–0.2225**); mean fold ROC-AUC is
**0.8408**. These are development OOF results after earlier pilot use of the
dataset. The interval holds the fitted models and selections fixed; it excludes
retraining variability and prior pilot adaptation. Fusion and the reserved final
test remain planned.

The original and [deduplicated metadata workflows](notebooks/Subset_dedup/README.md)
keep stages **05 models → 06 fixed NN → 07 tuning including NN → 08 robustness**.
Their shared [NN tuning protocol](docs/metadata_nn_tuning.md) specifies the frozen
36-setting, three-seed search and patient-safe selection. Training runs through
an explicit resumable CLI; the updated notebooks read verified versioned results.
Historical notebooks and outputs are preserved, and deduplicated evidence remains
the main clean-data comparison.

Audit or resume the new run with the project environment:

```powershell
.venv/Scripts/python.exe scripts/run_image_nested.py --audit-only
.venv/Scripts/python.exe scripts/run_image_nested.py --resume
```

The larger architecture/transformation search remains deferred. The raw stage
fits no image calibration or decision threshold. The separate
`b0_nested_calibration_5h_v1` stage also completed all five folds, fitting maps and
academic recall cutoffs from training-only OOF predictions. Calibration lowered
mean fold log loss **0.0894 → 0.0791** and Brier **0.0209 → 0.0174**, preserving
within-fold AP/ROC-AUC. The training-selected 80%-recall scenario achieved
**82.81% outer recall**, with **3.95% precision** and **39.10% of images flagged**.
These are development results for an academic scenario; the reserved test remains
unopened. No new MC uncertainty run is part of either stage.

[Image uncertainty](notebooks/Subset/13_image_uncertainty.ipynb) reads existing
checkpoints; its default is the weak EfficientNet-B0 control. New inference outputs
go to separate directories under `results/image_uncertainty_runs`.
[Historical notebook guides](notebooks/archive/README.md) explain the preserved
original experiments. Renaming notebook folders leaves data, models, training
histories and saved results unchanged; `_d1`/`_d2` in artifact names remain their
original identifiers.

[Project notes](https://docs.google.com/document/d/1rMOSNjIrVSib8rnjD8M53RhmcC24fubRem6pFThek0M/edit?usp=sharing)
