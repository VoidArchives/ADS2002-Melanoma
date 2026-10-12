# Clean development workflow

Open stages in order. Executing these notebooks starts no training.

These notebooks use the original Markdown cells and section order as presentation templates. Colours, figure composition and explanatory detail are retained, while clean inputs, saved numerical findings and methodological qualifications are adapted.

All stages explicitly use the frozen dedup_v1 manifest and original patient assignments.

Figures are generated from clean records by editable plotting code. The template reports export PNG/PDF under `results/dedup_v1/clean_workflow/figures/template_v3/<stage>/`; the updated shared metadata 06/07 reports export under `results/dedup_v1/metadata_nn_tuning/v1/nn_grid36_v1/report/figures/`. Incomplete image outputs are explicitly pending.

See [method and figure coverage](METHOD_COVERAGE.md) for correspondence with the originals and deliberate adaptations.

Run `.venv/Scripts/python.exe -B scripts/continue_clean_workflow.py` to complete only missing authorised stages.

Affected Subset metadata notebooks were archived before their report update. Historical predictions, models, figures, metrics and receipts remain unchanged.

Both metadata workflows keep **05 models → 06 fixed NN → 07 tuning including NN → 08 robustness**. The [shared NN tuning protocol](../../docs/metadata_nn_tuning.md) documents the completed frozen 36-setting search, explicit runner and patient-safe selection. Updated metadata reports write into the separate versioned `metadata_nn_tuning` namespace. Both five-fold conditions are complete; verified completion receipts determine whether results can be displayed.

- [01 · Clean development metadata and retain exact copies once](01_data_cleaning.ipynb)
- [02 · Explore images and patients in the retained development data](02_clean_data_eda.ipynb)
- [03 · Verify the preserved patient folds](03_patient_fold_verification.ipynb)
- [04 · Compare clean dummy and logistic baselines](04_metadata_baselines.ipynb)
- [05 · Compare clean metadata models and feature sets](05_metadata_models.ipynb)
- [06 · Examine the clean three-seed metadata NN](06_metadata_neural_network.ipynb)
- [07 · Inspect clean nested tuning including NN](07_metadata_tuning.ipynb)
- [08 · Read clean metadata uncertainty, calibration and decisions](08_metadata_robustness.ipynb)
- [09 · Clean B0 screening and nested evaluation](09_image_modelling.ipynb)
- [10 · Bounded clean pilots and deferred transformations](10_bounded_image_pilots.ipynb)
- [11 · Clean image uncertainty](11_image_uncertainty.ipynb)
- [12 · Combined clean model evaluation](12_combined_evaluation.ipynb)
