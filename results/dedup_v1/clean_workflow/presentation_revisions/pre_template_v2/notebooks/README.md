# Clean development workflow

Open stages in order. Executing these notebooks starts no training.

These notebooks preserve the original explanatory depth, analysis methods and figure families where they apply to the authorised clean workflow.

All stages explicitly use the frozen dedup_v1 manifest and original patient assignments.

Figures are generated from clean records by editable plotting code and exported as PNG/PDF under `results/dedup_v1/clean_workflow/figures/<stage>/`. Incomplete image outputs are explicitly pending.

See [method and figure coverage](METHOD_COVERAGE.md) for correspondence with the originals and deliberate adaptations.

Run `.venv/Scripts/python.exe -B scripts/continue_clean_workflow.py` to complete only missing authorised stages.

The existing Subset notebooks and outputs are historical and remain unchanged.

- [01 · Clean development metadata and retain exact copies once](01_data_cleaning.ipynb)
- [02 · Explore images and patients in the retained development data](02_clean_data_eda.ipynb)
- [03 · Verify the preserved patient folds](03_patient_fold_verification.ipynb)
- [04 · Compare clean dummy and logistic baselines](04_metadata_baselines.ipynb)
- [05 · Compare clean metadata models and feature sets](05_metadata_models.ipynb)
- [06 · Examine the clean three-seed metadata NN](06_metadata_neural_network.ipynb)
- [07 · Inspect clean nested metadata tuning](07_metadata_tuning.ipynb)
- [08 · Check clean metadata uncertainty, calibration and decisions](08_metadata_robustness.ipynb)
- [09 · Clean B0: preparation, nested selection and OOF evaluation](09_image_modelling.ipynb)
- [10 · Bounded duration and ConvNeXt-Tiny pilots](10_bounded_image_pilots.ipynb)
- [11 · Clean outer-fold uncertainty and referral diagnostics](11_image_uncertainty.ipynb)
- [12 · Compare clean models and state the supported conclusions](12_combined_evaluation.ipynb)
