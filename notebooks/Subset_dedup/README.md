# Clean development workflow

**Current v2 reporting contract.** Unweighted mean outer-fold **ROC-AUC** is the primary ranking summary. Mean-fold AP, log loss and Brier score are secondary measures; pooled metrics are supporting summaries. Each fold has equal weight, and patient-bootstrap intervals resample whole patients within folds. A single selected inner split is exploratory and cannot supply a five-fold estimate. Scores describe individual images; patient grouping does not turn them into patient-level diagnoses.

The metadata v2 experiment selects settings by inner ROC-AUC. Existing image recipes, checkpoints, durations and transformation-admission gates retain their historical AP-based rules. Reporting their saved ROC-AUC does not retrospectively make them ROC-selected. Calibration and recall cutoffs remain attached to their original procedures. All active report notebooks read saved artifacts; they start no training, image smoke or checkpoint inference and never open the reserved test. Final fitting and independent testing wait until the entire procedure is frozen.

Presentation snapshots are preserved under `results/metadata_nn_tuning/v2/archive_before_update/`. New image figures/tables use `results/metadata_nn_tuning/v2/report_verification/image_reports/<condition>/<stage>/`; current metadata assessment tables and figures use the explicit v2 `roc_auc_grid36_v1` run's `report/` directory. v1 AP results and historical image outputs retain their original identities.


Open stages in order. Executing these notebooks starts no training.

These notebooks use the original Markdown cells and section order as presentation templates. Colours, figure composition and explanatory detail are retained, while clean inputs, saved numerical findings and methodological qualifications are adapted.

All stages explicitly use the frozen dedup_v1 manifest and original patient assignments.

The historical v1 figures were generated from clean records by editable plotting code. Their preserved PNG/PDF exports remain under `results/dedup_v1/clean_workflow/figures/template_v3/<stage>/` and `results/dedup_v1/metadata_nn_tuning/v1/nn_grid36_v1/report/figures/`. Preparation 01–03 retains its existing saved outputs. Active 04–12 reports use the separate v2 destinations described above; they do not overwrite these historical exports. Incomplete image experiments are explicitly pending.

See [method and figure coverage](METHOD_COVERAGE.md) for correspondence with the originals and deliberate adaptations.

The historical controller `.venv/Scripts/python.exe -B scripts/continue_clean_workflow.py` owns fitting/inference stages. It is not required for the current saved-result presentation refresh.

Rebuild the current clean reports with `.venv/Scripts/python.exe -B scripts/build_dedup_workflow_notebooks.py`; this preserves existing preparation 01–03. The direct metadata builder is `.venv/Scripts/python.exe -B scripts/build_metadata_tuning_v2_notebooks.py`, and the image builder is `.venv/Scripts/python.exe -B scripts/build_image_roc_reports.py`. Execute saved-result reports separately with `.venv/Scripts/python.exe -B scripts/execute_metadata_roc_reports.py`. Its guards require the completed ROC-selected receipts, block fitting and reserved-test access, and confine new exports to v2 report namespaces. The compatibility NN builder and executor now dispatch to these current reports; they do not restore the historical AP presentation.

The [ROC-AUC workflow and verification guide](../../docs/metadata_roc_auc_workflow.md) explains the version-2 protocol, evidence boundaries, execution checks and preserved history. These report commands read completed experiment artifacts; they do not launch another tuning run.

Affected Subset metadata notebooks were archived before their report update. Historical predictions, models, figures, metrics and receipts remain unchanged.

Both metadata workflows keep **05 models → 06 fixed NN → 07 tuning including NN → 08 robustness**. The historical [shared NN tuning protocol](../../docs/metadata_nn_tuning.md) documents the completed AP-selected version-1 search. The separately declared version-2 experiment searches the same 36 NN settings and three classical families by inner ROC-AUC, with patient-safe duration selection and three-seed averaging. Both original and clean version-2 five-fold experiment receipts are complete. Current reports require those verified receipts; no AP result is substituted for missing ROC evidence. The original dataset retains 6,864 development images, and the frozen deduplicated dataset retains 6,706; the clean condition supplies the main modelling evidence.

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
