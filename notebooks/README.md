# Notebook workflows

Notebook filenames identify the dataset they use:

- `*_d1.ipynb`: subset dataset under `data/subset`
- `*_d2.ipynb`: full dataset under `data/raw` and `data/processed`

The numbered stages are kept separate so d1 and d2 results, folds, models, and evaluations are not accidentally mixed.

Reusable Python utilities under `src/melanoma_detection` remain dataset-neutral and do not receive a d1/d2 suffix.

`train_legacy.ipynb` contains older mixed scratch work and should not be used as part of either active workflow.
