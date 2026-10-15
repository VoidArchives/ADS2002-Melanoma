# Melanoma Detection

Project Notes: [Here](https://docs.google.com/document/d/1rMOSNjIrVSib8rnjD8M53RhmcC24fubRem6pFThek0M/edit?usp=sharing)

Presentation: [Here](https://canva.link/928zttfn7td7g22)

Image model experiments: [guide](docs/image_experiments.md) and [notebook](notebooks/04_image_experiments.ipynb).

CNN screening now compares all eight architectures using shared patient folds,
canonical TorchVision crop sizes, training-fold-only square-root class weighting,
one head-only epoch plus two full-backbone epochs with frozen BatchNorm, and
validation ROC-AUC checkpoints. CUDA AMP, batch preflight and accumulation target
an effective batch of 128. Compare pooled OOF ROC-AUC in `architecture_summary.csv`.

```powershell
.venv\Scripts\python.exe scripts/train_image_screening.py --device cuda --workers 0
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

To compare 0.5 against validation thresholds targeting 90% and 95% recall without
retraining, open [the saved-results notebook](notebooks/05_image_thresholds.ipynb),
or run `.venv\Scripts\python.exe scripts/evaluate_image_thresholds.py`.
Completed training folds are now reused by default. Use `--reuse-only` to forbid
training or `--force-retrain` to explicitly replace a run; see the guide for
checkpoint loading and the limits of validation threshold estimates.

New training runs retain `best_auc.pt` and `best_specificity_recall90.pt` per fold,
each with its own frozen validation-selected threshold. The training notebooks
display `checkpoint_comparison.csv`; every epoch logs recall-targeted specificity
alongside the fixed-0.5 metrics. Old runs need one retraining to create both files.
