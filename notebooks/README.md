# Notebook workflows

Notebook filenames identify the dataset they use:

- `*_d1.ipynb`: subset dataset under `data/subset`
- `*_d2.ipynb`: full dataset under `data/raw` and `data/processed`

The numbered stages are kept separate so d1 and d2 results, folds, models, and evaluations are not accidentally mixed.

Reusable Python utilities under `src/melanoma_detection` remain dataset-neutral and do not receive a d1/d2 suffix.

`train_legacy.ipynb` contains older mixed scratch work and should not be used as part of either active workflow.

## Read the notebooks

Each notebook opens by explaining its purpose, the question it answers and how it
connects to the earlier work. Guides beside the figures explain the axes, lines,
colours and comparisons. Every notebook ends with `Limitations`, followed by
`Summary`. The summary gives the finding, the decision about keeping the method
or model, and the next step. Planned notebooks state that they have no results yet.

For d1, begin with cleaning, exploration and patient splitting. The dummy/logistic
notebook introduces the basic comparison; the metadata-model notebook compares
the wider set of models. The NN and tuning notebooks test possible alternatives,
and 04b checks uncertainty, cutoffs, calibration and subgroup performance.

The main 05 image notebook reports the archived B0 pilot. Its WIP notebook and
05a uncertainty notebook preserve the historical experiment. The separate 05e
notebook explores DenseNet121 and ConvNeXt-Tiny on the same inner patient split.
The separate 05b notebook reports the fresh twelve-epoch weak B0 control and
the stronger profile's image-quality failure. No A/B performance comparison
was completed. The optional 05c lighting notebook records CLAHE as deferred.
Safer augmentation, preprocessing experiments, longer confirmation, fusion
and final nested evaluation remain later stages. For d2, cleaning and exploration form a
separate branch; the image-inventory notebook supports that branch by checking
the raw filenames. The legacy notebook is an archive rather than a next step.

## Metadata evaluation workflow (d1)

Run `03b_metadata_neural_network_d1.ipynb`, then `04b_metadata_robustness_d1.ipynb`.
The first notebook evaluates three predeclared NN runs. The second uses training-only
cutoff and calibration selection, patient-bootstrap confidence intervals, subgroup
checks and a frozen spline model fitted without opening the reserved test labels.

See [the plain-language methodology guide](../docs/metadata_methodology.md) for why
each step is needed, how it works and what its results can establish. The earlier
03/03a threshold examples are exploratory; use 04b for held-out evaluation of a
training-selected cutoff. All development OOF results remain development evidence.

The dependency snapshot for 03b/04b is `requirements-metadata.txt`. Run the notebooks
with the project environment, for example:

```powershell
.venv/Scripts/python.exe -m jupyter execute --inplace --timeout=900 notebooks/03b_metadata_neural_network_d1.ipynb
.venv/Scripts/python.exe -m jupyter execute --inplace --timeout=900 notebooks/04b_metadata_robustness_d1.ipynb
```

The later final-test script is `scripts/evaluate_metadata_holdout.py`; the guide
explains its frozen-model and one-attempt controls. It has been tested with synthetic
data, but the actual reserved test remains unevaluated.

## Image pilot workflow (d1)

`05_image_model_d1.ipynb` uses shared utilities and `configs/image_model.yaml`.
The first pilot compares four B0 learning-rate/loss settings on one inner
patient split, excluding outer fold 0 entirely. Its predictions are inner
validation scores, not final OOF evidence. It does not read the reserved test.

See [the image methodology and staged roadmap](../docs/image_methodology.md).
The existing `.venv` has CUDA enabled; its dependency snapshot is
`requirements-image.txt`. The original pilot training command is retained in its
archived source snapshot. Current code changes must not bypass the historical
manifest's source mismatch. The architecture experiment has its own runner:

```powershell
.venv/Scripts/python.exe scripts/run_image_architecture.py --config configs/image_architecture_pilot.yaml --session-file results/image_d1_runs/architecture_pilot_3h_v1/session.json --resume
```

That command respects the supplied session deadline and training allowance; an
expired session cannot silently allocate more time. Future resumption requires
an explicitly allocated new session. Both image notebooks default to saved
results. Six-epoch comparisons do not remove slower families from later work.

To save all result tables and figures using the explicitly selected environment:

```powershell
.venv/Scripts/python.exe scripts/execute_image_notebook.py
.venv/Scripts/python.exe scripts/execute_image_notebook.py --notebook 05e_image_architecture_comparison_d1.ipynb
```

The controlled augmentation utilities use `configs/image_augmentation.yaml`
and `scripts/run_image_conditions.py`. Quality approval, source-compatible
checkpoints and the fixed session deadline are mandatory. The current stronger
profile remains blocked; a later safer profile requires a new declared experiment.
CLAHE was not installed or trained because the complete safe A/B prerequisite
was unavailable. Do not bypass the gate or extend an expired session silently.

Rebuild and execute the new saved-result reports without training:

```powershell
.venv/Scripts/python.exe scripts/build_image_condition_notebooks.py --execute
```
