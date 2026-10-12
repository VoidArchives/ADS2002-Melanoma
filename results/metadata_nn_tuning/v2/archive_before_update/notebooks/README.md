# Notebook workflows

Active workflows are separated by dataset directory and numbered consecutively:

- `Subset/`: subset dataset under `data/subset`, notebooks 01–15, with the new
  `14_image_nested_evaluation.ipynb` preceding the planned fusion stage.
- `Full/`: full dataset under `data/raw` and `data/processed`, notebooks 01–03.
- `Subset_dedup/`: the twelve clean reports using the frozen retained-development
  manifest; stages 06/07 share a presentation template with `Subset`.
- `archive/`: historical experiments, outside the active sequence.

`Subset` was previously called d1; `Full` was d2. Existing data files, saved runs
and models retain their original `_d1`/`_d2` identifiers so provenance still
matches. The folder name identifies the notebook's dataset; keep the two branches'
results and evaluation splits separate.

## Start here

Read a notebook's opening question, saved figures, `Limitations` and `Summary`
before running its code. Each code section explains the operation and why it is
needed; figure guides explain the comparison to make and its limits. The
[concept guide](concepts.md) defines shared terms such as out-of-fold predictions,
average precision and calibration.

- **Understand the data:** Subset 01–03 show cleaning, exploration and patient
  splitting. Full 01–03 provide a separate preparation and inventory reference.
- **Understand metadata models:** Subset 04 introduces simple baselines; 05–07
  compare alternatives; 08 checks decisions, probability estimates and subgroups.
- **Understand image models:** Subset 09–10 compare saved pilots; 11–12 document
  transformation quality checks and incomplete experiments; 13 studies variation
  in predictions and whether it helps prioritise images for review. The new
  14 image evaluation reads the bounded five-outer-fold B0 run and reports
  performance only after all folds complete.
- **See what remains:** fusion and final evaluation remain planned stages. The
  active folders contain no fusion or reserved-test results.

“Executed” means outputs are saved, not that a method is a final model. Image
pilots reuse one inner validation split, while metadata out-of-fold (OOF)
predictions combine held-out development folds. Neither replaces an independent
reserved test. Opening saved outputs does not require training; running preparation
or modelling cells can write files, as described in each notebook.

## Subset sequence

| Notebook | Purpose | Status |
|---|---|---|
| [01 Data cleaning](Subset/01_data_cleaning.ipynb) | Validate and prepare subset data | Implemented; file-writing preparation |
| [02 Data exploration](Subset/02_data_exploration.ipynb) | Inspect labels, metadata and images | Executed |
| [03 Patient splitting](Subset/03_patient_splitting.ipynb) | Keep every patient's images in one fold | Implemented; patient-grouped folds |
| [04 Dummy/logistic baseline](Subset/04_dummy_logistic_baseline.ipynb) | Establish simple prediction baselines | Executed; exploratory cutoff example |
| [05 Metadata models](Subset/05_metadata_models.ipynb) | Compare metadata model families | Executed; development OOF comparisons |
| [06 Metadata neural network](Subset/06_metadata_neural_network.ipynb) | Evaluate three predeclared NN runs | Executed; development OOF predictions |
| [07 Metadata tuning](Subset/07_metadata_tuning.ipynb) | Compare nested tuning across four metadata families, including NN | Executed; development experiment |
| [08 Metadata robustness](Subset/08_metadata_robustness.ipynb) | Evaluate training-selected cutoffs, calibration and subgroups | Executed; reserved test untouched |
| [09 Image B0 pilot](Subset/09_image_b0_pilot.ipynb) | Report four saved B0 recipes | Executed; inner-validation diagnostics |
| [10 Image architecture comparison](Subset/10_image_architecture_comparison.ipynb) | Compare saved B0, DenseNet121 and ConvNeXt-Tiny pilots | Executed; exploratory inner split |
| [11 Image augmentation](Subset/11_image_augmentation.ipynb) | Report the weak B0 control and stronger-profile quality gate | Executed; stronger condition blocked |
| [12 Image lighting](Subset/12_image_lighting.ipynb) | Record the optional CLAHE experiment | Deferred; displays the weak control |
| [13 Image uncertainty](Subset/13_image_uncertainty.ipynb) | Compare dropout/TTA variation and referral baselines | Executed; exploratory inner-validation diagnostics |
| [14 Image outer evaluation](Subset/14_image_nested_evaluation.ipynb) | Report five outer folds with independent B0 recipe and duration selection | Executed; raw and calibrated development OOF complete |
| 14 Fusion model | Combine image and metadata predictions | Planned; no notebook or results |
| 15 Final evaluation | Evaluate the frozen workflow | Planned; reserved test unevaluated |

## Full sequence

| Notebook | Purpose | Status |
|---|---|---|
| [01 Data cleaning](Full/01_data_cleaning.ipynb) | Validate and prepare full-dataset metadata | Implemented; separate Full preparation |
| [02 Data exploration](Full/02_data_exploration.ipynb) | Inspect the full dataset | Executed |
| [03 Image inventory](Full/03_image_inventory.ipynb) | Check raw image filenames and availability | Image-file utility |

## Historical archive

| Notebook | Historical purpose |
|---|---|
| [Image model: three-epoch fold 0](archive/image_model_three_epoch_fold0.ipynb) | Original EfficientNet-B0 prototype |
| [Image uncertainty: three-epoch fold 0](archive/image_uncertainty_three_epoch_fold0.ipynb) | Original MC-dropout/TTA prototype |
| [Legacy training](archive/legacy_training.ipynb) | Older mixed scratch work |

Read these archives to inspect their original code and execution. They retain
historical paths and settings and are not steps to rerun in the current workflow.
Their models, predictions, metrics, figures and training histories remain
unchanged. Older preservation receipts keep their original notebook names and
verify the corresponding archives against the recorded original hashes.
Use the [archive reading guides](archive/README.md) for context and figure
interpretation while the original notebooks remain unchanged.

## Metadata evaluation workflow (Subset)

The order is **05 metadata models → 06 fixed NN → 07 tuning including NN → 08
robustness**, in both `Subset` and `Subset_dedup`. The [NN tuning protocol](../docs/metadata_nn_tuning.md)
documents the explicit runner and frozen 36-setting search. Updated 06/07/08
notebooks read completed artifacts and generate reports without fitting models.
Stage 08 preserves historical training-only spline calibration and decision
analyses, then adds separate tuned-NN comparisons. Those calibration maps and
cutoffs do not transfer to the new network. Final all-development fitting and
the reserved test remain separate later steps.

See [the plain-language methodology guide](../docs/metadata_methodology.md) for why
each step is needed, how it works and what its results can establish. The earlier
04/05 threshold examples are exploratory; use 08 for held-out evaluation of a
training-selected cutoff. All development OOF results remain development evidence.

The dependency snapshot for 06/08 is `requirements-metadata.txt`. Run the notebooks
with the project environment, for example:

```powershell
.venv/Scripts/python.exe -m jupyter execute --inplace --timeout=900 notebooks/Subset/06_metadata_neural_network.ipynb
.venv/Scripts/python.exe -m jupyter execute --inplace --timeout=900 notebooks/Subset/07_metadata_tuning.ipynb
.venv/Scripts/python.exe -m jupyter execute --inplace --timeout=900 notebooks/Subset/08_metadata_robustness.ipynb
```

The later final-test script is `scripts/evaluate_metadata_holdout.py`; the guide
explains its frozen-model and one-attempt controls. It has been tested with synthetic
data, but the actual reserved test remains unevaluated.

## Image pilot workflow (Subset)

`Subset/09_image_b0_pilot.ipynb` uses shared utilities and `configs/image_model.yaml`.
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
.venv/Scripts/python.exe scripts/execute_image_notebook.py --notebook Subset/10_image_architecture_comparison.ipynb
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

## Five-fold image evaluation workflow (Subset)

`Subset/14_image_nested_evaluation.ipynb` reads the completed raw
`b0_nested_5fold_5h_v1` run under `results/image_d1_runs`. It contains no training
loop. `configs/image_nested.yaml` declares B0, the existing weak augmentation,
unchanged deterministic preparation and seed 42. Every outer fold rebuilds
three inner patient groups using only its outer training pool.

All five raw folds completed on 6,864 images from 422 patients, including 128
melanoma images. Primary mean fold AP is **0.1447**, with a 95% conditional
patient-bootstrap interval **0.1033–0.2225**; mean fold ROC-AUC is **0.8408**.
These remain development OOF results after earlier pilots, including archived
fold 0. The interval conditions on saved fits and selections, excluding
retraining variability and prior adaptation. The reserved test remains unopened.

The runner screens all four learning-rate/loss recipes for six epochs on the
first inner fold. It chooses by screening AP, with ties within 0.0001 favouring
unweighted loss and then lower learning rate. The selected recipe continues
to twelve epochs on that split and runs twelve epochs on each other inner fold.
The median of the three best-AP epochs fixes the outer refit duration. That
refit uses all outer training patients and never monitors outer labels.

```powershell
.venv/Scripts/python.exe scripts/run_image_nested.py --audit-only
.venv/Scripts/python.exe scripts/run_image_nested.py --resume
```

The persistent 300-minute session clock includes implementation and reserves
20 minutes for reporting. The supervisor enforces the absolute training cutoff;
resumption preserves the original deadline and complete epoch-boundary state.
An expired session requires an explicitly allocated new session file. Partial
folds remain incomplete and cannot supply a final five-fold performance claim.

The raw report is finalised. To reproduce its saved report and execute the
read-only notebook:

```powershell
.venv/Scripts/python.exe scripts/finalize_image_nested.py
.venv/Scripts/python.exe scripts/execute_image_nested_notebook.py --timeout 600
```

The report verifies exact canonical coverage, shows mean fold AP as primary and
pooled AP as secondary, and uses 2,000 patient-bootstrap draws within outer folds.
It records valid/rejected draws, raw reliability, fixed-0.5 confusion counts and
verified matched metadata comparisons when available. Intervals condition on
the saved models and selections; earlier pilot use of this development dataset,
including archived fold 0, still limits independence.

The future 235-fit architecture/transformation search remains deferred. The
raw B0 stage adds no calibration fit or tuned cutoff. The separate
`b0_nested_calibration_5h_v1` stage completed after every raw fold, using the same
remaining session allocation. For each outer pool it
adds three fresh fixed-duration inner refits with the already selected recipe
and median epoch count. Their training-only OOF predictions fit sigmoid
recalibration (`C=1`, rejecting non-positive slope) and raw/calibrated cutoffs
targeting 80% training recall. Frozen maps and cutoffs are applied to the
unchanged raw outer predictions. Actual outer recall can differ from 80%.

Notebook 14's separate calibration branch reports mean fold log loss
**0.0894 → 0.0791** and Brier **0.0209 → 0.0174**. Within-fold AP and ROC-AUC
remained unchanged. The training-selected 80%-recall scenario achieved
**106/128 = 82.81%** outer recall, **3.95%** precision and
**2,684/6,864 = 39.10%** images flagged, including 2,578 benign images; 22
melanoma images were missed. Raw and calibrated cutoffs produced identical
decisions because the fitted maps preserve score ordering. This is an academic
development scenario, without clinical validation. Bootstrap intervals hold
models, calibration maps and thresholds fixed; the prior-pilot caveat still applies.

To reproduce the complete separate calibration report before executing Notebook 14:

```powershell
.venv/Scripts/python.exe scripts/finalize_image_nested_calibration.py
.venv/Scripts/python.exe scripts/execute_image_nested_notebook.py --timeout 600
```

Both stages add no new MC uncertainty, fusion or reserved-test evaluation.
Existing historical notebooks and outputs remain preserved.

## Image uncertainty workflow (Subset)

`Subset/13_image_uncertainty.ipynb` selects an existing run and recipe explicitly.
Its default is condition A of `b0_augmentation_12e_v1`, recipe `lr3e5_sqrt`, using
the recorded best checkpoint at epoch 6. Supported DenseNet121 and ConvNeXt-Tiny
recipes from `architecture_pilot_3h_v1` can be selected explicitly; the notebook
does not promote the highest pilot AP to a final architecture choice.

The notebook reconstructs the checkpoint's deterministic preparation and saved
inner-validation membership, and checks that evaluation patients were absent
from fitting. The current split has 1,834 images from 113 patients, including 33
malignant images, and excludes outer fold 0. These are inner-validation
diagnostics, not full OOF or final-test results.

Run a short smoke check before the full default notebook:

```powershell
.venv/Scripts/python.exe scripts/run_image_uncertainty.py --smoke
.venv/Scripts/python.exe scripts/execute_image_notebook.py --notebook Subset/13_image_uncertainty.ipynb
```

Alternative smoke checks require an explicit run, architecture and recipe:

```powershell
.venv/Scripts/python.exe scripts/run_image_uncertainty.py --smoke --run-id architecture_pilot_3h_v1 --architecture densenet121 --recipe-id low_none
.venv/Scripts/python.exe scripts/run_image_uncertainty.py --smoke --run-id architecture_pilot_3h_v1 --architecture convnext_tiny --recipe-id low_none
```

Set the matching `SOURCE_RUN`, `ARCHITECTURE` and `RECIPE_ID` in the notebook
before full alternative inference. Architecture recipe slots are `low_none`,
`low_sqrt`, `high_none` and `high_sqrt`.

Each execution writes to a separate directory under
`results/image_uncertainty_runs`, with raw predictions, summaries, figures,
checks and provenance. It uses 30 dropout passes with only explicit dropout
active, and separate identity/horizontal/vertical/both-flip TTA views. The
referral comparisons use common deterministic scores and fractions for dropout
SD, TTA SD, ordinary score, predictive entropy and repeated seeded random
ranking, with paired patient-bootstrap intervals. Retained classification error
uses the fixed raw cutoff 0.5; Brier error checks the score scale. These and the
reliability checks are descriptive; no calibrator or referral threshold is
fitted on pilot labels.

The historical archive retains the original code, execution and result paths.
Use it to read the old three-epoch fold-0 experiment. New inference does not
rewrite its results, checkpoints or training histories, and does not train,
implement fusion or evaluate the reserved test.
