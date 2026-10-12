# Metadata neural-network tuning

Both workflows retain the order **05 metadata models → 06 fixed neural network → 07 tuning, including NN → 08 robustness**. Stage 06 describes the fixed three-seed network. Stage 07 tests whether choosing its size, learning rate and dropout within training patients improves outer-fold predictions. This is an additional development experiment after earlier outer scores were examined; its search is frozen rather than expanded in response to the new results.

The original condition contains 6,864 development images. The primary clean condition uses the frozen `dedup_v1` manifest containing 6,706 images from the same 422 patients. Both keep age, sex and anatomical site, and the existing five patient outer folds. Neither entrypoint fits a final all-development model nor opens the reserved test.

## Run and resume

```powershell
.venv\Scripts\python.exe -B scripts/run_metadata_nn_tuning.py --config configs/metadata_nn_tuning.yaml --resume
```

Use `--condition original` or `--condition dedup_v1` to run one condition. Omitting it runs both sequentially. Completed stopping and refit phases are checkpointed independently, so a resumed trial does not repeat verified work. Configuration bytes, development inputs, membership files, reusable receipts, fitting sources and runtime versions contribute to the compatibility identity. A mismatch is rejected; choose a separately named run instead of modifying the frozen experiment in place. A latest pointer is published only after all five outer folds and every prediction have passed verification.

New original outputs are under `results/metadata_nn_tuning/v1/original/nn_grid36_v1/`; clean outputs are under `results/dedup_v1/metadata_nn_tuning/v1/nn_grid36_v1/`. Their receipts distinguish newly executed fits from historical clean fits reused. The search requires 3,270 stopping/refit calls per condition, or 6,540 across both. Original reconstruction adds 30 fixed-NN calls and 660 conventional-model fits. The clean reconstruction reuses 30 fixed-NN calls and 660 relevant conventional fits, with provenance verified before reuse.

## Frozen setting search

| Setting | Values | What it controls |
|---|---|---|
| Hidden units | 8, 16, 32, 64 | The number of learned hidden representations; larger widths allow more flexibility. |
| Learning rate | 0.0003, 0.001, 0.003 | The size of each optimiser update. |
| Dropout | 0.0, 0.2, 0.4 | The fraction of hidden activations randomly masked during training; disabled for prediction. |

The current recipe, 32 units / 0.001 / 0.2, is candidate `r00`. All other combinations follow ascending width, learning rate and dropout. Exact AP ties follow this declared order. Every candidate uses seeds 42, 142 and 242. Their probabilities are averaged before AP is calculated; no seed is selected for a favourable score.

One ReLU hidden layer, AdamW weight decay 0.001, unweighted binary cross-entropy, batch size 128 and at most 100 epochs remain fixed. The existing stopping loss, minimum ten attempted epochs, patience 12 and improvement tolerance 0.00001 are retained. CPU fits run sequentially with four Torch threads. Feature engineering, image modelling and network depth are outside this experiment.

## Patient-safe selection

The metadata-suite memberships were reconstructed from the original notebook's CSV ordering and `42 + outer_fold` splitting protocol. Their historical provenance is **reconstructed**, rather than an original contemporaneous membership receipt. The clean membership file must equal the original file filtered to retained image IDs. The image workflow's different inner memberships are not used here.

For an outer fold, all its patients are excluded from setting and duration selection. Each recipe is assessed on three saved inner validation groups. When group `v` is assessed, the remaining groups are `[(v+1)%3, (v+2)%3]`. Seed indices 0 and 2 stop on the first remaining group and fit on the second; index 1 reverses those roles. Preprocessing is fitted only on epoch-fitting patients. The stopping loss chooses duration, then that network and preprocessor are discarded. A fresh preprocessor and network fit on both inner-training groups for the chosen duration and predict the excluded inner group.

The unweighted mean of the three inner-fold ensemble AP values selects the recipe. For its outer prediction, the preserved stopping fold is `(outer_fold + 1 + seed_index) % 5`, and the training seed is `base_seed + outer_fold`. Duration is selected within outer training; a fresh network then refits on all outer-training patients. Each development image receives exactly one outer prediction from a procedure that excluded its patient. This follows the distinction between setting selection and evaluation in [nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## Reports and interpretation

```powershell
.venv\Scripts\python.exe -B scripts/build_metadata_nn_tuning_notebooks.py --run-id nn_grid36_v1
.venv\Scripts\python.exe -B scripts/update_metadata_nn_tuning_integration.py
.venv\Scripts\python.exe -B scripts/verify_metadata_nn_tuning_results.py
.venv\Scripts\python.exe -B scripts/execute_metadata_nn_tuning_reports.py
.venv\Scripts\python.exe -B scripts/verify_metadata_nn_tuning_reports.py --require-complete
```

These commands build report notebooks without training. Execute their cells to read verified completed outputs and regenerate figures. After execution, `build_metadata_nn_tuning_notebooks.py --html-only` exports saved reading previews without clearing outputs. Original and clean stages 06/07 share one presentation template based on the archived originals. The tuning bars retain fixed blue `#2563EB` and tuned gold `#C58A00`; the four-row ROC/PR grid keeps dashed fixed and solid tuned curves. Added sections explain patient support, every candidate, selected recipes, frequency, duration, score margins and per-fold gains.

The result verifier independently reconstructs all 540 inner ensemble AP scores per condition, matches each prediction file to its saved membership group, checks the complete frozen grid and winning records, and sums the fit log. Report acceptance checks notebook execution, valid local links, matching templates and a recorded visual review of the current notebook hashes. The final check requires all eight affected reports and the visual review to be complete. Coverage and verification receipts are stored in `results/metadata_nn_tuning/v1/report_verification/`, outside directories reserved for new training sessions.

Mean outer-fold AP is primary. Pooled metrics and the original-style pooled curves remain labelled secondary views. Two thousand paired bootstrap draws resample whole patients within each original outer fold. Tuned NN is compared with fixed NN, fixed spline and tuned spline using AP, ROC-AUC, log loss and Brier score. These intervals condition on saved predictions and do not include uncertainty from repeating the entire training search. An interval crossing zero does not establish equivalence.

The robustness and downstream reports distinguish the new tuned NN from the fixed NN. Historical spline/image calibration maps and training-selected thresholds remain preserved; they provide no calibration or decision evidence for the new tuned network. The experiment supplies model-choice evidence and does not automatically replace the spline.

## Preservation and checks

`results/metadata_nn_tuning/v1/archive_before_update/` contains all affected notebook snapshots and their hashes, plus a preservation receipt for historical development inputs and results. Historical code-hash checks resolve to the verified archived metadata notebook when its active report has been edited. Historical predictions, models, metrics, figures and receipts are not overwritten.

The historical clean combined-report source was recovered by reverting its reporting-only verification edit. Its exact bytes match the unchanged historical evaluation receipt's source hash. A separate archive binding checks that snapshot against both the historical receipt and its pre-update preservation hash; this reporting-only exception does not permit changed training sources.

Checks cover patient isolation, both-class support, fitting-only preprocessing, complete finite probabilities, probability averaging before AP, outer-label independence, deterministic ties, interrupted-phase recovery and mismatched-input rejection. Reports remain pending until complete five-fold receipts exist. The original duplicate-bearing condition provides a comparison; deduplicated evidence remains the main clean-data assessment.

## Completed experiment: `nn_grid36_v1`

Both conditions completed all five outer folds. The original condition executed 3,960 new fits; the clean condition executed 3,270 new fits and reused 690 verified historical clean fits. The independently verified totals are 7,230 newly executed fits and 690 fits reused. Mean outer-fold AP is the primary comparison:

| Dataset | Fixed spline AP | Tuned spline AP | Fixed NN AP | Tuned NN AP |
|---|---:|---:|---:|---:|
| Original | 0.0649 | 0.0689 | 0.0581 | 0.0657 |
| Deduplicated | 0.0648 | 0.0688 | 0.0596 | 0.0681 |

On the main clean dataset, tuned NN minus fixed NN AP is +0.0085, with a 95% conditional paired patient interval of [-0.0015, +0.0262]. Tuned NN minus fixed spline is +0.0032 [-0.0077, +0.0145]; tuned NN minus tuned spline is -0.0007 [-0.0140, +0.0064]. All three AP intervals include zero. The point improvement over the fixed network remains uncertain, and these results do not establish a dependable advantage over spline or justify automatic replacement.

Both assessments used 2,000 valid paired patient draws; no draws were rejected for missing class support. The detailed reports also retain ROC-AUC, log loss, Brier score, pooled secondary summaries, each fold and every selected recipe. Historical calibration and threshold results remain attached to their original fixed procedures. No final all-development model was fitted and the reserved test was not accessed.
