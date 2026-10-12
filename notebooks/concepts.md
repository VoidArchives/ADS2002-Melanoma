# Concepts used in the notebooks

**Current v2 reporting contract.** Unweighted mean outer-fold **ROC-AUC** is the primary ranking summary; mean-fold AP, pooled metrics, log loss and Brier score remain supporting measures. Each fold has equal weight, and patient-bootstrap intervals resample whole patients within folds. A single selected inner split is exploratory and cannot supply a five-fold estimate. Scores describe individual images; patient grouping does not turn them into patient-level diagnoses.

The metadata v2 experiment selects settings by inner ROC-AUC. Existing image recipes, checkpoints, durations and transformation-admission gates retain their historical AP-based rules. Reporting their saved ROC-AUC does not retrospectively make them ROC-selected. Calibration and recall cutoffs remain attached to their original procedures. All active report notebooks read saved artifacts; they start no training, image smoke or checkpoint inference and never open the reserved test. Final fitting and independent testing wait until the entire procedure is frozen.

Presentation snapshots are preserved under `results/metadata_nn_tuning/v2/archive_before_update/`. New image figures/tables use `results/metadata_nn_tuning/v2/report_verification/image_reports/<condition>/<stage>/`; current metadata assessment tables and figures use the explicit v2 `roc_auc_grid36_v1` run's `report/` directory. v1 AP results and historical image outputs retain their original identities.


Use this page when a term is unfamiliar. Each notebook explains the terms needed
for its own task; you can read its explanation first and return here for context.

## Data and evaluation

- **Metadata** means recorded information such as age, sex and lesion location.
  An image model instead learns from pixels.
- **Target** is the recorded label: 1 for malignant and 0 for benign. One row is
  usually one image; a patient may contribute several images.
- **Patient grouping** keeps a person's images together when dividing data.
  Otherwise, a model could recognise information about a patient it has already
  seen, making evaluation look better than performance on new patients.
- **Fold** means one group used for validation. **Out-of-fold (OOF) predictions**
  come from models fitted without the predicted patient's fold. Pooling all folds
  gives broader development evidence than inspecting just one fold.
- **Inner validation** is a separate patient group inside the training portion.
  It helps compare settings or choose a saved model. Results on that same group
  remain development diagnostics because those decisions used its labels.
- **Reserved test** is held back until the whole workflow is fixed. The planned
  final-evaluation notebook has no test results yet.

## Reading model results

| Term | What to read | Why it matters |
|---|---|---|
| Prediction score | Higher means the model ranks an image as more likely malignant | A score between 0 and 1 is not automatically a reliable risk probability |
| Average precision (AP) | Higher means malignant images tend to appear earlier in the ranking | Useful with rare malignant cases; compare with the malignant fraction of the evaluated sample |
| ROC-AUC | Higher means stronger separation of malignant and benign scores; 0.5 is a chance-ranking reference | It measures ranking, rather than the percentage classified correctly |
| Sensitivity / recall | Fraction of malignant images flagged at a chosen cutoff | Shows how many labelled malignant images the classification rule captures |
| Specificity | Fraction of benign images left unflagged | Shows the other side of the same cutoff tradeoff |
| Calibration | Agreement between scores and observed malignant frequency | On a reliability diagram, below the diagonal means scores exceed observed frequency |
| Brier score / log loss | Lower means closer agreement between scores and labels | These assess the score scale as well as prediction errors |
| Patient-bootstrap interval | Range obtained by repeatedly resampling whole patients | Keeps a patient's images together when describing uncertainty in a result; it is not an individual risk interval |

## Repeated image predictions

A **checkpoint** stores a trained model and its recorded settings. An **epoch**
is one pass through the training data. **Augmentation** changes training images
to vary their appearance; those changes must preserve the lesion and its label.

A **logit** is the model's raw output before a sigmoid converts it to a score
between 0 and 1. **BatchNorm** standardises features using stored training
statistics during evaluation. **LayerNorm** standardises a selected set of
features within each example. These normalisation layers stay in evaluation mode
during the dropout experiment so the intended change comes from dropout alone.

**MC dropout** repeatedly predicts the same image while randomly disabling some
units in explicit dropout layers. **TTA**, or test-time augmentation, instead
predicts a fixed set of permitted image views. Their **standard deviations (SDs)**
measure different sources of prediction variation. A stable prediction can still
be wrong, and a larger SD does not mean a larger melanoma risk.

**Referral** sets aside a fraction of images for hypothetical review.
**Coverage** is the fraction retained by the model. Compare referral rankings at
the same fraction using the same underlying scores. Lower error in the retained
group does not measure what happens to referred images or prove that review helps.


## Selection, reporting and freezing

**Selection metric** chooses a procedure using inner-training validation groups. **Reporting metric** assesses its saved predictions on outer-held-out patients. They need not be identical: the preserved image procedure used AP for its epochs, while the current reports lead with ROC-AUC. Changing a displayed metric does not change historical training. The new metadata experiment explicitly uses ROC-AUC for its inner search; network duration still uses stopping binary cross-entropy loss.

**Freeze the procedure** means record the duplicate policy, patient splits, features, preprocessing, search grid and objective, seed averaging, stopping/refit rules, calibration and decision cutoff before final testing. Fit the chosen procedure using development data, then evaluate the reserved test once. If a rule changes after that result is seen, the test has entered development. No current notebook performs that final fit or test.
