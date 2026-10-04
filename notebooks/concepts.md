# Concepts used in the notebooks

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
