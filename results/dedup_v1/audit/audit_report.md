# Development dataset audit and exact-copy variant

The audit selects only the original development image IDs before converting source metadata. No reserved-test files were opened and no training occurred.

## Verified retention

158 exact pixel-identical pairs were confirmed against current JPEGs and historical byte/pixel fingerprints. The smallest image ID in each group is retained. 157 benign and 1 malignant copies are excluded through a manifest; original files remain intact. Retained: 6706 images, 127 malignant images, 422 patients and 88 malignant patients.

| Fold | Images original -> retained | Malignant original -> retained | Retained patients |
|---|---:|---:|---:|
| 0 | 1370 -> 1344 | 25 -> 24 | 84 |
| 1 | 1372 -> 1366 | 26 -> 26 | 84 |
| 2 | 1381 -> 1328 | 26 -> 26 | 87 |
| 3 | 1371 -> 1371 | 25 -> 25 | 82 |
| 4 | 1370 -> 1297 | 26 -> 26 | 85 |

Original outer folds and every saved inner patient assignment are preserved. Source metadata matches after the historical missing-value cleaning and numeric age formatting are accounted for. Duplicate groups agree on all labels, patient IDs, folds and metadata fields. Pixel equality is explicitly checked after fingerprint matching.

## Data-quality findings

- 204 patients have multiple recorded ages; 41 span at least 15 years. Values remain unchanged: capture dates are unavailable, so longitudinal differences cannot be resolved here.
- 196 images have missing site; fold/label and patient concentration are exported separately. Sites are not filled from other images belonging to the patient.
- Patients contribute 3 to 115 images; the ten largest patients supply 13.48% of original rows. Patient bootstrap handles dependence; equal-patient image weighting is a separate sensitivity diagnostic.
- 39 coarse average-hash groups contain distinct pixels. They are review candidates and remain in the dataset.
- Deterministic quality contact sheets show extremes of near-white content, low grayscale gradients and dark borders. These proxies do not automatically establish unusable images or exclude them.
- The historical weak rotation has nearest interpolation, black fill and no expansion. The fixed +20-degree preview illustrates corner fill and possible edge clipping. Replication retains the original transform; revisions belong to a separate experiment.

## Limits and next stage

Diagnosis `unknown` for benign images is allowed. Multiple patient ages/sites and coarse similarity are not duplicate rules. This audit cannot verify annotation truth, imaging dates or clinical diagnoses. Frozen-prediction filtering measures evaluation weighting only; clean retraining is needed to remove training influence. Existing uncertainty outputs describe historical checkpoints until regenerated.

The variant loader verifies canonical-source bytes, exported ledgers/metadata/memberships and retained-ID identity before any model or report can use this dataset.
