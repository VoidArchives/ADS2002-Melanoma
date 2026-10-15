import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

from melanoma_detection.splitting import (
    iter_patient_folds, load_splits, patient_level_split, save_splits,
)


class PatientFoldTests(unittest.TestCase):
    def setUp(self):
        self.patients = pd.DataFrame({
            "patient_id": [f"p{i}" for i in range(18)],
            "has_malignant": [0] * 12 + [1] * 6,
        })

    def test_rotating_validation_and_patient_isolation(self):
        folds = patient_level_split(self.patients)
        self.assertEqual(len(folds), 3)
        self.assertEqual(folds, patient_level_split(self.patients))
        validation_counts = Counter()
        images = self.patients.loc[self.patients.index.repeat(2)]
        for fold, rows in iter_patient_folds(images, folds):
            self.assertTrue(rows.groupby("patient_id")["split"].nunique().eq(1).all())
            mapping = folds[fold - 1]
            self.assertEqual(set(mapping), set(self.patients.patient_id))
            validation = self.patients[self.patients.patient_id.map(mapping).eq("validation")]
            self.assertEqual(len(validation), 6)
            self.assertEqual(validation.has_malignant.sum(), 2)
            validation_counts.update(validation.patient_id)
        self.assertEqual(validation_counts, Counter({p: 1 for p in self.patients.patient_id}))

    def test_invalid_inputs(self):
        for n in (1, 0, 2.5, True, 7):
            with self.assertRaises(ValueError):
                patient_level_split(self.patients, n_splits=n)
        for frame in (pd.concat([self.patients, self.patients.iloc[:1]]),
                      self.patients.assign(patient_id=None),
                      self.patients.assign(has_malignant=None)):
            with self.assertRaises(ValueError):
                patient_level_split(frame)

    def test_saved_folds_and_missing_assignments(self):
        folds = patient_level_split(self.patients)
        with tempfile.TemporaryDirectory() as directory:
            for i, mapping in enumerate(folds):
                path = save_splits(mapping, Path(directory) / f"fold_{i}.csv")
                self.assertEqual(load_splits(path), mapping)
        with self.assertRaises(ValueError):
            list(iter_patient_folds(self.patients, [{}]))


if __name__ == "__main__":
    unittest.main()
