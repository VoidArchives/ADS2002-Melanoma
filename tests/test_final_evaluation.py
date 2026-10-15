import unittest

import numpy as np
import pandas as pd

from melanoma_detection.final_evaluation import (
    align_test_predictions, patient_bootstrap_report, validate_test_cohort,
)


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({"image_name": [f"i{i}" for i in range(8)],
            "patient_id": np.repeat(["p1", "p2", "p3", "p4"], 2),
            "target": [0, 0, 0, 1, 1, 1, 0, 1],
            "score": [.01, .2, .3, .8, .9, .7, .4, .6]})

    def test_test_patients_must_be_untouched(self):
        with self.assertRaisesRegex(ValueError, "Patient leakage"):
            validate_test_cohort(self.rows, self.rows)
        development = self.rows.assign(image_name=lambda x: "dev_" + x.image_name,
                                       patient_id=lambda x: "dev_" + x.patient_id)
        validate_test_cohort(self.rows, development)

    def test_alignment_and_corruption(self):
        aligned = align_test_predictions(self.rows.iloc[::-1], self.rows, ["score"])
        np.testing.assert_array_equal(aligned.score, self.rows.score)
        for invalid in (self.rows.iloc[:-1], self.rows.assign(target=1-self.rows.target),
                        self.rows.assign(score=np.nan), self.rows.assign(score=2)):
            with self.assertRaises(ValueError):
                align_test_predictions(invalid, self.rows, ["score"])

    def test_reproducible_patient_intervals_and_fixed_thresholds(self):
        thresholds = {"half": .5, "high": 1.0}
        first = patient_bootstrap_report(self.rows, thresholds, n_boot=30)
        second = patient_bootstrap_report(self.rows, thresholds, n_boot=30)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first.patients.eq(4).all())
        self.assertTrue(first.ROC_AUC.eq(1).all())
        self.assertEqual(first.set_index("policy").loc["high", "TP"], 0)
        self.assertEqual(first.set_index("policy").loc["high", "Precision_n_valid"], 0)
        for metric in ("ROC_AUC", "AP", "Sensitivity", "Specificity", "Balanced accuracy"):
            self.assertTrue(first[f"{metric}_lower"].between(0, 1).all())
            self.assertTrue(first[f"{metric}_upper"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
