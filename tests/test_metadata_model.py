import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from sklearn.dummy import DummyClassifier

from melanoma_detection.metadata_model import (
    DEFAULT_METADATA_FEATURES,
    binary_probability_metrics,
    build_logistic_metadata_pipeline,
    generate_oof_probabilities,
)


class MetadataModelTests(unittest.TestCase):
    @staticmethod
    def make_folded_frame() -> pd.DataFrame:
        rows = []
        sites = ["torso", "head/neck", "lower extremity"]
        for patient_number in range(12):
            positive_patient = patient_number >= 6
            for image_number in range(2):
                rows.append(
                    {
                        "image_name": f"ISIC_{patient_number:03d}_{image_number}",
                        "patient_id": f"IP_{patient_number:03d}",
                        "sex": "female" if patient_number % 2 else "male",
                        "age_approx": str(30 + 5 * (patient_number % 4)),
                        "anatom_site_general_challenge": sites[patient_number % 3],
                        "target": int(positive_patient and image_number == 0),
                        "fold": patient_number % 3,
                    }
                )
        return pd.DataFrame(rows)

    def test_pipeline_produces_binary_probabilities(self):
        frame = self.make_folded_frame()
        pipeline = build_logistic_metadata_pipeline()

        pipeline.fit(frame[list(DEFAULT_METADATA_FEATURES)], frame["target"])
        probabilities = pipeline.predict_proba(frame[list(DEFAULT_METADATA_FEATURES)])[:, 1]

        self.assertEqual(probabilities.shape, (len(frame),))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_oof_predictions_are_aligned_stable_and_non_mutating(self):
        frame = self.make_folded_frame()
        original = frame.copy(deep=True)
        estimator = build_logistic_metadata_pipeline()

        first = generate_oof_probabilities(frame, estimator)
        second = generate_oof_probabilities(frame, estimator)

        self.assertTrue(frame.equals(original))
        self.assertEqual(first.name, "probability")
        self.assertEqual(len(first), len(frame))
        self.assertFalse(first.isna().any())
        self.assertTrue(first.between(0, 1).all())
        assert_series_equal(first, second)

    def test_oof_predictions_support_a_probability_dummy(self):
        frame = self.make_folded_frame()
        estimator = DummyClassifier(strategy="prior")

        probabilities = generate_oof_probabilities(frame, estimator)

        self.assertEqual(len(probabilities), len(frame))
        self.assertFalse(probabilities.isna().any())

    def test_oof_predictions_reject_patient_leakage(self):
        frame = self.make_folded_frame()
        patient_rows = frame.index[frame["patient_id"].eq("IP_000")]
        frame.loc[patient_rows[0], "fold"] = 1

        with self.assertRaisesRegex(ValueError, "patient appears in multiple folds"):
            generate_oof_probabilities(frame, build_logistic_metadata_pipeline())

    def test_oof_predictions_validate_columns_and_targets(self):
        missing_feature = self.make_folded_frame().drop(columns="sex")
        with self.assertRaisesRegex(ValueError, "Required columns are missing"):
            generate_oof_probabilities(
                missing_feature,
                build_logistic_metadata_pipeline(),
            )

        invalid_target = self.make_folded_frame()
        invalid_target.loc[0, "target"] = 2
        with self.assertRaisesRegex(ValueError, "binary values"):
            generate_oof_probabilities(
                invalid_target,
                build_logistic_metadata_pipeline(),
            )

    def test_binary_probability_metrics_match_known_values(self):
        metrics = binary_probability_metrics(
            targets=[0, 0, 1, 1],
            probabilities=[0.1, 0.2, 0.8, 0.9],
        )

        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.025)
        self.assertGreater(metrics["log_loss"], 0)


if __name__ == "__main__":
    unittest.main()
