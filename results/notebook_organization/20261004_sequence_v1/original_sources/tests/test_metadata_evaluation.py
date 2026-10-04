import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from melanoma_detection.metadata_evaluation import (
    calibration_bins,
    decision_metrics,
    patient_bootstrap,
    probability_metrics,
    select_recall_threshold,
    subgroup_metrics,
    validate_probabilities,
)


class MetadataPredictionChecksTests(unittest.TestCase):
    def test_rejects_fractional_targets_before_casting(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_probabilities([0, 1.5], [0.2, 0.8])

    def test_rejects_missing_out_of_range_or_misaligned_predictions(self):
        for p in ([0.2, np.nan], [0.2, np.inf], [-0.1, 0.2], [0.2, 1.01], [0.2], np.array([0.2, 0.8 + 1j])):
            with self.subTest(p=p), self.assertRaises(ValueError):
                validate_probabilities([0, 1], p)
        for y in ([], [[0, 1]], [0, np.nan]):
            with self.subTest(y=y), self.assertRaises(ValueError):
                validate_probabilities(y, [0.2, 0.8])

    def test_probability_metrics_match_independent_sklearn_calculations(self):
        y, p = np.array([0, 1, 0, 1, 0]), np.array([0, 0.6, 0.6, 1, 0.1])
        metrics = probability_metrics(y, p)
        self.assertAlmostEqual(metrics["average_precision"], average_precision_score(y, p))
        self.assertAlmostEqual(metrics["roc_auc"], roc_auc_score(y, p))
        self.assertAlmostEqual(metrics["log_loss"], log_loss(y, p, labels=[0, 1]))
        self.assertAlmostEqual(metrics["brier"], brier_score_loss(y, p))

    def test_missing_denominators_are_nan_not_zero_performance(self):
        metrics = probability_metrics([0, 0], [0.1, 0.2])
        self.assertTrue(np.isnan(metrics["average_precision"]))
        self.assertTrue(np.isnan(metrics["roc_auc"]))
        self.assertTrue(np.isfinite(metrics["log_loss"]))
        decisions = decision_metrics([0, 0], [0, 0])
        self.assertTrue(np.isnan(decisions["sensitivity"]))
        self.assertTrue(np.isnan(decisions["precision"]))
        self.assertEqual(decisions["specificity"], 1)
        self.assertEqual(decisions["flagged_fraction"], 0)
        self.assertTrue(np.isnan(decision_metrics([1, 1], [1, 0])["specificity"]))


class MetadataThresholdTests(unittest.TestCase):
    def test_highest_recall_threshold_keeps_all_tied_scores_together(self):
        y = np.array([1, 1, 1, 1, 0, 0])
        p = np.array([0.9, 0.7, 0.7, 0.1, 0.7, 0.2])
        threshold = select_recall_threshold(y, p, target_recall=0.5)
        self.assertEqual(threshold, 0.7)
        metrics = decision_metrics(y, p >= threshold)
        self.assertEqual(metrics["sensitivity"], 0.75)
        self.assertEqual(metrics["TP"], 3)
        self.assertEqual(metrics["FP"], 1)
        self.assertLess(decision_metrics(y, p >= np.nextafter(threshold, np.inf))["sensitivity"], 0.5)
        self.assertEqual(select_recall_threshold(y, p, 1), 0.1)

    def test_threshold_rejects_unachievable_or_invalid_recall(self):
        with self.assertRaisesRegex(ValueError, "malignant"):
            select_recall_threshold([0, 0], [0.1, 0.2])
        for value in (0, -0.1, 1.1, np.nan, True, "0.8", complex(0.8, 0), None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_recall_threshold([0, 1], [0.1, 0.2], value)

    def test_recall_boundary_does_not_require_an_extra_positive_due_to_rounding(self):
        p = np.arange(50) / 50
        threshold = select_recall_threshold(np.ones(50), p, target_recall=0.14)
        self.assertEqual(threshold, 0.86)
        self.assertEqual(decision_metrics(np.ones(50), p >= threshold)["sensitivity"], 0.14)


class PatientBootstrapTests(unittest.TestCase):
    def test_whole_patient_paired_intervals_match_explicit_repeated_rows(self):
        # Patients have different image counts, mixed targets, and tied scores.
        y = np.array([0, 1, 0, 0, 1, 0, 0])
        groups = np.array(["a", "a", "b", "b", "b", "c", "d"])
        candidate = np.array([0.2, 0.7, 0.1, 0.7, 0.7, 0.4, 0.1])
        reference = np.array([0.3, 0.6, 0.1, 0.6, 0.5, 0.4, 0.1])
        n_bootstrap, seed = 100, 13
        result = patient_bootstrap(y, groups, {"candidate": candidate, "reference": reference}, n_bootstrap, seed, [("candidate", "reference")])
        rng = np.random.default_rng(seed)
        patients = pd.unique(groups)
        explicit = {"candidate": [], "reference": []}
        for _ in range(n_bootstrap):
            sampled = rng.integers(len(patients), size=len(patients))
            indices = np.concatenate([np.flatnonzero(groups == patients[index]) for index in sampled])
            if np.unique(y[indices]).size < 2:
                continue
            for name, p in [("candidate", candidate), ("reference", reference)]:
                explicit[name].append([
                    average_precision_score(y[indices], p[indices]),
                    roc_auc_score(y[indices], p[indices]),
                    log_loss(y[indices], p[indices], labels=[0, 1]),
                    brier_score_loss(y[indices], p[indices]),
                ])
        names = ["average_precision", "roc_auc", "log_loss", "brier"]
        for row in result.itertuples():
            values = np.asarray(explicit[row.model_name])[:, names.index(row.metric)]
            if row.estimate_kind == "difference":
                values -= np.asarray(explicit[row.reference_model])[:, names.index(row.metric)]
            lower, upper = np.quantile(values, [0.025, 0.975])
            self.assertAlmostEqual(row.ci_lower, lower, places=12)
            self.assertAlmostEqual(row.ci_upper, upper, places=12)
            self.assertEqual(row.n_valid, len(values))
            self.assertEqual(row.n_rejected_single_class, n_bootstrap - len(values))

    def test_identical_predictions_have_exactly_zero_paired_difference(self):
        result = patient_bootstrap([0, 1, 0, 1], ["a", "b", "c", "d"], {"a": [0.2, 0.8, 0.1, 0.7], "b": [0.2, 0.8, 0.1, 0.7]}, n_bootstrap=100, comparisons=[("a", "b")])
        differences = result.query("estimate_kind == 'difference'")
        self.assertTrue(differences[["estimate", "ci_lower", "ci_upper"]].eq(0).all().all())
        repeated = patient_bootstrap([0, 1, 0, 1], ["a", "b", "c", "d"], {"a": [0.2, 0.8, 0.1, 0.7], "b": [0.2, 0.8, 0.1, 0.7]}, n_bootstrap=100, comparisons=[("a", "b")])
        assert_frame_equal(result, repeated)

    def test_all_single_class_draws_report_no_valid_interval(self):
        result = patient_bootstrap([0, 0, 0], ["a", "a", "b"], {"model": [0.1, 0.2, 0.3]}, n_bootstrap=10)
        self.assertTrue(result["n_valid"].eq(0).all())
        self.assertTrue(result["n_rejected_single_class"].eq(10).all())
        self.assertTrue(result[["ci_lower", "ci_upper"]].isna().all().all())
        self.assertTrue(np.isfinite(result.loc[result.metric == "brier", "estimate"]).all())

    def test_invalid_groups_comparisons_and_configuration_are_rejected(self):
        y, p = [0, 1], {"model": [0.1, 0.8]}
        for groups in (["a", "a"], ["a", None], ["a", " "], ["a"]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                patient_bootstrap(y, groups, p)
        with self.assertRaisesRegex(ValueError, "comparison"):
            patient_bootstrap(y, ["a", "b"], p, comparisons=[("missing", "model")])
        with self.assertRaisesRegex(ValueError, "positive integer"):
            patient_bootstrap(y, ["a", "b"], p, n_bootstrap=True)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            patient_bootstrap(y, ["a", "b"], p, seed=-1)


class CalibrationAndSubgroupTests(unittest.TestCase):
    def test_calibration_preserves_ties_and_reconciles_counts(self):
        y = [0, 1, 0, 0, 1, 0]
        p = [0.1, 0.1, 0.1, 0.6, 0.6, 0.9]
        result = calibration_bins(y, p, n_bins=5, groups=["a", "a", "b", "c", "d", "e"])
        self.assertEqual(result["n_images"].sum(), 6)
        self.assertEqual(result["n_malignant_images"].sum(), 2)
        for value in set(p):
            containing = (result.min_probability <= value) & (result.max_probability >= value)
            self.assertEqual(containing.sum(), 1)
        self.assertEqual(calibration_bins(y, [0.2] * 6, n_bins=5).shape[0], 1)
        self.assertEqual(result.iloc[0]["n_patients"], 2)

    def test_subgroups_use_image_targets_and_unique_patient_support(self):
        data = pd.DataFrame({
            "target": [1, 0, 0, 1], "patient_id": ["a", "a", "b", "c"],
            "age_approx": [40, 40, np.nan, 65], "sex": ["male", "male", None, "female"],
            "anatom_site_general": ["torso", "head/neck", None, "torso"],
        })
        result = subgroup_metrics(data, [0.8, 0.4, 0.2, 0.7], [1, 0, 0, 1])
        younger = result.query("attribute == 'age' and subgroup == 'Under 50'").iloc[0]
        self.assertEqual(younger["n_images"], 2)
        self.assertEqual(younger["n_patients"], 1)
        self.assertEqual(younger["n_malignant_images"], 1)
        self.assertEqual(younger["n_positive_patients"], 1)
        missing_age = result.query("attribute == 'age' and subgroup == 'Missing'").iloc[0]
        self.assertTrue(np.isnan(missing_age["sensitivity"]))
        self.assertTrue(np.isnan(missing_age["average_precision"]))
        challenge_data = data.rename(columns={"anatom_site_general": "anatom_site_general_challenge"})
        assert_frame_equal(result, subgroup_metrics(challenge_data, [0.8, 0.4, 0.2, 0.7], [1, 0, 0, 1]))


if __name__ == "__main__":
    unittest.main()
