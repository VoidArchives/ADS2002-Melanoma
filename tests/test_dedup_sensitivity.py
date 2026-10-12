"""Paired-population sensitivity must not silently alter coverage or sampling."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.dedup_sensitivity import (
    align_predictions, filter_saved_predictions, frozen_decision_table,
    metric_tables, paired_population_bootstrap, subgroup_table,
)
from melanoma_detection.image_nested_reporting import _cache, _weighted_metrics


def synthetic():
    records = []
    for fold in range(5):
        for patient in range(4):
            for image in range(3):
                records.append({"image_id": f"f{fold}p{patient}i{image}", "patient_id": f"f{fold}p{patient}",
                    "fold": fold, "target": int(patient < 2 and image == 1),
                    "score": [0.2, 0.7, 0.75][image]+0.01*fold,
                    "age_approx": 30+10*patient, "sex": "female" if patient%2 else "male",
                    "anatom_site_general_challenge": np.nan if patient == 3 else "torso"})
    return pd.DataFrame(records).sort_values("image_id").reset_index(drop=True)


class SensitivityTests(unittest.TestCase):
    def test_alignment_accepts_shuffle_and_rejects_coverage_and_identity_errors(self):
        frame = synthetic()
        pd.testing.assert_frame_equal(align_predictions(frame.iloc[::-1], frame, ["score"]), frame, check_dtype=False)
        for changed in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]),
                        frame.assign(patient_id=frame.patient_id+"x"), frame.assign(target=.5),
                        frame.assign(fold=.5), frame.assign(score=np.inf)):
            with self.subTest(shape=changed.shape), self.assertRaises(ValueError):
                align_predictions(changed, frame, ["score"])

    def test_filter_preserves_every_retained_score_and_rejects_changed_identity(self):
        frame = synthetic()
        retained = frame[~frame.image_id.str.endswith("i2")]
        actual = filter_saved_predictions(frame.iloc[::-1], frame, retained, ["score"])
        self.assertTrue(np.array_equal(actual.score, retained.score))
        with self.assertRaises(ValueError):
            filter_saved_predictions(frame, frame, retained.assign(target=1-retained.target), ["score"])

    def test_shared_patients_with_separate_row_masks_are_reproducible(self):
        frame = synthetic()
        ids = frame.loc[~frame.image_id.str.endswith("i2"), "image_id"]
        intervals, comparisons = paired_population_bootstrap(frame, ids,
            {"b0_raw": frame.score, "spline_raw": frame.score}, n_bootstrap=100, seed=42)
        same = comparisons[comparisons.kind == "model_comparison"]
        self.assertTrue(same[["difference", "ci_lower", "ci_upper"]].eq(0).all().all())
        changes = comparisons[(comparisons.kind == "retained_minus_original") &
                              (comparisons.metric == "average_precision")]
        self.assertTrue(changes.difference.gt(0).all())
        self.assertGreater(intervals.valid_replicates.iloc[0], 1)
        self.assertLess(intervals.valid_replicates.iloc[0], 100)
        self.assertTrue((intervals.valid_replicates+intervals.rejected_single_class_replicates == 100).all())
        again = paired_population_bootstrap(frame, ids,
            {"b0_raw": frame.score, "spline_raw": frame.score}, n_bootstrap=100, seed=42)
        pd.testing.assert_frame_equal(intervals, again[0])
        pd.testing.assert_frame_equal(comparisons, again[1])

    def test_same_population_has_zero_changes_and_identical_clean_fit_zero_effect(self):
        frame = synthetic()
        score = {"b0_raw": frame.score, "spline_raw": frame.score,
                 "metadata_nn_mean": frame.score}
        _, comparisons = paired_population_bootstrap(frame, frame.image_id, score,
            n_bootstrap=100, seed=42, clean_scores=score)
        self.assertTrue(comparisons[["difference", "ci_lower", "ci_upper"]].eq(0).all().all())

    def test_retention_cannot_remove_patient_or_outcome_and_unsorted_scores_rejected(self):
        frame = synthetic()
        for ids in (frame.loc[frame.target == 0, "image_id"],
                    frame.loc[frame.patient_id != "f0p0", "image_id"],
                    pd.concat([frame.image_id, frame.image_id.iloc[:1]])):
            with self.assertRaises(ValueError):
                paired_population_bootstrap(frame, ids, {"b0_raw": frame.score}, n_bootstrap=10)
        with self.assertRaises(ValueError):
            paired_population_bootstrap(frame.iloc[::-1], frame.image_id,
                {"b0_raw": frame.score}, n_bootstrap=10)

    def test_weighted_cache_matches_literal_patient_replication_and_sklearn(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        p = np.array([.2, .2, .7, .7, 0, 1])
        w = np.array([3, 3, 0, 0, 2, 2])
        repeated = np.repeat(np.arange(6), w)
        observed = _weighted_metrics(_cache(y, p), w)
        self.assertAlmostEqual(observed[0], average_precision_score(y[repeated], p[repeated]))
        self.assertAlmostEqual(observed[1], roc_auc_score(y[repeated], p[repeated]))
        fractional = np.array([.2, .2, .5, .5, 1, 1])
        observed = _weighted_metrics(_cache(y, p), fractional)
        self.assertAlmostEqual(observed[0], average_precision_score(y, p, sample_weight=fractional))
        self.assertAlmostEqual(observed[1], roc_auc_score(y, p, sample_weight=fractional))

    def test_patient_weighting_and_missing_subgroup_ranking(self):
        frame = synthetic()
        frame = frame[~((frame.patient_id == "f0p0") & (frame.image_id.str.endswith("i2")))].reset_index(drop=True)
        scores = {"b0_raw": frame.score.to_numpy()}
        table = metric_tables(frame, scores, "retained")
        pooled = table[(table.weighting == "equal_patient") & (table.aggregation == "pooled")].iloc[0]
        weights = 1/frame.groupby("patient_id").image_id.transform("size")
        self.assertAlmostEqual(pooled.average_precision, average_precision_score(frame.target, frame.score, sample_weight=weights))
        groups = subgroup_table(frame, scores, "retained")
        missing = groups[(groups.axis == "site_knownness") & (groups.subgroup == "unknown")].iloc[0]
        self.assertEqual(missing.n_malignant_images, 0)
        self.assertEqual(missing.n_malignant_patients, 0)
        self.assertTrue(np.isnan(missing.average_precision))
        self.assertTrue(np.isnan(missing.roc_auc))

    def test_frozen_decisions_reject_changed_cutoff_or_changed_decision(self):
        frame = synthetic().assign(raw_threshold=.6, calibrated_threshold=.6,
                                   calibrated_probability=lambda x: x.score)
        frame["decision_raw"] = (frame.score >= frame.raw_threshold).astype(int)
        frame["decision_calibrated"] = frame.decision_raw
        table = frozen_decision_table(frame, "score", "calibrated_probability", "b0", "retained")
        pooled = table[(table.fold == "pooled") & (table.score_scale == "raw")].iloc[0]
        self.assertEqual(pooled.n_flagged, frame.decision_raw.sum())
        self.assertAlmostEqual(pooled.recall, 1)
        for changed in (frame.assign(decision_raw=1-frame.decision_raw),
                        frame.assign(raw_threshold=np.linspace(.5, .7, len(frame)))):
            with self.assertRaises(ValueError):
                frozen_decision_table(changed, "score", "calibrated_probability", "b0", "retained")

    def test_cleaned_missing_sentinels_are_unknown_without_changing_features(self):
        frame = synthetic()
        variants = [np.nan, "Unknown", "  uNkNoWn ", "Missing", " missing ", "", "  ", "None", "NULL"]
        for value in variants:
            changed = frame.copy()
            changed["anatom_site_general_challenge"] = changed.anatom_site_general_challenge.fillna(value)
            changed["age_approx"] = changed.age_approx.astype(object)
            changed.loc[changed.patient_id.str.endswith("p3"), "age_approx"] = " Unknown "
            before = changed.copy(deep=True)
            groups = subgroup_table(changed, {"b0_raw": changed.score}, "retained")
            knownness = groups[groups.axis == "site_knownness"].set_index("subgroup")
            with self.subTest(value=value):
                self.assertEqual(knownness.loc["unknown", "n_images"], 15)
                self.assertEqual(knownness.loc["known", "n_images"], 45)
                self.assertEqual(knownness.loc["unknown", "n_malignant_images"], 0)
                self.assertTrue(np.isnan(knownness.loc["unknown", "average_precision"]))
                self.assertEqual(groups[(groups.axis == "age_group") & (groups.subgroup == "Missing")].iloc[0].n_images, 15)
                pd.testing.assert_frame_equal(changed, before)


if __name__ == "__main__":
    unittest.main()
