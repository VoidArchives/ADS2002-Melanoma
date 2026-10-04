"""Independent examples of uncertainty-input and paired-referral contracts."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from melanoma_detection.image_uncertainty_evaluation import (
    aggregate_raw, convergence_summary, evaluate_referral, predictive_entropy,
    referral_patient_bootstrap, validate_raw,
)


class RawPredictionValidationTests(unittest.TestCase):
    def setUp(self):
        self.expected = pd.DataFrame({"image_id": ["a", "b", "c"],
                                     "patient_id": ["p", "p", "q"], "target": [0, 1, 0]})
        self.raw = pd.concat([self.expected.assign(pass_id=pass_id, score=score)
                              for pass_id, score in [(1, [.1, .6, .2]), (2, [.3, .8, .2])]], ignore_index=True)

    def test_checks_before_aggregation_and_keeps_exact_expected_order(self):
        expected_before, raw_before = self.expected.copy(), self.raw.copy()
        checked = validate_raw(self.raw.iloc[::-1], self.expected, ["1", "2"])
        self.assertEqual(checked.image_id.tolist(), ["a", "a", "b", "b", "c", "c"])
        result = aggregate_raw(self.raw, self.expected, [1, 2])
        np.testing.assert_allclose(result.mean_score, [.2, .7, .2])
        np.testing.assert_allclose(result.score_sd, [np.sqrt(.02), np.sqrt(.02), 0])
        self.assertTrue(result.n_passes.eq(2).all())
        assert_frame_equal(self.expected, expected_before)
        assert_frame_equal(self.raw, raw_before)

    def test_rejects_every_invalid_score_before_mean_can_hide_it(self):
        for score in [np.nan, np.inf, -0.1, 1.1, .5 + 1j]:
            altered = self.raw.copy()
            altered["score"] = altered.score.astype(complex if isinstance(score, complex) else float)
            altered.loc[0, "score"] = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                aggregate_raw(altered, self.expected, [1, 2])

    def test_exact_keys_passes_image_coverage_and_metadata_are_required(self):
        invalid = [pd.concat([self.raw, self.raw.iloc[:1]], ignore_index=True),
                   self.raw.iloc[:-1], self.raw.loc[self.raw.image_id.ne("c")],
                   self.raw.assign(image_id=self.raw.image_id.replace({"a": "unknown"})),
                   self.raw.assign(pass_id=self.raw.pass_id.replace({2: 3})),
                   self.raw.assign(patient_id="wrong"),
                   self.raw.assign(target=1-self.raw.target),
                   self.raw.assign(target="1.00000000000000000001"),
                   self.raw.assign(image_id=" ")]
        for index, raw in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                aggregate_raw(raw, self.expected, [1, 2])
        for passes in ([1, "1"], [], [True, 2]):
            with self.assertRaises(ValueError):
                validate_raw(self.raw, self.expected, passes)
        with self.assertRaisesRegex(ValueError, "unique"):
            aggregate_raw(self.raw, pd.concat([self.expected, self.expected.iloc[:1]]), [1, 2])

    def test_string_tta_ids_and_single_deterministic_pass_are_supported(self):
        tta = self.raw.assign(pass_id=self.raw.pass_id.map({1: "identity", 2: "hflip"}))
        self.assertTrue(aggregate_raw(tta, self.expected, ["identity", "hflip"]).n_passes.eq(2).all())
        deterministic = self.raw.loc[self.raw.pass_id.eq(1)]
        self.assertTrue(aggregate_raw(deterministic, self.expected, [1]).score_sd.eq(0).all())

    def test_nested_convergence_prefix_uses_existing_passes_only(self):
        raw = pd.concat([self.expected.assign(pass_id=index, score=np.array([.1, .5, .3]) + index * .001)
                         for index in range(1, 31)], ignore_index=True)
        result = convergence_summary(raw, self.expected, range(1, 31))
        self.assertEqual(result.n_passes.tolist(), [5, 10, 20])
        expected_mae = abs(raw.loc[raw.pass_id.le(5)].groupby("image_id").score.mean()
                           - raw.groupby("image_id").score.mean()).mean()
        self.assertAlmostEqual(result.mean_score_mae.iloc[0], expected_mae)
        self.assertTrue(result.reference_passes.eq(30).all())
        with self.assertRaises(ValueError): convergence_summary(raw, self.expected, range(1, 31), [30])


class ReferralComparisonTests(unittest.TestCase):
    def frame(self):
        # Mixed patient groups, including two images for a and three for b.
        return pd.DataFrame({"image_id": list("abcdefg"), "patient_id": ["a", "a", "b", "b", "b", "c", "d"],
                             "target": [0, 1, 0, 0, 1, 0, 0], "base_score": [.2, .6, .2, .4, .7, .1, .3],
                             "mc_mean_score": [.2, .6, .1, .7, .7, .4, .1], "mc_score_sd": [.1, .4, .05, .3, .4, .2, .01],
                             "tta_mean_score": [.3, .6, .1, .6, .5, .4, .1], "tta_score_sd": [.1, .2, .05, .3, .2, .4, .01]})

    def test_entropy_boundaries_and_order(self):
        entropy = predictive_entropy([0, .1, .5, .9, 1])
        self.assertEqual(entropy[0], 0)
        self.assertEqual(entropy[-1], 0)
        self.assertAlmostEqual(entropy[2], np.log(2))
        self.assertAlmostEqual(entropy[1], entropy[3])

    def test_same_predictions_fractions_and_random_seeds_per_method(self):
        frame = self.frame()
        result = evaluate_referral(frame, fractions=[0, .4], n_random=3, seed=17)
        self.assertTrue(result.fixed_score_cutoff.eq(.5).all())
        self.assertTrue(result.loc[result.referral_fraction.eq(.4)].n_referred.eq(2).all())
        for method in ["mc", "tta"]:
            zero = result.loc[result.method.eq(method) & result.referral_fraction.eq(0)]
            self.assertEqual(zero.retained_error.nunique(), 1)
            self.assertAlmostEqual(zero.retained_error.iloc[0], ((frame.base_score >= .5) != frame.target).mean())
            self.assertTrue(zero.score_column.eq("base_score").all())
            self.assertEqual(set(zero.loc[zero.ranking.eq("random"), "random_seed"]), {17, 18, 19})
        random = result.loc[result.ranking.eq("random")].pivot(index=["random_seed", "referral_fraction"], columns="method", values="n_malignant_images_referred")
        np.testing.assert_array_equal(random.mc, random.tta)
        mc = result.loc[result.method.eq("mc") & result.ranking.eq("uncertainty_sd") & result.referral_fraction.eq(.4)].iloc[0]
        self.assertEqual(mc.n_malignant_images_referred, 2)
        self.assertEqual(mc.malignant_referred_fraction, 1)
        self.assertEqual(mc.n_malignant_patients_referred, 2)
        self.assertAlmostEqual(mc.retained_error, 0)
        self.assertAlmostEqual(mc.retained_brier, np.mean(np.array([.2, .2, .4, .1, .3])**2))
        self.assertAlmostEqual(mc.retained_score_mae, np.mean([.2, .2, .4, .1, .3]))

    def test_common_base_prediction_baselines_and_optional_mean_sensitivity(self):
        frame = self.frame()
        common = evaluate_referral(frame, fractions=[0, .4], n_random=2)
        baselines = common.loc[common.ranking.ne("uncertainty_sd")].drop(columns="method")
        assert_frame_equal(baselines.iloc[:len(baselines)//2].reset_index(drop=True),
                           baselines.iloc[len(baselines)//2:].reset_index(drop=True))
        means = evaluate_referral(frame, fractions=[0], n_random=2, score_source="mean")
        for method in ["mc", "tta"]:
            rows = means.loc[means.method.eq(method)]
            self.assertTrue(rows.score_column.eq(f"{method}_mean_score").all())
            self.assertAlmostEqual(rows.retained_error.iloc[0], ((frame[f"{method}_mean_score"] >= .5) != frame.target).mean())

    def test_labels_do_not_change_ranking_and_input_order_does_not_change_ties(self):
        frame = self.frame()
        original = evaluate_referral(frame, fractions=[.4], n_random=3)
        reordered = evaluate_referral(frame.iloc[::-1], fractions=[.4], n_random=3)
        assert_frame_equal(original, reordered)
        changed = evaluate_referral(frame.assign(target=1-frame.target), fractions=[.4], n_random=3)
        np.testing.assert_array_equal(original.n_referred, changed.n_referred)
        np.testing.assert_array_equal(original.n_patients_referred, changed.n_patients_referred)

    def test_bootstrap_matches_independent_explicit_patient_resamples(self):
        frame = self.frame()
        n_bootstrap, n_random, seed, fraction = 40, 3, 13, .4
        intervals = referral_patient_bootstrap(frame, fractions=[fraction], n_bootstrap=n_bootstrap, n_random=n_random, seed=seed)
        rng = np.random.default_rng(seed)
        groups = [np.flatnonzero(frame.patient_id.to_numpy() == patient) for patient in frame.patient_id.unique()]
        values = {method: {comparison: {metric: [] for metric in ["retained_error", "malignant_referred_fraction"]}
                          for comparison in ["prediction_score", "predictive_entropy", "random"]} for method in ["mc", "tta"]}
        for replicate in range(n_bootstrap):
            sampled = rng.integers(len(groups), size=len(groups))
            indices = np.concatenate([groups[index] for index in sampled])
            sample = frame.iloc[indices].sort_values("image_id", kind="stable").reset_index(drop=True)
            random_rng = np.random.default_rng(np.random.SeedSequence([seed, replicate, 104729]))
            orders = random_rng.permuted(np.tile(np.arange(len(sample)), (n_random, 1)), axis=1)
            n = int(np.floor(len(sample) * fraction))
            y = sample.target.to_numpy()
            for method in ["mc", "tta"]:
                scores = sample.base_score.to_numpy()
                errors = (scores >= .5) != y
                entropy = -(np.clip(scores, 1e-15, 1-1e-15)*np.log(np.clip(scores, 1e-15, 1-1e-15))
                            + (1-scores)*np.log(np.clip(1-scores, 1e-15, 1)))
                rankings = {"uncertainty_sd": np.argsort(-sample[f"{method}_score_sd"].to_numpy(), kind="stable"),
                            "prediction_score": np.argsort(-scores, kind="stable"),
                            "predictive_entropy": np.argsort(-entropy, kind="stable")}
                computed = {name: [errors[order[n:]].mean(), y[order[:n]].sum()/y.sum() if y.sum() else np.nan]
                            for name, order in rankings.items()}
                computed["random"] = np.mean([[errors[order[n:]].mean(), y[order[:n]].sum()/y.sum() if y.sum() else np.nan] for order in orders], axis=0)
                for comparator in ["prediction_score", "predictive_entropy", "random"]:
                    for metric_index, metric in enumerate(["retained_error", "malignant_referred_fraction"]):
                        values[method][comparator][metric].append(computed["uncertainty_sd"][metric_index] - computed[comparator][metric_index])
        for row in intervals.itertuples():
            draws = np.asarray(values[row.method][row.comparator][row.metric])
            valid = draws[np.isfinite(draws)]
            expected = np.quantile(valid, [.025, .975])
            np.testing.assert_allclose([row.ci_lower, row.ci_upper], expected, atol=1e-12)
            self.assertEqual(row.n_valid, len(valid))
            self.assertEqual(row.n_undefined, n_bootstrap-len(valid))
        repeated = referral_patient_bootstrap(frame, fractions=[fraction], n_bootstrap=n_bootstrap, n_random=n_random, seed=seed)
        assert_frame_equal(intervals, repeated)

    def test_zero_referral_and_identical_rankings_have_zero_paired_difference(self):
        frame = self.frame().assign(mc_score_sd=self.frame().base_score, tta_score_sd=self.frame().base_score)
        result = referral_patient_bootstrap(frame, fractions=[0, .4], n_bootstrap=25, n_random=3)
        identical = result.loc[result.comparator.eq("prediction_score")]
        self.assertTrue(identical[["estimate", "ci_lower", "ci_upper"]].eq(0).all().all())
        zero = result.loc[result.referral_fraction.eq(0)]
        self.assertTrue(zero[["estimate", "ci_lower", "ci_upper"]].eq(0).all().all())

    def test_bootstrap_point_estimates_match_curves_with_distinct_seeds(self):
        frame = self.frame()
        curves = evaluate_referral(frame, fractions=[.4], n_random=7, seed=17042)
        intervals = referral_patient_bootstrap(frame, fractions=[.4], n_bootstrap=10,
                                              n_random=7, seed=20242, referral_seed=17042)
        for row in intervals.itertuples():
            method_rows = curves.loc[curves.method.eq(row.method)]
            candidate = method_rows.loc[method_rows.ranking.eq("uncertainty_sd"), row.metric].iloc[0]
            reference = method_rows.loc[method_rows.ranking.eq(row.comparator), row.metric].mean()
            self.assertAlmostEqual(row.estimate, candidate-reference)
            self.assertEqual(row.bootstrap_seed, 20242)
            self.assertEqual(row.referral_seed, 17042)

    def test_no_malignant_denominator_stays_undefined_but_error_is_valid(self):
        frame = self.frame().assign(target=0)
        curves = evaluate_referral(frame, fractions=[.4], n_random=2)
        self.assertTrue(curves.malignant_referred_fraction.isna().all())
        self.assertTrue(curves.malignant_patients_referred_fraction.isna().all())
        self.assertTrue(np.isfinite(curves.retained_error).all())
        result = referral_patient_bootstrap(frame, fractions=[.4], n_bootstrap=10, n_random=2)
        malignant = result.loc[result.metric.eq("malignant_referred_fraction")]
        self.assertTrue(malignant.n_valid.eq(0).all())
        self.assertTrue(malignant[["estimate", "ci_lower", "ci_upper"]].isna().all().all())
        self.assertTrue(result.loc[result.metric.eq("retained_error"), "n_valid"].eq(10).all())

    def test_invalid_summary_and_configuration_fail_explicitly(self):
        frame = self.frame()
        for invalid in [frame.assign(mc_score_sd=np.nan), frame.assign(tta_score_sd=-1),
                        frame.assign(mc_mean_score=1.1), frame.assign(target=.5),
                        pd.concat([frame, frame.iloc[:1]])]:
            with self.assertRaises(ValueError): evaluate_referral(invalid)
        for kwargs in [{"fractions": [1]}, {"fractions": [np.nan]}, {"fractions": [.1, .1]}, {"n_random": True}, {"seed": -1}, {"score_source": "bad"}]:
            with self.assertRaises(ValueError): evaluate_referral(frame, **kwargs)
        with self.assertRaises(ValueError): referral_patient_bootstrap(frame, n_bootstrap=0)
        with self.assertRaisesRegex(ValueError, "two"):
            referral_patient_bootstrap(frame.assign(patient_id="same"))


if __name__ == "__main__":
    unittest.main()
