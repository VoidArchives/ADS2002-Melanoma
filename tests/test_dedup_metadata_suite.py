"""Meaningful evidence for clean metadata fitting, frozen groups and safe reuse."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.dedup_metadata_suite import (
    FEATURES, GRIDS, KEYS, categorical_frame, inner_splits_from_memberships,
    metrics_table, model_setups, reconstruct_inner_memberships, run_outer_fold,
    run_suite, verify_reuse, verify_suite_receipt,
)
from melanoma_detection.development_dataset import load_development_dataset
from melanoma_detection.metadata_model import file_sha256


def synthetic():
    rows = []
    for fold in range(5):
        for patient in range(12):
            for lesion in range(2):
                rows.append({"image_name": f"image_{fold}_{patient:02d}_{lesion}",
                             "patient_id": f"patient_{fold}_{patient:02d}", "fold": fold,
                             "target": int(patient < 3 and lesion == 0),
                             "age_approx": str(20 + patient * 6 + fold),
                             "sex": "male" if patient % 2 else "female",
                             "anatom_site_general_challenge": "torso" if patient % 3 else "head/neck",
                             "diagnosis": "outcome proxy must never be used"})
    # Deliberately non-lexicographic CSV order tests reconstruction ordering.
    return pd.DataFrame(rows).sample(frac=1, random_state=19).reset_index(drop=True)


def reused_scores(frame):
    result = frame[KEYS].copy()
    for column in ("p_spline", "p_nn_seed_42", "p_nn_seed_142", "p_nn_seed_242", "p_nn_mean"):
        result[column] = .08
    return result


def quick_models():
    result = model_setups(threads=1)
    result["random_forest"].set_params(classifier__n_estimators=5)
    result["xgboost"].set_params(classifier__n_estimators=5)
    return result


def quick_grids():
    return {"spline": {"classifier__C": [.1, 1.]},
            "random_forest": {"classifier__max_depth": [2, 4]},
            "xgboost": {"classifier__max_depth": [2, 3]}}


class MetadataSuiteTests(unittest.TestCase):
    def test_original_csv_order_and_seed_rule_precede_filtering(self):
        original = synthetic()
        retained = set(original.loc[original.image_name.str.endswith("_0"), "image_name"])
        full, filtered = reconstruct_inner_memberships(original, retained)
        for fold in range(5):
            pool = original.loc[original.fold.ne(fold)].reset_index(drop=True)
            expected = np.full(len(pool), -1)
            for inner, (_, positions) in enumerate(StratifiedGroupKFold(
                    n_splits=3, shuffle=True, random_state=42 + fold).split(pool[FEATURES], pool.target, pool.patient_id)):
                expected[positions] = inner
            saved = full.loc[full.outer_fold.eq(fold)]
            np.testing.assert_array_equal(saved.image_name, pool.image_name)
            np.testing.assert_array_equal(saved.inner_fold, expected)
        expected_filtered = full.loc[full.image_name.isin(retained)].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_filtered, filtered)
        self.assertTrue(filtered.groupby(["outer_fold", "patient_id"]).inner_fold.nunique().eq(1).all())

    def test_frozen_memberships_need_no_splitter_and_keep_patients_disjoint(self):
        frame = synthetic()
        _, memberships = reconstruct_inner_memberships(frame, set(frame.image_name))
        with patch("melanoma_detection.dedup_metadata_suite.StratifiedGroupKFold", side_effect=AssertionError("Cannot resplit retained rows")):
            splits, audit = inner_splits_from_memberships(frame, 0, memberships)
        self.assertEqual(len(splits), 3)
        self.assertTrue(audit.patient_overlap.eq(0).all())
        self.assertTrue(audit.outer_patient_overlap.eq(0).all())
        self.assertEqual(sum(len(validation) for _, validation in splits), len(frame.loc[frame.fold.ne(0)]))

    def test_bad_membership_coverage_and_patient_boundary_are_rejected(self):
        frame = synthetic()
        _, memberships = reconstruct_inner_memberships(frame, set(frame.image_name))
        with self.assertRaisesRegex(ValueError, "coverage"):
            inner_splits_from_memberships(frame, 0, memberships.iloc[1:])
        bad = memberships.copy()
        row = bad.index[bad.outer_fold.eq(0)][0]
        bad.loc[row, "inner_fold"] = (bad.loc[row, "inner_fold"] + 1) % 3
        with self.assertRaisesRegex(ValueError, "patient"):
            inner_splits_from_memberships(frame, 0, bad)

    def test_outer_labels_cannot_change_predictions_or_selected_parameters(self):
        frame = synthetic()
        _, memberships = reconstruct_inner_memberships(frame, set(frame.image_name))
        initial = run_outer_fold(frame, 0, memberships, reused_scores(frame),
                                 threads=1, grids=quick_grids(), setups=quick_models())
        changed_frame = frame.copy()
        changed_frame.loc[changed_frame.fold.eq(0), "target"] = 1 - changed_frame.loc[changed_frame.fold.eq(0), "target"]
        changed = run_outer_fold(changed_frame, 0, memberships, reused_scores(frame),
                                 threads=1, grids=quick_grids(), setups=quick_models())
        np.testing.assert_array_equal(initial["predictions"].filter(like="p_"), changed["predictions"].filter(like="p_"))
        pd.testing.assert_frame_equal(initial["selected_parameters"], changed["selected_parameters"])
        self.assertEqual(int(initial["fit_log"].fits.sum()), 31)
        self.assertTrue(set(initial["models"]["fit_patient_ids"]).isdisjoint(frame.loc[frame.fold.eq(0), "patient_id"]))

    def test_outer_features_never_enter_preprocessing_fit(self):
        frame = synthetic()
        _, memberships = reconstruct_inner_memberships(frame, set(frame.image_name))
        initial = run_outer_fold(frame, 0, memberships, reused_scores(frame),
                                 threads=1, grids=quick_grids(), setups=quick_models())
        changed_frame = frame.copy()
        changed_frame.loc[changed_frame.fold.eq(0), "age_approx"] = "119"
        changed_frame.loc[changed_frame.fold.eq(0), "anatom_site_general_challenge"] = "outer_only_category"
        changed = run_outer_fold(changed_frame, 0, memberships, reused_scores(frame),
                                 threads=1, grids=quick_grids(), setups=quick_models())
        pd.testing.assert_frame_equal(initial["selected_parameters"], changed["selected_parameters"])
        for name in ("spline_tuned", "ablation_age", "ablation_age_site"):
            left = initial["models"]["estimators"][name].named_steps["preprocessor"].named_transformers_["smooth_age"]
            right = changed["models"]["estimators"][name].named_steps["preprocessor"].named_transformers_["smooth_age"]
            np.testing.assert_array_equal(left.named_steps["imputer"].statistics_, right.named_steps["imputer"].statistics_)
            np.testing.assert_array_equal(left.named_steps["scaler"].mean_, right.named_steps["scaler"].mean_)
        for name in ("categorical_logistic", "random_forest_tuned", "xgboost_tuned"):
            encoder = changed["models"]["estimators"][name].named_steps["preprocessor"].named_transformers_["categorical"]
            self.assertNotIn("outer_only_category", encoder.categories_[2])
            self.assertEqual(list(changed["models"]["estimators"][name].feature_names_in_), FEATURES)

    def test_historical_candidate_counts_and_fitting_resource_setting(self):
        self.assertEqual({name: len(list(ParameterGrid(grid))) for name, grid in GRIDS.items()},
                         {"spline": 18, "random_forest": 12, "xgboost": 12})
        models = model_setups(threads=4)
        self.assertEqual(models["random_forest"].get_params()["classifier__n_estimators"], 300)
        self.assertEqual(models["xgboost"].get_params()["classifier__n_estimators"], 200)
        self.assertEqual(models["random_forest"].get_params()["classifier__n_jobs"], 4)
        self.assertEqual(5 * (4 + 6 + 3 * (18 + 12 + 12) + 3), 695)

    def test_undefined_subgroup_metrics_are_missing_and_mean_fold_is_unweighted(self):
        frame = synthetic()
        predictions = frame[KEYS].copy()
        predictions["p_dummy"] = .04
        table = metrics_table(frame, predictions)
        mean = table.loc[table.scope.eq("mean_fold"), "average_precision"].iloc[0]
        self.assertAlmostEqual(mean, table.loc[table.scope.eq("fold"), "average_precision"].mean())
        self.assertEqual(len(table), 7)

    def test_receipt_rejects_hash_drift_and_escaping_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results/dedup_v1/metadata_suite/run/predictions.csv"
            output.parent.mkdir(parents=True)
            output.write_text("original")
            receipt = {"status": "complete", "compatibility_sha256": "fixed",
                       "output_sha256": {output.relative_to(root).as_posix(): file_sha256(output)}}
            verify_suite_receipt(root, receipt, compatibility_sha256="fixed")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_suite_receipt(root, receipt, compatibility_sha256="different")
            output.write_text("changed")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                verify_suite_receipt(root, receipt)
            receipt["output_sha256"] = {"results/historical.csv": "invalid"}
            with self.assertRaisesRegex(ValueError, "remain within"):
                verify_suite_receipt(root, receipt)

    def test_verified_existing_clean_receipt_and_wrong_identity_rejection(self):
        # Active 06 is now a report. Historical reading binds the original
        # fitting source to its archive; the legacy training runner remains strict.
        from melanoma_detection.historical_metadata_artifacts import verify_historical_reuse
        manifest = "data/subset/processed/dedup_v1/dataset_manifest.json"
        receipt = ROOT / "results/dedup_v1/metadata/comparator_receipt.json"
        if not (ROOT / manifest).exists() or not receipt.exists():
            self.skipTest("Integration artifacts are not supplied in this checkout.")
        frame, identity = load_development_dataset(ROOT, manifest)
        reused, predictions, calibration = verify_historical_reuse(ROOT, frame, identity, receipt)
        self.assertEqual(len(predictions), 6706)
        self.assertEqual(reused["rows"], 6706)
        np.testing.assert_array_equal(predictions.p_spline, calibration.p_spline_raw)
        changed = deepcopy(identity)
        changed["dataset_manifest_sha256"] = "invalid"
        with self.assertRaisesRegex(ValueError, "different development dataset"):
            verify_historical_reuse(ROOT, frame, changed, receipt)

    def test_missing_manifest_fails_before_any_dataset_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text("run_id: unsafe\n")
            with patch("melanoma_detection.dedup_metadata_suite.load_development_dataset", side_effect=AssertionError("No dataset read")):
                with self.assertRaisesRegex(ValueError, "explicit frozen"):
                    run_suite(root, config)


if __name__ == "__main__":
    unittest.main()
