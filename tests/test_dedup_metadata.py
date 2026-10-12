"""Evidence that clean replication cannot fit/select using heldout patients."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.dedup_metadata import (
    NN_SPECIFICATION, _prepare, _verify_predictions, fit_network,
    neural_preprocessor, predict_network, run_outer_fold, verify_saved_receipt,
)
from melanoma_detection.metadata_model import file_sha256


def synthetic_metadata():
    rows = []
    for fold in range(5):
        for patient in range(8):
            for lesion in range(2):
                rows.append({
                    "image_name": f"image_{fold}_{patient}_{lesion}",
                    "patient_id": f"patient_{fold}_{patient}", "fold": fold,
                    "target": int(patient < 2 and lesion == 0),
                    "age_approx": 20 + patient * 8 + fold,
                    "sex": "male" if patient % 2 else "female",
                    "anatom_site_general_challenge": "torso" if patient % 3 else "head/neck",
                    "diagnosis": "excluded outcome proxy", "arbitrary_leakage": int(patient < 2),
                })
    return pd.DataFrame(rows)


def quick_spec():
    spec = deepcopy(NN_SPECIFICATION)
    spec.update(max_epochs=3, min_epochs=1, patience=2)
    return spec


class DedupMetadataTests(unittest.TestCase):
    def test_outer_labels_do_not_change_nn_predictions_selection_or_calibration(self):
        data = synthetic_metadata()
        initial = run_outer_fold(data, 0, nn_spec=quick_spec())
        altered = data.copy()
        mask = altered.fold.eq(0)
        altered.loc[mask, "target"] = 1 - altered.loc[mask, "target"]
        changed = run_outer_fold(altered, 0, nn_spec=quick_spec())
        columns = list(initial["predictions"].filter(like="p_"))
        np.testing.assert_array_equal(initial["predictions"][columns], changed["predictions"][columns])
        calibration_columns = ["p_spline_raw", "p_spline_calibrated", "raw_threshold", "calibrated_threshold", "decision_raw", "decision_calibrated"]
        np.testing.assert_array_equal(initial["calibration"][calibration_columns], changed["calibration"][calibration_columns])
        pd.testing.assert_frame_equal(initial["epochs"], changed["epochs"])
        pd.testing.assert_frame_equal(initial["inner_audit"], changed["inner_audit"])
        pd.testing.assert_frame_equal(initial["selection_memberships"], changed["selection_memberships"])
        self.assertTrue(initial["inner_audit"].patient_overlap.eq(0).all())

    def test_outer_features_cannot_change_fitted_preprocessors_or_epoch_selection(self):
        data = synthetic_metadata()
        initial = run_outer_fold(data, 0, nn_spec=quick_spec())
        altered = data.copy()
        mask = altered.fold.eq(0)
        altered.loc[mask, "age_approx"] = 119
        altered.loc[mask, "anatom_site_general_challenge"] = "outer_only_category"
        changed = run_outer_fold(altered, 0, nn_spec=quick_spec())
        pd.testing.assert_frame_equal(initial["epochs"], changed["epochs"])
        for seed in NN_SPECIFICATION["base_seeds"]:
            left, right = initial["models"]["networks"][seed], changed["models"]["networks"][seed]
            np.testing.assert_array_equal(left["preprocessor"].named_transformers_["age"].named_steps["impute"].statistics_, right["preprocessor"].named_transformers_["age"].named_steps["impute"].statistics_)
            self.assertNotIn("outer_only_category", right["preprocessor"].named_transformers_["category"].categories_[1])
            for parameter in left["state_dict"]:
                self.assertTrue(torch.equal(left["state_dict"][parameter], right["state_dict"][parameter]))

    def test_training_reinitializes_and_refits_fixed_epochs(self):
        frame = _prepare(synthetic_metadata())
        fitting = frame.loc[~frame.fold.eq(0)]
        prep = neural_preprocessor().fit(fitting[["age_approx", "sex", "anatom_site_general_challenge"]])
        x = prep.transform(fitting)
        left, _, _, history = fit_network(x, fitting.target, 42, epochs=2, spec=quick_spec())
        # Deliberately alter an earlier fitted model; a fresh fit cannot reuse it.
        for parameter in left.parameters():
            with torch.no_grad():
                parameter.add_(100)
        right, _, _, second_history = fit_network(x, fitting.target, 42, epochs=2, spec=quick_spec())
        repeat, _, _, repeat_history = fit_network(x, fitting.target, 42, epochs=2, spec=quick_spec())
        self.assertEqual(history, second_history)
        self.assertEqual(second_history, repeat_history)
        self.assertEqual(len(history), 2)
        np.testing.assert_array_equal(predict_network(right, x), predict_network(repeat, x))

    def test_ids_targets_fold_diagnosis_and_unrequested_columns_never_become_features(self):
        frame = _prepare(synthetic_metadata())
        result = run_outer_fold(frame, 0, spline_calibration=False, nn_spec=quick_spec())
        for bundle in result["models"]["networks"].values():
            self.assertEqual(list(bundle["preprocessor"].feature_names_in_), ["age_approx", "sex", "anatom_site_general_challenge"])
        self.assertEqual(list(result["models"]["spline"].feature_names_in_), ["age_approx", "sex", "anatom_site_general_challenge"])
        self.assertEqual(len(result["epochs"]), 3)
        expected = result["predictions"][["p_nn_seed_42", "p_nn_seed_142", "p_nn_seed_242"]].mean(axis=1)
        np.testing.assert_array_equal(result["predictions"].p_nn_mean, expected)

    def test_coverage_rejects_missing_repeated_or_mismatched_ids(self):
        frame = _prepare(synthetic_metadata())
        expected = frame[["image_name", "patient_id", "target", "fold"]].copy()
        expected["p_nn_mean"] = .1
        # CSV round-trip may widen target dtype; content must still agree.
        expected.target = expected.target.astype("int64")
        _verify_predictions(frame, expected)
        for bad in (expected.iloc[:-1], pd.concat([expected.iloc[:-1], expected.iloc[:1]])):
            with self.assertRaisesRegex(ValueError, "coverage"):
                _verify_predictions(frame, bad)
        bad = expected.copy()
        bad.loc[0, "patient_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "do not match"):
            _verify_predictions(frame, bad)

    def test_resume_rejects_changed_spec_or_changed_output_and_escaping_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results/dedup_v1/metadata/fold_0/predictions.csv"
            output.parent.mkdir(parents=True)
            output.write_text("original", encoding="utf-8")
            receipt = {"status": "complete", "compatibility_sha256": "fixed", "output_sha256": {
                output.relative_to(root).as_posix(): file_sha256(output)}}
            verify_saved_receipt(root, receipt, compatibility_sha256="fixed")
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                verify_saved_receipt(root, receipt, compatibility_sha256="changed")
            output.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                verify_saved_receipt(root, receipt)
            receipt["output_sha256"] = {"results/historical.csv": "irrelevant"}
            with self.assertRaisesRegex(ValueError, "remain under"):
                verify_saved_receipt(root, receipt)


if __name__ == "__main__":
    unittest.main()
