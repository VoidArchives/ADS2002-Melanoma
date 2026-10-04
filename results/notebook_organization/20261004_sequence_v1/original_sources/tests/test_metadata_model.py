"""Check that outer labels cannot influence fitted metadata decisions."""

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.metadata_model import evaluate_outer_fold, file_sha256, fit_final_bundle, fit_recalibrator

script_spec = importlib.util.spec_from_file_location("metadata_holdout", ROOT / "scripts/evaluate_metadata_holdout.py")
holdout_module = importlib.util.module_from_spec(script_spec)
script_spec.loader.exec_module(holdout_module)


def synthetic_metadata():
    rows = []
    for fold in range(5):
        for person in range(8):
            for lesion in range(2):
                rows.append({
                    "image_name": f"image_{fold}_{person}_{lesion}",
                    "patient_id": f"patient_{fold}_{person}", "fold": fold,
                    "target": int(person < 2 and lesion == 0),
                    "age_approx": str(20 + person * 8 + fold),
                    "sex": "male" if person % 2 else "female",
                    "anatom_site_general_challenge": "torso" if person % 3 else "head/neck",
                })
    return pd.DataFrame(rows)


class MetadataProcedureTests(unittest.TestCase):
    def test_outer_targets_do_not_change_predictions_calibration_or_threshold(self):
        data = synthetic_metadata()
        initial, audit = evaluate_outer_fold(data, 0)
        altered = data.copy()
        mask = altered.fold.eq(0)
        altered.loc[mask, "target"] = 1 - altered.loc[mask, "target"]
        changed, changed_audit = evaluate_outer_fold(altered, 0)
        columns = ["p_spline_raw", "p_spline_calibrated", "raw_threshold", "calibrated_threshold", "decision_raw", "decision_calibrated"]
        np.testing.assert_array_equal(initial[columns], changed[columns])
        pd.testing.assert_frame_equal(audit, changed_audit)
        self.assertTrue(audit.patient_overlap.eq(0).all())

    def test_heldout_features_cannot_change_training_thresholds(self):
        data = synthetic_metadata()
        initial, _ = evaluate_outer_fold(data, 0)
        data.loc[data.fold.eq(0), "age_approx"] = "95"
        changed, _ = evaluate_outer_fold(data, 0)
        np.testing.assert_array_equal(initial[["raw_threshold", "calibrated_threshold"]], changed[["raw_threshold", "calibrated_threshold"]])

    def test_frozen_evaluation_rejects_patient_overlap(self):
        data = synthetic_metadata()
        bundle = fit_final_bundle(data, input_sha256="synthetic")
        with self.assertRaisesRegex(ValueError, "development patients"):
            holdout_module.evaluate_bundle(bundle, data.drop(columns="fold"))

    def test_frozen_evaluation_uses_stored_threshold(self):
        data = synthetic_metadata()
        bundle = fit_final_bundle(data, input_sha256="synthetic")
        heldout = data.drop(columns="fold").copy()
        heldout.image_name = "new_" + heldout.image_name
        heldout.patient_id = "new_" + heldout.patient_id
        predictions, metrics = holdout_module.evaluate_bundle(bundle, heldout)
        np.testing.assert_array_equal(predictions.decision, (predictions.probability >= bundle["threshold"]).astype(int))
        changed = heldout.copy()
        changed.target = 1 - changed.target
        changed_predictions, _ = holdout_module.evaluate_bundle(bundle, changed)
        np.testing.assert_array_equal(predictions.probability, changed_predictions.probability)

    def test_recalibration_cannot_reverse_ranking(self):
        with self.assertRaisesRegex(ValueError, "reverse or erase"):
            fit_recalibrator([1, 1, 0, 0], [.1, .2, .8, .9])

    def test_manifest_rejects_changed_code_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "model.joblib"
            source = root / "transform.py"
            bundle.write_bytes(b"fixed weights")
            source.write_text("original preprocessing", encoding="utf-8")
            manifest = {"bundle_sha256": file_sha256(bundle), "source_sha256": {"transform.py": file_sha256(source)}, "versions": {}}
            holdout_module.verify_manifest(bundle, manifest, root)
            source.write_text("changed preprocessing", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Source changed"):
                holdout_module.verify_manifest(bundle, manifest, root)
            bundle.write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "bundle differs"):
                holdout_module.verify_manifest(bundle, manifest, root)

    def test_manifest_rejects_changed_dependency_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "model.joblib"
            bundle.write_bytes(b"weights")
            manifest = {"bundle_sha256": file_sha256(bundle), "source_sha256": {}, "versions": {"numpy": "impossible-version"}}
            with self.assertRaisesRegex(ValueError, "Runtime differs"):
                holdout_module.verify_manifest(bundle, manifest, root)

    def test_cli_records_one_attempt_even_with_different_output_directory(self):
        data = synthetic_metadata()
        bundle = fit_final_bundle(data, input_sha256="synthetic")
        heldout = data.drop(columns="fold").copy()
        heldout.image_name = "new_" + heldout.image_name
        heldout.patient_id = "new_" + heldout.patient_id
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, data_path = root / "model.joblib", root / "heldout.csv"
            joblib.dump(bundle, model_path)
            manifest = {"bundle_sha256": file_sha256(model_path), "source_sha256": {}, "versions": {},
                        "input_sha256": "synthetic", "threshold": bundle["threshold"], "specification": bundle["specification"]}
            model_path.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
            heldout.to_csv(data_path, index=False)
            args = ["evaluate_metadata_holdout.py", "--bundle", str(model_path), "--data", str(data_path), "--output-dir", str(root / "first")]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                holdout_module.main()
            receipt = json.loads(model_path.with_suffix(".holdout_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "complete")
            args[-1] = str(root / "second")
            with patch.object(sys, "argv", args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                holdout_module.main()
            self.assertFalse((root / "second").exists())

    def test_cli_rejects_fractional_csv_label_at_original_precision(self):
        data = synthetic_metadata()
        bundle = fit_final_bundle(data, input_sha256="synthetic")
        heldout = data.drop(columns="fold").copy()
        heldout.image_name = "new_" + heldout.image_name
        heldout.patient_id = "new_" + heldout.patient_id
        heldout.target = heldout.target.astype(str)
        heldout.loc[0, "target"] = "1.000000000000000001"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, data_path = root / "model.joblib", root / "heldout.csv"
            joblib.dump(bundle, model_path)
            manifest = {"bundle_sha256": file_sha256(model_path), "source_sha256": {}, "versions": {},
                        "input_sha256": "synthetic", "threshold": bundle["threshold"], "specification": bundle["specification"]}
            model_path.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
            heldout.to_csv(data_path, index=False)
            args = ["evaluate_metadata_holdout.py", "--bundle", str(model_path), "--data", str(data_path), "--output-dir", str(root / "output")]
            with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
                holdout_module.main()
            receipt = json.loads(model_path.with_suffix(".holdout_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "failed")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
