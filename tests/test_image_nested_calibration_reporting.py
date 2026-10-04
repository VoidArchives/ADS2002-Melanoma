"""Frozen-score pairing, monotonic ranking and threshold-scenario report checks."""
import sys
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.image_nested_calibration_reporting import (
    build_calibration_evaluation, load_saved_calibration_run, validate_calibration_predictions,
)
from melanoma_detection.image_nested_reporting import fold_metric_table
from melanoma_detection.metadata_evaluation import decision_metrics
from melanoma_detection.metadata_evaluation import select_recall_threshold
from melanoma_detection.image_evaluation import sha256
sys.path.insert(0, str(ROOT / "scripts"))
from finalize_image_nested_calibration import finalize


def fixtures():
    records = []
    for fold in range(5):
        for patient in range(4):
            for image in range(2):
                probability = [.12, .6, .5, .2][patient] + .02 * fold + .06 * image
                records.append({"image_id": f"f{fold}p{patient}i{image}", "patient_id": f"f{fold}p{patient}",
                    "fold": fold, "target": int(patient < 2 and image == 1), "base_probability": probability})
    raw = pd.DataFrame(records).sort_values("image_id").reset_index(drop=True)
    calibrated = raw.copy()
    # Different positive-slope maps across folds preserve within-fold ranks
    # while potentially changing cross-fold score ordering and pooled AP.
    calibrated["calibrated_probability"] = expit((.5 + .2 * raw.fold) * logit(raw.base_probability) - .7 + .4 * raw.fold)
    calibrated["raw_threshold"] = .3 + .04 * raw.fold
    calibrated["calibrated_threshold"] = expit((.5 + .2 * raw.fold) * logit(calibrated.raw_threshold) - .7 + .4 * raw.fold)
    calibrated["decision_raw"] = (calibrated.base_probability >= calibrated.raw_threshold).astype(int)
    calibrated["decision_calibrated"] = (calibrated.calibrated_probability >= calibrated.calibrated_threshold).astype(int)
    return raw, calibrated


class CalibrationReportingTests(unittest.TestCase):
    def test_positive_monotonic_maps_preserve_each_fold_ap_and_auc(self):
        raw, calibrated = fixtures()
        raw_metrics = fold_metric_table(raw)
        calibrated_metrics = fold_metric_table(calibrated, probability_column="calibrated_probability")
        np.testing.assert_allclose(raw_metrics.average_precision, calibrated_metrics.average_precision, rtol=0, atol=1e-12)
        np.testing.assert_allclose(raw_metrics.roc_auc, calibrated_metrics.roc_auc, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(calibrated.decision_raw, calibrated.decision_calibrated)

    def test_rejects_changed_raw_scores_partial_coverage_and_unfrozen_decisions(self):
        raw, calibrated = fixtures()
        self.assertEqual(len(validate_calibration_predictions(calibrated.iloc[::-1], raw, raw)), len(raw))
        variants = [calibrated.iloc[:-1], calibrated.assign(base_probability=calibrated.base_probability + .001),
            calibrated.assign(decision_raw=.5), calibrated.assign(decision_calibrated=1 - calibrated.decision_calibrated),
            calibrated.assign(raw_threshold=-.1), calibrated.assign(calibrated_probability=np.inf)]
        changed = calibrated.copy()
        changed.loc[0, "raw_threshold"] += .01
        variants.append(changed)
        for variant in variants:
            with self.assertRaises(ValueError):
                validate_calibration_predictions(variant, raw, raw)

    def test_report_records_observed_recall_and_preserves_reference_files(self):
        raw, calibrated = fixtures()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            sentinel = output / "reference_predictions.csv"
            raw.to_csv(sentinel, index=False)
            protected = sentinel.read_bytes()
            summary = build_calibration_evaluation(calibrated, raw, raw, output, bootstrap_replicates=100, seed=142)
            self.assertEqual(sentinel.read_bytes(), protected)
            self.assertTrue(summary["mean_fold_ap_invariant"])
            self.assertTrue(summary["fold_roc_auc_invariant"])
            self.assertIsNone(summary["ranking_warning"])
            self.assertFalse(summary["reserved_test_loaded"])
            pooled = pd.read_csv(output / "evaluation/threshold_scenario_pooled.csv")
            expected = decision_metrics(calibrated.target, calibrated.decision_raw)
            raw_record = pooled[pooled.model == "raw"].iloc[0]
            self.assertAlmostEqual(raw_record.recall, expected["sensitivity"])
            self.assertEqual(raw_record.training_target_recall, .8)
            self.assertNotEqual(raw_record.recall, .8)
            self.assertTrue(np.isnan(raw_record.threshold))  # Different cutoffs across folds.
            by_fold = pd.read_csv(output / "evaluation/threshold_scenario_by_fold.csv")
            self.assertEqual(len(by_fold), 10)
            self.assertEqual(len(summary["evaluation_artifact_sha256"]), 10)
            differences = pd.read_csv(output / "evaluation/paired_calibration_differences.csv")
            self.assertTrue(differences.comparison.eq("calibrated minus raw").all())

    def test_numerical_or_nonmonotonic_rank_change_is_visible(self):
        raw, calibrated = fixtures()
        calibrated["calibrated_probability"] = .5
        calibrated["calibrated_threshold"] = .5
        calibrated["decision_calibrated"] = 1
        with tempfile.TemporaryDirectory() as folder:
            summary = build_calibration_evaluation(calibrated, raw, raw, folder, bootstrap_replicates=100, seed=142)
            self.assertFalse(summary["mean_fold_ap_invariant"])
            self.assertIn("ranking metrics changed", summary["ranking_warning"])

    def test_missing_stage_stays_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            result = load_saved_calibration_run(folder)
            self.assertFalse(result["complete"])
            self.assertEqual(result["status"], "not_started")

    def test_complete_saved_stage_finalizes_without_altering_raw_outputs(self):
        raw, _ = fixtures()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            canonical = root / "data/subset/processed/train_folds.csv"
            canonical.parent.mkdir(parents=True)
            raw.drop(columns="base_probability").rename(columns={"image_id": "image_name"}).to_csv(canonical, index=False)
            raw_dir = root / "results/image_d1_runs/raw"
            cal_dir = root / "results/image_d1_runs/cal"
            raw_dir.mkdir(parents=True)
            cal_dir.mkdir(parents=True)
            raw.to_csv(raw_dir / "outer_oof_predictions.csv", index=False)
            raw_manifest = {"development_csv": "data/subset/processed/train_folds.csv", "development_csv_sha256": sha256(canonical),
                            "config": {"training": {"seed": 142}}}
            (raw_dir / "run_manifest.json").write_text(json.dumps(raw_manifest))
            (raw_dir / "summary.json").write_text(json.dumps({"status": "complete", "completed_outer_folds": list(range(5)),
                "outer_oof_predictions_sha256": sha256(raw_dir / "outer_oof_predictions.csv")}))
            records, predictions = [], []
            for fold in range(5):
                raw_folder, cal_folder = raw_dir / f"folds/{fold}", cal_dir / f"folds/{fold}"
                raw_folder.mkdir(parents=True)
                cal_folder.mkdir(parents=True)
                selection = {"recipe": {"id": "recipe"}, "refit_epochs": 6}
                (raw_folder / "selection.json").write_text(json.dumps(selection))
                selection_hash = sha256(raw_folder / "selection.json")
                training = raw[raw.fold != fold].copy().reset_index(drop=True)
                codes, _ = pd.factorize(training.patient_id, sort=True)
                training["inner_fold"] = codes % 3
                training.to_csv(cal_folder / "training_oof_predictions.csv", index=False)
                training_hash = sha256(cal_folder / "training_oof_predictions.csv")
                slope, intercept = .5 + .2 * fold, -.7 + .4 * fold
                train_scores = expit(slope * logit(training.base_probability) + intercept)
                raw_threshold = select_recall_threshold(training.target, training.base_probability, .8)
                calibrated_threshold = select_recall_threshold(training.target, train_scores, .8)
                calibrator = {"status": "complete", "slope": slope, "intercept": intercept, "logit_clip": 1e-7,
                    "C": 1, "target_recall": .8, "outer_labels_used": False, "training_oof_sha256": training_hash,
                    "training_selection_sha256": selection_hash, "fixed_epochs": 6, "recipe": selection["recipe"],
                    "raw_threshold": raw_threshold, "calibrated_threshold": calibrated_threshold}
                (cal_folder / "calibrator.json").write_text(json.dumps(calibrator))
                outer = raw[raw.fold == fold].copy()
                outer["calibrated_probability"] = expit(slope * logit(outer.base_probability) + intercept)
                outer["raw_threshold"], outer["calibrated_threshold"] = raw_threshold, calibrated_threshold
                outer["decision_raw"] = (outer.base_probability >= raw_threshold).astype(int)
                outer["decision_calibrated"] = (outer.calibrated_probability >= calibrated_threshold).astype(int)
                outer.to_csv(cal_folder / "outer_predictions.csv", index=False)
                predictions.append(outer)
                records.append({"fold": fold, "status": "complete", "slope": slope, "intercept": intercept,
                    "raw_threshold": raw_threshold, "calibrated_threshold": calibrated_threshold, "fixed_epochs": 6,
                    "training_oof_sha256": training_hash, "training_selection_sha256": selection_hash,
                    "calibrator_sha256": sha256(cal_folder / "calibrator.json"),
                    "predictions_sha256": sha256(cal_folder / "outer_predictions.csv")})
            pd.concat(predictions).sort_values("image_id").to_csv(cal_dir / "outer_oof_predictions.csv", index=False)
            manifest = {**raw_manifest, "config": {"evaluation": {"bootstrap_replicates": 100}}}
            (cal_dir / "run_manifest.json").write_text(json.dumps(manifest))
            summary = {"status": "complete", "reference_run_id": "raw", "completed_outer_folds": list(range(5)),
                "folds": records, "outer_oof_predictions_sha256": sha256(cal_dir / "outer_oof_predictions.csv")}
            (cal_dir / "summary.json").write_text(json.dumps(summary))
            pending = load_saved_calibration_run(root, "cal")
            self.assertFalse(pending["complete"])
            self.assertTrue(pending["ready_for_reporting"])
            self.assertEqual(pending["status"], "training_complete_reporting_pending")
            protected = {path: sha256(path) for path in raw_dir.rglob("*") if path.is_file()}
            result = finalize(root, "cal")
            self.assertTrue(result["evaluation"]["mean_fold_ap_invariant"])
            self.assertEqual(protected, {path: sha256(path) for path in protected})
            loaded = load_saved_calibration_run(root, "cal")
            self.assertTrue(loaded["complete"])
            self.assertEqual(len(loaded["predictions"]), len(raw))
            (cal_dir / "folds/0/calibrator.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed: calibrator"):
                load_saved_calibration_run(root, "cal")


if __name__ == "__main__":
    unittest.main()
