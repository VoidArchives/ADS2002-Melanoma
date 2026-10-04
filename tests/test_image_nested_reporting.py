"""Coverage, clustered uncertainty and integrity checks for final image OOF reports."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.image_nested_reporting import (
    _cache, _weighted_metrics, build_evaluation, load_saved_nested_run,
    patient_fold_bootstrap, reliability_table, validate_outer_predictions,
)
from melanoma_detection.image_evaluation import sha256
from melanoma_detection.metadata_evaluation import probability_metrics
sys.path.insert(0, str(ROOT / "scripts"))
from finalize_image_nested import finalize


def synthetic_predictions():
    records = []
    for fold in range(5):
        for patient in range(4):
            for image in range(2):
                target = int(patient < 2 and image == 1)
                # Deliberately imperfect, tied scores and heterogeneous folds.
                probability = [.12, .6, .5, .2][patient] + .03 * fold + .07 * image
                records.append({"image_id": f"f{fold}p{patient}i{image}", "patient_id": f"f{fold}p{patient}",
                                "fold": fold, "target": target, "base_probability": probability})
    return pd.DataFrame(records).sort_values("image_id").reset_index(drop=True)


class NestedReportingTests(unittest.TestCase):
    def test_canonical_exact_coverage_accepts_order_and_rejects_identity_changes(self):
        frame = synthetic_predictions()
        expected = frame[ ["image_id", "patient_id", "fold", "target"] ].rename(columns={"image_id": "image_name"})
        actual = validate_outer_predictions(frame.iloc[::-1], expected)
        self.assertTrue(actual.image_id.equals(frame.image_id))
        for changed in [frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]),
                        frame.assign(patient_id="same_patient"), frame.assign(target=.5),
                        frame.assign(fold=1.5), frame.assign(base_probability=np.inf),
                        frame.assign(image_id=frame.image_id + " "), frame.assign(patient_id=frame.patient_id + "x"),
                        pd.concat([frame, frame.iloc[:1].assign(image_id="extra")])]:
            with self.subTest(changed=changed.shape):
                with self.assertRaises(ValueError):
                    validate_outer_predictions(changed, expected)

    def test_weighted_metrics_match_repeated_rows_with_ties_and_zero_weights(self):
        target = np.array([0, 1, 0, 1, 0, 1])
        probability = np.array([.2, .2, .6, .6, 0., 1.])
        weights = np.array([3, 2, 0, 1, 4, 2])
        repeated = np.repeat(np.arange(len(target)), weights)
        expected = list(probability_metrics(target[repeated], probability[repeated]).values())
        np.testing.assert_allclose(_weighted_metrics(_cache(target, probability), weights), expected, rtol=1e-12)

    def test_paired_bootstrap_identical_models_zero_differences_and_rejection_counts(self):
        frame = synthetic_predictions()
        intervals, comparisons = patient_fold_bootstrap(frame, {"image": frame.base_probability,
            "same": frame.base_probability}, n_bootstrap=100, seed=142)
        self.assertTrue((comparisons[["difference", "ci_lower", "ci_upper"]] == 0).all().all())
        valid = intervals.valid_replicates.iloc[0]
        self.assertGreater(valid, 1)
        self.assertLess(valid, 100)
        self.assertTrue((intervals.valid_replicates + intervals.rejected_single_class_replicates == 100).all())
        again, paired_again = patient_fold_bootstrap(frame, {"image": frame.base_probability,
            "same": frame.base_probability}, n_bootstrap=100, seed=142)
        pd.testing.assert_frame_equal(intervals, again)
        pd.testing.assert_frame_equal(comparisons, paired_again)

    def test_mean_fold_ap_is_distinct_from_pooled_ap(self):
        frame = synthetic_predictions()
        intervals, _ = patient_fold_bootstrap(frame, {"image": frame.base_probability}, n_bootstrap=100, seed=142)
        observed = intervals[intervals.metric == "average_precision"].set_index("aggregation").estimate
        expected_mean = np.mean([probability_metrics(rows.target, rows.base_probability)["average_precision"]
            for _, rows in frame.groupby("fold")])
        self.assertAlmostEqual(observed["mean_fold"], expected_mean)
        self.assertAlmostEqual(observed["pooled"], probability_metrics(frame.target, frame.base_probability)["average_precision"])
        self.assertNotAlmostEqual(observed["mean_fold"], observed["pooled"])

    def test_empty_reliability_bins_remain_missing_and_endpoint_scores_count(self):
        table = reliability_table([0, 1, 0], [0, 1, .52])
        self.assertEqual(table.n_images.sum(), 3)
        self.assertEqual(table.iloc[-1].n_images, 1)
        self.assertTrue(table[table.n_images == 0].observed_prevalence.isna().all())
        self.assertTrue(table[table.n_images == 0].mean_probability.isna().all())

    def test_incomplete_run_does_not_promote_partial_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run_dir = root / "results/image_d1_runs/run"
            run_dir.mkdir(parents=True)
            (run_dir / "run_manifest.json").write_text("{}")
            (run_dir / "summary.json").write_text(json.dumps({"status": "budget_exhausted", "completed_folds": [0]}))
            (run_dir / "outer_oof_predictions.csv").write_text("malformed partial CSV")
            loaded = load_saved_nested_run(root, "run")
            self.assertFalse(loaded["complete"])
            self.assertNotIn("predictions", loaded)
            self.assertEqual(loaded["status"], "budget_exhausted")

    def test_complete_report_roundtrip_independent_metrics_and_hash_tamper(self):
        frame = synthetic_predictions()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run_dir = root / "results/image_d1_runs/run"
            run_dir.mkdir(parents=True)
            canonical = root / "data/subset/processed/train_folds.csv"
            canonical.parent.mkdir(parents=True)
            frame.drop(columns="base_probability").rename(columns={"image_id": "image_name"}).to_csv(canonical, index=False)
            frame.to_csv(run_dir / "outer_oof_predictions.csv", index=False)
            manifest = {"development_csv": "data/subset/processed/train_folds.csv", "development_csv_sha256": sha256(canonical)}
            manifest_path = run_dir / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            manifest_before = manifest_path.read_bytes()
            evaluation = build_evaluation(frame, frame, run_dir, bootstrap_replicates=100, seed=142)
            self.assertEqual(manifest_path.read_bytes(), manifest_before)
            self.assertEqual(evaluation["outer_folds"], [0, 1, 2, 3, 4])
            self.assertEqual(evaluation["raw_threshold_05"]["threshold"], .5)
            self.assertFalse(evaluation["reserved_test_loaded"])
            summary = {"status": "complete", "outer_oof_predictions_sha256": sha256(run_dir / "outer_oof_predictions.csv"), "evaluation": evaluation}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            loaded = load_saved_nested_run(root, "run")
            self.assertTrue(loaded["complete"])
            self.assertEqual(len(loaded["predictions"]), len(frame))
            self.assertEqual(len(loaded["fold_metrics"]), 5)
            path = run_dir / "evaluation/fold_metrics.csv"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                load_saved_nested_run(root, "run")

    def test_finalizer_refuses_partial_run_without_writing_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run_dir = root / "results/image_d1_runs/run"
            run_dir.mkdir(parents=True)
            (run_dir / "run_manifest.json").write_text(json.dumps({"run_id": "run", "config": {"run": {"run_id": "run"}}}))
            summary = {"status": "incomplete", "run_id": "run", "completed_outer_folds": [0]}
            summary_path = run_dir / "summary.json"
            summary_path.write_text(json.dumps(summary))
            before = summary_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "five completed"):
                finalize(root, "run", include_metadata=False)
            self.assertEqual(summary_path.read_bytes(), before)
            self.assertFalse((run_dir / "evaluation").exists())

    def test_finalizer_complete_roundtrip_preserves_training_artifacts(self):
        from melanoma_detection.image_nested_reporting import fold_metric_table
        frame = synthetic_predictions().assign(run_id="run", model_name="efficientnet_b0", recipe_id="recipe", refit_epochs=6)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run_dir = root / "results/image_d1_runs/run"
            run_dir.mkdir(parents=True)
            canonical = root / "data/subset/processed/train_folds.csv"
            canonical.parent.mkdir(parents=True)
            frame[["image_id", "patient_id", "target", "fold"]].rename(columns={"image_id": "image_name"}).to_csv(canonical, index=False)
            frame.to_csv(run_dir / "outer_oof_predictions.csv", index=False)
            folds = fold_metric_table(frame).assign(status="complete")
            folds.to_csv(run_dir / "fold_metrics.csv", index=False)
            config = {"run": {"run_id": "run"}, "training": {"seed": 142},
                      "evaluation": {"reserved_test_allowed": False, "bootstrap_replicates": 100}}
            manifest = {"run_id": "run", "config": config, "development_csv": "data/subset/processed/train_folds.csv",
                        "development_csv_sha256": sha256(canonical)}
            manifest_path = run_dir / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            summary = {"run_id": "run", "status": "complete", "completed_outer_folds": list(range(5)),
                "folds": folds.to_dict("records"), "outer_oof_predictions_sha256": sha256(run_dir / "outer_oof_predictions.csv"),
                "fold_metrics_sha256": sha256(run_dir / "fold_metrics.csv"),
                "mean_fold_average_precision": float(folds.average_precision.mean())}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            protected = {path: sha256(path) for path in [manifest_path, canonical, run_dir / "outer_oof_predictions.csv", run_dir / "fold_metrics.csv"]}
            updated = finalize(root, "run", include_metadata=False)
            self.assertEqual(updated["evaluation"]["metadata_comparator_status"], "not requested")
            self.assertEqual(protected, {path: sha256(path) for path in protected})
            self.assertTrue(load_saved_nested_run(root, "run")["complete"])


if __name__ == "__main__":
    unittest.main()
