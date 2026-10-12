"""Coverage, provenance and paired patient draws for the clean report layer."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.dedup_workflow import (MANIFEST, CORE_RECEIPT, _verify_hashes, build_evaluation, json_safe,
    load_context, load_image_report, primary_bootstrap, project_path)
from melanoma_detection.development_dataset import sha256_file
from melanoma_detection.dedup_sensitivity import align_predictions, metric_tables, subgroup_table
from melanoma_detection.historical_reporting_source import ARCHIVE, ARCHIVE_RECEIPT, REPORT_SOURCE


def toy_frame():
    rows = []
    for fold in range(5):
        for patient in range(6):
            for image in range(1 + patient % 2):
                rows.append({"image_id": f"f{fold}p{patient}i{image}", "patient_id": f"p{fold}_{patient}",
                    "fold": fold, "target": int(patient >= 3), "age_approx": 30 + patient,
                    "sex": "male", "anatom_site_general_challenge": "unknown" if patient < 3 else "torso"})
    return pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)


def recovered_reporting_fixture(root):
    """Copy the genuine recovered source and its independent historical bindings."""
    archive = root / ARCHIVE
    binding = json.loads((ROOT / ARCHIVE / ARCHIVE_RECEIPT).read_text(encoding="utf-8"))
    for relative in (binding["snapshot"], binding["historical_receipt"],
                     f"{ARCHIVE}/{ARCHIVE_RECEIPT}", f"{ARCHIVE}/preservation_receipt.json"):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    active = root / REPORT_SOURCE
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_bytes(b"# changed report-only implementation\n")
    output = root / "results/dedup_v1/report.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("unchanged historical report\n", encoding="utf-8")
    receipt = {"artifact_sha256": {output.relative_to(root).as_posix(): sha256_file(output)},
               "source_sha256": {REPORT_SOURCE: binding["source_sha256"]}}
    return archive, binding, receipt


class CleanWorkflowTests(unittest.TestCase):
    def test_live_frozen_counts_and_fold_memberships(self):
        context = load_context(ROOT)
        self.assertEqual(len(context["frame"]), 6706)
        self.assertEqual(int(context["frame"].target.sum()), 127)
        self.assertEqual(context["frame"].patient_id.nunique(), 422)
        folds = context["fold_counts"]
        self.assertTrue(folds.patients_before.equals(folds.patients_after))
        self.assertEqual(folds.images_after.tolist(), [1344, 1366, 1328, 1371, 1297])
        self.assertEqual(int(folds.excluded_malignant.sum()), 1)

    def test_prediction_pairing_rejects_missing_extra_and_relabelled(self):
        frame = toy_frame()
        predictions = frame.assign(score=.2)
        with self.assertRaises(ValueError):
            align_predictions(predictions.iloc[1:], frame, ["score"])
        changed = predictions.copy()
        changed.loc[0, "target"] = 1
        with self.assertRaises(ValueError):
            align_predictions(changed, frame, ["score"])
        changed = predictions.copy()
        changed.loc[0, "patient_id"] = "another_patient"
        with self.assertRaises(ValueError):
            align_predictions(changed, frame, ["score"])

    def test_primary_contrasts_pair_identical_predictions_exactly(self):
        frame = toy_frame()
        p = np.linspace(.1, .9, len(frame))
        intervals, comparisons = primary_bootstrap(frame,
            {"b0_raw": p, "spline_raw": p, "metadata_nn_mean": p}, replicates=80, seed=42)
        self.assertEqual(comparisons.comparison.nunique(), 3)
        self.assertTrue(np.allclose(comparisons[["difference", "ci_lower", "ci_upper"]], 0))
        self.assertEqual(intervals.seed.unique().tolist(), [42])
        first, second = primary_bootstrap(frame, {"spline_raw": p, "metadata_nn_mean": p}, replicates=80, seed=42)
        self.assertEqual(second.comparison.unique().tolist(), ["metadata_nn_mean minus spline_raw"])
        self.assertTrue(first.valid_replicates.le(80).all())

    def test_secondary_patient_weighting_matches_independent_metrics(self):
        frame = toy_frame()
        p = np.linspace(.1, .9, len(frame))
        metrics = metric_tables(frame, {"model": p}, "clean")
        selected = metrics[(metrics.aggregation == "pooled") & (metrics.weighting == "equal_patient")].iloc[0]
        weights = 1 / frame.groupby("patient_id").image_id.transform("size")
        self.assertAlmostEqual(selected.average_precision, average_precision_score(frame.target, p, sample_weight=weights))
        self.assertAlmostEqual(selected.roc_auc, roc_auc_score(frame.target, p, sample_weight=weights))

    def test_single_class_site_subgroup_auc_is_missing(self):
        frame = toy_frame()
        table = subgroup_table(frame, {"model": np.linspace(.1, .9, len(frame))}, "clean")
        unknown = table[(table.axis == "site_knownness") & (table.subgroup == "unknown")].iloc[0]
        self.assertTrue(pd.isna(unknown.roc_auc))
        self.assertEqual(unknown.n_malignant_patients, 0)

    def test_artifact_checks_fail_closed_and_reject_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "results/dedup_v1/output.csv"
            path.parent.mkdir(parents=True)
            path.write_text("verified\n")
            receipt = {"artifact_sha256": {path.relative_to(root).as_posix(): sha256_file(path)}}
            _verify_hashes(root, receipt)
            path.write_text("changed\n")
            with self.assertRaises(ValueError):
                _verify_hashes(root, receipt)
            with self.assertRaises(ValueError):
                project_path(root, "results/other.csv")

    def test_report_rejects_changed_upstream_input_even_if_its_own_csv_is_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results/dedup_v1/report.csv"
            output.parent.mkdir(parents=True)
            output.write_text("unchanged report\n")
            source = root / "results/image_d1_runs/clean/outer_oof_predictions.csv"
            source.parent.mkdir(parents=True)
            source.write_text("saved scores\n")
            receipt = {"artifact_sha256": {output.relative_to(root).as_posix(): sha256_file(output)},
                "input_sha256": {source.relative_to(root).as_posix(): sha256_file(source)}}
            _verify_hashes(root, receipt)
            source.write_text("changed scores\n")
            with self.assertRaisesRegex(ValueError, "Report input changed"):
                _verify_hashes(root, receipt)

    def test_exact_recovered_reporting_source_verifies_changed_active_self_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, binding, receipt = recovered_reporting_fixture(root)
            self.assertNotEqual(sha256_file(root / REPORT_SOURCE), binding["source_sha256"])
            self.assertEqual(sha256_file(root / binding["snapshot"]), binding["source_sha256"])
            _verify_hashes(root, receipt)

    def test_recovered_reporting_archive_rejects_tampered_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, binding, receipt = recovered_reporting_fixture(root)
            snapshot = root / binding["snapshot"]
            snapshot.write_bytes(snapshot.read_bytes() + b"\n# changed snapshot\n")
            with self.assertRaisesRegex(ValueError, "Recorded source changed"):
                _verify_hashes(root, receipt)

    def test_recovered_reporting_archive_rejects_misbound_source_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, original, receipt = recovered_reporting_fixture(root)
            for field, value in (("source_sha256", "0" * 64),
                                 ("historical_receipt_sha256", "0" * 64),
                                 ("snapshot", f"{ARCHIVE}/src/melanoma_detection/other.py")):
                with self.subTest(binding_field=field):
                    binding = {**original, field: value}
                    (archive / ARCHIVE_RECEIPT).write_text(json.dumps(binding), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Recorded source changed"):
                        _verify_hashes(root, receipt)

    def test_reporting_archive_cannot_authorize_an_arbitrary_changed_python_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, binding, receipt = recovered_reporting_fixture(root)
            relative = "src/melanoma_detection/arbitrary.py"
            active = root / relative
            original = b"# original arbitrary Python source\n"
            active.write_bytes(original)
            digest = sha256_file(active)
            active.write_bytes(b"# changed arbitrary Python source\n")
            snapshot = archive / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(original)
            # Even an internally consistent new binding chain must not widen
            # the reporting helper's single-module allowlist.
            historical = root / binding["historical_receipt"]
            historical.write_text(json.dumps({"source_sha256": {relative: digest}}), encoding="utf-8")
            historical_digest = sha256_file(historical)
            preserved = json.loads((archive / "preservation_receipt.json").read_text(encoding="utf-8"))
            preserved["protected_files"][binding["historical_receipt"]] = historical_digest
            (archive / "preservation_receipt.json").write_text(json.dumps(preserved), encoding="utf-8")
            binding.update(source=relative, source_sha256=digest,
                           snapshot=snapshot.relative_to(root).as_posix(),
                           historical_receipt_sha256=historical_digest)
            (archive / ARCHIVE_RECEIPT).write_text(json.dumps(binding), encoding="utf-8")
            receipt["source_sha256"] = {relative: digest}
            with self.assertRaisesRegex(ValueError, "Recorded source changed"):
                _verify_hashes(root, receipt)

    def test_training_complete_before_reporting_remains_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "results/image_d1_runs/b0_nested_dedup_v1/summary.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"status": "complete"}))
            with patch("melanoma_detection.dedup_workflow.load_saved_nested_run", side_effect=AssertionError("premature final reader")):
                bundle = load_image_report(root)
            self.assertFalse(bundle["complete"])
            self.assertEqual(bundle["status"], "training_complete_reporting_pending")

    def test_json_boolean_and_missing_values_keep_meaning(self):
        self.assertIs(json_safe(True), True)
        self.assertIsNone(json_safe(np.nan))

    def test_complete_report_exports_matching_calibrated_decisions_and_three_pairs(self):
        # Exercise the otherwise pending production image branch without GPU
        # inference, fitting, or modifying any real experiment.
        frame = toy_frame()
        raw = np.where(frame.target.eq(1), .8, .2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/melanoma_detection/dedup_workflow.py"
            source.parent.mkdir(parents=True)
            source.write_bytes((ROOT / "src/melanoma_detection/dedup_workflow.py").read_bytes())
            core = root / CORE_RECEIPT
            core.parent.mkdir(parents=True)
            meta = frame.rename(columns={"image_id": "image_name"}).assign(
                p_spline_raw=raw, p_spline_calibrated=raw, raw_threshold=.5,
                calibrated_threshold=.5, decision_raw=frame.target, decision_calibrated=frame.target)
            meta_path = core.parent / "metadata_calibrated_oof.csv"
            meta.to_csv(meta_path, index=False)
            core.write_text(json.dumps({"calibration_prediction_path": meta_path.relative_to(root).as_posix()}))
            image_dir = root / "results/image_d1_runs/b0_nested_calibration_dedup_v1"
            image_dir.mkdir(parents=True)
            (image_dir / "summary.json").write_text('{"status":"complete"}')
            identity = {"dataset_manifest_sha256": "frozen"}
            bundle = {"root": root, "identity": identity, "expected": frame,
                "scores": {"b0_raw": raw, "spline_raw": raw, "metadata_nn_mean": raw}, "inputs": {},
                "stages": {key: "complete" for key in ("metadata_core", "metadata_suite", "image_raw")}}
            image = frame.assign(base_probability=raw, calibrated_probability=raw,
                raw_threshold=.5, calibrated_threshold=.5, decision_raw=frame.target, decision_calibrated=frame.target)
            calibration = {"status": "complete", "complete": True, "run_dir": image_dir,
                "manifest": {"dataset_identity": identity}, "predictions": image}
            with patch("melanoma_detection.dedup_workflow.load_clean_scores", return_value=bundle), \
                 patch("melanoma_detection.dedup_workflow.load_saved_calibration_run", return_value=calibration), \
                 patch("melanoma_detection.dedup_workflow.__file__", str(source)):
                summary = build_evaluation(root, replicates=40)
            self.assertTrue(summary["comparison_complete"])
            output = root / summary["output_directory"]
            decisions = pd.read_csv(output / "training_only_cutoff_metrics.csv")
            self.assertEqual(set(decisions.model), {"b0", "fixed_spline"})
            self.assertTrue((decisions.FP == 0).all())
            paired = pd.read_csv(output / "paired_comparisons.csv")
            self.assertEqual(paired.comparison.nunique(), 3)
            self.assertTrue((paired.difference == 0).all())
            self.assertEqual(summary["status"], "complete")
            _verify_hashes(root, summary)


if __name__ == "__main__":
    unittest.main()
