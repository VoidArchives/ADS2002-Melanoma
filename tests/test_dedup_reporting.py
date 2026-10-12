"""Saved variant reports must retain complete, explicit comparator provenance."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from melanoma_detection.development_dataset import load_development_dataset
from melanoma_detection.dedup_audit import freeze_development_variant
from melanoma_detection.image_evaluation import sha256
from melanoma_detection.image_nested_reporting import fold_metric_table, load_matched_metadata, load_run_development
from finalize_image_nested import finalize
from test_dedup_audit import create_audit_fixture


class DedupReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original, _, counts = create_audit_fixture(self.root)
        freeze_development_variant(self.root, expected_counts=counts, progress=None)
        self.frame, self.identity = load_development_dataset(self.root,
            "data/subset/processed/dedup_v1/dataset_manifest.json")
        self.manifest = {"development_csv": "data/subset/processed/train_folds.csv",
            "development_csv_sha256": sha256(self.root / "data/subset/processed/train_folds.csv"),
            "dataset_identity": self.identity}
        self.predictions = self.frame[["image_name", "patient_id", "fold", "target"]].copy()
        # Imperfect fixed mock scores exercise exact pairing; no model is fitted.
        self.predictions["p_nn_mean"] = np.linspace(.01, .9, len(self.predictions))
        self.predictions["p_spline"] = np.linspace(.05, .7, len(self.predictions))
        self.directory = self.root / "results/dedup_v1/metadata"
        self.directory.mkdir(parents=True)
        self.prediction_path = self.directory / "metadata_nn_oof.csv"
        self.predictions.to_csv(self.prediction_path, index=False)
        sources = {}
        for name in ("dedup_metadata.py", "development_dataset.py", "metadata_model.py",
                     "metadata_validation.py", "metadata_evaluation.py"):
            relative = f"src/melanoma_detection/{name}"
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / relative).read_bytes())
            sources[relative] = sha256(path)
        self.receipt = {"status": "complete", "dataset_identity": self.identity,
            "input_path": "data/subset/processed/train_folds.csv",
            "input_sha256": self.manifest["development_csv_sha256"],
            "prediction_path": self.prediction_path.relative_to(self.root).as_posix(),
            "prediction_sha256": sha256(self.prediction_path), "source_sha256": sources}
        self.receipt_path = self.directory / "comparator_receipt.json"
        self.write_receipt()

    def write_receipt(self, receipt=None):
        self.receipt_path.write_text(json.dumps(self.receipt if receipt is None else receipt), encoding="utf-8")

    def load(self, expected=None, identity=None):
        return load_matched_metadata(self.root, self.frame if expected is None else expected,
            receipt_path=self.receipt_path.relative_to(self.root).as_posix(),
            dataset_identity=self.identity if identity is None else identity)

    def test_original_default_and_explicit_variant_resolve_different_exact_populations(self):
        legacy = {key: value for key, value in self.manifest.items() if key != "dataset_identity"}
        self.assertEqual(len(load_run_development(self.root, legacy)), 60)
        actual = load_run_development(self.root, self.manifest)
        pd.testing.assert_frame_equal(actual, self.frame)
        self.assertEqual(len(actual), 59)
        changed = copy.deepcopy(self.manifest)
        changed["dataset_identity"]["retained_ids_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "identity changed"):
            load_run_development(self.root, changed)

    def test_explicit_matching_metadata_preserves_scores_and_original_rows_rejected(self):
        shuffled = self.predictions.sample(frac=1, random_state=41)
        shuffled.to_csv(self.prediction_path, index=False)
        self.receipt["prediction_sha256"] = sha256(self.prediction_path)
        self.write_receipt()
        actual = self.load()
        for name, score in (("metadata_mlp_mean", "p_nn_mean"), ("metadata_spline_raw", "p_spline")):
            self.assertEqual(len(actual[name]), 59)
            np.testing.assert_allclose(actual[name].base_probability, self.predictions[score], rtol=0, atol=1e-15)
        with self.assertRaisesRegex(ValueError, "coverage/labels"):
            self.load(expected=self.original)
        with self.assertRaisesRegex(ValueError, "implicitly"):
            load_matched_metadata(self.root, self.original,
                receipt_path=self.receipt_path.relative_to(self.root).as_posix())

    def test_changed_missing_or_incomplete_comparator_provenance_is_rejected(self):
        changed = copy.deepcopy(self.receipt)
        changed["source_sha256"] = {}
        self.write_receipt(changed)
        with self.assertRaisesRegex(ValueError, "source provenance"):
            self.load()
        changed = copy.deepcopy(self.receipt)
        changed["status"] = "incomplete"
        self.write_receipt(changed)
        with self.assertRaisesRegex(ValueError, "complete"):
            self.load()
        self.write_receipt()
        source = self.root / "src/melanoma_detection/metadata_model.py"
        source.write_bytes(source.read_bytes()+b"\n# changed\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.load()

    def test_partial_predictions_rejected_even_when_their_file_digest_is_updated(self):
        self.predictions.iloc[:-1].to_csv(self.prediction_path, index=False)
        self.receipt["prediction_sha256"] = sha256(self.prediction_path)
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "coverage/labels"):
            self.load()

    def test_deduplicated_finalizer_cannot_silently_drop_missing_comparators(self):
        directory = self.root / "results/image_d1_runs/mock_clean"
        directory.mkdir(parents=True)
        predictions = self.predictions[["image_name", "patient_id", "fold", "target", "p_nn_mean"]].rename(
            columns={"image_name": "image_id", "p_nn_mean": "base_probability"}).assign(run_id="mock_clean")
        predictions.to_csv(directory / "outer_oof_predictions.csv", index=False)
        metrics = fold_metric_table(predictions).assign(status="complete")
        metrics.to_csv(directory / "fold_metrics.csv", index=False)
        manifest = {**self.manifest, "run_id": "mock_clean", "config": {
            "run": {"run_id": "mock_clean", "metadata_comparator_receipt": "results/missing_comparator.json"},
            "evaluation": {"reserved_test_allowed": False, "bootstrap_replicates": 100}, "training": {"seed": 42}}}
        (directory / "run_manifest.json").write_text(json.dumps(manifest))
        summary = {"run_id": "mock_clean", "status": "complete", "completed_outer_folds": list(range(5)),
            "folds": metrics.to_dict("records"), "outer_oof_predictions_sha256": sha256(directory / "outer_oof_predictions.csv"),
            "fold_metrics_sha256": sha256(directory / "fold_metrics.csv"),
            "mean_fold_average_precision": metrics.average_precision.mean()}
        path = directory / "summary.json"
        path.write_text(json.dumps(summary))
        digest = sha256(path)
        with self.assertRaisesRegex(ValueError, "matching metadata comparators"):
            finalize(self.root, "mock_clean")
        self.assertEqual(sha256(path), digest)
        self.assertFalse((directory / "evaluation").exists())


if __name__ == "__main__":
    unittest.main()
