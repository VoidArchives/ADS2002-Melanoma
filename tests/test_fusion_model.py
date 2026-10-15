"""CPU-only regression checks for score alignment and patient-separated stacking."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from melanoma_detection import fusion_model as fusion
from melanoma_detection.splitting import patient_level_split


class FusionTests(unittest.TestCase):
    def setUp(self):
        patients = pd.DataFrame({"patient_id": [f"p{i}" for i in range(18)],
                                 "target": [0] * 12 + [1] * 6})
        self.maps = patient_level_split(patients, label_col="target")
        rows = patients.loc[patients.index.repeat(2)].reset_index(drop=True)
        rows["image_name"] = [f"image{i}" for i in range(len(rows))]
        rng = np.random.default_rng(42)
        self.images = rows.assign(melanoma_score=rng.uniform(0.01, 0.99, len(rows)))
        self.metadata = rows.assign(p_nn_mean=rng.uniform(0.01, 0.99, len(rows)),
                                    p_spline=rng.uniform(0.01, 0.99, len(rows)),
                                    p_nn_seed_42=0.99)

    def prepare(self, images=None, metadata=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return fusion.prepare_fusion_data(
                self.images if images is None else images,
                self.metadata if metadata is None else metadata, **kwargs)

    def test_logit_endpoints_and_invalid_probabilities(self):
        values = fusion.safe_logit([0, 0.5, 1])
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(values[1], 0)
        for value in (np.nan, np.inf, -0.01, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fusion.safe_logit([value])
        for eps in (0, 0.5, np.nan):
            with self.assertRaises(ValueError):
                fusion.safe_logit([0.5], eps)

    def test_match_audit_and_mean_score_without_seed_averaging(self):
        images = self.images.iloc[:-1]
        metadata = self.metadata.iloc[1:].sample(frac=1, random_state=1)
        matched, audit = self.prepare(images, metadata, include_spline=True)
        self.assertEqual(audit["only_image_ids"], ["image0"])
        self.assertEqual(audit["only_metadata_ids"], ["image35"])
        self.assertEqual(len(matched), 34)
        expected = metadata.set_index("image_name").loc[matched.image_name, "p_nn_mean"]
        np.testing.assert_array_equal(matched.metadata_score, expected)
        self.assertIn("p_spline", matched)
        self.assertEqual(list(fusion.build_fusion_features(matched)), ["image_logit", "metadata_logit"])

    def test_conflicting_ids_labels_and_invalid_source_rows_raise(self):
        for column, value in (("target", 1), ("patient_id", "wrong-patient")):
            bad = self.metadata.copy()
            bad.loc[0, column] = value
            with self.assertRaisesRegex(ValueError, f"{column} mismatch"):
                self.prepare(metadata=bad)
        for source in ("images", "metadata"):
            table = getattr(self, source)
            with self.assertRaisesRegex(ValueError, "unique"):
                self.prepare(**{source: pd.concat([table, table.iloc[:1]])})
        for score in (np.nan, np.inf, -0.1, 1.1):
            for source, column in (("images", "melanoma_score"), ("metadata", "p_nn_mean")):
                bad = getattr(self, source).copy()
                bad.loc[0, column] = score
                with self.assertRaisesRegex(ValueError, "probabilities"):
                    self.prepare(**{source: bad})
        bad = self.metadata.assign(fold=np.arange(len(self.metadata)) % 3)
        with self.assertRaisesRegex(ValueError, "multiple base-model folds"):
            self.prepare(metadata=bad)

    def test_patient_recovery_requires_complete_clean_image_lookup(self):
        images = self.images.drop(columns="patient_id")
        clean = self.images[["image_name", "patient_id"]]
        matched, _ = self.prepare(images=images, clean_images=clean)
        np.testing.assert_array_equal(matched.patient_id, self.images.patient_id)
        with self.assertRaisesRegex(ValueError, "clean_images"):
            self.prepare(images=images)
        with self.assertRaisesRegex(ValueError, "No clean_images patient ID"):
            self.prepare(images=images, clean_images=clean.iloc[1:])

    def test_cv_fits_only_training_patients_and_predicts_once(self):
        rows, _ = self.prepare()
        rows = rows.sample(frac=1, random_state=2)  # Non-default index/order.
        with patch.object(fusion, "fit_fusion_model", wraps=fusion.fit_fusion_model) as fit:
            oof, metrics = fusion.cross_validate_fusion_model(rows, self.maps)
        self.assertEqual(fit.call_count, 3)
        self.assertEqual(len(metrics), 3)
        np.testing.assert_array_equal(oof.image_name, rows.image_name)
        for fold, call in enumerate(fit.call_args_list, 1):
            train_patients = set(call.args[0].patient_id)
            validation = oof.loc[oof.fusion_fold.eq(fold)]
            self.assertTrue(train_patients.isdisjoint(validation.patient_id))
            self.assertEqual(train_patients | set(validation.patient_id), set(rows.patient_id))
        fusion.validate_fusion_predictions(oof, rows, self.maps)
        metrics = fusion.evaluate_fusion_models(oof, {"fusion": "fusion_score"})
        self.assertAlmostEqual(metrics.loc["fusion", "ROC_AUC"], roc_auc_score(oof.target, oof.fusion_score))
        for corruption in (oof.iloc[:-1], oof.assign(target=1-oof.target),
                           oof.assign(fusion_score=np.nan), oof.assign(fusion_fold=1)):
            with self.assertRaises(ValueError):
                fusion.validate_fusion_predictions(corruption, rows, self.maps)

    def test_invalid_saved_folds_fail_before_any_fit(self):
        rows, _ = self.prepare()
        for maps in (self.maps[:2], [self.maps[0]] * 3, [{}, *self.maps[1:]]):
            with patch.object(fusion, "fit_fusion_model") as fit:
                with self.assertRaises(ValueError):
                    fusion.cross_validate_fusion_model(rows, maps)
                fit.assert_not_called()

    def test_final_artifact_round_trip_and_optional_spline(self):
        rows, _ = self.prepare(include_spline=True)
        fusion.cross_validate_fusion_model(rows, self.maps, metadata_score_col="p_spline")
        final = fusion.fit_final_fusion_model(rows, source_paths={"metadata_source_column": "p_nn_mean"})
        inference = rows[["image_score", "metadata_score"]]
        expected = fusion.predict_fusion_model(final, inference)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fusion.joblib"
            joblib.dump(final, path)
            restored = joblib.load(path)
        np.testing.assert_array_equal(expected, fusion.predict_fusion_model(restored, inference))
        self.assertEqual(restored.feature_names, ("image_logit", "metadata_logit"))
        self.assertEqual(restored.source_paths["metadata_source_column"], "p_nn_mean")

    def test_returned_fold_checkpoints_reproduce_oof_scores(self):
        rows, _ = self.prepare(include_spline=True)
        for column in ("metadata_score", "p_spline"):
            oof, _, models = fusion.cross_validate_fusion_model(
                rows, self.maps, metadata_score_col=column, return_models=True)
            self.assertEqual(set(models), {1, 2, 3})
            for fold, model in models.items():
                held_out = oof.loc[oof.fusion_fold.eq(fold)]
                np.testing.assert_allclose(
                    fusion.predict_fusion_model(model, held_out), held_out.fusion_score,
                    rtol=0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
