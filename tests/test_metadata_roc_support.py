"""Meaningful provenance and held-out baseline checks for ROC report migration."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.metadata_roc_archive import original_snapshot, verified_source_path
from melanoma_detection.metadata_roc_baselines import baseline_predictions


class ROCSupportTests(unittest.TestCase):
    def test_archive_binds_real_original_baseline_source(self):
        path = original_snapshot(ROOT, "notebooks/Subset/04_dummy_logistic_baseline.ipynb")
        self.assertTrue(path.is_file())

    def test_archive_rejects_unknown_source_and_escape(self):
        with self.assertRaises(ValueError):
            original_snapshot(ROOT, "../outside.py")
        with self.assertRaises(ValueError):
            verified_source_path(ROOT, "src/melanoma_detection/metadata_roc_baselines.py", "0" * 64)

    def test_heldout_labels_cannot_change_baseline_fits_or_probabilities(self):
        rows = []
        for fold in range(5):
            for patient in range(8):
                rows.append({"image_name": f"f{fold}p{patient}", "patient_id": f"f{fold}p{patient}",
                    "fold": fold, "target": patient % 2, "age_approx": float(30 + patient),
                    "sex": "male" if patient % 2 else "female", "anatom_site_general_challenge": "torso"})
        frame = pd.DataFrame(rows)
        first, model, audit, roles = baseline_predictions(frame, 0)
        changed = frame.copy()
        changed.loc[changed.fold.eq(0), "target"] = 1 - changed.loc[changed.fold.eq(0), "target"]
        second, second_model, _, _ = baseline_predictions(changed, 0)
        np.testing.assert_array_equal(first[["p_dummy", "p_categorical_logistic"]], second[["p_dummy", "p_categorical_logistic"]])
        np.testing.assert_array_equal(model.named_steps["classifier"].coef_, second_model.named_steps["classifier"].coef_)
        self.assertEqual(audit["patient_overlap"], 0)
        self.assertEqual(audit["fits"], 1)
        self.assertTrue((roles.loc[roles.fold.eq(0), "role"] == "heldout").all())
        self.assertTrue((first.p_dummy == frame.loc[frame.fold.ne(0), "target"].mean()).all())


if __name__ == "__main__":
    unittest.main()
