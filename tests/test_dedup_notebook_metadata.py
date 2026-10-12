"""Check report-only patient sampling, cutoff illustrations and sparse evidence."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.dedup_notebook_metadata import (
    decision_intervals, exploratory_cutoff, metadata_specs, patient_draws,
    reliability_table, subgroup_diagnostics,
)


def example():
    rows = []
    for fold in range(3):
        for patient in range(4):
            for image in range(2):
                rows.append({"patient_id": f"p_{fold}_{patient}", "fold": fold,
                             "target": int(patient == 0 and image == 0),
                             "age_approx": 40 if image == 0 else 60,
                             "sex": "female" if patient % 2 else "male",
                             "anatom_site_general_challenge": "Unknown" if image == 0 else "torso"})
    return pd.DataFrame(rows)


class MetadataNotebookTests(unittest.TestCase):
    def test_diagnostic_draws_preserve_fold_patient_totals_and_repeat_whole_patients(self):
        frame = example()
        draws, codes = patient_draws(frame, repetitions=25)
        patients = frame[["patient_id", "fold"]].drop_duplicates().sort_values("patient_id")
        for fold in range(3):
            mask = patients.fold.eq(fold).to_numpy()
            np.testing.assert_array_equal(draws[:, mask].sum(axis=1), np.full(25, 4))
        np.testing.assert_array_equal(draws[:, codes[0]], draws[:, codes[1]])
        duplicate_patient = frame.copy()
        duplicate_patient.loc[1, "fold"] = 1
        with self.assertRaisesRegex(ValueError, "crosses outer folds"):
            patient_draws(duplicate_patient, repetitions=2)

    def test_exploratory_cutoff_matches_original_balanced_accuracy_and_tie_rule(self):
        target = np.array([0, 1, 0, 1, 0, 1])
        score = np.array([.03, .03, .04, .08, .10, .10])
        candidates = np.unique(np.r_[0, score, 1])
        balanced = np.array([balanced_accuracy_score(target, score >= threshold) for threshold in candidates])
        best = np.flatnonzero(np.isclose(balanced, balanced.max()))
        expected = candidates[best[np.argmin(np.abs(candidates[best] - .5))]]
        self.assertEqual(exploratory_cutoff(target, score), expected)

    def test_equal_scores_stay_in_one_reliability_bin_and_counts_partition_images(self):
        frame = example()
        table = reliability_table(frame, {"constant": np.full(len(frame), .1)}, repetitions=25)
        self.assertEqual(len(table), 1)
        self.assertEqual(table.n_images.sum(), len(frame))
        self.assertEqual(table.n_malignant_images.sum(), frame.target.sum())
        self.assertTrue(table.valid_draws.eq(25).all())

    def test_no_flags_gives_missing_precision_interval_without_inventing_zero(self):
        frame = example()
        intervals = decision_intervals(frame, np.zeros(len(frame), dtype=int), repetitions=25)
        precision = intervals.loc[intervals.metric.eq("precision")].iloc[0]
        self.assertTrue(np.isnan(precision.estimate))
        self.assertTrue(np.isnan(precision.ci_lower))
        self.assertEqual(precision.valid_draws, 0)

    def test_sparse_group_intervals_are_suppressed_and_undefined_ap_is_missing(self):
        frame = example()
        table = subgroup_diagnostics(frame, np.linspace(.01, .1, len(frame)), np.zeros(len(frame)), repetitions=25)
        older = table.loc[table.field.eq("Age band") & table.group.eq(">=50")].iloc[0]
        self.assertEqual(older.malignant_patients, 0)
        self.assertTrue(np.isnan(older.average_precision))
        self.assertTrue(table.average_precision_lower.isna().all())
        self.assertTrue(table.valid_draws.eq(0).all())
        unknown = table.loc[table.field.eq("Site availability") & table.group.eq("Unknown")].iloc[0]
        self.assertEqual(unknown.malignant_patients, 3)
        self.assertTrue(np.isfinite(unknown.average_precision))
        self.assertIn("intervals suppressed", unknown.evidence_note)

    def test_all_expanded_metadata_cells_compile(self):
        specifications = metadata_specs()
        self.assertEqual(len(specifications), 5)
        self.assertEqual([name[:2] for name, _, _, _ in specifications], ["04", "05", "06", "07", "08"])
        for filename, _, _, sections in specifications:
            for heading, description, code in sections:
                self.assertIn(heading, {"__markdown__", "__code__"})
                if heading == "__markdown__":
                    self.assertTrue(description)
                else:
                    compile(code, filename + heading, "exec")


if __name__ == "__main__":
    unittest.main()
