import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from melanoma_detection.splitting import assign_patient_folds


class AssignPatientFoldsTests(unittest.TestCase):
    @staticmethod
    def make_frame() -> pd.DataFrame:
        rows = []
        for patient_number in range(12):
            patient_id = f"IP_{patient_number:03d}"
            is_positive_patient = patient_number >= 6
            for image_number in range(2):
                rows.append(
                    {
                        "image_name": f"ISIC_{patient_number:03d}_{image_number}",
                        "patient_id": patient_id,
                        "target": int(is_positive_patient and image_number == 0),
                        "feature": patient_number,
                    }
                )
        return pd.DataFrame(rows)

    def test_assignments_are_patient_safe_stable_and_non_mutating(self):
        frame = self.make_frame()
        original = frame.copy(deep=True)

        first = assign_patient_folds(frame, n_splits=3, random_state=42)
        second = assign_patient_folds(frame, n_splits=3, random_state=42)

        assert_frame_equal(frame, original)
        self.assertEqual(list(first.columns), [*original.columns, "fold"])
        assert_series_equal(first["image_name"], original["image_name"])
        assert_series_equal(first["target"], original["target"])
        self.assertEqual(first["fold"].dtype, "int8")
        self.assertEqual(set(first["fold"]), {0, 1, 2})
        self.assertFalse(first["fold"].isna().any())
        self.assertTrue(first.groupby("patient_id")["fold"].nunique().eq(1).all())
        assert_series_equal(first["fold"], second["fold"])

        for _, fold_frame in first.groupby("fold"):
            self.assertEqual(set(fold_frame["target"]), {0, 1})
            positive_patients = fold_frame.groupby("patient_id")["target"].max().sum()
            self.assertGreater(positive_patients, 0)

    def test_rejects_missing_required_columns(self):
        frame = self.make_frame().drop(columns="target")

        with self.assertRaisesRegex(ValueError, "Required columns are missing"):
            assign_patient_folds(frame, n_splits=3)

    def test_rejects_non_binary_or_missing_targets(self):
        invalid = self.make_frame()
        invalid.loc[0, "target"] = 2
        with self.assertRaisesRegex(ValueError, "must be binary"):
            assign_patient_folds(invalid, n_splits=3)

        missing = self.make_frame()
        missing.loc[0, "target"] = None
        with self.assertRaisesRegex(ValueError, "contains missing values"):
            assign_patient_folds(missing, n_splits=3)

    def test_rejects_missing_or_blank_patient_ids(self):
        missing = self.make_frame()
        missing.loc[0, "patient_id"] = None
        with self.assertRaisesRegex(ValueError, "contains missing values"):
            assign_patient_folds(missing, n_splits=3)

        blank = self.make_frame()
        blank.loc[0, "patient_id"] = "  "
        with self.assertRaisesRegex(ValueError, "contains blank values"):
            assign_patient_folds(blank, n_splits=3)

    def test_rejects_insufficient_patient_groups(self):
        frame = self.make_frame().query("patient_id in ['IP_000', 'IP_006']")

        with self.assertRaisesRegex(ValueError, "unique patients are required"):
            assign_patient_folds(frame, n_splits=3)

    def test_rejects_insufficient_patient_groups_per_class(self):
        frame = self.make_frame().query(
            "patient_id in ['IP_000', 'IP_001', 'IP_002', 'IP_003', 'IP_006']"
        )

        with self.assertRaisesRegex(ValueError, "positive and .* negative patient groups"):
            assign_patient_folds(frame, n_splits=3)


class SubsetPatientFoldIntegrationTests(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    TRAIN_PATH = PROJECT_ROOT / "data" / "subset" / "processed" / "train_clean.csv"
    TEST_PATH = PROJECT_ROOT / "data" / "subset" / "processed" / "test_clean.csv"

    @unittest.skipUnless(
        TRAIN_PATH.exists() and TEST_PATH.exists(),
        "The ignored subset data files are not available in this checkout.",
    )
    def test_real_subset_produces_complete_patient_safe_folds(self):
        train = pd.read_csv(self.TRAIN_PATH)
        test_patients = pd.read_csv(self.TEST_PATH, usecols=["patient_id"])

        assigned = assign_patient_folds(train)
        repeated = assign_patient_folds(train)

        self.assertEqual(len(assigned), 6_864)
        self.assertEqual(assigned["patient_id"].nunique(), 422)
        self.assertEqual(int(assigned["target"].sum()), 128)
        self.assertEqual(
            int(assigned.groupby("patient_id")["target"].max().sum()),
            88,
        )
        self.assertEqual(set(assigned["fold"]), {0, 1, 2, 3, 4})
        self.assertFalse(assigned["fold"].isna().any())
        self.assertTrue(assigned.groupby("patient_id")["fold"].nunique().eq(1).all())
        self.assertEqual(list(assigned.columns), [*train.columns, "fold"])
        assert_series_equal(assigned["image_name"], train["image_name"])
        assert_series_equal(assigned["target"], train["target"])
        assert_series_equal(assigned["fold"], repeated["fold"])

        for _, fold_frame in assigned.groupby("fold"):
            self.assertEqual(set(fold_frame["target"]), {0, 1})
            positive_patients = fold_frame.groupby("patient_id")["target"].max().sum()
            self.assertGreater(positive_patients, 0)

        self.assertTrue(
            set(assigned["patient_id"]).isdisjoint(test_patients["patient_id"])
        )


if __name__ == "__main__":
    unittest.main()
