import csv
import tempfile
import unittest
from pathlib import Path

from melanoma_detection.image_validation import check_image_inventory


class CheckImageInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.jpeg_directory = self.root / "jpeg"
        self.dicom_directory = self.root / "dicom"
        self.jpeg_directory.mkdir()
        self.dicom_directory.mkdir()
        self.train_csv = self.root / "train.csv"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_csv(self, image_names):
        with self.train_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["image_name"])
            writer.writeheader()
            for image_name in image_names:
                writer.writerow({"image_name": image_name})

    def touch_image(self, directory, filename):
        (directory / filename).touch()

    def test_reports_an_exact_inventory_match(self):
        self.write_csv(["ISIC_001", "ISIC_002"])
        self.touch_image(self.jpeg_directory, "ISIC_001.jpg")
        self.touch_image(self.jpeg_directory, "ISIC_002.jpeg")
        self.touch_image(self.dicom_directory, "ISIC_001.dcm")
        self.touch_image(self.dicom_directory, "ISIC_002.DCM")

        report = check_image_inventory(
            self.train_csv,
            self.jpeg_directory,
            self.dicom_directory,
        )

        self.assertTrue(report.all_csv_rows_have_both_formats)
        self.assertTrue(report.directories_exactly_match_csv)
        report.assert_complete(strict=True)

    def test_reports_missing_orphan_and_duplicate_files(self):
        self.write_csv(["ISIC_001", "ISIC_002"])
        self.touch_image(self.jpeg_directory, "ISIC_001.jpg")
        self.touch_image(self.jpeg_directory, "ISIC_001.jpeg")
        self.touch_image(self.jpeg_directory, "ORPHAN.jpg")
        self.touch_image(self.dicom_directory, "ISIC_001.dcm")

        report = check_image_inventory(
            self.train_csv,
            self.jpeg_directory,
            self.dicom_directory,
        )

        self.assertEqual(report.missing_jpeg, ("ISIC_002",))
        self.assertEqual(report.missing_dicom, ("ISIC_002",))
        self.assertEqual(report.orphan_jpeg, ("ORPHAN",))
        self.assertEqual(report.duplicate_jpeg_stems, ("ISIC_001",))
        self.assertFalse(report.directories_exactly_match_csv)
        with self.assertRaises(AssertionError):
            report.assert_complete()

    def test_rejects_blank_image_identifiers(self):
        self.write_csv([""])

        with self.assertRaisesRegex(ValueError, "CSV row 2"):
            check_image_inventory(
                self.train_csv,
                self.jpeg_directory,
                self.dicom_directory,
            )


if __name__ == "__main__":
    unittest.main()
