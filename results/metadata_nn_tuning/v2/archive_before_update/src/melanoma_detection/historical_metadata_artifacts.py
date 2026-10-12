"""Read historical metadata evidence after active notebooks become reports.

This reader does not alter the frozen legacy training runner or its resume
identity. Historical source checks bind edited notebooks to verified archives.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dedup_metadata import NN_SPECIFICATION, SPLINE_SPECIFICATION, _verify_predictions, verify_saved_receipt
from .dedup_metadata_suite import verify_suite_receipt
from .metadata_source_archive import verified_source_path


def _path(root, relative):
    root = Path(root).resolve()
    prefix = root / "results/dedup_v1/metadata"
    path = (root / relative).resolve()
    if not path.is_relative_to(prefix):
        raise ValueError("Historical comparator path escapes the metadata output namespace.")
    return path


def verify_historical_suite(root, receipt):
    """Verify immutable outputs and their actual original fitting sources."""
    verify_suite_receipt(root, receipt)
    for relative, digest in receipt["source_sha256"].items():
        verified_source_path(root, relative, digest)
    return receipt


def verify_historical_reuse(root, frame, identity, receipt_path):
    """Read exact fixed predictions and calibration without training reuse."""
    root = Path(root).resolve()
    receipt = json.loads(_path(root, receipt_path).read_text(encoding="utf-8"))
    if receipt.get("dataset_identity") != identity:
        raise ValueError("Historical metadata receipt has a different development dataset identity.")
    specification = receipt.get("specification", {})
    if specification.get("neural_network") != NN_SPECIFICATION or specification.get("spline") != SPLINE_SPECIFICATION:
        raise ValueError("Historical comparator differs from the prescribed fixed protocol.")
    verify_saved_receipt(root, receipt)
    for relative, digest in receipt["source_sha256"].items():
        verified_source_path(root, relative, digest)
    predictions = pd.read_csv(_path(root, receipt["prediction_path"]), float_precision="round_trip")
    calibration = pd.read_csv(_path(root, receipt["calibration_prediction_path"]), float_precision="round_trip")
    _verify_predictions(frame, predictions)
    _verify_predictions(frame, calibration)
    predictions = predictions.set_index("image_name").loc[frame.image_name].reset_index()
    calibration = calibration.set_index("image_name").loc[frame.image_name].reset_index()
    if not np.array_equal(predictions.p_spline.to_numpy(), calibration.p_spline_raw.to_numpy()):
        raise ValueError("Historical spline predictions disagree with saved calibration raw scores.")
    return receipt, predictions, calibration
