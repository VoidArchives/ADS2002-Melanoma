"""Evaluate a frozen metadata bundle once, after all modelling decisions stop.

Example (run from the repository root when ready for the final test):
python scripts/evaluate_metadata_holdout.py --bundle models/metadata_d1/frozen_spline.joblib --data PATH_TO_CLEANED_HELDOUT_CSV --output-dir results/metadata_final_holdout

The script has no default data path, refuses development-patient overlap and
records one attempted evaluation beside the frozen bundle. It never selects a
model or cutoff. A failed attempt remains recorded rather than silently retried.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from melanoma_detection.metadata_evaluation import decision_metrics, patient_bootstrap, probability_metrics
from melanoma_detection.metadata_model import FEATURES, file_sha256
from melanoma_detection.metadata_validation import validate_metadata_frame


def verify_manifest(bundle_path: Path, manifest: dict, project_root: Path) -> None:
    """Refuse stale weights, changed transformation code or a changed runtime."""
    if file_sha256(bundle_path) != manifest["bundle_sha256"]:
        raise ValueError("The bundle differs from its frozen manifest.")
    for relative_path, expected_hash in manifest["source_sha256"].items():
        path = (project_root / relative_path).resolve()
        if not path.is_relative_to(project_root.resolve()) or file_sha256(path) != expected_hash:
            raise ValueError(f"Source changed after freezing: {relative_path}")
    for package, expected_version in manifest["versions"].items():
        if importlib.metadata.version(package) != expected_version:
            raise ValueError(f"Runtime differs from the frozen manifest: {package}")
    if "python" in manifest and platform.python_version() != manifest["python"]:
        raise ValueError("Python differs from the frozen manifest.")


def evaluate_bundle(bundle: dict, heldout: pd.DataFrame):
    """Use only stored weights and threshold; reject overlap before scoring."""
    heldout = validate_metadata_frame(heldout, require_folds=False)
    if bundle.get("schema_version") != 1 or bundle["specification"]["primary_probability"] != "raw":
        raise ValueError("Unsupported frozen metadata bundle.")
    if set(heldout.patient_id.astype(str)) & set(bundle["training_patient_ids"]):
        raise ValueError("Held-out data includes development patients; evaluation refused.")
    if set(heldout.image_name.astype(str)) & set(bundle["training_image_ids"]):
        raise ValueError("Held-out data includes development images; evaluation refused.")
    probabilities = bundle["model"].predict_proba(heldout[FEATURES])[:, 1]
    decisions = probabilities >= bundle["threshold"]
    predictions = heldout[["image_name", "patient_id", "target"]].copy()
    predictions["probability"] = probabilities
    predictions["decision"] = decisions.astype(int)
    metrics = {**probability_metrics(heldout.target, probabilities), **decision_metrics(heldout.target, decisions)}
    return predictions, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists. Repeated final-test evaluation is refused.")
    attempt_path = args.bundle.with_suffix(".holdout_receipt.json")
    if attempt_path.exists():
        parser.error("This bundle already has a recorded holdout attempt. Evaluation is refused.")
    manifest_path = args.bundle.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest(args.bundle, manifest, Path(__file__).resolve().parents[1])
    # Load only this locally generated, trusted model bundle.
    bundle = joblib.load(args.bundle)
    if bundle["input_sha256"] != manifest["input_sha256"] or bundle["threshold"] != manifest["threshold"] or bundle["specification"] != manifest["specification"]:
        raise ValueError("Model decisions do not match the frozen manifest.")
    receipt = {"status": "started", "bundle_sha256": file_sha256(args.bundle),
               "evaluation_input_sha256": file_sha256(args.data), "output_dir": str(args.output_dir.resolve())}
    # Exclusive creation also prevents simultaneous attempts against this bundle.
    with attempt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    try:
        heldout = pd.read_csv(args.data, dtype={"age_approx": "string", "target": "string"})
        predictions, metrics = evaluate_bundle(bundle, heldout)
        intervals = patient_bootstrap(
            predictions.target, predictions.patient_id, {"frozen_spline": predictions.probability},
            n_bootstrap=2000, seed=42,
        )
        receipt.update({
            "status": "complete", "threshold": bundle["threshold"], "specification": bundle["specification"],
            "images": len(predictions), "patients": predictions.patient_id.nunique(),
            "metrics": metrics, "decisions_changed_using_test": False,
        })
        args.output_dir.mkdir(parents=True, exist_ok=False)
        predictions.to_csv(args.output_dir / "predictions.csv", index=False)
        intervals.to_csv(args.output_dir / "patient_bootstrap_intervals.csv", index=False)
        (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        attempt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt, indent=2))
    except Exception as error:
        receipt.update({"status": "failed", "error": str(error)})
        attempt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
