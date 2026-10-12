"""Finalise saved five-fold image results without training or reserved-test access.

  .venv/Scripts/python.exe scripts/finalize_image_nested.py

Training must already be complete. The existing manifest and prediction hashes
are verified before independent metrics, figures and bootstrap intervals are
written. Run the read-only evaluation notebook afterward to save its outputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.image_evaluation import sha256
from melanoma_detection.image_nested_reporting import (
    build_evaluation, fold_metric_table, load_matched_metadata, validate_outer_predictions, load_run_development,
)


def finalize(project_root, run_id="b0_nested_5fold_5h_v1", *, include_metadata=True):
    root = Path(project_root).resolve()
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one directory name.")
    run_dir = root / "results/image_d1_runs" / run_id
    manifest_path, summary_path = run_dir / "run_manifest.json", run_dir / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_before = sha256(summary_path)
    config = manifest["config"]
    if manifest.get("run_id") != run_id or config["run"].get("run_id") != run_id or summary.get("run_id") != run_id:
        raise ValueError("Run identities disagree between directory, manifest, configuration and summary.")
    folds = summary.get("folds", [])
    if (summary.get("status") != "complete" or summary.get("completed_outer_folds") != list(range(5))
            or len(folds) != 5 or {row.get("fold") for row in folds} != set(range(5))
            or any(row.get("status") != "complete" for row in folds)):
        raise ValueError("Final reporting requires all five completed outer folds; partial runs stay incomplete.")
    if config["evaluation"].get("reserved_test_allowed") is not False:
        raise ValueError("This finaliser permits development-only evaluation.")
    relative = manifest.get("development_csv")
    development_path = (root / relative).resolve() if isinstance(relative, str) else None
    if development_path != (root / "data/subset/processed/train_folds.csv").resolve():
        raise ValueError("Manifest must name the canonical d1 development CSV.")
    if sha256(development_path) != manifest.get("development_csv_sha256"):
        raise ValueError("Canonical development CSV changed since training.")
    predictions_path = run_dir / "outer_oof_predictions.csv"
    if sha256(predictions_path) != summary.get("outer_oof_predictions_sha256"):
        raise ValueError("Complete OOF prediction hash changed or is missing.")
    top_folds_path = run_dir / "fold_metrics.csv"
    if sha256(top_folds_path) != summary.get("fold_metrics_sha256"):
        raise ValueError("Runner fold-metric hash changed or is missing.")
    expected = load_run_development(root, manifest)
    predictions = validate_outer_predictions(pd.read_csv(predictions_path), expected)
    if "run_id" in predictions and not predictions.run_id.eq(run_id).all():
        raise ValueError("Prediction run_id does not match the completed run.")
    recomputed = fold_metric_table(predictions)
    recorded = pd.read_csv(top_folds_path).sort_values("fold").reset_index(drop=True)
    metric_names = ["average_precision", "roc_auc", "log_loss", "brier"]
    if (not recomputed.fold.equals(recorded.fold) or not np.allclose(
            recomputed[metric_names], recorded[metric_names], rtol=1e-10, atol=1e-12)):
        raise ValueError("Runner fold metrics disagree with recomputed OOF scores.")
    if not np.isclose(recomputed.average_precision.mean(), summary["mean_fold_average_precision"], rtol=1e-10, atol=1e-12):
        raise ValueError("Runner mean-fold AP disagrees with recomputed OOF scores.")
    comparators = None
    comparator_status = "not requested"
    comparator_provenance = {}
    if include_metadata:
        try:
            metadata_receipt = config["run"].get("metadata_comparator_receipt", "results/metrics/metadata_nn_run_config_d1.json")
            comparators = load_matched_metadata(root, expected, receipt_path=metadata_receipt,
                                               dataset_identity=manifest.get("dataset_identity"))
        except (ValueError, OSError, KeyError, TypeError) as error:
            if manifest.get("dataset_identity") is not None:
                raise ValueError("Deduplicated reporting requires its matching metadata comparators.") from error
            comparator_status = f"unavailable: {type(error).__name__}: {error}"
        else:
            comparator_status = "verified exact dataset coverage, saved input/prediction hashes and " + (
                "saved source hashes" if manifest.get("dataset_identity") is not None else "notebook code")
            metadata_receipt = root / metadata_receipt
            receipt = json.loads(metadata_receipt.read_text(encoding="utf-8"))
            comparator_provenance = {"receipt_path": metadata_receipt.relative_to(root).as_posix(),
                                    "receipt_sha256": sha256(metadata_receipt),
                                    "prediction_path": receipt["prediction_path"],
                                    "prediction_sha256": receipt["prediction_sha256"],
                                    "interpretation": receipt.get("interpretation", "development comparison")}
    evaluation = build_evaluation(predictions, expected, run_dir,
        bootstrap_replicates=config["evaluation"]["bootstrap_replicates"],
        seed=config["training"]["seed"], comparators=comparators)
    evaluation["metadata_comparator_status"] = comparator_status
    evaluation["metadata_comparator_provenance"] = comparator_provenance
    if sha256(summary_path) != summary_before:
        raise ValueError("Run summary changed during finalisation; rerun after training has stopped.")
    summary["evaluation"] = evaluation
    temporary = summary_path.with_name("summary.reporting.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default="b0_nested_5fold_5h_v1")
    parser.add_argument("--metadata", action=argparse.BooleanOptionalAction, default=True,
                        help="Include verified exact-coverage metadata comparisons when available.")
    args = parser.parse_args()
    summary = finalize(ROOT, args.run_id, include_metadata=args.metadata)
    evaluation = summary["evaluation"]
    print(json.dumps({"status": summary["status"], "run_id": args.run_id,
        "primary_mean_fold_average_precision": evaluation["primary_mean_fold_average_precision"],
        "primary_mean_fold_ap_ci95": evaluation["primary_mean_fold_ap_ci95"],
        "valid_bootstrap_replicates": evaluation["valid_bootstrap_replicates"],
        "metadata_comparator_status": evaluation["metadata_comparator_status"],
        "reserved_test_loaded": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
