"""Verify completed raw/calibrated image reports and write one new receipt.

Run only after calibration reporting and final notebook execution:
  .venv/Scripts/python.exe scripts/verify_image_nested_completion.py

All sources, configurations, model files, summaries and existing receipts are
read-only. The only output is a new raw-run completion_receipt.json. CPU loads
inspect checkpoint metadata; no model is constructed and no GPU work runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection import image_model as core
from melanoma_detection.image_evaluation import sha256, workspace_relative_path
from melanoma_detection.image_nested_reporting import load_saved_nested_run
from melanoma_detection.image_nested_calibration_reporting import load_saved_calibration_run


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sources(root, run_dir, hashes, expected_count):
    require(isinstance(hashes, dict) and len(hashes) == expected_count, "Unexpected training source inventory.")
    for relative, expected in hashes.items():
        current = workspace_relative_path(root, relative)
        archived = run_dir / "source_at_training" / relative
        require(sha256(current) == expected == sha256(archived), f"Current or archived source changed: {relative}")
    return len(hashes)


def tests_log(path):
    text = path.read_text(encoding="utf-8-sig")
    runs = list(re.finditer(r"(?m)^Ran (\d+) tests in ([0-9.]+)s\s*$", text))
    statuses = list(re.finditer(r"(?m)^OK(?: \(skipped=(\d+)\))?\s*$", text))
    require(bool(runs) and bool(statuses) and statuses[-1].start() > runs[-1].start(), "Tests log has no successful final result.")
    total, seconds = int(runs[-1].group(1)), float(runs[-1].group(2))
    skipped = int(statuses[-1].group(1) or 0)
    require(total == 244 and skipped == 1, f"Expected 244 tests and one skip; actual {total} and {skipped}.")
    require("FAILED (" not in text[runs[-1].start():], "Final test run failed.")
    return {"total": total, "passed": total - skipped, "skipped": skipped, "elapsed_seconds": seconds, "log_sha256": sha256(path)}


def notebook_execution(path):
    notebook = read_json(path)
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    require(bool(cells), "Evaluation notebook has no code cells.")
    require(all(type(cell.get("execution_count")) is int and cell["execution_count"] > 0 for cell in cells), "Evaluation notebook has unexecuted code cells.")
    counts = [cell["execution_count"] for cell in cells]
    require(counts == list(range(1, len(cells) + 1)), "Notebook must have one complete sequential execution.")
    outputs = [output for cell in cells for output in cell.get("outputs", [])]
    require(not any(output.get("output_type") == "error" for output in outputs), "Evaluation notebook contains execution errors.")
    calibrated_cells = [cell for cell in cells if 'if calibration["complete"]:' in "".join(cell.get("source", []))]
    require(len(calibrated_cells) >= 2 and all(cell.get("outputs") for cell in calibrated_cells), "Final calibrated report branch has not executed with results.")
    rendered = "\n".join("".join(output.get("data", {}).get("text/markdown", [])) for output in outputs)
    require("five-fold calibration report is complete and verified" in rendered, "Notebook still displays an incomplete calibration report.")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "executed_code_cells": len(cells),
            "error_outputs": 0, "complete_calibration_branch": True,
            "embedded_png_outputs": sum("image/png" in output.get("data", {}) for output in outputs)}


def pool_frame(root, raw_dir, raw_manifest, fold):
    frame = pd.read_csv(raw_dir / f"folds/{fold}/inner_memberships.csv", dtype={"image_name": str, "patient_id": str})
    image_dir = workspace_relative_path(root, raw_manifest["config"]["run"]["image_dir"])
    frame["image_path"] = frame.image_name.map(lambda name: image_dir / f"{name}.jpg")
    return frame


def verify(project_root, *, raw_run_id="b0_nested_5fold_5h_v1", calibration_run_id="b0_nested_calibration_5h_v1",
           notebook="notebooks/Subset/14_image_nested_evaluation.ipynb", expected_start_utc="2026-10-04T02:50:00+00:00"):
    root = Path(project_root).resolve()
    raw = load_saved_nested_run(root, raw_run_id)
    calibrated = load_saved_calibration_run(root, calibration_run_id)
    require(raw.get("complete") is True and calibrated.get("complete") is True, "Both saved five-fold reports must be complete and verified.")
    raw_dir, cal_dir = raw["run_dir"], calibrated["run_dir"]
    require(calibrated["summary"]["reference_run_id"] == raw_run_id, "Calibration references a different raw run.")
    preserved_receipt = raw_dir / "verification_receipt.json"
    existing_receipt_sha256 = sha256(preserved_receipt)
    roundtrip = read_json(preserved_receipt)
    require(roundtrip.get("status") == "passed" and roundtrip.get("outer_checkpoints_verified") == 5,
            "Prior independent raw-checkpoint roundtrip verification is missing.")
    require(sorted(row["fold"] for row in roundtrip.get("prediction_roundtrips", [])) == list(range(5)), "Prior roundtrip verification lacks all five folds.")
    require(all(row["maximum_absolute_difference"] <= 1e-12 for row in roundtrip["prediction_roundtrips"]), "Prior checkpoint reproduction exceeded tolerance.")

    raw_sources = read_json(raw_dir / "folds/0/fold_manifest.json")["compatibility"]["source_sha256"]
    raw_source_count = sources(root, raw_dir, raw_sources, 6)
    for fold in range(1, 5):
        require(read_json(raw_dir / f"folds/{fold}/fold_manifest.json")["compatibility"]["source_sha256"] == raw_sources,
                "Raw folds have different source provenance.")
    cal_source_count = sources(root, cal_dir, calibrated["manifest"]["source_sha256"], 8)
    historical = read_json(raw_dir / "historical_preservation.json")["files"]
    require(len(historical) == 408, "Historical preservation inventory must contain 408 files.")
    for relative, digest in historical.items():
        require(sha256(workspace_relative_path(root, relative)) == digest, f"Protected historical artifact changed: {relative}")

    raw_records = raw["summary"]["folds"]
    require(len(raw_records) == 5 and {item["fold"] for item in raw_records} == set(range(5)), "Raw checkpoint receipts lack five distinct folds.")
    for record in raw_records:
        path = raw_dir / f"folds/{record['fold']}/refit/latest.pt"
        require(sha256(path) == record["checkpoint_sha256"], "Raw final checkpoint hash changed.")

    cal_checkpoint_count = cal_artifact_count = 0
    for record in calibrated["summary"]["folds"]:
        fold = record["fold"]
        inner_refits = record["inner_refits"]
        require(len(inner_refits) == 3 and {row["inner_fold"] for row in inner_refits} == set(range(3)), "Calibration requires three distinct inner fixed refits per outer fold.")
        pool = pool_frame(root, raw_dir, raw["manifest"], fold)
        outer_patients = set(raw["expected"].loc[raw["expected"].fold.eq(fold), "patient_id"])
        require(not set(pool.patient_id) & outer_patients, "Outer patients leaked into the calibration training pool.")
        require(pool.groupby("patient_id").inner_fold.nunique().eq(1).all(), "Calibration patients cross inner folds.")
        for inner in inner_refits:
            inner_fold = inner["inner_fold"]
            folder = cal_dir / f"folds/{fold}/inner/{inner_fold}"
            for name, key in (("latest.pt", "checkpoint_sha256"), ("job_manifest.json", "job_manifest_sha256"),
                              ("history.csv", "history_sha256"), ("predictions.csv", "predictions_sha256")):
                require(sha256(folder / name) == inner[key], f"Calibration inner artifact changed: {folder.name}/{name}")
                cal_artifact_count += 1
            state = core._read_checkpoint(folder / "latest.pt", device="cpu")
            manifest = read_json(folder / "job_manifest.json")
            require(state["manifest"] == manifest, "Calibration checkpoint and job manifest disagree.")
            require(state["epoch"] == record["fixed_epochs"] == inner["fixed_epochs"] == manifest["fixed_epochs"], "Calibration checkpoint is not the selected fixed final epoch.")
            require(state["identity"]["validation"] is None and state["identity"].get("training_purpose") == "fixed_duration_refit",
                    "Calibration refit used validation checkpoint selection.")
            require(state["best_epoch"] is None and state["best_metrics"] is None, "Calibration refit selected a validation-best checkpoint.")
            require([row["epoch"] for row in state["history"]] == list(range(1, state["epoch"] + 1)), "Calibration history lacks complete fixed epochs.")
            require(all(not {"average_precision", "roc_auc", "log_loss", "brier"}.intersection(row) for row in state["history"]), "Calibration refit history contains validation metrics.")
            fitting = pool.loc[pool.inner_fold.ne(inner_fold)].copy().reset_index(drop=True)
            validation = pool.loc[pool.inner_fold.eq(inner_fold)]
            require(not set(fitting.patient_id) & (set(validation.patient_id) | outer_patients), "Calibration fitting patients overlap validation or outer patients.")
            require(state["identity"]["fitting"] == core._frame_identity(fitting), "Calibration checkpoint fitting membership differs from its inner split.")
            cal_checkpoint_count += 1
            del state
    require(cal_checkpoint_count == 15, "Calibration verification must cover all 15 fixed refits.")

    test_result = tests_log(raw_dir / "tests.log")
    notebook_result = notebook_execution(workspace_relative_path(root, notebook))
    session = read_json(raw_dir / "session.json")
    start = datetime.fromisoformat(session["started_utc"]).astimezone(timezone.utc)
    expected_start = datetime.fromisoformat(expected_start_utc).astimezone(timezone.utc)
    require(start == expected_start and session["budget"] == raw["manifest"]["config"]["budget"], "Original session start or budget changed.")
    require(session["budget"]["session_minutes"] == 300 and session["budget"]["reporting_reserve_minutes"] == 20, "The original five-hour budget/reporting reserve changed.")
    final = start + timedelta(minutes=300)
    cutoff = final - timedelta(minutes=20)
    require(datetime.fromisoformat(session["session_deadline_utc"]) == final and datetime.fromisoformat(session["training_cutoff_utc"]) == cutoff,
            "Original session deadlines changed.")
    cal_config = calibrated["manifest"]["config"]
    require(cal_config["budget"] == {"source": "reference_session_only", "allow_deadline_extension": False}, "Calibration has a separate or extended allocation.")
    require(workspace_relative_path(root, cal_config["run"]["reference_session"]) == raw_dir / "session.json", "Calibration uses a different session.")
    declaration = read_json(cal_dir / "protocol_declaration.json")
    require(declaration.get("declared_before_outer_fold0_prediction") is True and declaration.get("existing_outer_prediction_folds") == [],
            "Calibration was not predeclared before outer prediction.")
    require(sha256(root / "configs/image_nested_calibration.yaml") == declaration["config_sha256"], "Predeclared calibration configuration changed.")
    now = datetime.now(timezone.utc)
    elapsed = (now - start).total_seconds() / 60
    require(sha256(preserved_receipt) == existing_receipt_sha256, "Existing raw prediction roundtrip receipt changed.")
    return {"schema_version": 1, "status": "passed", "verified_utc": now.isoformat(),
            "raw_run_id": raw_run_id, "calibration_run_id": calibration_run_id,
            "raw_training_sources_verified": raw_source_count, "calibration_training_sources_verified": cal_source_count,
            "historical_files_unchanged": len(historical), "raw_final_checkpoints_verified": len(raw_records),
            "calibration_fixed_refit_checkpoints_verified": cal_checkpoint_count,
            "calibration_inner_artifact_hashes_verified": cal_artifact_count,
            "calibration_checkpoint_state_device": "cpu", "fitting_patient_isolation_verified": True,
            "existing_raw_roundtrip_receipt_sha256": existing_receipt_sha256,
            "tests": test_result, "notebook": notebook_result,
            "session": {"started_utc": start.isoformat(), "training_cutoff_utc": cutoff.isoformat(),
                        "session_deadline_utc": final.isoformat(), "budget_minutes": 300,
                        "elapsed_minutes_at_verification": elapsed, "remaining_minutes_at_verification": max(0, 300 - elapsed),
                        "verified_within_original_budget": elapsed <= 300},
            "raw_metrics": {key: raw["evaluation"][key] for key in ("primary_mean_fold_average_precision", "primary_mean_fold_ap_ci95", "pooled_metrics_secondary")},
            "calibration_metrics": {key: calibrated["evaluation"][key] for key in ("mean_fold_metrics", "mean_fold_ap_invariant", "training_selected_decisions_identical", "threshold_metrics_pooled")},
            "n_images": len(raw["predictions"]), "n_patients": int(raw["predictions"].patient_id.nunique()),
            "reserved_test_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-run-id", default="b0_nested_5fold_5h_v1")
    parser.add_argument("--calibration-run-id", default="b0_nested_calibration_5h_v1")
    parser.add_argument("--notebook", default="notebooks/Subset/14_image_nested_evaluation.ipynb")
    parser.add_argument("--expected-start-utc", default="2026-10-04T02:50:00+00:00")
    parser.add_argument("--check-only", action="store_true", help="Perform verification without writing the new completion receipt.")
    args = parser.parse_args()
    receipt = verify(ROOT, raw_run_id=args.raw_run_id, calibration_run_id=args.calibration_run_id,
                     notebook=args.notebook, expected_start_utc=args.expected_start_utc)
    if not args.check_only:
        path = ROOT / "results/image_d1_runs" / args.raw_run_id / "completion_receipt.json"
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
