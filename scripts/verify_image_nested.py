"""Verify preserved historical artifacts and completed nested checkpoint predictions.

Capture preservation before training with --capture. Verify later without training.
Use --prediction-roundtrip for exact deterministic outer checkpoint reconstruction.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.image_nested import _atomic_json, file_sha256, _read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/image_d1_runs/b0_nested_5fold_5h_v1")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--prediction-roundtrip", action="store_true")
    args = parser.parse_args()
    output = args.run_dir.resolve()
    if not output.is_relative_to(ROOT / "results/image_d1_runs"):
        raise ValueError("Run output must remain inside image runs.")
    preservation = output / "historical_preservation.json"
    if args.capture:
        if preservation.exists():
            raise FileExistsError("Existing baseline cannot be reset.")
        paths = []
        for parent in (ROOT / "results/image_d1_runs", ROOT / "results/image_uncertainty_runs"):
            if parent.exists():
                paths.extend(path for path in parent.rglob("*") if path.is_file() and not path.is_relative_to(output))
        paths.extend(path for path in (ROOT / "notebooks").rglob("*.ipynb") if path.name != "14_image_nested_evaluation.ipynb")
        hashes = {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}
        _atomic_json(preservation, {"files": hashes})
        print(f"Captured {len(hashes)} historical artifact hashes.")
        return 0
    hashes = _read_json(preservation)["files"]
    failures = [relative for relative, digest in hashes.items() if not (ROOT / relative).is_file() or file_sha256(ROOT / relative) != digest]
    if failures:
        raise ValueError(f"Historical artifacts changed: {failures}")
    summary = _read_json(output / "summary.json")
    if summary["status"] != "complete":
        raise ValueError("Five-fold run has not completed.")
    predictions = output / "outer_oof_predictions.csv"
    if file_sha256(predictions) != summary["outer_oof_predictions_sha256"]:
        raise ValueError("OOF prediction artifact changed.")
    roundtrips = []
    for record in summary["folds"]:
        checkpoint = Path(record["checkpoint"])
        if file_sha256(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("Final refit checkpoint changed.")
        if args.prediction_roundtrip:
            from melanoma_detection.image_model import load_model_checkpoint, predict_frame, seed_everything
            seed_everything(42)
            model, state = load_model_checkpoint(checkpoint, device="cuda")
            if state["identity"].get("training_purpose") != "fixed_duration_refit" or state["identity"]["validation"] is not None:
                raise ValueError("Outer checkpoint used validation outcomes.")
            if state["epoch"] != record["refit_epochs"] or state["best_epoch"] is not None:
                raise ValueError("Outer checkpoint was selected rather than fixed-duration refitted.")
            frame = pd.read_csv(ROOT / "data/subset/processed/train_folds.csv")
            frame = frame.loc[frame.fold == record["fold"]].sort_values("image_name").reset_index(drop=True)
            frame["image_path"] = frame.image_name.map(lambda name: ROOT / "data/subset/train" / f"{name}.jpg")
            actual = predict_frame(model, frame.assign(target=0), state["spec"], {**state["training"], "microbatch": state["microbatch"]}, device="cuda")
            saved = pd.read_csv(output / f"folds/{record['fold']}/outer_predictions.csv").sort_values("image_id").reset_index(drop=True)
            if actual.image_id.tolist() != saved.image_id.tolist() or not np.allclose(actual.base_probability, saved.base_probability, rtol=0, atol=1e-12):
                raise ValueError("Deterministic checkpoint predictions differ from saved outer scores.")
            roundtrips.append({"fold": record["fold"], "images": len(actual), "maximum_absolute_difference": float(np.max(np.abs(actual.base_probability.to_numpy() - saved.base_probability.to_numpy())))})
    receipt = {"status": "passed", "historical_files_unchanged": len(hashes), "outer_checkpoints_verified": 5, "prediction_roundtrips": roundtrips}
    _atomic_json(output / "verification_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
