"""Read-only checkpoint smoke check before the educational uncertainty notebook.

Choose a run, architecture and recipe explicitly. No training, calibration,
threshold fitting, fusion, outer-fold inference or reserved-test access occurs.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.image_uncertainty import (
    load_selected_checkpoint, run_smoke,
    snapshot_preserved_artifacts, assert_preserved_artifacts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--run-id", default="b0_augmentation_12e_v1")
    parser.add_argument("--recipe-id", default="lr3e5_sqrt")
    parser.add_argument("--architecture", choices=["efficientnet_b0", "densenet121", "convnext_tiny"],
                        default="efficientnet_b0")
    parser.add_argument("--condition-id", default="A")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    archive = ROOT / "notebooks/archive/image_uncertainty_three_epoch_fold0.ipynb"
    protected = snapshot_preserved_artifacts(ROOT, archive_path=archive)
    context = load_selected_checkpoint(ROOT, args.run_id, args.recipe_id,
        architecture=args.architecture, condition_id=args.condition_id, device=args.device)
    receipt = run_smoke(context, n_images=16, seed=20261004)
    receipt["preservation"] = assert_preserved_artifacts(protected)
    receipt["selection"] = {"run_id": args.run_id, "recipe_id": args.recipe_id,
        "architecture": args.architecture, "condition_id": args.condition_id}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ROOT / "results/image_uncertainty_runs" / f"smoke_{args.architecture}_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "smoke_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(receipt, indent=2, allow_nan=False))
    print(f"Smoke receipt: {folder / 'smoke_receipt.json'}", flush=True)


if __name__ == "__main__":
    main()
