"""Small training-pipeline diagnostic; never saves or changes trained checkpoints.

Run from the project root with .venv/Scripts/python.exe scripts/profile_image_pipeline.py.
"""
import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--augmentation", choices=["basic", "strong"], default="basic")
    args = parser.parse_args()
    if args.workers < 0 or args.batches < 1:
        parser.error("workers must be nonnegative and batches positive")

    import pandas as pd
    import torch
    from melanoma_detection.data import make_loaders
    from melanoma_detection.image_model import setup_image_model, _train_mode_with_frozen_batchnorm
    from melanoma_detection.preprocessing import image_transforms
    from melanoma_detection.model_registry import get_model_spec
    from melanoma_detection.image_model import unfreeze_backbone
    from melanoma_detection.utils import require_gpu

    device = require_gpu()
    args.size = args.size or get_model_spec(args.model).input_size
    rows = pd.read_pickle(ROOT / "results/clean_images.pkl")
    rows = rows.loc[rows["split"].eq("train")].head(8 * args.batches)
    transform, evaluation = image_transforms(args.size, args.augmentation, model_name=args.model)
    _, loaders, _ = make_loaders(rows, transform, evaluation, batch_size=8,
                                 num_workers=args.workers, worker_timeout=60)
    loader = loaders["train"]
    # Random weights avoid downloading anything; this is timing, not a model experiment.
    model, criterion, optimizer, _ = setup_image_model(
        args.model, [0, 1], pretrained=False, device=device)
    unfreeze_backbone(model)
    _train_mode_with_frozen_batchnorm(model)
    scaler = torch.amp.GradScaler("cuda")
    print(f"device={next(model.parameters()).device}, workers={args.workers}, "
          f"size={args.size}, augmentation={args.augmentation}", flush=True)
    print("First batch includes worker/CUDA startup; compare subsequent batches.", flush=True)
    started = time.perf_counter()
    iterator = iter(loader)
    for batch in range(min(args.batches, len(loader))):
        images, labels, _ = next(iterator)
        loaded = time.perf_counter()
        images, labels = images.to(device), labels.to(device)
        torch.cuda.synchronize()
        transferred = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = criterion(model(images).squeeze(1), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        finished = time.perf_counter()
        print(f"batch {batch + 1}: data wait={loaded-started:.3f}s, "
              f"transfer={transferred-loaded:.3f}s, "
              f"forward/backward/step={finished-transferred:.3f}s", flush=True)
        started = time.perf_counter()


if __name__ == "__main__":
    main()
