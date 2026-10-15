"""Small shared helpers: seeding, device selection, checkpoints, config, logging.

This module sits at the bottom of the dependency chain, so it must not import
any other module from the package. torch is imported lazily so that the
non-deep-learning modules (splitting, evaluation, ...) work without it.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


@dataclass(frozen=True)
class ProjectPaths:
    """Every file location the notebooks share, so no path is typed twice."""

    root: Path
    data: Path
    labels: Path
    images: Path
    results: Path
    checkpoints: Path
    predictions: Path
    features: Path
    models: Path
    splits: Path
    clean_images: Path
    image_audit: Path
    removal_log: Path


def get_project_paths(
    root: str | Path, data_dir: str | Path | None = None, images_dir: str | Path | None = None
) -> ProjectPaths:
    """Build (and create) the project's standard folders.

    Expects ``<data_dir>/train_labels.csv`` and an image folder of JPEGs (default
    ``<data_dir>/train/``, override with ``images_dir``); ``data_dir`` defaults to
    ``<root>/data``. Everything written by the notebooks goes under ``<root>/results``.
    """
    root = Path(root).resolve()
    data = Path(data_dir).resolve() if data_dir is not None else root / "data"
    images = Path(images_dir).resolve() if images_dir is not None else data / "train"
    results = root / "results"
    paths = ProjectPaths(
        root=root, data=data,
        labels=data / "train_labels.csv", images=images,
        results=results, checkpoints=results / "checkpoints",
        predictions=results / "predictions", features=results / "features",
        models=results / "models",
        splits=results / "splits.csv",
        clean_images=results / "clean_images.pkl",
        image_audit=results / "image_audit.pkl",
        removal_log=results / "duplicate_removal_log.csv",
    )
    for folder in (paths.results, paths.checkpoints, paths.predictions, paths.features, paths.models):
        folder.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(level: int = logging.INFO) -> None:
    """Make ``logger.info`` messages from the package visible in a notebook."""
    logging.basicConfig(level=level, format="%(message)s", force=True)


def set_seed(seed: int = RANDOM_STATE, loader_generator: Any = None) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU and all GPUs).

    ``loader_generator`` is an optional ``torch.Generator`` (the one passed to the
    training ``DataLoader``); it is re-seeded so shuffling restarts identically.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is optional for the tabular-only workflow
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if loader_generator is not None:
        loader_generator.manual_seed(seed)


def get_device(device: str | None = None):
    """Return a ``torch.device``: the one requested, else CUDA if present, else CPU."""
    import torch

    return torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def require_gpu():
    """Return the CUDA device, or raise a clear error if PyTorch cannot see a GPU.

    Use this instead of :func:`get_device` when training must NOT silently fall back
    to the CPU.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU visible to PyTorch "
            f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}). "
            "A '+cpu' build of torch is the usual cause: reinstall the CUDA build from "
            "https://pytorch.org/get-started/locally/ and check that `nvidia-smi` works."
        )
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    logger.info("Using GPU: %s (%.1f GB)", properties.name, properties.total_memory / 1e9)
    return device


def save_checkpoint(state_dict: dict, path: str | Path, **metadata: Any) -> Path:
    """Save a model ``state_dict`` plus any metadata (epoch, metric, model name...)."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    # Replace only after serialization succeeds, retaining the previous checkpoint
    # if a write fails or the process is interrupted.
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": cpu_state, "metadata": metadata}, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a checkpoint written by :func:`save_checkpoint`."""
    import torch

    return torch.load(path, map_location=map_location, weights_only=True)


def load_config(path: str | Path) -> dict:
    """Read a YAML config file (paths, hyperparameters, seeds) into a dict."""
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
