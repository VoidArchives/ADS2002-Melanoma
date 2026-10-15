"""Run fair architecture screening from the project root (never evaluates test)."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=ROOT / 'results/clean_images.pkl')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/experiments')
    parser.add_argument('--folds', type=Path, default=ROOT / 'results')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/image_model.yaml')
    parser.add_argument('--models', nargs='+', help='Defaults to all eight registry architectures')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument('--reuse-only', action='store_true', help='Load completed folds; never train missing runs')
    reuse.add_argument('--force-retrain', action='store_true', help='Retrain and replace existing results')
    args = parser.parse_args()

    import pandas as pd
    from melanoma_detection.experiments import ImageExperimentConfig, benchmark_image_models
    from melanoma_detection.model_registry import MODEL_REGISTRY
    from melanoma_detection.splitting import load_or_create_patient_folds
    from melanoma_detection.utils import load_config, setup_logging

    setup_logging()
    options = load_config(args.config) or {}
    if args.workers is not None:
        options['num_workers'] = args.workers
    configs = [ImageExperimentConfig(**{**options, 'model_name': name})
               for name in (args.models or list(MODEL_REGISTRY))]
    rows = pd.read_pickle(args.images)
    folds = load_or_create_patient_folds(rows, args.folds, seed=configs[0].seed)
    mode = 'never' if args.force_retrain else 'only' if args.reuse_only else 'auto'
    result = benchmark_image_models(rows, args.output, configs=configs, split_maps=folds,
                                    device=args.device, reuse=mode)
    print(result['summary'].to_string(index=False))


if __name__ == '__main__':
    main()
