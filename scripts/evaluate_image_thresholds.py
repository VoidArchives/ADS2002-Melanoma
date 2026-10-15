"""Compare validation thresholds from saved OOF scores, without training or a GPU."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results/cross_validation')
    parser.add_argument('--output', type=Path, help='Defaults to <results>/thresholds')
    parser.add_argument('--recall-targets', nargs='+', type=float, default=[0.90, 0.95])
    parser.add_argument('--models', nargs='+')
    args = parser.parse_args()

    import json
    import pandas as pd
    from melanoma_detection.evaluation import compare_validation_thresholds

    tables = {}
    paths = {}
    for path in sorted(args.results.glob('*/out_of_fold_predictions.csv')):
        table = pd.read_csv(path)
        if table['model'].nunique() != 1:
            raise ValueError(f'Expected one model in {path}')
        name = table['model'].iloc[0]
        if args.models and name not in args.models:
            continue
        if name in tables:
            raise ValueError(f'Multiple saved runs for {name}; put the intended runs in a separate results directory.')
        if 'split' in table and table['split'].eq('test').any():
            raise ValueError('Threshold tuning must not use test predictions.')
        if not table.groupby('patient_id').fold.nunique().eq(1).all():
            raise ValueError(f'Patients occur in multiple validation folds: {path}')
        tables[name] = table.assign(score=table['probability'])
        paths[name] = str(path.resolve())
    if not tables:
        raise FileNotFoundError(f'No saved OOF predictions found in {args.results}')
    if args.models and set(args.models) - set(tables):
        raise FileNotFoundError(f'Missing models: {set(args.models) - set(tables)}')
    reference = next(iter(tables.values())).set_index('image_name').target
    pooled = compare_validation_thresholds(tables, reference, args.recall_targets)
    pooled.insert(1, 'scope', 'pooled OOF tuning diagnostic')
    folds = []
    for name, table in tables.items():
        for fold, subset in table.groupby('fold'):
            report = compare_validation_thresholds(
                {name: subset}, subset.set_index('image_name').target, args.recall_targets)
            report.insert(1, 'fold', fold)
            report['checkpoint'] = str(Path(paths[name]).parent / f'fold_{fold}' /
                                       Path(paths[name]).parent.name / 'model.pt')
            folds.append(report)
    output = args.output or args.results / 'thresholds'
    output.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output / 'pooled_threshold_comparison.csv', index=False)
    pd.concat(folds, ignore_index=True).to_csv(output / 'fold_thresholds.csv', index=False)
    (output / 'report.json').write_text(json.dumps({
        'sources': paths, 'recall_targets': args.recall_targets,
        'interpretation': 'Thresholds selected and measured on validation scores. Development diagnostics only. '
                          'Freeze checkpoint-specific thresholds before independent test evaluation. '
                          'Pooled OOF thresholds do not calibrate a refitted model or fold ensemble.',
    }, indent=2), encoding='utf-8')
    print(pooled[['model', 'policy', 'threshold', 'recall', 'specificity', 'precision',
                  'TN', 'FP', 'FN', 'TP']].round(4).to_string(index=False))
    print(f'\nValidation tuning diagnostics only. Reports saved to {output}')


if __name__ == '__main__':
    main()
