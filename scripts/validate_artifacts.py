from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / 'artifacts'

REQUIRED = [
    'activations/metadata.json',
    'feature_catalog.csv',
    'layer_metrics.csv',
    'stability.csv',
    'selection_stability.csv',
    'causal_results.csv',
    'feature_set_results.csv',
    'study_feature_summary.csv',
    'study_summary.json',
    'summary.json',
    'report.md',
]


def _require_columns(path: Path, columns: set[str]) -> None:
    frame = pd.read_csv(path)
    missing = columns.difference(frame.columns)
    if missing:
        raise SystemExit(f'{path.relative_to(ROOT)} missing columns: {sorted(missing)}')
    if frame.empty:
        raise SystemExit(f'{path.relative_to(ROOT)} is empty.')


def main() -> None:
    missing = [name for name in REQUIRED if not (ARTIFACT_DIR / name).exists()]
    if missing:
        raise SystemExit(f'Missing offline-study artifacts: {missing}')

    metadata = json.loads((ARTIFACT_DIR / 'activations' / 'metadata.json').read_text(encoding='utf-8'))
    pooling = str(metadata.get('feature_pooling', ''))
    if 'prompt-wide' not in pooling:
        raise SystemExit(
            'Activation metadata is not v0.14 prompt-wide. Rerun experiments.collect_activations.'
        )

    _require_columns(
        ARTIFACT_DIR / 'feature_catalog.csv',
        {'layer', 'concept', 'feature_id', 'train_auroc', 'auroc', 'f1'},
    )
    _require_columns(
        ARTIFACT_DIR / 'selection_stability.csv',
        {'layer', 'concept', 'feature_id', 'resample_support', 'median_resample_rank'},
    )
    _require_columns(
        ARTIFACT_DIR / 'causal_results.csv',
        {
            'task_id',
            'concept',
            'feature_id',
            'intervention',
            'condition',
            'target_mean_logprob_delta',
            'js_divergence',
        },
    )
    _require_columns(
        ARTIFACT_DIR / 'study_feature_summary.csv',
        {
            'concept',
            'layer',
            'feature_id',
            'heldout_auroc',
            'heldout_f1',
            'candidate_resample_support',
            'target_specificity_ratio',
            'js_specificity_ratio',
        },
    )

    summary = json.loads((ARTIFACT_DIR / 'study_summary.json').read_text(encoding='utf-8'))
    if int(summary.get('n_concepts', 0)) < 1:
        raise SystemExit('study_summary.json has no concepts.')

    required_figures = [
        'feature_auroc.png',
        'layer_diagnostics.png',
        'causal_effects.png',
        'feature_set_effects.png',
        'association_vs_causality.png',
        'candidate_stability.png',
    ]
    missing_figures = [
        name for name in required_figures if not (ARTIFACT_DIR / 'figures' / name).exists()
    ]
    if missing_figures:
        raise SystemExit(f'Missing report figures: {missing_figures}')

    print('FeatureLens offline artifact validation: PASS')
    print(f"  concepts: {summary['n_concepts']}")
    print(f"  feature pooling: {summary['selected_feature_pooling']}")
    print('  report: artifacts/report.md')


if __name__ == '__main__':
    main()
