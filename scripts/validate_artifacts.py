from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / 'artifacts'

REQUIRED = [
    'feature_catalog.csv',
    'layer_metrics.csv',
    'stability.csv',
    'selection_stability.csv',
    'causal_results_final_token.csv',
    'causal_results_max_active.csv',
    'causal_position_summary.csv',
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

    _require_columns(ARTIFACT_DIR / 'feature_catalog.csv', {'layer','concept','feature_id','train_auroc','auroc','f1'})
    _require_columns(ARTIFACT_DIR / 'selection_stability.csv', {'layer','concept','feature_id','resample_support','median_resample_rank'})
    causal_columns = {
        'task_id','concept','feature_id','position_policy','intervention_token_index',
        'feature_active_at_intervention','feature_active_at_final_token','feature_active_anywhere',
        'intervention','condition','target_mean_logprob_delta','js_divergence',
    }
    _require_columns(ARTIFACT_DIR / 'causal_results_final_token.csv', causal_columns)
    _require_columns(ARTIFACT_DIR / 'causal_results_max_active.csv', causal_columns)
    _require_columns(ARTIFACT_DIR / 'causal_position_summary.csv', {
        'concept','position_policy','feature_active_at_intervention_rate','target_specificity_ratio',
        'target_paired_advantage','target_sign_flip_pvalue',
    })
    _require_columns(ARTIFACT_DIR / 'study_feature_summary.csv', {
        'concept','layer','feature_id','heldout_auroc','heldout_f1','candidate_resample_support',
        'final_target_specificity_ratio','max_active_target_specificity_ratio',
        'final_feature_active_at_intervention_rate','max_active_feature_active_at_intervention_rate',
    })

    summary = json.loads((ARTIFACT_DIR / 'study_summary.json').read_text(encoding='utf-8'))
    if int(summary.get('n_concepts', 0)) < 1:
        raise SystemExit('study_summary.json has no concepts.')
    if summary.get('primary_causal_position_policy') != 'max_feature_activation':
        raise SystemExit('study_summary.json must use max_feature_activation as the primary causal policy.')
    if 'causal task' not in str(summary.get('causal_statistical_unit', '')).lower():
        raise SystemExit('study_summary.json must document causal-task-level inference.')

    required_figures = [
        'feature_auroc.png','layer_diagnostics.png','causal_effects.png','feature_set_effects.png',
        'association_vs_causality.png','causal_position_sensitivity.png',
    ]
    missing_figures = [name for name in required_figures if not (ARTIFACT_DIR / 'figures' / name).exists()]
    if missing_figures:
        raise SystemExit(f'Missing report figures: {missing_figures}')

    print('FeatureLens offline artifact validation: PASS')
    print(f"  concepts: {summary['n_concepts']}")
    print(f"  primary causal policy: {summary['primary_causal_position_policy']}")
    print(f"  statistical unit: {summary['causal_statistical_unit']}")
    print('  report: artifacts/report.md')


if __name__ == '__main__':
    main()
