from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.common import ARTIFACT_DIR
from featurelens.stats import paired_bootstrap_difference_ci, paired_sign_flip_pvalue

POLICIES = ('final_token', 'max_feature_activation')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Aggregate held-out association, stability, and causal evidence by concept.'
    )
    parser.add_argument('--artifact-dir', type=Path, default=ARTIFACT_DIR)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def selected_features(catalog: pd.DataFrame) -> pd.DataFrame:
    scored = catalog.copy()
    scored['activation_contrast'] = (
        scored['activation_rate_pos'].astype(float) - scored['activation_rate_neg'].astype(float)
    )
    ordered = scored.sort_values(
        ['concept', 'train_auroc', 'activation_contrast'],
        ascending=[True, False, False],
    )
    return ordered.groupby('concept', as_index=False).first()


def causal_path(artifact_dir: Path, policy: str) -> Path:
    explicit = artifact_dir / f'causal_results_{policy if policy == "final_token" else "max_active"}.csv'
    if explicit.exists():
        return explicit
    if policy == 'final_token':
        legacy = artifact_dir / 'causal_results.csv'
        if legacy.exists():
            return legacy
    return explicit


def _task_level_specificity(
    frame: pd.DataFrame,
    *,
    effect_column: str,
    seed: int,
    active_only: bool = False,
) -> dict[str, float | list[float]]:
    if frame.empty:
        return _empty_specificity()

    sae_rows = frame[frame['condition'] == 'sae_feature'].copy()
    if active_only:
        active_column = (
            'feature_active_at_intervention'
            if 'feature_active_at_intervention' in sae_rows.columns
            else 'feature_activation'
        )
        if active_column == 'feature_activation':
            active_tasks = sae_rows.loc[sae_rows[active_column].astype(float) > 0.0, 'task_id'].unique()
        else:
            active_tasks = sae_rows.loc[sae_rows[active_column].astype(float) > 0.0, 'task_id'].unique()
        frame = frame[frame['task_id'].isin(active_tasks)].copy()
        sae_rows = frame[frame['condition'] == 'sae_feature'].copy()

    sae = (
        sae_rows.assign(
            _abs_effect=lambda data: np.abs(pd.to_numeric(data[effect_column], errors='coerce'))
        )
        .groupby('task_id', as_index=False)
        .agg(sae_abs_effect=('_abs_effect', 'mean'), sae_signed_effect=(effect_column, 'mean'))
    )
    random_rows = frame[frame['condition'] == 'random_norm_matched'].assign(
        _abs_effect=lambda data: np.abs(pd.to_numeric(data[effect_column], errors='coerce'))
    )
    if 'intervention' in random_rows.columns:
        random = (
            random_rows.groupby(['task_id', 'intervention'], as_index=False)['_abs_effect']
            .mean()
            .groupby('task_id', as_index=False)['_abs_effect']
            .mean()
            .rename(columns={'_abs_effect': 'random_abs_effect'})
        )
    else:
        random = (
            random_rows.groupby('task_id', as_index=False)['_abs_effect']
            .mean()
            .rename(columns={'_abs_effect': 'random_abs_effect'})
        )
    paired = sae.merge(random, on='task_id', how='inner')
    if paired.empty:
        return _empty_specificity()

    sae_abs = paired['sae_abs_effect'].to_numpy(dtype=float)
    sae_signed = paired['sae_signed_effect'].to_numpy(dtype=float)
    random_abs = paired['random_abs_effect'].to_numpy(dtype=float)
    ci_low, ci_high = paired_bootstrap_difference_ci(sae_abs, random_abs, seed=seed)
    return {
        'sae_abs_mean': float(np.mean(sae_abs)),
        'sae_signed_mean': float(np.mean(sae_signed)),
        'random_abs_mean': float(np.mean(random_abs)),
        'specificity_ratio': float(np.mean(sae_abs) / max(float(np.mean(random_abs)), 1e-12)),
        'paired_advantage': float(np.mean(sae_abs - random_abs)),
        'paired_advantage_ci_95': [float(ci_low), float(ci_high)],
        'paired_sign_flip_pvalue': float(paired_sign_flip_pvalue(sae_abs, random_abs, seed=seed + 1)),
        'n_tasks': int(len(paired)),
    }


def _empty_specificity() -> dict[str, float | list[float]]:
    return {
        'sae_abs_mean': float('nan'),
        'sae_signed_mean': float('nan'),
        'random_abs_mean': float('nan'),
        'specificity_ratio': float('nan'),
        'paired_advantage': float('nan'),
        'paired_advantage_ci_95': [float('nan'), float('nan')],
        'paired_sign_flip_pvalue': float('nan'),
        'n_tasks': 0,
    }


def _paired_specificity(
    frame: pd.DataFrame,
    *,
    effect_column: str,
    task_column: str = 'task_id',
    seed: int,
) -> dict[str, float | list[float]]:
    """Backward-compatible public helper using task-level inference."""
    if task_column != 'task_id':
        frame = frame.rename(columns={task_column: 'task_id'})
    return _task_level_specificity(frame, effect_column=effect_column, seed=seed)


def _safe_spearman(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    a = pd.to_numeric(x, errors='coerce').to_numpy(dtype=float)
    b = pd.to_numeric(y, errors='coerce').to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return {'rho': float('nan'), 'pvalue': float('nan'), 'n': int(mask.sum())}
    if np.unique(a[mask]).size < 2 or np.unique(b[mask]).size < 2:
        return {'rho': float('nan'), 'pvalue': float('nan'), 'n': int(mask.sum())}
    result = spearmanr(a[mask], b[mask])
    return {'rho': float(result.statistic), 'pvalue': float(result.pvalue), 'n': int(mask.sum())}


def _coverage(sae_rows: pd.DataFrame, column: str, fallback: str | None = None) -> float:
    if column in sae_rows.columns:
        return float(pd.to_numeric(sae_rows[column], errors='coerce').fillna(0).astype(float).gt(0).mean())
    if fallback and fallback in sae_rows.columns:
        return float(pd.to_numeric(sae_rows[fallback], errors='coerce').fillna(0).astype(float).gt(0).mean())
    return float('nan')


def _policy_summary(frame: pd.DataFrame, policy: str, seed: int) -> dict:
    target = _task_level_specificity(frame, effect_column='target_mean_logprob_delta', seed=seed)
    target_active = _task_level_specificity(
        frame, effect_column='target_mean_logprob_delta', seed=seed + 7, active_only=True
    )
    js = _task_level_specificity(frame, effect_column='js_divergence', seed=seed + 17)
    js_active = _task_level_specificity(
        frame, effect_column='js_divergence', seed=seed + 29, active_only=True
    )
    sae_rows = frame[frame['condition'] == 'sae_feature']
    return {
        'position_policy': policy,
        'tasks': int(target['n_tasks']),
        'feature_active_at_intervention_rate': _coverage(
            sae_rows, 'feature_active_at_intervention', fallback='feature_activation'
        ),
        'feature_active_at_final_token_rate': _coverage(
            sae_rows, 'feature_active_at_final_token', fallback='feature_activation'
        ),
        'feature_active_anywhere_rate': _coverage(
            sae_rows, 'feature_active_anywhere', fallback='feature_activation'
        ),
        'target_sae_abs_mean': target['sae_abs_mean'],
        'target_random_abs_mean': target['random_abs_mean'],
        'target_specificity_ratio': target['specificity_ratio'],
        'target_paired_advantage': target['paired_advantage'],
        'target_paired_ci_low': float(target['paired_advantage_ci_95'][0]),
        'target_paired_ci_high': float(target['paired_advantage_ci_95'][1]),
        'target_sign_flip_pvalue': target['paired_sign_flip_pvalue'],
        'active_target_sae_abs_mean': target_active['sae_abs_mean'],
        'active_target_random_abs_mean': target_active['random_abs_mean'],
        'active_target_specificity_ratio': target_active['specificity_ratio'],
        'active_target_paired_advantage': target_active['paired_advantage'],
        'active_target_paired_ci_low': float(target_active['paired_advantage_ci_95'][0]),
        'active_target_paired_ci_high': float(target_active['paired_advantage_ci_95'][1]),
        'active_target_sign_flip_pvalue': target_active['paired_sign_flip_pvalue'],
        'active_tasks': int(target_active['n_tasks']),
        'js_sae_mean': js['sae_abs_mean'],
        'js_random_mean': js['random_abs_mean'],
        'js_specificity_ratio': js['specificity_ratio'],
        'active_js_specificity_ratio': js_active['specificity_ratio'],
    }


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    catalog = pd.read_csv(artifact_dir / 'feature_catalog.csv')
    paraphrase = pd.read_csv(artifact_dir / 'stability.csv')
    stability_path = artifact_dir / 'selection_stability.csv'
    selection_stability = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()

    causal_by_policy: dict[str, pd.DataFrame] = {}
    for policy in POLICIES:
        path = causal_path(artifact_dir, policy)
        if path.exists():
            frame = pd.read_csv(path)
            if 'position_policy' not in frame.columns:
                frame['position_policy'] = policy
            causal_by_policy[policy] = frame

    if 'final_token' not in causal_by_policy or 'max_feature_activation' not in causal_by_policy:
        missing = [policy for policy in POLICIES if policy not in causal_by_policy]
        raise SystemExit(f'Missing causal position policies: {missing}')

    selected = selected_features(catalog)
    position_rows: list[dict] = []
    for policy_idx, policy in enumerate(POLICIES):
        overall = _policy_summary(causal_by_policy[policy], policy, args.seed + 1000 * policy_idx)
        overall['concept'] = '__all__'
        position_rows.append(overall)
        for concept_idx, concept in enumerate(sorted(selected['concept'].astype(str).unique())):
            subset = causal_by_policy[policy][causal_by_policy[policy]['concept'] == concept].copy()
            row = _policy_summary(
                subset,
                policy,
                args.seed + 1000 * policy_idx + 100 * (concept_idx + 1),
            )
            row['concept'] = concept
            position_rows.append(row)

    position_summary = pd.DataFrame(position_rows)
    position_summary.to_csv(artifact_dir / 'causal_position_summary.csv', index=False)

    rows: list[dict] = []
    for concept_idx, selected_row in selected.sort_values('concept').reset_index(drop=True).iterrows():
        concept = str(selected_row['concept'])
        layer = int(selected_row['layer'])
        feature_id = int(selected_row['feature_id'])
        para_rows = paraphrase[
            (paraphrase['concept'] == concept) & (paraphrase['layer'].astype(int) == layer)
        ]
        selection_row = pd.DataFrame()
        if not selection_stability.empty:
            selection_row = selection_stability[
                (selection_stability['concept'] == concept)
                & (selection_stability['layer'].astype(int) == layer)
                & (selection_stability['feature_id'].astype(int) == feature_id)
            ]

        base = {
            'concept': concept,
            'layer': layer,
            'feature_id': feature_id,
            'train_auroc': float(selected_row['train_auroc']),
            'heldout_auroc': float(selected_row['auroc']),
            'heldout_f1': float(selected_row['f1']),
            'activation_rate_pos_train': float(selected_row['activation_rate_pos']),
            'activation_rate_neg_train': float(selected_row['activation_rate_neg']),
            'candidate_resample_support': (
                float(selection_row.iloc[0]['resample_support']) if not selection_row.empty else 0.0
            ),
            'candidate_median_resample_rank': (
                float(selection_row.iloc[0]['median_resample_rank'])
                if not selection_row.empty
                else float('nan')
            ),
            'mean_paraphrase_topk_jaccard': (
                float(para_rows['topk_jaccard'].mean()) if not para_rows.empty else float('nan')
            ),
            'mean_paraphrase_sparse_cosine': (
                float(para_rows['sparse_cosine'].mean()) if not para_rows.empty else float('nan')
            ),
        }
        for policy in POLICIES:
            policy_frame = causal_by_policy[policy]
            subset = policy_frame[policy_frame['concept'] == concept].copy()
            summary = _policy_summary(
                subset,
                policy,
                args.seed + 10_000 + 1000 * POLICIES.index(policy) + 100 * concept_idx,
            )
            prefix = 'final' if policy == 'final_token' else 'max_active'
            for key, value in summary.items():
                if key in {'position_policy'}:
                    continue
                base[f'{prefix}_{key}'] = value
        base['target_specificity_gain_max_vs_final'] = (
            float(base['max_active_target_specificity_ratio'])
            - float(base['final_target_specificity_ratio'])
        )
        base['js_specificity_gain_max_vs_final'] = (
            float(base['max_active_js_specificity_ratio'])
            - float(base['final_js_specificity_ratio'])
        )
        rows.append(base)

    study = pd.DataFrame(rows)
    study_path = artifact_dir / 'study_feature_summary.csv'
    study.to_csv(study_path, index=False)

    correlations = {
        'heldout_auroc_vs_max_active_target_specificity': _safe_spearman(
            study['heldout_auroc'], study['max_active_target_specificity_ratio']
        ),
        'heldout_auroc_vs_max_active_js_specificity': _safe_spearman(
            study['heldout_auroc'], study['max_active_js_specificity_ratio']
        ),
        'heldout_f1_vs_max_active_target_specificity': _safe_spearman(
            study['heldout_f1'], study['max_active_target_specificity_ratio']
        ),
        'candidate_resample_support_vs_max_active_target_specificity': _safe_spearman(
            study['candidate_resample_support'], study['max_active_target_specificity_ratio']
        ),
    }

    overall = position_summary[position_summary['concept'] == '__all__'].set_index('position_policy')
    final_row = overall.loc['final_token']
    max_row = overall.loc['max_feature_activation']
    most_predictive = study.sort_values('heldout_auroc', ascending=False).iloc[0]
    most_target_specific = study.sort_values('max_active_target_specificity_ratio', ascending=False).iloc[0]
    most_js_specific = study.sort_values('max_active_js_specificity_ratio', ascending=False).iloc[0]

    summary = {
        'n_concepts': int(len(study)),
        'selected_feature_pooling': 'prompt-wide max SAE activation across non-padding prompt tokens',
        'dense_probe_pooling': 'final prompt token residual',
        'primary_causal_position_policy': 'max_feature_activation',
        'causal_statistical_unit': 'causal task; ablation and amplification are averaged within task before paired inference',
        'median_selected_feature_resample_support': float(study['candidate_resample_support'].median()),
        'final_token_feature_coverage': float(final_row['feature_active_at_intervention_rate']),
        'max_active_feature_coverage': float(max_row['feature_active_at_intervention_rate']),
        'final_token_target_specificity_ratio': float(final_row['target_specificity_ratio']),
        'max_active_target_specificity_ratio': float(max_row['target_specificity_ratio']),
        'final_token_target_paired_advantage': float(final_row['target_paired_advantage']),
        'max_active_target_paired_advantage': float(max_row['target_paired_advantage']),
        'final_token_target_paired_ci_95': [
            float(final_row['target_paired_ci_low']), float(final_row['target_paired_ci_high'])
        ],
        'max_active_target_paired_ci_95': [
            float(max_row['target_paired_ci_low']), float(max_row['target_paired_ci_high'])
        ],
        'final_token_target_sign_flip_pvalue': float(final_row['target_sign_flip_pvalue']),
        'max_active_target_sign_flip_pvalue': float(max_row['target_sign_flip_pvalue']),
        'most_predictive_concept': {
            'concept': str(most_predictive['concept']),
            'heldout_auroc': float(most_predictive['heldout_auroc']),
        },
        'highest_max_active_target_specificity': {
            'concept': str(most_target_specific['concept']),
            'ratio': float(most_target_specific['max_active_target_specificity_ratio']),
        },
        'highest_max_active_js_specificity': {
            'concept': str(most_js_specific['concept']),
            'ratio': float(most_js_specific['max_active_js_specificity_ratio']),
        },
        'correlations': correlations,
        'guardrail': (
            'Max-active causal positions are selected from SAE activation only, never from behavioral outcome. '
            'Cross-concept Spearman correlations are descriptive because the study contains seven concepts.'
        ),
    }
    (artifact_dir / 'study_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Wrote {artifact_dir / "causal_position_summary.csv"}')
    print(f'Wrote {study_path}')
    print(f'Wrote {artifact_dir / "study_summary.json"}')


if __name__ == '__main__':
    main()
