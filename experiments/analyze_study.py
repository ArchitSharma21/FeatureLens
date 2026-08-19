from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.common import ARTIFACT_DIR
from featurelens.stats import paired_bootstrap_difference_ci, paired_sign_flip_pvalue


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


def _paired_specificity(
    frame: pd.DataFrame,
    *,
    effect_column: str,
    task_column: str = 'task_id',
    seed: int,
) -> dict[str, float | list[float]]:
    sae = (
        frame[frame['condition'] == 'sae_feature']
        .groupby(task_column, as_index=False)[effect_column]
        .first()
        .rename(columns={effect_column: 'sae_effect'})
    )
    random = (
        frame[frame['condition'] == 'random_norm_matched']
        .assign(_abs_effect=lambda data: np.abs(data[effect_column].astype(float)))
        .groupby(task_column, as_index=False)['_abs_effect']
        .mean()
        .rename(columns={'_abs_effect': 'random_abs_effect'})
    )
    paired = sae.merge(random, on=task_column, how='inner')
    if paired.empty:
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

    sae_values = paired['sae_effect'].to_numpy(dtype=float)
    sae_abs = np.abs(sae_values)
    random_abs = paired['random_abs_effect'].to_numpy(dtype=float)
    ci_low, ci_high = paired_bootstrap_difference_ci(sae_abs, random_abs, seed=seed)
    return {
        'sae_abs_mean': float(np.mean(sae_abs)),
        'sae_signed_mean': float(np.mean(sae_values)),
        'random_abs_mean': float(np.mean(random_abs)),
        'specificity_ratio': float(np.mean(sae_abs) / max(float(np.mean(random_abs)), 1e-12)),
        'paired_advantage': float(np.mean(sae_abs - random_abs)),
        'paired_advantage_ci_95': [float(ci_low), float(ci_high)],
        'paired_sign_flip_pvalue': float(
            paired_sign_flip_pvalue(sae_abs, random_abs, seed=seed + 1)
        ),
        'n_tasks': int(len(paired)),
    }


def _safe_spearman(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    a = pd.to_numeric(x, errors='coerce').to_numpy(dtype=float)
    b = pd.to_numeric(y, errors='coerce').to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return {'rho': float('nan'), 'pvalue': float('nan'), 'n': int(mask.sum())}
    if np.unique(a[mask]).size < 2 or np.unique(b[mask]).size < 2:
        return {'rho': float('nan'), 'pvalue': float('nan'), 'n': int(mask.sum())}
    result = spearmanr(a[mask], b[mask])
    return {
        'rho': float(result.statistic),
        'pvalue': float(result.pvalue),
        'n': int(mask.sum()),
    }


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    catalog = pd.read_csv(artifact_dir / 'feature_catalog.csv')
    causal = pd.read_csv(artifact_dir / 'causal_results.csv')
    paraphrase = pd.read_csv(artifact_dir / 'stability.csv')
    stability_path = artifact_dir / 'selection_stability.csv'
    selection_stability = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()

    selected = selected_features(catalog)
    rows: list[dict] = []

    for concept_idx, selected_row in selected.sort_values('concept').reset_index(drop=True).iterrows():
        concept = str(selected_row['concept'])
        layer = int(selected_row['layer'])
        feature_id = int(selected_row['feature_id'])
        causal_concept = causal[
            (causal['concept'] == concept) & (causal['intervention'] == 'ablate')
        ].copy()
        target = _paired_specificity(
            causal_concept,
            effect_column='target_mean_logprob_delta',
            seed=args.seed + 100 * concept_idx,
        )
        js = _paired_specificity(
            causal_concept,
            effect_column='js_divergence',
            seed=args.seed + 100 * concept_idx + 17,
        )
        sae_rows = causal_concept[causal_concept['condition'] == 'sae_feature']

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

        rows.append(
            {
                'concept': concept,
                'layer': layer,
                'feature_id': feature_id,
                'train_auroc': float(selected_row['train_auroc']),
                'heldout_auroc': float(selected_row['auroc']),
                'heldout_f1': float(selected_row['f1']),
                'activation_rate_pos_train': float(selected_row['activation_rate_pos']),
                'activation_rate_neg_train': float(selected_row['activation_rate_neg']),
                'candidate_resample_support': (
                    float(selection_row.iloc[0]['resample_support'])
                    if not selection_row.empty
                    else 0.0
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
                'causal_feature_active_rate': (
                    float(np.mean(sae_rows['feature_activation'].astype(float) > 0.0))
                    if not sae_rows.empty
                    else float('nan')
                ),
                'target_sae_abs_mean': target['sae_abs_mean'],
                'target_sae_signed_mean': target['sae_signed_mean'],
                'target_random_abs_mean': target['random_abs_mean'],
                'target_specificity_ratio': target['specificity_ratio'],
                'target_paired_advantage': target['paired_advantage'],
                'target_paired_ci_low': float(target['paired_advantage_ci_95'][0]),
                'target_paired_ci_high': float(target['paired_advantage_ci_95'][1]),
                'target_sign_flip_pvalue': target['paired_sign_flip_pvalue'],
                'js_sae_mean': js['sae_abs_mean'],
                'js_random_mean': js['random_abs_mean'],
                'js_specificity_ratio': js['specificity_ratio'],
                'js_paired_advantage': js['paired_advantage'],
                'js_paired_ci_low': float(js['paired_advantage_ci_95'][0]),
                'js_paired_ci_high': float(js['paired_advantage_ci_95'][1]),
                'js_sign_flip_pvalue': js['paired_sign_flip_pvalue'],
                'causal_tasks': int(target['n_tasks']),
            }
        )

    study = pd.DataFrame(rows)
    study_path = artifact_dir / 'study_feature_summary.csv'
    study.to_csv(study_path, index=False)

    correlations = {
        'heldout_auroc_vs_target_specificity': _safe_spearman(
            study['heldout_auroc'], study['target_specificity_ratio']
        ),
        'heldout_auroc_vs_js_specificity': _safe_spearman(
            study['heldout_auroc'], study['js_specificity_ratio']
        ),
        'heldout_f1_vs_target_specificity': _safe_spearman(
            study['heldout_f1'], study['target_specificity_ratio']
        ),
        'candidate_resample_support_vs_target_specificity': _safe_spearman(
            study['candidate_resample_support'], study['target_specificity_ratio']
        ),
    }

    most_predictive = study.sort_values('heldout_auroc', ascending=False).iloc[0]
    most_target_specific = study.sort_values('target_specificity_ratio', ascending=False).iloc[0]
    most_js_specific = study.sort_values('js_specificity_ratio', ascending=False).iloc[0]
    median_support = float(study['candidate_resample_support'].median())

    summary = {
        'n_concepts': int(len(study)),
        'selected_feature_pooling': 'prompt-wide max SAE activation across non-padding prompt tokens',
        'dense_probe_pooling': 'final prompt token residual',
        'median_selected_feature_resample_support': median_support,
        'most_predictive_concept': {
            'concept': str(most_predictive['concept']),
            'heldout_auroc': float(most_predictive['heldout_auroc']),
        },
        'highest_target_specificity': {
            'concept': str(most_target_specific['concept']),
            'ratio': float(most_target_specific['target_specificity_ratio']),
        },
        'highest_js_specificity': {
            'concept': str(most_js_specific['concept']),
            'ratio': float(most_js_specific['js_specificity_ratio']),
        },
        'correlations': correlations,
        'guardrail': (
            'Cross-concept Spearman correlations are descriptive because the study contains only seven '
            'controlled concepts. Causal specificity remains evaluated per concept against matched random '
            'controls rather than inferred from correlation alone.'
        ),
    }
    (artifact_dir / 'study_summary.json').write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8',
    )
    print(f'Wrote {study_path}')
    print(f'Wrote {artifact_dir / "study_summary.json"}')


if __name__ == '__main__':
    main()
