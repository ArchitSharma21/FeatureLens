from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import ARTIFACT_DIR
from featurelens.stats import (
    bootstrap_mean_ci,
    paired_bootstrap_difference_ci,
    paired_sign_flip_pvalue,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a truthful experiment report from saved metrics.')
    parser.add_argument('--artifact-dir', type=Path, default=ARTIFACT_DIR)
    return parser.parse_args()


def _selected_features(catalog: pd.DataFrame) -> pd.DataFrame:
    scored = catalog.copy()
    scored['activation_contrast'] = scored['activation_rate_pos'] - scored['activation_rate_neg']
    ordered = scored.sort_values(
        ['concept', 'train_auroc', 'activation_contrast'],
        ascending=[True, False, False],
    )
    return ordered.groupby('concept', as_index=False).first()


def _effect_column(frame: pd.DataFrame) -> str:
    """Prefer the full-continuation length-normalized metric, with legacy fallback."""
    if 'target_mean_logprob_delta' in frame.columns:
        return 'target_mean_logprob_delta'
    return 'target_logprob_delta'


def _save_plots(
    artifact_dir: Path,
    selected: pd.DataFrame,
    layers: pd.DataFrame,
    causal: pd.DataFrame,
    feature_sets: pd.DataFrame | None,
) -> None:
    fig_dir = artifact_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(7.5, 4.2))
    ax = figure.add_subplot(111)
    ordered = selected.sort_values('auroc')
    ax.barh(ordered['concept'], ordered['auroc'])
    ax.axvline(0.5, linewidth=1, linestyle='--')
    ax.set_xlabel('Held-out AUROC')
    ax.set_title('Selected SAE feature predictiveness')
    figure.tight_layout()
    figure.savefig(fig_dir / 'feature_auroc.png', dpi=160)
    plt.close(figure)

    figure = plt.figure(figsize=(7.0, 4.2))
    ax = figure.add_subplot(111)
    ax.plot(
        layers['layer'],
        layers['linear_probe_macro_auroc'],
        marker='o',
        label='Linear probe AUROC',
    )
    ax.plot(
        layers['layer'],
        layers['reconstruction_cosine'],
        marker='o',
        label='SAE reconstruction cosine',
    )
    ax.set_xlabel('Layer')
    ax.set_ylim(0, 1.05)
    ax.set_title('Layer-wise representation diagnostics')
    ax.legend()
    figure.tight_layout()
    figure.savefig(fig_dir / 'layer_diagnostics.png', dpi=160)
    plt.close(figure)

    causal_metric = _effect_column(causal)
    grouped = (
        causal.groupby(['intervention', 'condition'])[causal_metric]
        .apply(lambda values: float(np.mean(np.abs(values))))
        .reset_index(name='mean_abs_effect')
    )
    pivot = grouped.pivot(index='intervention', columns='condition', values='mean_abs_effect')
    figure = plt.figure(figsize=(7.0, 4.2))
    ax = figure.add_subplot(111)
    pivot.plot(kind='bar', ax=ax)
    ax.set_ylabel('Mean |Δ mean log p/token|' if causal_metric == 'target_mean_logprob_delta' else 'Mean |Δ log p(target)|')
    ax.set_title('Single-feature SAE edits vs norm-matched controls')
    ax.tick_params(axis='x', rotation=0)
    figure.tight_layout()
    figure.savefig(fig_dir / 'causal_effects.png', dpi=160)
    plt.close(figure)

    if feature_sets is not None and not feature_sets.empty:
        set_metric = _effect_column(feature_sets)
        grouped_sets = (
            feature_sets.groupby(['set_size', 'condition'])[set_metric]
            .apply(lambda values: float(np.mean(np.abs(values))))
            .reset_index(name='mean_abs_effect')
        )
        set_pivot = grouped_sets.pivot(index='set_size', columns='condition', values='mean_abs_effect')
        figure = plt.figure(figsize=(7.0, 4.2))
        ax = figure.add_subplot(111)
        set_pivot.plot(kind='line', marker='o', ax=ax)
        ax.set_xlabel('Jointly ablated feature count')
        ax.set_ylabel('Mean |Δ mean log p/token|' if set_metric == 'target_mean_logprob_delta' else 'Mean |Δ log p(target)|')
        ax.set_title('Distributed feature-set causal effect')
        figure.tight_layout()
        figure.savefig(fig_dir / 'feature_set_effects.png', dpi=160)
        plt.close(figure)



def _save_study_plots(artifact_dir: Path, study: pd.DataFrame) -> None:
    if study.empty:
        return
    fig_dir = artifact_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(7.2, 4.6))
    ax = figure.add_subplot(111)
    ax.scatter(study['heldout_auroc'], study['target_specificity_ratio'])
    for row in study.itertuples():
        ax.annotate(str(row.concept), (row.heldout_auroc, row.target_specificity_ratio), fontsize=8)
    ax.set_xlabel('Held-out feature AUROC')
    ax.set_ylabel('Target causal specificity ratio')
    ax.set_title('Association evidence vs random-normalized causality')
    figure.tight_layout()
    figure.savefig(fig_dir / 'association_vs_causality.png', dpi=160)
    plt.close(figure)

    figure = plt.figure(figsize=(7.4, 4.6))
    ax = figure.add_subplot(111)
    ordered = study.sort_values('candidate_resample_support')
    ax.barh(ordered['concept'], ordered['candidate_resample_support'])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel('Selection support across activation resamples')
    ax.set_title('Selected-feature candidate stability')
    figure.tight_layout()
    figure.savefig(fig_dir / 'candidate_stability.png', dpi=160)
    plt.close(figure)

def _paired_stats(
    frame: pd.DataFrame,
    *,
    index: list[str],
    sae_condition: str,
    random_condition: str,
    seed: int,
) -> dict[str, float | list[float]]:
    """Pair one SAE effect with the mean absolute effect of its random-control ensemble."""
    metric = _effect_column(frame)
    sae = (
        frame[frame['condition'] == sae_condition]
        .groupby(index, as_index=False)[metric]
        .first()
        .rename(columns={metric: 'sae_effect'})
    )
    random = (
        frame[frame['condition'] == random_condition]
        .assign(_abs_effect=lambda data: np.abs(data[metric].astype(float)))
        .groupby(index, as_index=False)['_abs_effect']
        .mean()
        .rename(columns={'_abs_effect': 'random_abs_effect'})
    )
    paired = sae.merge(random, on=index, how='inner')
    sae_abs = np.abs(paired['sae_effect'].to_numpy(dtype=float))
    random_abs = paired['random_abs_effect'].to_numpy(dtype=float)
    if sae_abs.size == 0:
        return {
            'sae_abs': float('nan'),
            'random_abs': float('nan'),
            'ratio': float('nan'),
            'paired_advantage': float('nan'),
            'ci': [float('nan'), float('nan')],
            'pvalue': float('nan'),
            'n_pairs': 0,
        }
    diff = sae_abs - random_abs
    low, high = paired_bootstrap_difference_ci(sae_abs, random_abs, seed=seed)
    return {
        'sae_abs': float(np.mean(sae_abs)),
        'random_abs': float(np.mean(random_abs)),
        'ratio': float(np.mean(sae_abs) / max(float(np.mean(random_abs)), 1e-12)),
        'paired_advantage': float(np.mean(diff)),
        'ci': [float(low), float(high)],
        'pvalue': float(paired_sign_flip_pvalue(sae_abs, random_abs, seed=seed + 1)),
        'n_pairs': int(sae_abs.size),
    }


def main() -> None:
    args = parse_args()
    catalog = pd.read_csv(args.artifact_dir / 'feature_catalog.csv')
    layers = pd.read_csv(args.artifact_dir / 'layer_metrics.csv')
    stability = pd.read_csv(args.artifact_dir / 'stability.csv')
    causal = pd.read_csv(args.artifact_dir / 'causal_results.csv')
    feature_set_path = args.artifact_dir / 'feature_set_results.csv'
    feature_sets = pd.read_csv(feature_set_path) if feature_set_path.exists() else None
    study_path = args.artifact_dir / 'study_feature_summary.csv'
    study = pd.read_csv(study_path) if study_path.exists() else pd.DataFrame()
    study_summary_path = args.artifact_dir / 'study_summary.json'
    study_summary = (
        json.loads(study_summary_path.read_text(encoding='utf-8'))
        if study_summary_path.exists()
        else {}
    )
    selected = _selected_features(catalog)
    _save_plots(args.artifact_dir, selected, layers, causal, feature_sets)
    _save_study_plots(args.artifact_dir, study)

    mean_auc = float(selected['auroc'].mean())
    median_auc = float(selected['auroc'].median())
    auc_ci_low, auc_ci_high = bootstrap_mean_ci(selected['auroc'].to_numpy(), seed=42)
    best_layer_row = layers.sort_values('linear_probe_macro_auroc', ascending=False).iloc[0]
    mean_jaccard = float(stability['topk_jaccard'].mean())
    mean_sparse_cos = float(stability['sparse_cosine'].mean())

    single = _paired_stats(
        causal,
        index=['task_id', 'intervention'],
        sae_condition='sae_feature',
        random_condition='random_norm_matched',
        seed=43,
    )
    sae = causal[causal['condition'] == 'sae_feature']
    active_rate = float(np.mean(sae['feature_activation'] > 0))
    top1_change = float(sae['top1_changed'].mean())

    set_summary: dict[int, dict[str, float | list[float]]] = {}
    largest_set: dict[str, float | list[float]] | None = None
    largest_k: int | None = None
    if feature_sets is not None and not feature_sets.empty:
        for size in sorted(int(x) for x in feature_sets['set_size'].unique()):
            subset = feature_sets[feature_sets['set_size'] == size]
            set_summary[size] = _paired_stats(
                subset,
                index=['task_id', 'set_size'],
                sae_condition='sae_feature_set',
                random_condition='random_norm_matched',
                seed=100 + size,
            )
        largest_k = max(set_summary)
        largest_set = set_summary[largest_k]

    sae_abs = float(single['sae_abs'])
    random_abs = float(single['random_abs'])
    ratio = float(single['ratio'])

    single_ci_low = float(single['ci'][0])
    single_p = float(single['pvalue'])
    single_specific = ratio >= 1.5 and single_ci_low > 0.0 and single_p < 0.05

    if mean_auc >= 0.8 and sae_abs < 0.08:
        interpretation = (
            'The selected sparse features were strongly predictive on held-out prompts, but single-feature '
            'interventions produced only modest downstream changes. FeatureLens therefore treats the '
            'representation-level signal as correlational rather than automatically causal.'
        )
    elif mean_auc >= 0.8 and sae_abs >= 0.08 and single_specific:
        interpretation = (
            'The selected sparse features were strongly predictive and single-feature interventions produced '
            'larger target-continuation shifts than norm-matched random residual perturbations. The paired '
            'bootstrap interval excludes zero and the sign-flip test passes the configured 0.05 threshold, '
            'supporting a causal-specificity claim for at least some predictive features.'
        )
    elif mean_auc >= 0.8 and sae_abs >= 0.08 and ratio >= 1.5:
        interpretation = (
            'The selected sparse features were strongly predictive and their point-estimate intervention '
            'effects exceeded norm-matched random controls, but the paired uncertainty test does not support '
            'a strong causal-specificity claim at the 0.05 threshold. The result is reported as suggestive '
            'rather than conclusive.'
        )
    elif mean_auc < 0.65:
        interpretation = (
            'Feature/concept predictiveness was limited on held-out prompts, so strong causal claims would '
            'be premature. The main result is diagnostic: concept design or feature selection should be '
            'refined before interpreting intervention effects.'
        )
    else:
        interpretation = (
            'The results show mixed predictive and causal evidence. FeatureLens reports association, '
            'robustness, and intervention measurements separately rather than collapsing them into one score.'
        )

    if largest_set is not None and largest_k is not None:
        set_advantage = float(largest_set['paired_advantage'])
        set_ci_low = float(largest_set['ci'][0])
        set_p = float(largest_set['pvalue'])
        set_specific = set_ci_low > 0.0 and set_p < 0.05
        if set_advantage > float(single['paired_advantage']) + 0.02 and set_specific:
            interpretation += (
                f' Joint ablation of the top {largest_k} same-layer concept features produced a larger '
                'paired advantage over random controls than the single-feature edits, with paired uncertainty '
                'supporting the difference. This is consistent with causal influence being distributed across '
                'a sparse feature set rather than concentrated in one unit.'
            )
        elif set_advantage > float(single['paired_advantage']) + 0.02:
            interpretation += (
                f' The top-{largest_k} joint-ablation point estimate exceeded the single-feature advantage, '
                'but its paired uncertainty test does not support a strong distributed-causality claim at the '
                '0.05 threshold. The pattern is therefore treated as suggestive only.'
            )
        elif abs(set_advantage) <= 0.02:
            interpretation += (
                f' Expanding the intervention to the top {largest_k} same-layer features did not materially '
                'increase specificity over random controls, which argues against assuming that a broader '
                'concept-associated sparse subspace is automatically more causal.'
            )

    effect_label = 'mean log p/token' if _effect_column(causal) == 'target_mean_logprob_delta' else 'target log-probability'
    headline = (
        f'Selected SAE features averaged {mean_auc:.3f} held-out AUROC; single-feature SAE interventions '
        f'changed {effect_label} by {sae_abs:.3f} in absolute value on average versus {random_abs:.3f} '
        'for norm-matched random residual controls.'
    )
    highlights = [
        f'Median selected-feature held-out AUROC: {median_auc:.3f}; mean AUROC 95% bootstrap CI [{auc_ci_low:.3f}, {auc_ci_high:.3f}].',
        f'Best residual linear-probe layer: {int(best_layer_row["layer"])} with macro AUROC {best_layer_row["linear_probe_macro_auroc"]:.3f}.',
        f'Mean paraphrase TopK Jaccard: {mean_jaccard:.3f}; sparse activation cosine: {mean_sparse_cos:.3f}.',
        f'Selected feature active on {active_rate:.1%} of causal prompts; modified next-token top-1 on {top1_change:.1%}.',
        f'Single-feature mean absolute causal effect / random-control effect ratio: {ratio:.2f}×.',
        f'Single-feature paired mean |effect| advantage over random: {float(single["paired_advantage"]):+.3f}, 95% bootstrap CI [{float(single["ci"][0]):+.3f}, {float(single["ci"][1]):+.3f}], sign-flip p={float(single["pvalue"]):.4f}.',
    ]
    if largest_set is not None and largest_k is not None:
        highlights.append(
            f'Top-{largest_k} joint ablation: SAE/random mean absolute effect ratio {float(largest_set["ratio"]):.2f}×; '
            f'paired advantage {float(largest_set["paired_advantage"]):+.3f}, 95% CI '
            f'[{float(largest_set["ci"][0]):+.3f}, {float(largest_set["ci"][1]):+.3f}], '
            f'sign-flip p={float(largest_set["pvalue"]):.4f}.'
        )

    if study_summary:
        correlations = study_summary.get('correlations', {})
        assoc_target = correlations.get('heldout_auroc_vs_target_specificity', {})
        assoc_js = correlations.get('heldout_auroc_vs_js_specificity', {})
        highlights.extend(
            [
                f'Selected-feature median activation-resample support: {float(study_summary.get("median_selected_feature_resample_support", float("nan"))):.1%}.',
                f'Across concepts, held-out AUROC vs target-specificity Spearman ρ={float(assoc_target.get("rho", float("nan"))):+.3f} (n={int(assoc_target.get("n", 0))}); descriptive only.',
                f'Across concepts, held-out AUROC vs JS-specificity Spearman ρ={float(assoc_js.get("rho", float("nan"))):+.3f} (n={int(assoc_js.get("n", 0))}); descriptive only.',
            ]
        )

    summary = {
        'headline': headline,
        'highlights': highlights,
        'interpretation': interpretation,
        'metrics': {
            'mean_selected_feature_test_auroc': mean_auc,
            'mean_selected_feature_test_auroc_bootstrap_ci_95': [auc_ci_low, auc_ci_high],
            'median_selected_feature_test_auroc': median_auc,
            'best_linear_probe_layer': int(best_layer_row['layer']),
            'best_linear_probe_macro_auroc': float(best_layer_row['linear_probe_macro_auroc']),
            'mean_paraphrase_topk_jaccard': mean_jaccard,
            'mean_paraphrase_sparse_cosine': mean_sparse_cos,
            'single_feature_effect_metric': _effect_column(causal),
            'mean_abs_sae_effect': sae_abs,
            'mean_abs_random_effect': random_abs,
            'causal_to_random_effect_ratio': ratio,
            'paired_mean_abs_effect_advantage': float(single['paired_advantage']),
            'paired_mean_abs_effect_advantage_bootstrap_ci_95': single['ci'],
            'paired_sign_flip_pvalue': float(single['pvalue']),
            'causal_prompt_feature_active_rate': active_rate,
            'sae_top1_change_rate': top1_change,
            'feature_set_results': {str(k): value for k, value in set_summary.items()},
            'study_summary': study_summary,
        },
    }
    (args.artifact_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines = [
        '# FeatureLens experiment report',
        '',
        '## Research question',
        '',
        '**Do sparse features that predict a concept also causally influence model behaviour?**',
        '',
        '## Executive summary',
        '',
        headline,
        '',
        interpretation,
        '',
        '## Key measurements',
        '',
        *[f'- {item}' for item in highlights],
        '',
        '## Experimental design',
        '',
        '- Model: Qwen3-1.7B-Base.',
        '- SAEs: Qwen-Scope residual-stream TopK SAEs at configured early/middle/late layers.',
        '- Discovery set: controlled concept prompts with paired paraphrases.',
        '- SAE concept evidence: prompt-wide maximum activation per feature across non-padding prompt tokens; final-token activations are saved separately for local diagnostics.',
        '- Split discipline: paraphrase groups stay entirely in train or held-out test.',
        '- Candidate stability: deterministic balanced activation resamples estimate how often selected features survive small changes in the discovery sample.',
        '- Feature selection: training-split AUROC and activation contrast; held-out AUROC/F1 are reported separately.',
        '- Linear baseline: multinomial logistic regression on the dense residual stream.',
        '- Single-feature causal edit: reconstruction-preserving decoder-direction delta patched into the original residual.',
        '- Feature-set causal edit: joint ablation of top same-layer concept features, evaluated at k=1/3/5 by default.',
        '- Negative control: ensemble of deterministic random residual directions, each matched to the SAE perturbation L2 norm.',
        '- Target metric: exact full target continuation scored teacher-forced; mean log probability per target token is the primary length-comparable effect.',
        '- Secondary diagnostics: first-token probability/rank, next-token JS divergence, and top-1 changes.',
        '- Uncertainty: bootstrap 95% confidence intervals and paired sign-flip randomization tests.',
        '',
        '## Figures',
        '',
        '![Feature AUROC](figures/feature_auroc.png)',
        '',
        '![Layer diagnostics](figures/layer_diagnostics.png)',
        '',
        '![Single-feature causal effects](figures/causal_effects.png)',
    ]
    if feature_sets is not None and not feature_sets.empty:
        lines.extend(['', '![Feature-set causal effects](figures/feature_set_effects.png)'])
    if not study.empty:
        lines.extend(
            [
                '',
                '![Association vs causality](figures/association_vs_causality.png)',
                '',
                '![Candidate stability](figures/candidate_stability.png)',
                '',
                '## Association vs causality across concepts',
                '',
                "The offline study joins each concept-selected feature's held-out AUROC/F1 and activation-resample support with random-normalized causal specificity on the held-out causal task set. Cross-concept correlations are descriptive because there are only seven controlled concepts.",
            ]
        )
    lines.extend(
        [
            '',
            '## Interpretation guardrails',
            '',
            'A high feature/concept AUROC or high paraphrase overlap is correlational evidence only. Causal evidence requires a downstream change under intervention and is interpreted relative to a norm-matched random control. Feature-set effects are not assumed stronger a priori; they are separately measured. The narrative above is generated from saved metrics, with no hard-coded result values.',
            '',
            '## Reproducibility',
            '',
            'Run `python experiments/run_all.py` from the repository root for the full GPU + CPU study. If model/SAE activations and causal rows already exist, run `python experiments/run_analysis_only.py` to regenerate CPU evaluation, stability, study synthesis, figures, validation, and this report without another model download or inference pass.',
            '',
        ]
    )
    (args.artifact_dir / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(headline)
    print(f'Wrote {args.artifact_dir / "report.md"}')


if __name__ == '__main__':
    main()
