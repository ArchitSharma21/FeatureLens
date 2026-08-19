from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class OfflineStudy:
    """Read committed offline-study artifacts without invoking the model."""

    REQUIRED = (
        'summary.json',
        'study_summary.json',
        'study_feature_summary.csv',
        'causal_position_summary.csv',
        'selection_stability.csv',
        'feature_catalog.csv',
        'layer_metrics.csv',
        'stability.csv',
        'causal_results_final_token.csv',
        'causal_results_max_active.csv',
        'feature_set_results.csv',
        'report.md',
    )

    def __init__(self, artifact_dir: str | Path = 'artifacts') -> None:
        self.artifact_dir = Path(artifact_dir)

    @property
    def missing(self) -> list[str]:
        return [name for name in self.REQUIRED if not (self.artifact_dir / name).exists()]

    @property
    def complete(self) -> bool:
        return not self.missing

    def _json(self, name: str) -> dict:
        path = self.artifact_dir / name
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding='utf-8'))

    def dataframe(self, name: str) -> pd.DataFrame:
        path = self.artifact_dir / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def figure(self, name: str) -> str | None:
        path = self.artifact_dir / 'figures' / name
        return str(path) if path.exists() else None

    def overview_markdown(self) -> str:
        if not self.complete:
            missing = ', '.join(f'`{name}`' for name in self.missing[:6])
            suffix = '…' if len(self.missing) > 6 else ''
            return (
                '### Offline study not materialized yet\n\n'
                'The live workbench is usable now, but the finalized position-sensitivity study artifacts '
                f'have not been committed. Missing: {missing}{suffix}\n\n'
                'For a fresh study run `python -m experiments.run_all --resume`. If the v0.15 final-token '
                'study already exists, use the v0.16 causal-addendum notebook or '
                '`python -m experiments.run_causal_addendum --resume` to add max-active causal results '
                'without recollecting discovery activations.'
            )

        summary = self._json('summary.json')
        study = self._json('study_summary.json')
        correlations = study.get('correlations', {})
        target_corr = correlations.get('heldout_auroc_vs_max_active_target_specificity', {})
        js_corr = correlations.get('heldout_auroc_vs_max_active_js_specificity', {})
        return (
            '### Offline study results\n\n'
            f"{summary.get('headline', 'Benchmark completed.')}\n\n"
            f"{summary.get('interpretation', '')}\n\n"
            '**Position sensitivity**\n\n'
            f"- Final-token feature coverage: **{float(study.get('final_token_feature_coverage', float('nan'))):.1%}**.\n"
            f"- Max-active feature coverage: **{float(study.get('max_active_feature_coverage', float('nan'))):.1%}**.\n"
            f"- Final-token target specificity: **{float(study.get('final_token_target_specificity_ratio', float('nan'))):.2f}×**.\n"
            f"- Max-active target specificity: **{float(study.get('max_active_target_specificity_ratio', float('nan'))):.2f}×**.\n\n"
            '**Study-level diagnostics**\n\n'
            f"- Selected-feature median resample support: **{float(study.get('median_selected_feature_resample_support', float('nan'))):.1%}**.\n"
            f"- Held-out AUROC ↔ max-active target-specificity Spearman ρ: **{float(target_corr.get('rho', float('nan'))):+.3f}** (n={int(target_corr.get('n', 0))}).\n"
            f"- Held-out AUROC ↔ max-active JS-specificity Spearman ρ: **{float(js_corr.get('rho', float('nan'))):+.3f}** (n={int(js_corr.get('n', 0))}).\n\n"
            'Cross-concept correlations are descriptive; per-concept causal evidence remains anchored to '
            'norm-matched random controls.'
        )

    def readiness_markdown(self) -> str:
        if self.complete:
            return '**Artifact status:** complete and ready for the public study view.'
        return (
            f'**Artifact status:** {len(self.REQUIRED) - len(self.missing)}/{len(self.REQUIRED)} required '
            f'files present; {len(self.missing)} missing.'
        )
