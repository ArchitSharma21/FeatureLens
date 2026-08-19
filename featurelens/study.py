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
        'selection_stability.csv',
        'feature_catalog.csv',
        'layer_metrics.csv',
        'stability.csv',
        'causal_results.csv',
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
                'The live workbench is usable now, but held-out study artifacts have not been committed. '
                f'Missing: {missing}{suffix}\n\n'
                'Run `python experiments/run_all.py` on a CUDA machine for the full study. If activation and '
                'causal artifacts already exist, use `python experiments/run_analysis_only.py` to rerun the '
                'CPU-only evaluation, stability analysis, evidence synthesis, figures, and report. '
                'Then run `python -m scripts.validate_artifacts` before committing the small CSV/JSON/report '
                'outputs. Do not commit `artifacts/activations/`.'
            )

        summary = self._json('summary.json')
        study = self._json('study_summary.json')
        correlations = study.get('correlations', {})
        target_corr = correlations.get('heldout_auroc_vs_target_specificity', {})
        js_corr = correlations.get('heldout_auroc_vs_js_specificity', {})
        return (
            '### Offline study results\n\n'
            f"{summary.get('headline', 'Benchmark completed.')}\n\n"
            f"{summary.get('interpretation', '')}\n\n"
            '**Study-level diagnostics**\n\n'
            f"- Selected-feature median resample support: **{float(study.get('median_selected_feature_resample_support', float('nan'))):.1%}**.\n"
            f"- Held-out AUROC ↔ target-specificity Spearman ρ: **{float(target_corr.get('rho', float('nan'))):+.3f}** (n={int(target_corr.get('n', 0))}).\n"
            f"- Held-out AUROC ↔ JS-specificity Spearman ρ: **{float(js_corr.get('rho', float('nan'))):+.3f}** (n={int(js_corr.get('n', 0))}).\n\n"
            'The cross-concept correlations are descriptive because there are only seven controlled concepts. '
            'Per-concept causal claims remain anchored to matched random-control comparisons.'
        )

    def readiness_markdown(self) -> str:
        if self.complete:
            return '**Artifact status:** complete and ready for the public study view.'
        return (
            f'**Artifact status:** {len(self.REQUIRED) - len(self.missing)}/{len(self.REQUIRED)} required '
            f'files present; {len(self.missing)} missing.'
        )
