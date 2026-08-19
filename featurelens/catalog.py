from __future__ import annotations

import csv
import json
from pathlib import Path


class FeatureCatalog:
    def __init__(self, artifact_dir: str | Path = 'artifacts') -> None:
        self.artifact_dir = Path(artifact_dir)
        self.rows: list[dict] = []
        path = self.artifact_dir / 'feature_catalog.csv'
        if path.exists():
            with path.open(newline='', encoding='utf-8') as handle:
                self.rows = list(csv.DictReader(handle))

    def hint(self, layer: int, feature_id: int) -> str:
        matches = [
            row
            for row in self.rows
            if int(row.get('layer', -1)) == int(layer)
            and int(row.get('feature_id', -1)) == int(feature_id)
        ]
        if not matches:
            return 'unlabeled'
        best = max(
            matches,
            key=lambda row: float(row.get('train_auroc', row.get('auroc', 0.0)) or 0.0),
        )
        concept = best.get('concept', 'unlabeled')
        auc = float(best.get('auroc', 0.0) or 0.0)
        return f'{concept} (AUROC {auc:.2f})'

    def benchmark_markdown(self) -> str:
        summary_path = self.artifact_dir / 'summary.json'
        if not summary_path.exists():
            return (
                '### Offline benchmark\n\n'
                'No benchmark artifacts are committed yet. Run `python experiments/run_all.py` '
                'on a CUDA machine, then commit `artifacts/summary.json`, `feature_catalog.csv`, '
                'and `report.md`. The live workbench is fully usable without them.'
            )
        data = json.loads(summary_path.read_text(encoding='utf-8'))
        headline = data.get('headline', 'Benchmark completed.')
        bullets = data.get('highlights', [])
        body = '\n'.join(f'- {item}' for item in bullets)
        return f'### Offline benchmark\n\n{headline}\n\n{body}'
