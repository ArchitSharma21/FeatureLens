from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import ARTIFACT_DIR

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run only the v0.16 max-active causal addendum and CPU reanalysis.'
    )
    parser.add_argument('--artifact-dir', type=Path, default=ARTIFACT_DIR)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--random-controls', type=int, default=8)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print('\n$', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def migrate_final_token_baseline(source: Path, destination: Path) -> None:
    """Preserve a v0.15 causal CSV while adding v0.16 position-policy metadata."""
    frame = pd.read_csv(source)
    if 'position_policy' not in frame.columns:
        active = pd.to_numeric(frame['feature_activation'], errors='coerce').fillna(0.0) > 0.0
        frame['position_policy'] = 'final_token'
        frame['intervention_token_index'] = -1
        frame['intervention_token_text'] = ''
        frame['final_token_index'] = -1
        frame['final_token_text'] = ''
        frame['max_prompt_feature_token_index'] = -1
        frame['max_prompt_feature_token_text'] = ''
        frame['final_token_feature_activation'] = pd.to_numeric(
            frame['feature_activation'], errors='coerce'
        ).fillna(0.0)
        frame['max_prompt_feature_activation'] = np.nan
        frame['feature_active_at_intervention'] = active.astype(int)
        frame['feature_active_at_final_token'] = active.astype(int)
        # Prompt-wide activity cannot be reconstructed from the legacy final-token CSV.
        frame['feature_active_anywhere'] = np.nan
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    legacy = artifact_dir / 'causal_results.csv'
    final = artifact_dir / 'causal_results_final_token.csv'
    if not final.exists():
        if not legacy.exists():
            raise SystemExit(
                'Missing final-token baseline. Expected artifacts/causal_results.csv or '
                'artifacts/causal_results_final_token.csv from the completed v0.15 study.'
            )
        migrate_final_token_baseline(legacy, final)
        print(f'Preserved v0.15 baseline as {final.name}.')
    elif 'position_policy' not in pd.read_csv(final, nrows=1).columns:
        migrate_final_token_baseline(final, final)
        print(f'Upgraded {final.name} with v0.16 position metadata.')

    output = artifact_dir / 'causal_results_max_active.csv'
    causal_command = [
        sys.executable, '-m', 'experiments.run_causal',
        '--position-policy', 'max_feature_activation',
        '--output', str(output),
        '--random-controls', str(args.random_controls),
    ]
    if args.resume:
        causal_command.append('--resume')
    run(causal_command)
    run([sys.executable, '-m', 'experiments.analyze_study'])
    run([sys.executable, '-m', 'experiments.make_report'])
    run([sys.executable, '-m', 'scripts.validate_artifacts'])
    print('\nFeatureLens v0.16 causal addendum complete.')


if __name__ == '__main__':
    main()
