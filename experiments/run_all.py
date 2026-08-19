from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from featurelens.config import SETTINGS

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the full FeatureLens offline study.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip stages whose expected outputs already exist.',
    )
    parser.add_argument(
        '--activation-batch-size',
        type=int,
        default=16,
        help='Batch size used only by experiments.collect_activations.',
    )
    parser.add_argument(
        '--activation-max-length',
        type=int,
        default=192,
        help='Maximum prompt length used only by experiments.collect_activations.',
    )
    return parser.parse_args()


def run(
    module: str,
    *,
    outputs: list[Path],
    resume: bool,
    extra_args: list[str] | None = None,
) -> None:
    if resume and outputs and all(path.exists() for path in outputs):
        print(f'\nSKIP {module}: expected outputs already exist.', flush=True)
        return
    command = [sys.executable, '-m', module, *(extra_args or [])]
    print('\n$', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    artifact_dir = ROOT / 'artifacts'
    activation_dir = artifact_dir / 'activations'

    run(
        'experiments.build_dataset',
        outputs=[ROOT / 'data' / 'prompts.jsonl', ROOT / 'data' / 'causal_tasks.jsonl'],
        resume=args.resume,
    )
    run(
        'experiments.collect_activations',
        outputs=[
            activation_dir / 'metadata.json',
            *[activation_dir / f'features_layer{layer}.npz' for layer in SETTINGS.layers],
            *[activation_dir / f'features_final_layer{layer}.npz' for layer in SETTINGS.layers],
        ],
        resume=args.resume,
        extra_args=[
            '--batch-size', str(args.activation_batch_size),
            '--max-length', str(args.activation_max_length),
        ],
    )
    run(
        'experiments.evaluate_features',
        outputs=[
            artifact_dir / 'feature_catalog.csv',
            artifact_dir / 'layer_metrics.csv',
            artifact_dir / 'stability.csv',
            artifact_dir / 'split.json',
        ],
        resume=args.resume,
    )
    causal_output = artifact_dir / 'causal_results.csv'
    run(
        'experiments.run_causal',
        outputs=[causal_output, causal_output.with_suffix(causal_output.suffix + '.complete')],
        resume=args.resume,
        extra_args=['--resume'] if args.resume else None,
    )
    feature_set_output = artifact_dir / 'feature_set_results.csv'
    run(
        'experiments.run_feature_sets',
        outputs=[
            feature_set_output,
            feature_set_output.with_suffix(feature_set_output.suffix + '.complete'),
        ],
        resume=args.resume,
        extra_args=['--resume'] if args.resume else None,
    )
    run(
        'experiments.analyze_stability',
        outputs=[artifact_dir / 'selection_stability.csv'],
        resume=args.resume,
    )
    run(
        'experiments.analyze_study',
        outputs=[artifact_dir / 'study_feature_summary.csv', artifact_dir / 'study_summary.json'],
        resume=args.resume,
    )
    run(
        'experiments.make_report',
        outputs=[artifact_dir / 'summary.json', artifact_dir / 'report.md'],
        resume=args.resume,
    )
    subprocess.run(
        [sys.executable, '-m', 'scripts.validate_artifacts'],
        cwd=ROOT,
        check=True,
    )
    print('\nFeatureLens experiment pipeline complete. See artifacts/report.md')


if __name__ == '__main__':
    main()
