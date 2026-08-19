from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(module: str, *args: str) -> None:
    command = [sys.executable, '-m', module, *args]
    print('\n$', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run('experiments.evaluate_features')
    run('experiments.analyze_stability')
    run('experiments.analyze_study')
    run('experiments.make_report')
    run('scripts.validate_artifacts')
    print('\nFeatureLens CPU analysis pipeline complete. See artifacts/report.md')


if __name__ == '__main__':
    main()
