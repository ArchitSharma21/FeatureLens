from __future__ import annotations

import csv
import sys
import types
from pathlib import Path


def _stub_transformers() -> None:
    if 'transformers' in sys.modules:
        return
    stub = types.ModuleType('transformers')
    stub.AutoModelForCausalLM = type('AutoModelForCausalLM', (), {})
    stub.AutoTokenizer = type('AutoTokenizer', (), {})
    sys.modules['transformers'] = stub


def test_run_all_forwards_activation_collection_tuning(monkeypatch, tmp_path: Path) -> None:
    from experiments import run_all

    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_all.subprocess, 'run', fake_run)
    missing = tmp_path / 'not-created'
    run_all.run(
        'experiments.collect_activations',
        outputs=[missing],
        resume=True,
        extra_args=['--batch-size', '8', '--max-length', '192'],
    )
    assert calls
    assert calls[0][-4:] == ['--batch-size', '8', '--max-length', '192']


def test_causal_checkpoint_helpers_roundtrip(tmp_path: Path) -> None:
    _stub_transformers()
    from experiments import run_causal

    output = tmp_path / 'causal_results.csv'
    rows = [
        {'task_id': 'task-1', 'concept': 'mathematics', 'condition': 'sae_feature'},
        {'task_id': 'task-1', 'concept': 'mathematics', 'condition': 'random_norm_matched'},
    ]
    run_causal._write_rows_atomic(output, rows)
    loaded = run_causal._load_checkpoint_rows(output)
    assert loaded == rows
    assert run_causal._completion_marker(output).name == 'causal_results.csv.complete'
    assert not output.with_suffix('.csv.tmp').exists()


def test_feature_set_checkpoint_helpers_roundtrip(tmp_path: Path) -> None:
    _stub_transformers()
    from experiments import run_feature_sets

    output = tmp_path / 'feature_set_results.csv'
    rows = [
        {'task_id': 'task-1', 'set_size': '1', 'condition': 'sae_feature_set'},
        {'task_id': 'task-1', 'set_size': '1', 'condition': 'random_norm_matched'},
    ]
    run_feature_sets._write_rows_atomic(output, rows)
    with output.open(newline='', encoding='utf-8') as handle:
        saved = list(csv.DictReader(handle))
    assert saved == rows
    assert run_feature_sets._load_checkpoint_rows(output) == rows
    assert run_feature_sets._completion_marker(output).name == 'feature_set_results.csv.complete'
