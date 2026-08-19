from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def test_discovery_dataset_is_balanced_and_paired() -> None:
    rows = read_jsonl(ROOT / 'data' / 'prompts.jsonl')
    assert len(rows) == 224
    concept_counts = Counter(row['concept'] for row in rows)
    assert set(concept_counts.values()) == {32}
    assert 'german_language' in concept_counts
    assert 'french_language' not in concept_counts
    german = [row for row in rows if row['concept'] == 'german_language']
    assert len(german) == 32
    assert any('Tisch' in row['text'] or 'Deutsch' in row['text'] or 'Hamburg' in row['text'] for row in german)
    pair_counts = Counter(row['pair_id'] for row in rows)
    assert set(pair_counts.values()) == {2}


def test_causal_dataset_covers_every_discovery_concept() -> None:
    discovery = read_jsonl(ROOT / 'data' / 'prompts.jsonl')
    causal = read_jsonl(ROOT / 'data' / 'causal_tasks.jsonl')
    assert len(causal) == 28
    assert {row['concept'] for row in causal} == {row['concept'] for row in discovery}
    german = [row for row in causal if row['concept'] == 'german_language']
    assert len(german) == 4
    assert all(row['target'].strip() in {'hallo', 'danke', 'ja', 'guten Abend'} for row in german)
