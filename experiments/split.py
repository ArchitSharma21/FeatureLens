from __future__ import annotations

import random


def grouped_concept_split(rows: list[dict], test_fraction: float = 0.25, seed: int = 42):
    """Split paraphrase groups within each concept so paired prompts never leak across splits."""
    rng = random.Random(seed)
    train_ids: list[int] = []
    test_ids: list[int] = []
    concepts = sorted({row['concept'] for row in rows})
    for concept in concepts:
        concept_rows = [(idx, row) for idx, row in enumerate(rows) if row['concept'] == concept]
        pair_ids = sorted({row['pair_id'] for _, row in concept_rows})
        rng.shuffle(pair_ids)
        n_test = max(1, round(len(pair_ids) * test_fraction))
        test_pairs = set(pair_ids[:n_test])
        for idx, row in concept_rows:
            (test_ids if row['pair_id'] in test_pairs else train_ids).append(idx)
    return sorted(train_ids), sorted(test_ids)
