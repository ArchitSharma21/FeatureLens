from __future__ import annotations

from experiments.split import grouped_concept_split


def test_grouped_split_keeps_paraphrase_pairs_together() -> None:
    rows = []
    for concept in ('a', 'b'):
        for pair in range(4):
            for variant in range(2):
                rows.append(
                    {
                        'concept': concept,
                        'pair_id': f'{concept}-{pair}',
                        'variant': variant,
                    }
                )
    train, test = grouped_concept_split(rows, test_fraction=0.25, seed=42)
    train_pairs = {rows[idx]['pair_id'] for idx in train}
    test_pairs = {rows[idx]['pair_id'] for idx in test}
    assert train_pairs.isdisjoint(test_pairs)
    assert {rows[idx]['concept'] for idx in train} == {'a', 'b'}
    assert {rows[idx]['concept'] for idx in test} == {'a', 'b'}
