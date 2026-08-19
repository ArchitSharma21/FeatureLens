from __future__ import annotations

from pathlib import Path

from featurelens.selection import load_feature_sets


def test_feature_set_selection_stays_within_one_best_layer(tmp_path: Path) -> None:
    catalog = tmp_path / 'feature_catalog.csv'
    catalog.write_text(
        'layer,concept,feature_id,train_auroc,auroc,f1,activation_rate_pos,activation_rate_neg\n'
        '4,math,10,0.80,0.75,0.70,0.8,0.2\n'
        '14,math,20,0.92,0.82,0.80,0.9,0.1\n'
        '14,math,21,0.90,0.81,0.79,0.8,0.2\n'
        '14,math,22,0.88,0.80,0.78,0.7,0.2\n'
        '26,math,30,0.85,0.79,0.76,0.8,0.3\n',
        encoding='utf-8',
    )
    selected = load_feature_sets(catalog, max_size=3)
    assert selected['math']['layer'] == 14
    assert selected['math']['feature_ids'] == [20, 21, 22]
