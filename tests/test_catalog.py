from __future__ import annotations

from pathlib import Path

from featurelens.catalog import FeatureCatalog


def test_catalog_returns_best_hint(tmp_path: Path) -> None:
    (tmp_path / 'feature_catalog.csv').write_text(
        'layer,concept,feature_id,auroc\n14,math,123,0.80\n14,code,123,0.65\n',
        encoding='utf-8',
    )
    catalog = FeatureCatalog(tmp_path)
    assert catalog.hint(14, 123) == 'math (AUROC 0.80)'
    assert catalog.hint(4, 123) == 'unlabeled'
