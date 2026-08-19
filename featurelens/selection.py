from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def load_feature_sets(path: str | Path, max_size: int) -> dict[str, dict]:
    """
    Select a same-layer feature set for each concept from a feature catalog.

    The layer whose best training feature is strongest is selected first. The top
    distinct features from that same SAE dictionary are then returned. Keeping a
    set within one layer is essential because decoder directions from different
    residual spaces should not be summed into one intervention.
    """
    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        item = dict(row)
        item['layer'] = int(row['layer'])
        item['feature_id'] = int(row['feature_id'])
        item['train_auroc'] = float(row['train_auroc'])
        item['activation_contrast'] = (
            float(row['activation_rate_pos']) - float(row['activation_rate_neg'])
        )
        grouped[row['concept']].append(item)

    result: dict[str, dict] = {}
    for concept, concept_rows in grouped.items():
        best_by_layer: dict[int, tuple[float, float]] = {}
        for row in concept_rows:
            key = (row['train_auroc'], row['activation_contrast'])
            best_by_layer[row['layer']] = max(
                best_by_layer.get(row['layer'], (-1.0, -1.0)),
                key,
            )
        chosen_layer = max(best_by_layer, key=best_by_layer.get)
        layer_rows = [row for row in concept_rows if row['layer'] == chosen_layer]
        layer_rows.sort(
            key=lambda row: (row['train_auroc'], row['activation_contrast']),
            reverse=True,
        )

        seen: set[int] = set()
        feature_ids: list[int] = []
        for row in layer_rows:
            feature_id = int(row['feature_id'])
            if feature_id not in seen:
                seen.add(feature_id)
                feature_ids.append(feature_id)
            if len(feature_ids) >= int(max_size):
                break

        result[concept] = {
            'layer': int(chosen_layer),
            'feature_ids': feature_ids,
        }
    return result
