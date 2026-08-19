from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from experiments.common import ARTIFACT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Estimate prompt-wide candidate selection stability from saved SAE activations.'
    )
    parser.add_argument('--activation-dir', type=Path, default=ARTIFACT_DIR / 'activations')
    parser.add_argument('--output', type=Path, default=ARTIFACT_DIR / 'selection_stability.csv')
    parser.add_argument('--resamples', type=int, default=128)
    parser.add_argument('--top-features', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def _vector_mean(matrix: sp.csr_matrix, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    return np.asarray(matrix[indices].mean(axis=0)).ravel().astype(np.float32, copy=False)


def _vector_rate(binary: sp.csr_matrix, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.zeros(binary.shape[1], dtype=np.float32)
    return np.asarray(binary[indices].mean(axis=0)).ravel().astype(np.float32, copy=False)


def balanced_candidate_score(
    target_mean: np.ndarray,
    other_mean: np.ndarray,
    target_rate: np.ndarray,
) -> np.ndarray:
    """Live-compatible selectivity × coverage × log-magnitude candidate score."""
    target = np.asarray(target_mean, dtype=np.float64)
    other = np.asarray(other_mean, dtype=np.float64)
    rate = np.asarray(target_rate, dtype=np.float64)
    difference = target - other
    denom = np.abs(target) + np.abs(other) + 1e-12
    selectivity = np.where(difference > 0.0, difference / denom, 0.0)
    return selectivity * np.clip(rate, 0.0, 1.0) * np.log1p(np.maximum(target, 0.0))


def _rank_top(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(scores > 0.0)
    if positive.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    k = min(int(k), int(positive.size))
    candidate_scores = scores[positive]
    local = np.argpartition(candidate_scores, -k)[-k:]
    ids = positive[local]
    ordered = np.argsort(scores[ids])[::-1]
    ids = ids[ordered]
    return ids.astype(int), scores[ids].astype(float)


def _sample_indices(
    concept_indices: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    return {
        concept: rng.choice(indices, size=indices.size, replace=True)
        for concept, indices in concept_indices.items()
    }


def _concept_statistics(
    matrix: sp.csr_matrix,
    binary: sp.csr_matrix,
    concept_indices: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    means = {
        concept: _vector_mean(matrix, indices)
        for concept, indices in concept_indices.items()
    }
    rates = {
        concept: _vector_rate(binary, indices)
        for concept, indices in concept_indices.items()
    }
    return means, rates


def _score_for_concept(
    concept: str,
    means: dict[str, np.ndarray],
    rates: dict[str, np.ndarray],
) -> np.ndarray:
    other_concepts = [name for name in means if name != concept]
    other_mean = np.mean(np.stack([means[name] for name in other_concepts]), axis=0)
    return balanced_candidate_score(means[concept], other_mean, rates[concept])


def main() -> None:
    args = parse_args()
    if args.resamples < 1:
        raise ValueError('--resamples must be at least 1.')
    if args.top_features < 1:
        raise ValueError('--top-features must be at least 1.')

    metadata = json.loads((args.activation_dir / 'metadata.json').read_text(encoding='utf-8'))
    pooling = metadata.get('feature_pooling', '')
    if 'prompt-wide' not in pooling:
        raise RuntimeError(
            'Selection stability requires v0.14 prompt-wide activation artifacts. '
            'Rerun experiments.collect_activations before this analysis.'
        )

    rows = metadata['rows']
    layers = [int(layer) for layer in metadata['layers']]
    concepts = sorted({row['concept'] for row in rows})
    concept_indices = {
        concept: np.array(
            [idx for idx, row in enumerate(rows) if row['concept'] == concept],
            dtype=int,
        )
        for concept in concepts
    }
    rng = np.random.default_rng(args.seed)
    output_rows: list[dict] = []

    for layer in layers:
        matrix = sp.load_npz(args.activation_dir / f'features_layer{layer}.npz').tocsr()
        binary = matrix.copy()
        binary.data = np.ones_like(binary.data, dtype=np.float32)

        full_means, full_rates = _concept_statistics(matrix, binary, concept_indices)
        full_scores = {
            concept: _score_for_concept(concept, full_means, full_rates)
            for concept in concepts
        }
        full_orders: dict[str, np.ndarray] = {
            concept: np.argsort(scores)[::-1]
            for concept, scores in full_scores.items()
        }
        support: dict[tuple[str, int], int] = defaultdict(int)
        ranks: dict[tuple[str, int], list[int]] = defaultdict(list)

        for _ in range(args.resamples):
            sampled = _sample_indices(concept_indices, rng)
            means, rates = _concept_statistics(matrix, binary, sampled)
            for concept in concepts:
                scores = _score_for_concept(concept, means, rates)
                ids, _ = _rank_top(scores, args.top_features)
                for rank, feature_id in enumerate(ids.tolist(), start=1):
                    key = (concept, int(feature_id))
                    support[key] += 1
                    ranks[key].append(rank)

        for concept in concepts:
            full_rank_map = np.empty(matrix.shape[1], dtype=np.int32)
            full_rank_map[full_orders[concept]] = np.arange(1, matrix.shape[1] + 1, dtype=np.int32)
            seen = {
                feature_id
                for (seen_concept, feature_id), count in support.items()
                if seen_concept == concept and count > 0
            }
            full_ids, _ = _rank_top(full_scores[concept], max(args.top_features, 50))
            seen.update(int(feature_id) for feature_id in full_ids.tolist())

            for feature_id in seen:
                key = (concept, feature_id)
                feature_ranks = ranks.get(key, [])
                output_rows.append(
                    {
                        'layer': layer,
                        'concept': concept,
                        'feature_id': feature_id,
                        'full_score': float(full_scores[concept][feature_id]),
                        'full_rank': int(full_rank_map[feature_id]),
                        'resample_support': float(support.get(key, 0) / args.resamples),
                        'median_resample_rank': (
                            float(np.median(feature_ranks)) if feature_ranks else float('nan')
                        ),
                        'mean_resample_rank': (
                            float(np.mean(feature_ranks)) if feature_ranks else float('nan')
                        ),
                        'resamples': int(args.resamples),
                        'top_features_per_resample': int(args.top_features),
                    }
                )
        print(f'Stability analysis complete for layer {layer}', flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'layer',
        'concept',
        'feature_id',
        'full_score',
        'full_rank',
        'resample_support',
        'median_resample_rank',
        'mean_resample_rank',
        'resamples',
        'top_features_per_resample',
    ]
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f'Wrote {len(output_rows)} selection-stability rows to {args.output}')


if __name__ == '__main__':
    main()
