from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from experiments.common import ARTIFACT_DIR
from experiments.split import grouped_concept_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate predictive SAE features and residual probes.')
    parser.add_argument('--activation-dir', type=Path, default=ARTIFACT_DIR / 'activations')
    parser.add_argument('--output-dir', type=Path, default=ARTIFACT_DIR)
    parser.add_argument('--top-features', type=int, default=20)
    parser.add_argument('--min-train-fires', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.0
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def _evaluate_feature(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float]:
    threshold = _best_threshold(y_train, train_scores)
    train_auc = float(roc_auc_score(y_train, train_scores))
    test_auc = float(roc_auc_score(y_test, test_scores))
    pred = (test_scores >= threshold).astype(int)
    f1 = float(f1_score(y_test, pred, zero_division=0))
    return train_auc, test_auc, f1, threshold


def _sparse_cosine(a: sp.csr_matrix, b: sp.csr_matrix) -> float:
    numerator = float(a.multiply(b).sum())
    denom = float(np.sqrt(a.multiply(a).sum()) * np.sqrt(b.multiply(b).sum()))
    return numerator / denom if denom > 0 else 1.0


def _jaccard(a: sp.csr_matrix, b: sp.csr_matrix) -> float:
    sa = set(a.indices.tolist())
    sb = set(b.indices.tolist())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.activation_dir / 'metadata.json').read_text(encoding='utf-8'))
    if 'prompt-wide' not in str(metadata.get('feature_pooling', '')):
        raise RuntimeError(
            'v0.14 evaluation requires prompt-wide activation artifacts. '
            'Rerun experiments.collect_activations before evaluating features.'
        )
    rows = metadata['rows']
    layers = [int(x) for x in metadata['layers']]
    train_idx, test_idx = grouped_concept_split(rows, seed=args.seed)
    labels = np.array([row['concept'] for row in rows])
    concepts = sorted(set(labels.tolist()))

    feature_rows: list[dict] = []
    layer_rows: list[dict] = []
    stability_rows: list[dict] = []

    encoder = LabelEncoder().fit(labels)
    y_all = encoder.transform(labels)
    y_train_multi = y_all[train_idx]
    y_test_multi = y_all[test_idx]

    for layer in layers:
        x = sp.load_npz(args.activation_dir / f'features_layer{layer}.npz').tocsr()
        x_csc = x.tocsc()
        residuals = np.load(args.activation_dir / f'residuals_layer{layer}.npy').astype(np.float32)
        recon = json.loads(
            (args.activation_dir / f'reconstruction_layer{layer}.json').read_text(encoding='utf-8')
        )

        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2500, class_weight='balanced', random_state=args.seed),
        )
        probe.fit(residuals[train_idx], y_train_multi)
        pred = probe.predict(residuals[test_idx])
        probs = probe.predict_proba(residuals[test_idx])
        probe_f1 = float(f1_score(y_test_multi, pred, average='macro'))
        probe_auc = float(
            roc_auc_score(y_test_multi, probs, multi_class='ovr', average='macro')
        )

        layer_rows.append(
            {
                'layer': layer,
                'linear_probe_macro_auroc': probe_auc,
                'linear_probe_macro_f1': probe_f1,
                'reconstruction_cosine': recon['mean_cosine'],
                'reconstruction_nmse': recon['mean_nmse'],
                'mean_active_features': recon['mean_active_features'],
            }
        )

        train_matrix = x[train_idx]
        candidate_ids, counts = np.unique(train_matrix.indices, return_counts=True)
        candidate_ids = candidate_ids[counts >= args.min_train_fires]

        for concept in concepts:
            y_train = (labels[train_idx] == concept).astype(int)
            y_test = (labels[test_idx] == concept).astype(int)
            concept_results: list[dict] = []
            for feature_id in candidate_ids.tolist():
                train_scores = x_csc[train_idx, feature_id].toarray().ravel()
                if int((train_scores > 0).sum()) < args.min_train_fires:
                    continue
                test_scores = x_csc[test_idx, feature_id].toarray().ravel()
                train_auc, test_auc, f1, threshold = _evaluate_feature(
                    train_scores, test_scores, y_train, y_test
                )
                pos_train = train_scores[y_train == 1]
                neg_train = train_scores[y_train == 0]
                result = {
                    'layer': layer,
                    'concept': concept,
                    'feature_id': int(feature_id),
                    'train_auroc': train_auc,
                    'auroc': test_auc,
                    'f1': f1,
                    'threshold': threshold,
                    'activation_rate_pos': float(np.mean(pos_train > 0)),
                    'activation_rate_neg': float(np.mean(neg_train > 0)),
                    'mean_activation_pos': float(np.mean(pos_train)),
                    'mean_activation_neg': float(np.mean(neg_train)),
                }
                concept_results.append(result)
            concept_results.sort(
                key=lambda item: (item['train_auroc'], item['activation_rate_pos'] - item['activation_rate_neg']),
                reverse=True,
            )
            feature_rows.extend(concept_results[: args.top_features])

        pair_map: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            pair_map[row['pair_id']].append(idx)
        for pair_id, indices in pair_map.items():
            if len(indices) != 2:
                continue
            a, b = indices
            stability_rows.append(
                {
                    'layer': layer,
                    'pair_id': pair_id,
                    'concept': rows[a]['concept'],
                    'topk_jaccard': _jaccard(x.getrow(a), x.getrow(b)),
                    'sparse_cosine': _sparse_cosine(x.getrow(a), x.getrow(b)),
                }
            )
        print(f'Evaluated layer {layer}', flush=True)

    with (args.output_dir / 'feature_catalog.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)

    with (args.output_dir / 'layer_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_rows)

    with (args.output_dir / 'stability.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stability_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stability_rows)

    split_payload = {
        'seed': args.seed,
        'train_indices': train_idx,
        'test_indices': test_idx,
        'n_train': len(train_idx),
        'n_test': len(test_idx),
    }
    (args.output_dir / 'split.json').write_text(json.dumps(split_payload, indent=2), encoding='utf-8')
    print(f'Wrote evaluation artifacts to {args.output_dir}')


if __name__ == '__main__':
    main()
