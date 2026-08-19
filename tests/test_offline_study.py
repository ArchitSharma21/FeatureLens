from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.analyze_stability import balanced_candidate_score
from experiments.analyze_study import _paired_specificity, _safe_spearman
from experiments.collect_activations import _promptwide_max_encoding
from featurelens.sae import SparseEncoding
from featurelens.study import OfflineStudy


def test_promptwide_max_encoding_ignores_padding_and_max_pools() -> None:
    encoding = SparseEncoding(
        indices=torch.tensor(
            [
                [[1, 2], [1, 3], [2, 4]],
                [[5, 6], [5, 7], [7, 8]],
            ]
        ),
        values=torch.tensor(
            [
                [[1.0, 2.0], [4.0, 3.0], [5.0, 1.0]],
                [[9.0, 9.0], [2.0, 4.0], [7.0, 6.0]],
            ]
        ),
    )
    mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
    pooled = _promptwide_max_encoding(encoding, mask)

    row0 = dict(zip(pooled[0].indices.tolist(), pooled[0].values.tolist(), strict=True))
    row1 = dict(zip(pooled[1].indices.tolist(), pooled[1].values.tolist(), strict=True))
    assert row0 == {1: 4.0, 2: 5.0, 3: 3.0, 4: 1.0}
    assert row1 == {5: 2.0, 7: 7.0, 8: 6.0}


def test_balanced_candidate_score_rewards_selectivity_not_raw_scale() -> None:
    target = np.array([30.0, 1000.0])
    other = np.array([0.0, 950.0])
    rate = np.array([1.0, 1.0])
    score = balanced_candidate_score(target, other, rate)
    assert score[0] > score[1]


def test_paired_specificity_uses_random_ensemble_mean_per_task() -> None:
    frame = pd.DataFrame(
        [
            {'task_id': 'a', 'condition': 'sae_feature', 'target_mean_logprob_delta': 0.4},
            {'task_id': 'a', 'condition': 'random_norm_matched', 'target_mean_logprob_delta': 0.1},
            {'task_id': 'a', 'condition': 'random_norm_matched', 'target_mean_logprob_delta': -0.1},
            {'task_id': 'b', 'condition': 'sae_feature', 'target_mean_logprob_delta': -0.2},
            {'task_id': 'b', 'condition': 'random_norm_matched', 'target_mean_logprob_delta': 0.05},
            {'task_id': 'b', 'condition': 'random_norm_matched', 'target_mean_logprob_delta': -0.15},
        ]
    )
    result = _paired_specificity(
        frame,
        effect_column='target_mean_logprob_delta',
        seed=7,
    )
    assert np.isclose(result['sae_abs_mean'], 0.3)
    assert np.isclose(result['random_abs_mean'], 0.1)
    assert np.isclose(result['specificity_ratio'], 3.0)
    assert result['n_tasks'] == 2


def test_safe_spearman_handles_small_samples() -> None:
    small = _safe_spearman(pd.Series([1.0, 2.0]), pd.Series([2.0, 1.0]))
    assert small['n'] == 2
    assert np.isnan(small['rho'])
    enough = _safe_spearman(pd.Series([1.0, 2.0, 3.0]), pd.Series([3.0, 2.0, 1.0]))
    assert enough['n'] == 3
    assert np.isclose(enough['rho'], -1.0)


def test_offline_study_reports_missing_and_complete(tmp_path: Path) -> None:
    study = OfflineStudy(tmp_path)
    assert not study.complete
    assert 'not materialized yet' in study.overview_markdown()

    for name in OfflineStudy.REQUIRED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.json'):
            if name == 'study_summary.json':
                payload = {
                    'median_selected_feature_resample_support': 0.8,
                    'correlations': {
                        'heldout_auroc_vs_target_specificity': {'rho': -0.2, 'n': 7},
                        'heldout_auroc_vs_js_specificity': {'rho': 0.4, 'n': 7},
                    },
                }
            else:
                payload = {'headline': 'Synthetic headline.', 'interpretation': 'Synthetic interpretation.'}
            path.write_text(json.dumps(payload), encoding='utf-8')
        elif name.endswith('.csv'):
            path.write_text('x\n1\n', encoding='utf-8')
        else:
            path.write_text('# report\n', encoding='utf-8')

    study = OfflineStudy(tmp_path)
    assert study.complete
    text = study.overview_markdown()
    assert 'Synthetic headline.' in text
    assert '80.0%' in text
