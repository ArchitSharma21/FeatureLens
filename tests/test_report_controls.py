from __future__ import annotations

import math

import pandas as pd

from experiments.make_report import _paired_stats


def test_paired_stats_uses_random_control_ensemble_mean_absolute_effect() -> None:
    frame = pd.DataFrame(
        [
            {'task_id': 'a', 'intervention': 'ablate', 'condition': 'sae_feature', 'target_mean_logprob_delta': 0.30},
            {'task_id': 'a', 'intervention': 'ablate', 'condition': 'random_norm_matched', 'target_mean_logprob_delta': 0.10},
            {'task_id': 'a', 'intervention': 'ablate', 'condition': 'random_norm_matched', 'target_mean_logprob_delta': -0.20},
            {'task_id': 'b', 'intervention': 'ablate', 'condition': 'sae_feature', 'target_mean_logprob_delta': -0.40},
            {'task_id': 'b', 'intervention': 'ablate', 'condition': 'random_norm_matched', 'target_mean_logprob_delta': 0.05},
            {'task_id': 'b', 'intervention': 'ablate', 'condition': 'random_norm_matched', 'target_mean_logprob_delta': -0.15},
        ]
    )
    stats = _paired_stats(
        frame,
        index=['task_id', 'intervention'],
        sae_condition='sae_feature',
        random_condition='random_norm_matched',
        seed=7,
    )
    # Per-task random absolute means are 0.15 and 0.10 -> overall 0.125.
    assert math.isclose(float(stats['random_abs']), 0.125, rel_tol=1e-9)
    assert math.isclose(float(stats['sae_abs']), 0.35, rel_tol=1e-9)
    assert int(stats['n_pairs']) == 2


def test_paired_stats_ignores_missing_random_pairs() -> None:
    frame = pd.DataFrame(
        [
            {'task_id': 'a', 'intervention': 'ablate', 'condition': 'sae_feature', 'target_mean_logprob_delta': 0.30},
            {'task_id': 'a', 'intervention': 'ablate', 'condition': 'random_norm_matched', 'target_mean_logprob_delta': 0.10},
            {'task_id': 'b', 'intervention': 'ablate', 'condition': 'sae_feature', 'target_mean_logprob_delta': 0.90},
        ]
    )
    stats = _paired_stats(
        frame,
        index=['task_id', 'intervention'],
        sae_condition='sae_feature',
        random_condition='random_norm_matched',
        seed=8,
    )
    assert int(stats['n_pairs']) == 1
    assert math.isclose(float(stats['sae_abs']), 0.30, rel_tol=1e-9)
