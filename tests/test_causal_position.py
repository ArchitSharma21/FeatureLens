from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _stub_transformers() -> None:
    if 'transformers' in sys.modules:
        return
    stub = types.ModuleType('transformers')
    stub.AutoModelForCausalLM = type('AutoModelForCausalLM', (), {})
    stub.AutoTokenizer = type('AutoTokenizer', (), {})
    sys.modules['transformers'] = stub


def test_feature_trace_and_position_policy() -> None:
    _stub_transformers()
    from experiments.run_causal import choose_intervention_position, feature_activation_trace
    from featurelens.sae import SparseEncoding

    enc = SparseEncoding(
        indices=torch.tensor([[1, 8], [7, 2], [7, 3]]),
        values=torch.tensor([[2.0, 1.0], [4.0, 3.0], [9.0, 1.0]]),
    )
    trace = feature_activation_trace(enc, 7)
    assert trace.tolist() == [0.0, 4.0, 9.0]
    idx, activation, active = choose_intervention_position(
        trace, prompt_len=3, position_policy='max_feature_activation'
    )
    assert (idx, activation, active) == (2, 9.0, True)
    idx, activation, active = choose_intervention_position(
        trace, prompt_len=3, position_policy='final_token'
    )
    assert (idx, activation, active) == (2, 9.0, True)


def test_max_active_inactive_falls_back_to_final_token() -> None:
    _stub_transformers()
    from experiments.run_causal import choose_intervention_position

    idx, activation, active = choose_intervention_position(
        torch.zeros(4), prompt_len=4, position_policy='max_feature_activation'
    )
    assert idx == 3
    assert activation == 0.0
    assert not active


def test_legacy_causal_baseline_migration(tmp_path: Path) -> None:
    from experiments.run_causal_addendum import migrate_final_token_baseline

    source = tmp_path / 'causal_results.csv'
    destination = tmp_path / 'causal_results_final_token.csv'
    pd.DataFrame(
        [
            {'task_id': 'a', 'feature_activation': 3.0, 'condition': 'sae_feature'},
            {'task_id': 'a', 'feature_activation': 3.0, 'condition': 'random_norm_matched'},
            {'task_id': 'b', 'feature_activation': 0.0, 'condition': 'sae_feature'},
        ]
    ).to_csv(source, index=False)
    migrate_final_token_baseline(source, destination)
    frame = pd.read_csv(destination)
    assert set(frame['position_policy']) == {'final_token'}
    assert frame['feature_active_at_intervention'].tolist() == [1, 1, 0]
    assert frame['feature_active_at_final_token'].tolist() == [1, 1, 0]
    assert frame['feature_active_anywhere'].isna().all()


def test_task_level_specificity_averages_interventions_within_task() -> None:
    from experiments.analyze_study import _task_level_specificity

    rows = []
    for intervention, sae_effect, random_effects in [
        ('ablate', 0.4, [0.1, -0.1]),
        ('amplify_2x', -0.2, [0.05, -0.15]),
    ]:
        rows.append({'task_id': 'a', 'intervention': intervention, 'condition': 'sae_feature', 'target_mean_logprob_delta': sae_effect})
        for value in random_effects:
            rows.append({'task_id': 'a', 'intervention': intervention, 'condition': 'random_norm_matched', 'target_mean_logprob_delta': value})
    result = _task_level_specificity(pd.DataFrame(rows), effect_column='target_mean_logprob_delta', seed=1)
    # One task: SAE mean absolute intervention effect=(.4+.2)/2=.3;
    # random means=(.1,.1), then averaged within task=.1.
    assert result['n_tasks'] == 1
    assert np.isclose(result['sae_abs_mean'], 0.3)
    assert np.isclose(result['random_abs_mean'], 0.1)
    assert np.isclose(result['specificity_ratio'], 3.0)
