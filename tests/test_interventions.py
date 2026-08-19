from __future__ import annotations

import pytest
import torch

from featurelens.interventions import (
    InterventionSpec,
    joint_residual_delta,
    normalized_random_control,
    residual_delta,
)


def test_ablation_delta() -> None:
    spec = InterventionSpec('ablate', 0.0)
    assert spec.delta_activation(3.5) == -3.5


def test_scale_delta() -> None:
    spec = InterventionSpec('scale', 2.0)
    assert spec.delta_activation(3.5) == 3.5
    assert InterventionSpec('scale', 0.0).delta_activation(3.5) == -3.5


def test_injection_delta() -> None:
    spec = InterventionSpec('inject', -4.0)
    assert spec.delta_activation(123.0) == -4.0


def test_residual_delta_is_decoder_direction_times_coefficient_delta() -> None:
    direction = torch.tensor([1.0, 2.0, -1.0])
    delta = residual_delta(direction, 3.0, InterventionSpec('ablate', 0.0))
    assert torch.allclose(delta, torch.tensor([-3.0, -6.0, 3.0]))


def test_joint_ablation_sums_feature_deltas() -> None:
    directions = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, -1.0],
        ]
    )
    delta, coefficient_deltas = joint_residual_delta(
        directions,
        [2.0, 3.0],
        InterventionSpec('ablate', 0.0),
    )
    assert coefficient_deltas == [-2.0, -3.0]
    assert torch.allclose(delta, torch.tensor([-2.0, -3.0, -1.0]))


def test_joint_scale_uses_same_multiplier_per_feature() -> None:
    directions = torch.eye(2)
    delta, coefficient_deltas = joint_residual_delta(
        directions,
        [2.0, 4.0],
        InterventionSpec('scale', 1.5),
    )
    assert coefficient_deltas == [1.0, 2.0]
    assert torch.allclose(delta, torch.tensor([1.0, 2.0]))


def test_joint_injection_is_rejected() -> None:
    with pytest.raises(ValueError, match='only'):
        joint_residual_delta(
            torch.eye(2),
            [1.0, 1.0],
            InterventionSpec('inject', 2.0),
        )


def test_random_control_matches_norm_and_is_deterministic() -> None:
    delta = torch.tensor([3.0, 4.0, 0.0])
    a = normalized_random_control(delta, seed=7)
    b = normalized_random_control(delta, seed=7)
    assert torch.allclose(a, b)
    assert torch.allclose(torch.linalg.vector_norm(a), torch.linalg.vector_norm(delta), atol=1e-5)
