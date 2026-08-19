from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InterventionSpec:
    mode: str
    coefficient: float

    def delta_activation(self, original_activation: float) -> float:
        mode = self.mode.lower().strip()
        original = float(original_activation)
        coefficient = float(self.coefficient)
        if mode == 'ablate':
            return -original
        if mode == 'scale':
            return (coefficient - 1.0) * original
        if mode == 'inject':
            return coefficient
        raise ValueError("mode must be one of: 'ablate', 'scale', 'inject'.")


def residual_delta(
    decoder_direction: torch.Tensor,
    original_activation: float,
    spec: InterventionSpec,
) -> torch.Tensor:
    """Return the reconstruction-preserving SAE delta applied to the original residual."""
    return decoder_direction * spec.delta_activation(original_activation)


def joint_residual_delta(
    decoder_directions: torch.Tensor,
    original_activations: Sequence[float] | torch.Tensor,
    spec: InterventionSpec,
) -> tuple[torch.Tensor, list[float]]:
    """
    Sum reconstruction-preserving deltas for a set of SAE features.

    ``decoder_directions`` must have shape ``[n_features, d_model]``. The same
    ablation/scale intervention is applied to every selected feature. ``inject``
    is intentionally rejected for feature sets because a shared additive
    coefficient has ambiguous semantics across unrelated decoder directions.
    """
    mode = spec.mode.lower().strip()
    if mode not in {'ablate', 'scale'}:
        raise ValueError("Feature-set interventions support only 'ablate' or 'scale'.")

    directions = decoder_directions
    if directions.ndim != 2:
        raise ValueError('decoder_directions must have shape [n_features, d_model].')

    if isinstance(original_activations, torch.Tensor):
        activations = original_activations.detach().float().reshape(-1).tolist()
    else:
        activations = [float(x) for x in original_activations]

    if len(activations) != directions.shape[0]:
        raise ValueError('Number of activations must match decoder directions.')
    if not activations:
        raise ValueError('Select at least one feature for a feature-set intervention.')

    coefficient_deltas = [spec.delta_activation(value) for value in activations]
    coeff = torch.tensor(
        coefficient_deltas,
        device=directions.device,
        dtype=directions.dtype,
    )
    delta = torch.sum(directions * coeff[:, None], dim=0)
    return delta, coefficient_deltas


def normalized_random_control(delta: torch.Tensor, seed: int) -> torch.Tensor:
    """Generate a deterministic random residual perturbation with identical L2 norm."""
    norm = torch.linalg.vector_norm(delta.float())
    if float(norm.item()) == 0.0:
        return torch.zeros_like(delta)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    random_vec = torch.randn(delta.shape, generator=generator, dtype=torch.float32)
    random_vec = random_vec / torch.linalg.vector_norm(random_vec)
    random_vec = random_vec * norm.cpu()
    return random_vec.to(device=delta.device, dtype=delta.dtype)
