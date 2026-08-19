from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch


def reconstruction_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    x = original.float().reshape(-1)
    x_hat = reconstructed.float().reshape(-1)
    mse = torch.mean((x - x_hat) ** 2)
    denom = torch.mean(x**2).clamp_min(1e-12)
    nmse = mse / denom
    cosine = torch.nn.functional.cosine_similarity(x.unsqueeze(0), x_hat.unsqueeze(0)).item()
    return {
        'mse': float(mse.item()),
        'nmse': float(nmse.item()),
        'cosine': float(cosine),
    }


def js_divergence_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    p = torch.softmax(logits_a.float(), dim=-1)
    q = torch.softmax(logits_b.float(), dim=-1)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)))
    kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)))
    return float((0.5 * (kl_pm + kl_qm)).item())


def safe_log_probability(probability: float) -> float:
    return math.log(max(float(probability), 1e-12))


def target_token_logprobs(
    logits: torch.Tensor,
    *,
    prompt_length: int,
    target_ids: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """
    Return teacher-forced log probabilities for an exact target continuation.

    ``logits`` must be ``[sequence, vocab]`` for the concatenated prompt + target
    sequence. The token at target position ``j`` is predicted by the logit row
    immediately before that token.
    """
    if logits.ndim != 2:
        raise ValueError('logits must have shape [sequence, vocab].')
    ids = torch.as_tensor(target_ids, device=logits.device, dtype=torch.long).reshape(-1)
    if ids.numel() == 0:
        raise ValueError('target_ids must contain at least one token.')
    start = int(prompt_length) - 1
    stop = start + int(ids.numel())
    if start < 0 or stop > logits.shape[0]:
        raise ValueError('Prompt/target lengths are incompatible with logits sequence length.')
    rows = logits[start:stop].float()
    return torch.log_softmax(rows, dim=-1).gather(1, ids[:, None]).squeeze(1)


def sequence_logprob_summary(
    logits: torch.Tensor,
    *,
    prompt_length: int,
    target_ids: Sequence[int] | torch.Tensor,
) -> tuple[float, float, list[float]]:
    """Return total log p, mean log p/token, and token-level log probabilities."""
    token_values = target_token_logprobs(
        logits,
        prompt_length=prompt_length,
        target_ids=target_ids,
    )
    total = float(token_values.sum().item())
    mean = float(token_values.mean().item())
    return total, mean, [float(x) for x in token_values.detach().cpu().tolist()]


def sparse_jaccard(
    indices_a: np.ndarray,
    values_a: np.ndarray,
    indices_b: np.ndarray,
    values_b: np.ndarray,
) -> float:
    a = set(indices_a[np.asarray(values_a) > 0].tolist())
    b = set(indices_b[np.asarray(values_b) > 0].tolist())
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def sparse_topk_cosine(
    indices_a: Sequence[int] | torch.Tensor,
    values_a: Sequence[float] | torch.Tensor,
    indices_b: Sequence[int] | torch.Tensor,
    values_b: Sequence[float] | torch.Tensor,
) -> float:
    """Cosine similarity between two sparse TopK vectors without densifying SAE width."""
    idx_a = torch.as_tensor(indices_a, dtype=torch.long).reshape(-1).cpu().tolist()
    val_a = torch.as_tensor(values_a, dtype=torch.float64).reshape(-1).cpu().tolist()
    idx_b = torch.as_tensor(indices_b, dtype=torch.long).reshape(-1).cpu().tolist()
    val_b = torch.as_tensor(values_b, dtype=torch.float64).reshape(-1).cpu().tolist()

    a = {int(i): float(v) for i, v in zip(idx_a, val_a, strict=True) if float(v) > 0}
    b = {int(i): float(v) for i, v in zip(idx_b, val_b, strict=True) if float(v) > 0}
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    dot = sum(value * b.get(feature_id, 0.0) for feature_id, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def contrastive_log_odds(
    baseline_a: float,
    modified_a: float,
    baseline_b: float,
    modified_b: float,
) -> tuple[float, float, float]:
    """Return baseline A-vs-B log-odds, modified log-odds, and causal shift."""
    baseline = float(baseline_a) - float(baseline_b)
    modified = float(modified_a) - float(modified_b)
    return baseline, modified, modified - baseline


def decoder_cosine_matrix(directions: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine matrix for decoder directions shaped ``[features, d_model]``."""
    if directions.ndim != 2 or directions.shape[0] < 1:
        raise ValueError('directions must have shape [features, d_model].')
    values = directions.float()
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(1e-12)
    normalized = values / norms
    return normalized @ normalized.T


def joint_direction_norm_ratio(deltas: torch.Tensor) -> tuple[float, float, float]:
    """Compare the norm of a summed edit with the root-sum-square independent reference."""
    if deltas.ndim != 2 or deltas.shape[0] < 1:
        raise ValueError('deltas must have shape [features, d_model].')
    values = deltas.float()
    individual_norms = torch.linalg.vector_norm(values, dim=1)
    joint_norm = float(torch.linalg.vector_norm(values.sum(dim=0)).item())
    independent_norm = float(torch.sqrt(torch.sum(individual_norms ** 2)).item())
    ratio = joint_norm / max(independent_norm, 1e-12)
    return joint_norm, independent_norm, float(ratio)
