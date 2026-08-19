from __future__ import annotations

import math

import torch

from featurelens.metrics import (
    js_divergence_from_logits,
    reconstruction_metrics,
    safe_log_probability,
    sequence_logprob_summary,
    sparse_topk_cosine,
    target_token_logprobs,
)


def test_reconstruction_metrics_perfect_match() -> None:
    x = torch.tensor([1.0, 2.0, 3.0])
    metrics = reconstruction_metrics(x, x.clone())
    assert metrics['mse'] == 0.0
    assert metrics['nmse'] == 0.0
    assert abs(metrics['cosine'] - 1.0) < 1e-6


def test_js_divergence_zero_for_identical_logits() -> None:
    logits = torch.tensor([1.0, 2.0, -1.0])
    assert abs(js_divergence_from_logits(logits, logits)) < 1e-8


def test_safe_log_probability_is_finite_at_zero() -> None:
    assert safe_log_probability(0.0) < 0.0


def test_target_logprobs_score_every_continuation_token() -> None:
    # prompt length = 2, target ids = [1, 0].
    # Target token 0 is predicted from logits row 1; token 1 from row 2.
    logits = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [0.0, 0.0],
        ]
    )
    values = target_token_logprobs(logits, prompt_length=2, target_ids=[1, 0])
    expected_0 = torch.log_softmax(logits[1], dim=-1)[1]
    expected_1 = torch.log_softmax(logits[2], dim=-1)[0]
    assert torch.allclose(values, torch.stack([expected_0, expected_1]))


def test_sequence_summary_total_and_mean_are_consistent() -> None:
    logits = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [0.0, 0.0],
        ]
    )
    total, mean, token_values = sequence_logprob_summary(
        logits,
        prompt_length=2,
        target_ids=[1, 0],
    )
    assert len(token_values) == 2
    assert math.isclose(total, sum(token_values), rel_tol=1e-6)
    assert math.isclose(mean, total / 2.0, rel_tol=1e-6)


def test_sparse_topk_cosine_is_one_for_identical_sparse_vectors() -> None:
    cosine = sparse_topk_cosine([1, 3], [2.0, 1.0], [1, 3], [2.0, 1.0])
    assert abs(cosine - 1.0) < 1e-9


def test_sparse_topk_cosine_is_zero_for_disjoint_support() -> None:
    cosine = sparse_topk_cosine([1, 3], [2.0, 1.0], [2, 4], [5.0, 7.0])
    assert cosine == 0.0


def test_contrastive_log_odds_reports_preference_shift() -> None:
    from featurelens.metrics import contrastive_log_odds

    baseline, modified, delta = contrastive_log_odds(
        baseline_a=-4.0,
        modified_a=-3.5,
        baseline_b=-2.0,
        modified_b=-2.2,
    )
    assert math.isclose(baseline, -2.0)
    assert math.isclose(modified, -1.3)
    assert math.isclose(delta, 0.7)


def test_decoder_cosine_matrix_and_joint_norm_ratio() -> None:
    from featurelens.metrics import decoder_cosine_matrix, joint_direction_norm_ratio

    orthogonal = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    matrix = decoder_cosine_matrix(orthogonal)
    assert torch.allclose(matrix, torch.eye(2), atol=1e-6)

    joint_norm, independent_norm, ratio = joint_direction_norm_ratio(orthogonal)
    assert math.isclose(joint_norm, math.sqrt(5.0), rel_tol=1e-6)
    assert math.isclose(independent_norm, math.sqrt(5.0), rel_tol=1e-6)
    assert math.isclose(ratio, 1.0, rel_tol=1e-6)


def test_joint_norm_ratio_detects_cancellation() -> None:
    from featurelens.metrics import joint_direction_norm_ratio

    opposing = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    joint_norm, independent_norm, ratio = joint_direction_norm_ratio(opposing)
    assert math.isclose(joint_norm, 0.0, abs_tol=1e-8)
    assert independent_norm > 0
    assert math.isclose(ratio, 0.0, abs_tol=1e-8)
