import numpy as np

from featurelens.stats import (
    bootstrap_mean_ci,
    paired_bootstrap_difference_ci,
    paired_sign_flip_pvalue,
)


def test_bootstrap_mean_ci_contains_sample_mean():
    values = np.array([0.7, 0.8, 0.9, 0.85, 0.75])
    low, high = bootstrap_mean_ci(values, n_resamples=1000, seed=1)
    assert low <= values.mean() <= high


def test_paired_bootstrap_detects_positive_difference():
    a = np.array([2.0, 2.2, 1.8, 2.1, 2.4])
    b = np.array([0.5, 0.6, 0.4, 0.7, 0.5])
    low, high = paired_bootstrap_difference_ci(a, b, n_resamples=1000, seed=2)
    assert low > 0
    assert high > low


def test_sign_flip_small_for_consistent_effect():
    a = np.array([2.0, 2.1, 2.3, 2.2, 2.4, 2.5, 2.1, 2.2])
    b = np.array([0.2, 0.4, 0.3, 0.5, 0.2, 0.4, 0.3, 0.2])
    p = paired_sign_flip_pvalue(a, b, n_permutations=5000, seed=3)
    assert p < 0.05


def test_sign_flip_one_for_identical_pairs():
    x = np.array([1.0, 2.0, 3.0])
    assert paired_sign_flip_pvalue(x, x, n_permutations=1000, seed=4) == 1.0


def test_sign_flip_is_exact_for_small_effective_sample() -> None:
    from featurelens.stats import paired_sign_flip_pvalue

    # With two positive non-zero differences, only the ++ and -- assignments
    # are as extreme as the observed all-positive mean: p = 2 / 4.
    assert paired_sign_flip_pvalue([1.0, 1.0], [0.0, 0.0]) == 0.5
