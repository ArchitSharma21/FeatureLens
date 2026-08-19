from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray | list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a sample mean."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float('nan'), float('nan')
    if x.size == 1:
        value = float(x[0])
        return value, value
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, x.size, size=(int(n_resamples), x.size))
    means = x[sample_idx].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def paired_bootstrap_difference_ci(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for mean(a - b), preserving paired rows."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return float('nan'), float('nan')
    diff = x - y
    return bootstrap_mean_ci(
        diff,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


def paired_sign_flip_pvalue(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    *,
    n_permutations: int = 20000,
    seed: int = 42,
    exact_max_nonzero_pairs: int = 18,
) -> float:
    """Two-sided paired sign-flip test for a non-zero mean difference.

    For small effective samples the test is enumerated exactly. Zero-difference
    pairs are removed before deciding whether exact enumeration is feasible;
    their sign cannot change the statistic. Larger samples use a deterministic
    Monte-Carlo approximation.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    diff = (x - y)[mask]
    if diff.size == 0:
        return float('nan')
    observed = abs(float(diff.mean()))
    if observed == 0.0:
        return 1.0

    nonzero = diff[np.abs(diff) > 0.0]
    if nonzero.size == 0:
        return 1.0

    # Preserve the original mean denominator: zero-difference pairs contribute
    # to n but never to a sign-flipped numerator.
    denominator = float(diff.size)
    if nonzero.size <= int(exact_max_nonzero_pairs):
        count = 1 << int(nonzero.size)
        extreme = 0
        for bits in range(count):
            signs = np.fromiter(
                (1.0 if bits & (1 << idx) else -1.0 for idx in range(nonzero.size)),
                dtype=float,
                count=nonzero.size,
            )
            statistic = abs(float(np.sum(signs * nonzero) / denominator))
            if statistic >= observed - 1e-15:
                extreme += 1
        return float(extreme / count)

    rng = np.random.default_rng(seed)
    extreme = 0
    batch = 2000
    remaining = int(n_permutations)
    while remaining > 0:
        n = min(batch, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, nonzero.size))
        permuted = np.abs((signs * nonzero).sum(axis=1) / denominator)
        extreme += int(np.count_nonzero(permuted >= observed))
        remaining -= n
    return float((extreme + 1) / (int(n_permutations) + 1))
