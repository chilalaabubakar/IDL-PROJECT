"""Bootstrap confidence intervals over samples.  [Ticket E-4]

Always resample whole samples, never atoms: atoms inside one sample are correlated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def bootstrap_ci(
    per_sample: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) percentile CI by resampling samples with replacement."""
    x = np.asarray(per_sample, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    if statistic is np.mean:
        stats = x[idx].mean(axis=1)
    else:
        stats = np.array([statistic(x[i]) for i in idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(statistic(x)), float(lo), float(hi)


def bootstrap_ci_items(
    items: Sequence[Any],
    statistic: Callable[[list[Any]], float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Same, for statistics of a whole set of per-sample objects (e.g. a pooled W1 distance)."""
    rng = np.random.default_rng(seed)
    n = len(items)
    stats = np.array(
        [statistic([items[i] for i in rng.integers(0, n, size=n)]) for _ in range(n_resamples)]
    )
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(statistic(list(items))), float(lo), float(hi)
