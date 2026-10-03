"""Bootstrap confidence intervals over samples.  [Ticket E-4]"""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    per_sample: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) percentile CI by resampling samples with replacement."""
    raise NotImplementedError("Ticket E-4")
