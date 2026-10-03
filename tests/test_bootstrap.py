"""Acceptance tests for Ticket E-4 (bootstrap CIs)."""

import numpy as np
import pytest

from glassdiff.eval.bootstrap import bootstrap_ci

pytestmark = pytest.mark.skip(reason="Ticket E-4: delete this line when implementing")


def test_constant_sample_has_zero_width():
    est, lo, hi = bootstrap_ci(np.full(256, 0.3))
    assert est == lo == hi == pytest.approx(0.3)


def test_interval_contains_estimate_and_is_reproducible():
    x = np.random.default_rng(0).binomial(1, 0.2, size=256).astype(float)
    a = bootstrap_ci(x, seed=1)
    b = bootstrap_ci(x, seed=1)
    assert a == b
    est, lo, hi = a
    assert lo <= est <= hi and est == pytest.approx(x.mean())
    assert 0.03 < hi - lo < 0.15  # ~ 2 * 1.96 * sqrt(0.2 * 0.8 / 256) = 0.098
