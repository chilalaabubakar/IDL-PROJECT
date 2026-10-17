"""Host recovery and host-cluster statistics for physical credibility follow-up."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from glassdiff.types import Structures

SPEC = importlib.util.spec_from_file_location(
    "physical", Path(__file__).resolve().parents[1] / "scripts/check_physical_credibility.py"
)
physical = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(physical)


def test_host_recovery_rejects_changed_original_coordinates():
    hosts = Structures(
        torch.tensor([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], dtype=torch.float64),
        torch.tensor([[0, 1], [0, 1]]),
        torch.full((2, 2), 10.0, dtype=torch.float64),
    )
    ids = np.array([0, 1, 0])
    saved = dict(
        pos=hosts.pos[ids].numpy(),
        types=hosts.types[ids].numpy(),
        box=hosts.box[ids].numpy(),
        pin_pos=hosts.pos[ids].numpy().copy(),
        mode="host",
    )
    assert np.array_equal(physical.recover_hosts(saved, hosts), ids)
    saved["pin_pos"][0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="coordinate"):
        physical.recover_hosts(saved, hosts)


def test_weighted_host_quantiles_keep_repeated_requests_together():
    calc, n = physical.weighted_quantiles(
        [np.array([0.0]), np.array([2.0]), np.array([10.0])], np.array([7, 7, 9])
    )
    assert n == 2
    # Selecting the first host twice repeats both of its requests, not just one.
    expected = np.quantile(np.array([0.0, 2.0, 0.0, 2.0]), physical.QUANTILE_LEVELS)
    assert np.allclose(calc(np.array([2, 0])), expected)


def test_cluster_w1_translation_and_seed_are_reproducible():
    a = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    b = [np.array([2.0, 3.0]), np.array([2.0, 3.0])]
    ids = np.array([4, 5])
    result = physical.cluster_w1(a, ids, b, ids, 30, 17)
    assert result == pytest.approx([2.0, 2.0, 2.0])
    assert result == physical.cluster_w1(a, ids, b, ids, 30, 17)


def test_zero_observed_losses_still_has_finite_host_uncertainty():
    lo, hi = physical.wilson_interval(32, 32)
    assert 0.89 < lo < 0.90
    assert hi == pytest.approx(1.0)
    lo, hi = physical.wilson_interval(0, 32)
    assert lo == pytest.approx(0.0)
    assert 0.10 < hi < 0.11
