"""Acceptance tests for Ticket M-3 (noise process and masked loss)."""

import pytest
import torch

from glassdiff.diffusion.noise import add_noise, masked_displacement_loss, sample_sigma


def test_sigma_range():
    g = torch.Generator().manual_seed(0)
    sigma = sample_sigma(1000, 1e-3, 0.5, generator=g)
    assert sigma.shape == (1000,) and sigma.min() >= 1e-3 and sigma.max() <= 0.5


def test_add_noise_respects_pins(make_lattice):
    s = make_lattice(2, 8, seed=0)
    pin = torch.zeros(2, 64, dtype=torch.bool)
    pin[:, :5] = True
    noisy, disp = add_noise(s, torch.tensor([0.3, 0.3], dtype=s.pos.dtype), pin)
    assert torch.allclose(noisy.pos - s.pos, disp)
    assert (disp[pin] == 0).all() and (disp[~pin].abs() > 0).all()


def test_loss_ignores_pinned_atoms():
    target = torch.randn(2, 10, 2)
    pin = torch.zeros(2, 10, dtype=torch.bool)
    pin[:, :3] = True
    pred = target.clone()
    assert masked_displacement_loss(pred, target, pin).item() == pytest.approx(0.0)
    pred[pin] += 100.0
    assert masked_displacement_loss(pred, target, pin).item() == pytest.approx(0.0)
    pred[0, 5] += 1.0  # one unpinned atom of graph 0 off by (1, 1): squared error 2
    # per graph: mean over unpinned atoms AND both coordinates (like DM2's F.mse_loss);
    # then mean over graphs: ((2 / (7 * 2)) + 0) / 2
    expected = (2.0 / (7 * 2)) / 2
    assert masked_displacement_loss(pred, target, pin).item() == pytest.approx(expected)
