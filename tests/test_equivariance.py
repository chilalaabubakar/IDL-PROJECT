"""Acceptance tests for Ticket M-2 (EGNN-PBC symmetries). Run in float64 for tight tolerances."""

import pytest
import torch

from glassdiff.models.egnn_pbc import EGNNPBC
from glassdiff.types import Structures, defect_token

ROT90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return EGNNPBC(hidden=32, n_layers=2, cutoff=2.5, k_max=48).double().eval()


@pytest.fixture
def s(make_lattice):
    return make_lattice(2, 8, jitter=0.15, seed=0)  # square box, 64 atoms


def test_output_shape_with_conditions(model, s):
    sigma = torch.tensor([0.1, 0.3], dtype=torch.float64)
    pin = torch.zeros(2, 64, dtype=torch.bool)
    pin[:, 0] = True
    label = torch.zeros(2, 64, dtype=torch.long)
    label[:, 0] = defect_token(3)
    out = model(s, sigma, pin, label)
    assert out.shape == (2, 64, 2)
    assert (out[:, 0] == 0).all()  # pinned atoms never move


def test_translation_invariance_including_wrap(model, s):
    sigma = torch.tensor([0.2, 0.2], dtype=torch.float64)
    out = model(s, sigma)
    assert out.abs().max() > 1e-4  # the symmetry checks below must not be trivially 0 == 0
    for shift in (torch.tensor([0.31, -2.7]), s.box[0] * torch.tensor([1.0, -2.0])):
        shifted = model(s.with_pos(s.pos + shift.double()), sigma)
        assert torch.allclose(out, shifted, atol=1e-9)
    assert torch.allclose(out, model(s.wrapped(), sigma), atol=1e-9)


def test_rotation_by_90_degrees(model, s):
    sigma = torch.tensor([0.2, 0.2], dtype=torch.float64)
    out = model(s, sigma)
    rotated = model(s.with_pos(s.pos @ ROT90.T), sigma)  # square box maps onto itself
    assert torch.allclose(rotated, out @ ROT90.T, atol=1e-9)


def test_permutation_equivariance(model, s):
    sigma = torch.tensor([0.2, 0.2], dtype=torch.float64)
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(1))
    out = model(s, sigma)
    out_p = model(Structures(s.pos[:, perm], s.types[:, perm], s.box), sigma)
    assert torch.allclose(out_p, out[:, perm], atol=1e-9)


def test_arbitrary_rotation_of_isolated_cluster(model):
    """Away from the box edges the model must be equivariant to any rotation, not just 90°."""
    g = torch.Generator().manual_seed(3)
    centre = torch.tensor([15.0, 15.0], dtype=torch.float64)
    pos = centre + 2.0 * (torch.rand(1, 20, 2, generator=g, dtype=torch.float64) - 0.5)
    s = Structures(pos, (torch.arange(20) % 2)[None], torch.full((1, 2), 30.0, dtype=torch.float64))
    sigma = torch.tensor([0.2], dtype=torch.float64)
    theta = torch.tensor(0.7, dtype=torch.float64)
    rot = torch.stack(
        [torch.stack([theta.cos(), -theta.sin()]), torch.stack([theta.sin(), theta.cos()])]
    )
    out = model(s, sigma)
    out_r = model(s.with_pos((pos - centre) @ rot.T + centre), sigma)
    assert out.abs().max() > 1e-4
    assert torch.allclose(out_r, out @ rot.T, atol=1e-9)
