"""Acceptance tests for Ticket M-2 (EGNN-PBC symmetries). Run in float64 for tight tolerances."""

import pytest
import torch

from glassdiff.models.egnn_pbc import EGNNPBC
from glassdiff.types import Structures, defect_token

pytestmark = pytest.mark.skip(reason="Ticket M-2: delete this line when implementing")

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
    assert model(s, sigma, pin, label).shape == (2, 64, 2)


def test_translation_invariance_including_wrap(model, s):
    sigma = torch.tensor([0.2, 0.2], dtype=torch.float64)
    out = model(s, sigma)
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
