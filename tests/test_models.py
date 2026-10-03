"""Tests for Tickets M-5 (MPNN) and M-6 (flat MLP): shapes, pins and the symmetries each
model is supposed to have (and, for the ladder, the ones it is supposed to lack)."""

import torch

from glassdiff.models.mlp import FlatMLP
from glassdiff.models.mpnn import MPNN
from glassdiff.models.registry import build_model
from glassdiff.types import Structures, defect_token

ROT90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)


def _inputs(make_lattice):
    s = make_lattice(2, 8, jitter=0.15, seed=0)
    sigma = torch.tensor([0.1, 0.3], dtype=torch.float64)
    pin = torch.zeros(2, 64, dtype=torch.bool)
    pin[:, 3] = True
    label = torch.zeros(2, 64, dtype=torch.long)
    label[:, 3] = defect_token(1)
    return s, sigma, pin, label


def test_mpnn_shapes_translation_and_permutation(make_lattice):
    torch.manual_seed(0)
    model = MPNN(hidden=32, n_layers=2).double().eval()
    s, sigma, pin, label = _inputs(make_lattice)
    out = model(s, sigma, pin, label)
    assert out.shape == (2, 64, 2) and (out[:, 3] == 0).all() and out.abs().max() > 1e-4
    shifted = model(s.with_pos(s.pos + torch.tensor([0.4, -3.1], dtype=torch.float64)), sigma)
    assert torch.allclose(model(s, sigma), shifted, atol=1e-9)
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(1))
    out_p = model(Structures(s.pos[:, perm], s.types[:, perm], s.box), sigma)
    assert torch.allclose(out_p, model(s, sigma)[:, perm], atol=1e-9)
    rotated = model(s.with_pos(s.pos @ ROT90.T), sigma)
    assert not torch.allclose(rotated, model(s, sigma) @ ROT90.T, atol=1e-6)  # no rotation symmetry


def test_flat_mlp_shapes_and_order_handling(make_lattice):
    torch.manual_seed(0)
    model = FlatMLP(n_atoms=64, hidden=64, n_layers=2).double().eval()
    s, sigma, pin, label = _inputs(make_lattice)
    out = model(s, sigma, pin, label)
    assert out.shape == (2, 64, 2) and (out[:, 3] == 0).all()
    # periodic inputs: shifting by a whole box changes nothing
    assert torch.allclose(model(s.with_pos(s.pos + s.box[:, None, :]), sigma), model(s, sigma))


def test_registry_builds_all_models():
    for cfg in (
        {"name": "egnn_pbc", "hidden": 16, "n_layers": 1},
        {"name": "mpnn", "hidden": 16, "n_layers": 1, "rotation_augmentation": True},
        {"name": "flat_mlp", "n_atoms": 64, "hidden": 32, "n_layers": 1},
    ):
        assert build_model(cfg) is not None
