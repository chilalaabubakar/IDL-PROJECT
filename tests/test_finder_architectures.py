"""The other finders (docs/POINT_DEFECTS.md, "Other finders"): the symmetries they must have.

AtomTransformerClassifier sees only distances across the periodic box, so it is exactly
unchanged by any shift, by turning the box by 90 degrees, by reflecting it and by moving an
atom by a box vector, and renumbering the atoms renumbers its output. GridUNetClassifier
draws the glass on a grid: at evaluation it averages over the 8 symmetries of the square box,
which makes it exactly invariant to them. Its resolution is halved len(channels) - 1 times, so
only shifts by whole coarsest pixels change nothing; other shifts change it a little.
Run in float64 for tight tolerances.
"""

import pytest
import torch

from glassdiff.models.attention import AtomTransformerClassifier
from glassdiff.models.cnn import GridUNetClassifier, d4, rasterize, sample_at_atoms
from glassdiff.models.registry import build_model
from glassdiff.types import Structures

ROT90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)


@pytest.fixture
def s(make_lattice):
    return make_lattice(2, 8, jitter=0.15, seed=0)  # square box of side 7.3, 64 atoms


def transformer():
    torch.manual_seed(0)
    m = AtomTransformerClassifier(hidden=32, n_layers=2, n_heads=4, ffn=64, pair_dim=16)
    return m.double().eval()


def unet():
    torch.manual_seed(0)
    return GridUNetClassifier(grid=32, channels=(8, 8, 8), hidden=16).double().eval()


def reflect(s: Structures) -> Structures:
    return s.with_pos(torch.stack([-s.pos[..., 0], s.pos[..., 1]], -1))


@pytest.mark.parametrize("make", [transformer, unet])
def test_turning_reflecting_and_renumbering(make, s):
    model = make()
    out = model(s)
    assert out.shape == (2, 64, 5) and out.abs().max() > 1e-3  # not trivially 0 == 0
    assert torch.allclose(model(s.with_pos(s.pos @ ROT90.T)), out, atol=1e-8)
    assert torch.allclose(model(reflect(s)), out, atol=1e-8)
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(1))
    out_p = model(Structures(s.pos[:, perm], s.types[:, perm], s.box))
    assert torch.allclose(out_p, out[:, perm], atol=1e-8)


def test_transformer_ignores_any_shift_and_box_vectors(s):
    model = transformer()
    out = model(s)
    for shift in (torch.tensor([0.31, -2.7]), s.box[0] * torch.tensor([1.0, -2.0])):
        assert torch.allclose(model(s.with_pos(s.pos + shift.double())), out, atol=1e-9)
    assert torch.allclose(model(s.wrapped()), out, atol=1e-9)


def test_unet_ignores_shifts_by_whole_coarse_pixels_and_box_vectors(s):
    model = unet()  # grid 32, resolution halved twice: coarse pixels of 4 fine pixels
    out = model(s)
    pixel = s.box[0, 0] / 32
    shifts = (
        4 * pixel * torch.tensor([3.0, -5.0], dtype=torch.float64),
        s.box[0] * torch.tensor([1.0, -2.0]),
    )
    for shift in shifts:
        assert torch.allclose(model(s.with_pos(s.pos + shift.double())), out, atol=1e-8)
    # any other shift changes the output, but only a little
    for shift in (0.37 * pixel, 3 * pixel):
        nudged = model(s.with_pos(s.pos + shift))
        assert 0 < (nudged - out).abs().max() < 0.5 * out.abs().max()


def test_unet_drawing_and_reading_back(s):
    img = rasterize(s.pos, s.types, s.box, grid=32, blob=0.15, n_species=2)
    assert img.shape == (2, 2, 32, 32)
    # each atom contributes about the same total weight (it varies a little with where the
    # atom sits inside its pixel), so the two channels count the species
    per_atom = img.sum((1, 2, 3)) / 64
    assert torch.allclose(
        img.sum((2, 3)) / per_atom[:, None],
        (s.types[..., None] == torch.arange(2)).sum(1).double(),
        rtol=1e-3,
    )
    # a smooth field sampled at the atoms equals the field at their positions
    g = 32
    x = (torch.arange(g, dtype=torch.float64) + 0.5) / g
    field = torch.cos(2 * torch.pi * x)[:, None].expand(g, g)
    feat = field[None, None].expand(2, 1, g, g)
    got = sample_at_atoms(feat, s.pos, s.box)[..., 0]
    want = torch.cos(2 * torch.pi * torch.remainder(s.pos[..., 0], s.box[:, :1]) / s.box[:, :1])
    assert (got - want).abs().max() < 0.01


def test_d4_keeps_the_box_and_the_distances(s):
    from glassdiff.geometry import pair_vectors

    d0 = pair_vectors(s.pos, s.box).norm(dim=-1)
    for k in range(4):
        for f in (False, True):
            t = d4(s.pos, s.box, torch.full((2,), k), torch.full((2,), f))
            assert ((t >= 0) & (t < s.box[:, None, :])).all()
            assert torch.allclose(pair_vectors(t, s.box).norm(dim=-1), d0, atol=1e-9)


def test_unet_training_mode_augments():
    torch.manual_seed(0)
    model = build_model(dict(name="grid_unet", grid=32, channels=[8, 8, 8], hidden=16)).train()
    assert model.augment and model.tta
    from glassdiff.types import Structures as S

    pos = torch.rand(2, 20, 2) * 7.3
    s = S(pos, torch.randint(0, 2, (2, 20)), torch.full((2, 2), 7.3))
    a, b = model(s), model(s)  # a new random symmetry and shift every call
    assert torch.isfinite(a).all() and not torch.allclose(a, b)


def test_registry_builds_both():
    t = build_model(dict(name="atom_transformer"))
    u = build_model(dict(name="grid_unet"))
    for m in (t, u):
        assert m.n_classes == 5 and not m.sigma_aware
        assert 0.45e6 < sum(p.numel() for p in m.parameters()) < 0.6e6  # like the EGNN (0.52 M)
