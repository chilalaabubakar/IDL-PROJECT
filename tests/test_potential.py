"""Acceptance tests for Ticket P-2 (Kob–Andersen potential)."""

from pathlib import Path

import pytest
import torch

from glassdiff.physics.ka_potential import ka_energy, ka_forces, ka_virial
from glassdiff.types import Structures

DATA = Path("data/ka2d_256/test.npz")


def lj_shifted(r, sigma, eps, rc_factor=2.5):
    def lj(x):
        return 4 * eps * ((sigma / x) ** 12 - (sigma / x) ** 6)

    return lj(r) - lj(rc_factor * sigma) if r < rc_factor * sigma else 0.0


@pytest.mark.parametrize("ta,tb,sigma,eps", [(0, 0, 1.0, 1.0), (0, 1, 0.8, 1.5), (1, 1, 0.88, 0.5)])
@pytest.mark.parametrize("r", [0.75, 0.95, 1.3, 2.1, 3.0])
def test_pair_energy_matches_analytic(ta, tb, sigma, eps, r):
    box = torch.tensor([[20.0, 20.0]], dtype=torch.float64)
    pos = torch.tensor([[[5.0, 5.0], [5.0 + r, 5.0]]], dtype=torch.float64)
    s = Structures(pos, torch.tensor([[ta, tb]]), box)
    expected = lj_shifted(r, sigma, eps)
    assert ka_energy(s).item() == pytest.approx(expected, abs=1e-8)
    per_atom = ka_energy(s, per_atom=True)
    assert per_atom[0, 0].item() == pytest.approx(expected / 2, abs=1e-8)


def test_pair_across_periodic_boundary():
    box = torch.tensor([[10.0, 10.0]], dtype=torch.float64)
    pos = torch.tensor([[[0.4, 3.0], [9.4, 3.0]]], dtype=torch.float64)  # distance 1.0 via wrap
    s = Structures(pos, torch.zeros(1, 2, dtype=torch.long), box)
    assert ka_energy(s).item() == pytest.approx(lj_shifted(1.0, 1.0, 1.0), abs=1e-8)


def test_forces_match_finite_differences(make_lattice):
    s = make_lattice(1, 8, jitter=0.08, seed=0)  # float64
    f = ka_forces(s)
    h = 1e-6
    for atom, axis in [(0, 0), (5, 1), (37, 0), (63, 1)]:
        plus, minus = s.pos.clone(), s.pos.clone()
        plus[0, atom, axis] += h
        minus[0, atom, axis] -= h
        de = ka_energy(s.with_pos(plus)) - ka_energy(s.with_pos(minus))
        assert f[0, atom, axis].item() == pytest.approx(-(de / (2 * h)).item(), rel=1e-5, abs=1e-6)


def test_virial_gives_the_pressure(make_lattice):
    """The per-atom virials sum to d V P, with P = -dE/dV from scaling box and positions."""
    s = make_lattice(2, 8, jitter=0.1, seed=3)
    area = s.box.prod(-1)
    h = 1e-6
    up = ka_energy(Structures(s.pos * (1 + h), s.types, s.box * (1 + h)))
    down = ka_energy(Structures(s.pos * (1 - h), s.types, s.box * (1 - h)))
    pressure = -(up - down) / (area * ((1 + h) ** 2 - (1 - h) ** 2))
    w = ka_virial(s)
    assert w.shape == s.types.shape
    assert torch.allclose(w.sum(-1) / (2 * area), pressure, rtol=1e-6, atol=1e-8)
    # two atoms closer than the minimum of the potential push each other apart: positive
    box = torch.tensor([[20.0, 20.0]], dtype=torch.float64)
    pair = torch.tensor([[[5.0, 5.0], [5.9, 5.0]]], dtype=torch.float64)
    close = ka_virial(Structures(pair, torch.zeros(1, 2, dtype=torch.long), box))
    assert (close > 0).all() and close[0, 0] == close[0, 1]
    pair[0, 1, 0] = 6.5  # beyond the minimum (1.12 sigma): they attract, negative
    assert (ka_virial(Structures(pair, torch.zeros(1, 2, dtype=torch.long), box)) < 0).all()


def test_per_atom_energy_matches_lammps_dataset():
    """The dataset stores LAMMPS `compute pe/atom` for every glass (Ticket P-1)."""
    if not DATA.exists():
        pytest.skip("dataset not generated")
    from glassdiff.data.dataset import load_split

    split = load_split(DATA)
    s = split.structures
    s = Structures(s.pos[:10].double(), s.types[:10], s.box[:10].double())
    ours = ka_energy(s, per_atom=True)
    assert torch.allclose(ours, split.pe_atom[:10].double(), atol=1e-5)
