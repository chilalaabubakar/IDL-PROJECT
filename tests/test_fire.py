"""Acceptance tests for Ticket P-3 (batched FIRE relaxation)."""

import torch

from glassdiff.physics.fire import fire_minimize
from glassdiff.physics.ka_potential import ka_energy


def test_relaxation_lowers_energy_and_converges(make_lattice):
    s = make_lattice(3, 8, jitter=0.1, seed=1)
    res = fire_minimize(s, fmax=1e-6)
    assert res.converged.all()
    assert (res.max_force <= 1e-6).all()
    assert (ka_energy(res.structures) < ka_energy(s)).all()


def test_frozen_atoms_do_not_move(make_lattice):
    s = make_lattice(2, 8, jitter=0.1, seed=2)
    frozen = torch.zeros(2, 64, dtype=torch.bool)
    frozen[:, :20] = True
    res = fire_minimize(s, frozen=frozen, fmax=1e-6)
    assert (res.displacement[frozen] == 0).all()
    assert torch.equal(res.structures.pos[frozen], s.pos[frozen])


def test_batch_members_are_independent(make_lattice):
    s = make_lattice(2, 8, jitter=0.1, seed=3)
    both = fire_minimize(s, fmax=1e-6).structures.pos
    first = fire_minimize(type(s)(s.pos[:1], s.types[:1], s.box[:1]), fmax=1e-6).structures.pos
    assert torch.allclose(both[:1], first, atol=1e-6)


def test_neighbor_list_forces_match_dense(make_lattice):
    from glassdiff.geometry import neighbor_list
    from glassdiff.physics.ka_potential import ka_forces, ka_forces_neighbors

    s = make_lattice(2, 8, jitter=0.12, seed=4)
    nl = neighbor_list(s.pos, s.box, 2.8, 64)
    assert torch.allclose(ka_forces_neighbors(s, nl.idx, nl.mask), ka_forces(s), atol=1e-10)


def test_neighbor_list_and_dense_relaxation_agree(make_lattice):
    s = make_lattice(2, 8, jitter=0.1, seed=5)
    dense = fire_minimize(s, fmax=1e-8, skin=None)
    sparse = fire_minimize(s, fmax=1e-8, skin=0.3)
    assert torch.allclose(dense.structures.pos, sparse.structures.pos, atol=1e-7)
