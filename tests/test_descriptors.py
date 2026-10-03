"""Acceptance tests for Ticket E-1 (local descriptors)."""

import pytest
import torch

from glassdiff.analysis.descriptors import (
    cutoff_coordination,
    psi6_global,
    psi6_local,
    same_species_neighbors,
    voronoi_coordination,
)
from glassdiff.types import Structures

pytestmark = pytest.mark.skip(reason="Ticket E-1: delete this line when implementing")


def test_triangular_lattice(make_triangular):
    s = make_triangular(8, 8, a=1.1)  # all A; 2nd neighbours at 1.905 > 1.4 cutoff
    assert (cutoff_coordination(s) == 6).all()
    assert (voronoi_coordination(s) == 6).all()
    assert torch.allclose(psi6_local(s), torch.ones(1, 64, dtype=s.pos.dtype), atol=1e-6)
    assert psi6_global(s).item() == pytest.approx(1.0, abs=1e-6)


def test_same_species_neighbors():
    pos = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [6.0, 6.0], [1.5, 1.8]]], dtype=torch.float64)
    types = torch.tensor([[1, 1, 1, 0]])
    s = Structures(pos, types, torch.tensor([[10.0, 10.0]], dtype=torch.float64))
    assert same_species_neighbors(s, species=1, rc=1.2).tolist() == [[1, 1, 0, 0]]
