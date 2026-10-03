"""Per-atom local descriptors.  [Ticket E-1]

Reference implementations (numpy/scipy) are in docs/pilot/pilot_ka2d.py. Acceptance:
tests/test_descriptors.py, plus reproducing the pilot statistics on the 12 pilot glasses.
"""

from __future__ import annotations

from torch import Tensor

from glassdiff.types import Structures


def cutoff_coordination(s: Structures, factor: float = 1.4) -> Tensor:
    """[B, N] long: neighbours within factor * sigma_ab (approx. first minimum of g_ab(r))."""
    raise NotImplementedError("Ticket E-1")


def same_species_neighbors(s: Structures, species: int = 1, rc: float = 1.2) -> Tensor:
    """[B, N] long: neighbours of the same species within rc (zero for other species)."""
    raise NotImplementedError("Ticket E-1")


def voronoi_coordination(s: Structures) -> Tensor:
    """[B, N] long: Voronoi (Delaunay) neighbour count with periodic images (scipy, CPU)."""
    raise NotImplementedError("Ticket E-1")


def psi6_local(s: Structures) -> Tensor:
    """[B, N] float: |psi_6| over Voronoi neighbours."""
    raise NotImplementedError("Ticket E-1")


def psi6_global(s: Structures) -> Tensor:
    """[B] float: |mean over atoms of complex psi_6|; crystallization gate (< 0.3)."""
    raise NotImplementedError("Ticket E-1")


def delaunay_voids(s: Structures) -> list[tuple[Tensor, Tensor]]:
    """Per structure: (circumcentres [T, 2], circumradii [T]) of Delaunay triangles (D3)."""
    raise NotImplementedError("Ticket E-1")
