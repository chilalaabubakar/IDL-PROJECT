"""Per-atom local descriptors.  [Ticket E-1]

Distance-based descriptors are batched torch (dense minimum-image pairs). Voronoi-based ones
use scipy's Delaunay triangulation on a 3x3 periodic tiling, on the CPU, one structure at a
time. Acceptance: tests/test_descriptors.py.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import Delaunay
from torch import Tensor

from glassdiff.geometry import pair_vectors
from glassdiff.physics.ka_potential import KA_SIGMA
from glassdiff.types import Structures


def _pair_distances(s: Structures) -> Tensor:
    """[B, N, N] minimum-image distances with +inf on the diagonal."""
    dist = pair_vectors(s.pos, s.box).norm(dim=-1)
    eye = torch.eye(s.n_atoms, dtype=torch.bool, device=s.pos.device)
    return dist.masked_fill(eye, float("inf"))


def cutoff_coordination(s: Structures, factor: float = 1.4) -> Tensor:
    """[B, N] long: neighbours within factor * sigma_ab (approx. first minimum of g_ab(r))."""
    sigma = torch.tensor(KA_SIGMA, dtype=s.pos.dtype, device=s.pos.device)
    rc = factor * sigma[s.types[:, :, None], s.types[:, None, :]]
    return (_pair_distances(s) < rc).sum(-1)


def same_species_neighbors(s: Structures, species: int = 1, rc: float = 1.2) -> Tensor:
    """[B, N] long: neighbours of the same species within rc (zero for other species)."""
    is_sp = s.types == species
    count = ((_pair_distances(s) < rc) & is_sp[:, None, :]).sum(-1)
    return count * is_sp


def _tiled_delaunay(pos: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, Delaunay]:
    """Delaunay triangulation of the 3x3 periodic tiling; indices 0..N-1 are the central copy."""
    pos = np.mod(pos, box)
    shifts = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
    tiled = np.concatenate([pos] + [pos + np.array(sh) * box for sh in shifts])
    return tiled, Delaunay(tiled)


def voronoi_neighbors(s: Structures) -> list[list[np.ndarray]]:
    """Per structure, per atom: [k, 2] vectors to its Voronoi (Delaunay) neighbours."""
    out = []
    for b in range(s.batch_size):
        pos = s.pos[b].detach().cpu().double().numpy()
        box = s.box[b].detach().cpu().double().numpy()
        tiled, tri = _tiled_delaunay(pos, box)
        indptr, indices = tri.vertex_neighbor_vertices
        out.append([tiled[indices[indptr[i] : indptr[i + 1]]] - tiled[i] for i in range(len(pos))])
    return out


def voronoi_coordination(s: Structures) -> Tensor:
    """[B, N] long: Voronoi (Delaunay) neighbour count with periodic images (scipy, CPU)."""
    counts = [[len(v) for v in per_atom] for per_atom in voronoi_neighbors(s)]
    return torch.tensor(counts, dtype=torch.long, device=s.pos.device)


def _psi6_complex(s: Structures) -> np.ndarray:
    """[B, N] complex: mean over Voronoi neighbours of exp(6 i theta_ij)."""
    out = []
    for per_atom in voronoi_neighbors(s):
        out.append([np.exp(6j * np.arctan2(v[:, 1], v[:, 0])).mean() for v in per_atom])
    return np.array(out)


def psi6_local(s: Structures) -> Tensor:
    """[B, N] float: |psi_6| over Voronoi neighbours."""
    return torch.tensor(np.abs(_psi6_complex(s)), dtype=s.pos.dtype, device=s.pos.device)


def psi6_global(s: Structures) -> Tensor:
    """[B] float: |mean over atoms of complex psi_6|; crystallization gate (< 0.3)."""
    val = np.abs(_psi6_complex(s).mean(axis=1))
    return torch.tensor(val, dtype=s.pos.dtype, device=s.pos.device)


def delaunay_voids(s: Structures) -> list[tuple[Tensor, Tensor]]:
    """Per structure: (circumcentres [T, 2], circumradii [T]) of Delaunay triangles (D3).

    Only triangles whose circumcentre lies in the central box are kept, so each void of the
    periodic structure is counted once.
    """
    out = []
    for b in range(s.batch_size):
        box = s.box[b].detach().cpu().double().numpy()
        tiled, tri = _tiled_delaunay(s.pos[b].detach().cpu().double().numpy(), box)
        p = tiled[tri.simplices]
        a, bb, c = p[:, 0], p[:, 1], p[:, 2]
        d = 2 * (
            a[:, 0] * (bb[:, 1] - c[:, 1])
            + bb[:, 0] * (c[:, 1] - a[:, 1])
            + c[:, 0] * (a[:, 1] - bb[:, 1])
        )
        sa, sb, sc = (a**2).sum(1), (bb**2).sum(1), (c**2).sum(1)
        ux = (sa * (bb[:, 1] - c[:, 1]) + sb * (c[:, 1] - a[:, 1]) + sc * (a[:, 1] - bb[:, 1])) / d
        uy = (sa * (c[:, 0] - bb[:, 0]) + sb * (a[:, 0] - c[:, 0]) + sc * (bb[:, 0] - a[:, 0])) / d
        r = np.hypot(a[:, 0] - ux, a[:, 1] - uy)
        keep = (ux >= 0) & (ux < box[0]) & (uy >= 0) & (uy < box[1])
        centres = torch.tensor(np.stack([ux[keep], uy[keep]], 1), dtype=s.pos.dtype)
        out.append((centres, torch.tensor(r[keep], dtype=s.pos.dtype)))
    return out
